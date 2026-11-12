"""Ask and Answer for admins: questions about the team, answered only from Plane's data.

1. Route (mixture of experts at the system level): every data section Plane sends is an "expert" with its own
   instructions. A cheap lexical router scores the sections against the question; at medium and high effort the
   model also picks. Only the chosen sections go into the prompt, so a small context holds what matters.
2. Answer from those sections.
3. Verify: every number in the answer must appear in the data (or the question). At medium and high effort an
   answer with invented numbers is rewritten once; the result says whether it passed.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from app.llm.effort import Effort, EffortProfile
from app.llm.generate import GenerationError, Generator, Usage
from app.llm.structured import generate_structured
from app.llm.tokens import count_tokens, input_budget, truncate_to_tokens
from app.prompts import ASK, ASK_FIX, ROUTER
from app.search.ranking import BM25

logger = logging.getLogger(__name__)

# The experts: what each section is good for (used by both routers) and how to read it.
EXPERTS: dict[str, tuple[str, str]] = {
    "person": (
        "everything about the people named in the question: hours, tasks, quality, notes, evidence, rows",
        "Answer about these people from their own data first.",
    ),
    "people": (
        "hours reported per person, days, reports sent, missing or late, overtime, who worked how much, most, least "
        "ساعت کار کارکرد بیشترین کمترین گزارش چند ساعت چه کسی",
        "Read per-person totals; compare people only on the same measure.",
    ),
    "quality": (
        "quality of work, finished tasks, on time, late, deadline, reopened, rework, estimates, overrun, focus, "
        "meetings, documents, comments, trend, compared with before, improving, worse, performance "
        "کیفیت عملکرد تاخیر دیر به‌موقع تحویل برگشت دوباره باز تخمین برآورد جلسه مستند کامنت روند بهتر بدتر",
        "Quality notes show where to look; a late or reopened task can have a good reason. Say which numbers "
        "support each point.",
    ),
    "daily_notes": (
        "what each person wrote in their Day Reports: tasks, minutes, notes, what they did on a day, research, config "
        "یادداشت گزارش روزانه چه کار کرد تحقیق کانفیگ تنظیم توضیح امروز دیروز",
        "Quote what people wrote when it matters; judge time against the work described.",
    ),
    "evidence": (
        "whether reported work has traces: activity, documents, tasks marked done but still open, real work, fake "
        "واقعا کار کرده شواهد ردپا ثبت فعالیت الکی دروغ واقعی",
        "Low evidence is a reason to ask, not proof.",
    ),
    "projects": (
        "time per project per person, where the time went, which project, split "
        "پروژه زمان روی کدام پروژه تقسیم",
        "",
    ),
    "risks": (
        "open tasks at risk: overdue, blocked, stalled, unassigned, due soon, deadline "
        "ریسک عقب افتاده مسدود گیر متوقف بدون مسئول",
        "Name the task keys.",
    ),
    "workload": (
        "open tasks per person, planned time left against capacity, overloaded, free, busy "
        "حجم کار سرش شلوغ آزاد بار ظرفیت",
        "",
    ),
    "docs": ("work item descriptions and documents matching the question, how something works, specs", "Cite titles."),
}


class Section(BaseModel):
    title: str = ""
    text: str = Field(default="", max_length=200_000)


class RouterChoice(BaseModel):
    sections: list[str] = Field(default_factory=list, max_length=3)


_NUMBER = re.compile(r"\d+(?:[.,/:]\d+)*")
_PERSIAN = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", ".") for m in _NUMBER.finditer(text.translate(_PERSIAN))}


def unsupported_numbers(answer: str, data: str, question: str) -> list[str]:
    """Numbers in the answer that are neither in the data nor in the question. Small counts (0-10) are allowed:
    the model may count rows it was given."""
    known = numbers(data) | numbers(question)
    # A number written "8.0" in the data may be "8" in the answer, and the other way round.
    known |= {n[:-2] for n in known if n.endswith(".0")}
    bad = []
    for n in sorted(numbers(answer)):
        if n in known:
            continue
        try:
            if float(n) <= 10:
                continue
        except ValueError:
            pass
        bad.append(n)
    return bad


def lexical_route(question: str, available: list[str], limit: int = 3) -> list[str]:
    docs = [f"{name} {EXPERTS.get(name, ('', ''))[0]}" for name in available]
    scores = BM25(docs).scores(question)
    ranked = [name for name, score in sorted(zip(available, scores), key=lambda p: -p[1]) if score > 0]
    # Nothing matched: the per-person totals answer most general questions.
    if not ranked:
        ranked = [name for name in ("people", "quality", "daily_notes") if name in available]
    return ranked[:limit]


async def route(generator: Generator, profile: EffortProfile, question: str, available: list[str], usage: Usage) -> tuple[list[str], str]:
    lexical = lexical_route(question, available)
    # A question about named people: their own data always comes first (it is the most precise context).
    if "person" in available:
        lexical = ["person", *[name for name in lexical if name != "person"]][:3]
    if profile.effort is Effort.LOW:
        return lexical, "lexical"
    described = "; ".join(f"{name} = {EXPERTS.get(name, ('', ''))[0]}" for name in available)
    try:
        choice = await generate_structured(
            generator, ROUTER.render(sections=described, question=question), RouterChoice, profile, usage
        )
        picked = [name for name in choice.sections if name in available]
    except GenerationError:
        picked = []
    # The model's picks first, then the lexical ones, at most three.
    merged = list(dict.fromkeys((["person"] if "person" in available else []) + picked + lexical))[:3]
    return merged, "model" if picked else "lexical"


def build_data(sections: dict[str, Section], chosen: list[str], budget: int) -> str:
    parts = []
    # The person section is the question's subject: it gets most of the room.
    shares = {name: max(200, budget // max(1, len(chosen))) for name in chosen}
    if chosen and chosen[0] == "person" and len(chosen) > 1:
        shares["person"] = budget * 3 // 5
        rest = max(200, (budget - shares["person"]) // (len(chosen) - 1))
        shares.update({name: rest for name in chosen[1:]})
    for name in chosen:
        section = sections[name]
        title = section.title or name
        parts.append(f"## {title}\n{truncate_to_tokens(section.text.strip(), shares[name])}")
    return "\n\n".join(parts)


async def answer_question(
    generator: Generator, profile: EffortProfile, question: str, period: str, sections: dict[str, Section]
) -> dict:
    usage = Usage()
    available = [name for name, s in sections.items() if s.text.strip()]
    if not available:
        return {"answer": "There is no data for this period to answer from.", "sections": [], "verified": True,
                "unverified_numbers": [], "router": "none", "usage": {"prompt_tokens": 0, "answer_tokens": 0}}
    chosen, router = await route(generator, profile, question, available, usage)
    expert = " ".join(EXPERTS[name][1] for name in chosen if name in EXPERTS and EXPERTS[name][1])
    budget = input_budget(profile.context_tokens, profile.max_answer_tokens) - count_tokens(question) - 300
    data = build_data(sections, chosen, budget)

    result = await generator.complete(ASK.render(expert=expert, period=period, data=data, question=question), profile)
    usage.prompt_tokens += result.usage.prompt_tokens
    usage.answer_tokens += result.usage.answer_tokens
    answer = result.text.strip()

    bad = unsupported_numbers(answer, data, question)
    if bad and profile.effort is not Effort.LOW:
        try:
            fixed = await generator.complete(
                ASK_FIX.render(bad=", ".join(bad), data=data, question=question, answer=answer), profile
            )
            usage.prompt_tokens += fixed.usage.prompt_tokens
            usage.answer_tokens += fixed.usage.answer_tokens
            if fixed.text.strip():
                answer = fixed.text.strip()
                bad = unsupported_numbers(answer, data, question)
        except GenerationError as exc:
            logger.warning("answer verification skipped: %s", exc)

    return {
        "answer": answer,
        "sections": chosen,
        "router": router,
        "verified": not bad,
        "unverified_numbers": bad,
        "usage": {"prompt_tokens": usage.prompt_tokens, "answer_tokens": usage.answer_tokens},
    }
