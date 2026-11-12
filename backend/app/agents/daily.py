"""Daily review: is the time each person reported proportionate to the work they describe?

Plane sends every row of the day's reports with the evidence it recorded. Rows the numbers already settle (well
documented, recorded activity, normal length) get a rule verdict and cost nothing. The rest go to the model, one
call per person with all their open rows, which judges each row: plausible, inflated, vague or unrelated, and
how long the work would normally take. The person's verdict is then computed from the rows, not guessed.
"""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field

from app.llm.effort import Effort, EffortProfile
from app.llm.generate import GenerationError, Generator, Usage, current_trace
from app.llm.structured import generate_structured
from app.llm.tokens import truncate_to_tokens
from app.prompts import DAILY_REVIEW

logger = logging.getLogger(__name__)

NOTE_MIN_CHARS = 20
LONG_ROW_MINUTES = 240
# Rows per person sent to the model (the longest first): keeps one call inside a small context.
MAX_ROWS = {Effort.HIGH: 14, Effort.MEDIUM: 10, Effort.LOW: 6}


class RowVerdict(str, Enum):
    PLAUSIBLE = "plausible"
    INFLATED = "inflated"
    VAGUE = "vague"
    UNRELATED = "unrelated"
    UNCHECKED = "unchecked"


class Evidence(BaseModel):
    activity: bool = False
    document: bool = False
    note: bool = False


class DayRow(BaseModel):
    id: str
    key: str = ""
    task: str
    description: str = ""
    project: str = ""
    minutes: int = Field(ge=0, le=24 * 60)
    note: str = ""
    finished: bool = False
    kind: str = "issue"  # issue | micro
    is_meeting: bool = False
    evidence: Evidence = Field(default_factory=Evidence)
    done_but_open: bool = False


class DayPerson(BaseModel):
    user_id: str
    name: str
    rows: list[DayRow] = Field(default_factory=list, max_length=100)


class ModelRow(BaseModel):
    n: int
    verdict: RowVerdict
    expected_minutes: int = Field(ge=0, le=24 * 60)
    reason: str = Field(max_length=400)
    question: str = Field(default="", max_length=300)


class ModelReview(BaseModel):
    rows: list[ModelRow] = Field(default_factory=list)
    summary: str = Field(default="", max_length=600)


def _has_note(row: DayRow) -> bool:
    return len(row.note.strip()) >= NOTE_MIN_CHARS


def rule_check(row: DayRow) -> tuple[RowVerdict | None, str]:
    """A verdict when the numbers settle it, else `None` (the model should look)."""
    if row.kind == "micro" and row.is_meeting and row.minutes <= 180:
        return RowVerdict.PLAUSIBLE, "A meeting of normal length."
    documented = row.evidence.activity and (row.evidence.document or _has_note(row))
    if documented and row.minutes < LONG_ROW_MINUTES and not row.done_but_open:
        return RowVerdict.PLAUSIBLE, "Recorded activity and a written result back the time."
    if row.minutes <= 30 and (_has_note(row) or row.evidence.activity):
        return RowVerdict.PLAUSIBLE, "A short row with a trace."
    return None, ""


def rule_fallback(row: DayRow) -> tuple[RowVerdict, str]:
    """Used when the model is not asked (low effort) or not answering."""
    if not _has_note(row) and not row.evidence.document:
        return RowVerdict.VAGUE, "The note does not say what was done and there is no document."
    return RowVerdict.UNCHECKED, "Not checked by the model; look at the note."


def _row_text(n: int, row: DayRow) -> str:
    recorded = ", ".join(
        label
        for label, on in (
            ("activity on the task", row.evidence.activity),
            ("a document", row.evidence.document),
        )
        if on
    ) or "nothing recorded"
    parts = [
        f"[{n}] {row.key + ' ' if row.key else ''}{row.task}"
        + (f" ({row.project})" if row.project else "")
        + (" - meeting" if row.is_meeting else ""),
        f"    reported: {row.minutes} min{' - marked done' if row.finished else ''}"
        + (" - but the task is still open" if row.done_but_open else ""),
        f"    note: {row.note.strip() or '(none)'}",
        f"    recorded: {recorded}",
    ]
    if row.description.strip():
        parts.append(f"    task description: {row.description.strip()[:400]}")
    return "\n".join(parts)


def person_verdict(rows: list[dict], total: int) -> tuple[str, int]:
    """`ok`, `check` or `concern`, and the minutes in question (inflated beyond the estimate, plus vague rows)."""
    questioned = 0
    for row in rows:
        if row["verdict"] == RowVerdict.INFLATED.value:
            questioned += max(0, row["minutes"] - (row["expected_minutes"] or 0))
        elif row["verdict"] in (RowVerdict.VAGUE.value, RowVerdict.UNRELATED.value):
            questioned += row["minutes"]
    if total and questioned >= total * 0.5:
        return "concern", questioned
    if questioned >= 60:
        return "check", questioned
    return "ok", questioned


async def review_person(
    generator: Generator, profile: EffortProfile, person: DayPerson, day: str, usage: Usage
) -> dict:
    total = sum(r.minutes for r in person.rows)
    results: dict[str, dict] = {}
    open_rows: list[DayRow] = []
    for row in person.rows:
        verdict, reason = rule_check(row)
        if verdict is None:
            open_rows.append(row)
        else:
            results[row.id] = {"verdict": verdict.value, "expected_minutes": None, "reason": reason, "question": "", "source": "rules"}

    # At low effort the model sees only the doubtful rows; at medium and high all of them (but the longest first).
    if profile.effort is Effort.LOW:
        to_model = open_rows
    else:
        to_model = open_rows + [r for r in person.rows if r not in open_rows and r.minutes >= 60]
    to_model = sorted(to_model, key=lambda r: -r.minutes)[: MAX_ROWS[profile.effort]]
    summary, summary_source = "", "rules"
    trace_index = None

    if to_model:
        text = "\n".join(_row_text(n, row) for n, row in enumerate(to_model, start=1))
        messages = DAILY_REVIEW.render(
            name=person.name,
            day=day,
            total_minutes=total,
            rows=truncate_to_tokens(text, profile.context_tokens - profile.max_answer_tokens - 400),
        )
        trace = current_trace.get()
        try:
            # The position of this person's prompt among the job's recorded calls (for ratings / training data).
            trace_at = len(trace) if trace is not None else None
            review = await generate_structured(generator, messages, ModelReview, profile, usage)
            trace_index = (len(trace) - 1) if trace is not None and trace_at is not None and len(trace) > trace_at else None
            by_n = {r.n: r for r in review.rows}
            for n, row in enumerate(to_model, start=1):
                got = by_n.get(n)
                if got is None:
                    continue
                expected = min(got.expected_minutes, 24 * 60)
                verdict = got.verdict
                # A model "inflated" with an estimate at or above the reported time is not inflated.
                if verdict is RowVerdict.INFLATED and expected >= row.minutes:
                    verdict = RowVerdict.PLAUSIBLE
                results[row.id] = {
                    "verdict": verdict.value,
                    "expected_minutes": expected,
                    "reason": got.reason,
                    "question": got.question,
                    "source": "ai",
                }
            summary, summary_source = review.summary, "ai"
        except GenerationError as exc:
            logger.warning("daily review of %s fell back to rules: %s", person.user_id, exc)

    rows = []
    for row in person.rows:
        found = results.get(row.id)
        if found is None:
            verdict, reason = rule_fallback(row)
            found = {"verdict": verdict.value, "expected_minutes": None, "reason": reason, "question": "", "source": "rules"}
        rows.append({"id": row.id, "key": row.key, "task": row.task, "minutes": row.minutes, "note": row.note, **found})

    verdict, questioned = person_verdict(rows, total)
    if not summary:
        flagged = [r for r in rows if r["verdict"] in ("inflated", "vague", "unrelated")]
        summary = (
            "Every row looks proportionate to the work described."
            if not flagged
            else f"{len(flagged)} of {len(rows)} rows need a closer look ({questioned} min in question)."
        )
    return {
        "user_id": person.user_id,
        "name": person.name,
        "total_minutes": total,
        "questioned_minutes": questioned,
        "verdict": verdict,
        "summary": summary,
        "summary_source": summary_source,
        "trace": trace_index,
        "rows": rows,
    }


async def review_day(generator: Generator, profile: EffortProfile, day: str, people: list[DayPerson], progress) -> dict:
    usage = Usage()
    results = []
    for index, person in enumerate(people, start=1):
        progress(f"{index} / {len(people)} people")
        if not person.rows:
            continue
        results.append(await review_person(generator, profile, person, day, usage))
    order = {"concern": 0, "check": 1, "ok": 2}
    results.sort(key=lambda p: (order[p["verdict"]], -p["questioned_minutes"]))
    return {
        "day": day,
        "people": results,
        "team": {
            "people_reviewed": len(results),
            "concern": sum(1 for p in results if p["verdict"] == "concern"),
            "check": sum(1 for p in results if p["verdict"] == "check"),
            "reported_minutes": sum(p["total_minutes"] for p in results),
            "questioned_minutes": sum(p["questioned_minutes"] for p in results),
        },
        "usage": {"prompt_tokens": usage.prompt_tokens, "answer_tokens": usage.answer_tokens},
    }
