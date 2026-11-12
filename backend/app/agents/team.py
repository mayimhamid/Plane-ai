"""Team analysis: a supervisor splits the team into per-person workers, then combines their reviews.

Plane computes the evidence (numbers and flags) deterministically; nothing here guesses facts. The model only
reads those facts and explains them. At low effort the model is asked only about people whose score is
unclear; clear cases get a rule-based explanation, which costs nothing.

    supervisor
      |- worker(person 1) -> review
      |- worker(person 2) -> review   (sequential: one generation at a time on a small machine)
      |- ...
      '- combine(reviews) -> team summary
"""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field

from app.llm.effort import Effort, EffortProfile
from app.llm.generate import GenerationError, Generator, Usage
from app.llm.structured import generate_structured
from app.llm.tokens import truncate_to_tokens
from app.prompts import MEMBER_REVIEW, TEAM_REVIEW

logger = logging.getLogger(__name__)

CLEAR_HIGH = 80
CLEAR_LOW = 20


class Verdict(str, Enum):
    WELL = "well_supported"
    PARTLY = "partly_supported"
    WEAKLY = "weakly_supported"
    NO_DATA = "no_data"


class MemberEvidence(BaseModel):
    user_id: str
    name: str
    score: int = Field(ge=0, le=100)
    facts: dict[str, float | int | str] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)


class MemberReview(BaseModel):
    verdict: Verdict
    explanation: str = Field(max_length=1200)
    questions: list[str] = Field(default_factory=list, max_length=3)


class TeamSummary(BaseModel):
    bullets: list[str] = Field(default_factory=list, max_length=8)


def rule_verdict(score: int, has_data: bool) -> Verdict:
    if not has_data:
        return Verdict.NO_DATA
    if score >= 70:
        return Verdict.WELL
    if score >= 40:
        return Verdict.PARTLY
    return Verdict.WEAKLY


def _has_data(member: MemberEvidence) -> bool:
    return bool(member.facts.get("reported_minutes") or member.facts.get("reported_tasks"))


def rule_review(member: MemberEvidence) -> MemberReview:
    verdict = rule_verdict(member.score, _has_data(member))
    if verdict is Verdict.NO_DATA:
        return MemberReview(verdict=verdict, explanation="Nothing was reported in this period.", questions=[])
    if not member.flags:
        text = f"Evidence score {member.score}/100. The reported work matches recorded activity; nothing stands out."
        return MemberReview(verdict=verdict, explanation=text, questions=[])
    text = f"Evidence score {member.score}/100. " + " ".join(f"{flag}." for flag in member.flags[:4])
    return MemberReview(
        verdict=verdict,
        explanation=text,
        questions=["Can you walk me through the work behind the flagged items?"],
    )


def _facts_text(member: MemberEvidence) -> str:
    return "\n".join(f"- {key.replace('_', ' ')}: {value}" for key, value in member.facts.items())


async def review_member(
    generator: Generator, profile: EffortProfile, member: MemberEvidence, period: str, usage: Usage
) -> tuple[MemberReview, str]:
    """One worker. Returns the review and where it came from: `ai` or `rules`."""
    clear = member.score >= CLEAR_HIGH or member.score <= CLEAR_LOW or not _has_data(member)
    if profile.llm_only_when_unclear and clear:
        return rule_review(member), "rules"
    messages = MEMBER_REVIEW.render(
        name=member.name,
        period=period,
        score=member.score,
        facts=truncate_to_tokens(_facts_text(member), profile.context_tokens // 3),
        flags="\n".join(f"- {f}" for f in member.flags) or "- none",
    )
    try:
        return await generate_structured(generator, messages, MemberReview, profile, usage), "ai"
    except GenerationError as exc:
        logger.warning("member review fell back to rules: %s", exc)
        return rule_review(member), "rules"


async def analyze_team(
    generator: Generator, profile: EffortProfile, members: list[MemberEvidence], period: str, team_facts: dict
) -> dict:
    usage = Usage()
    results = []
    for member in members:
        review, source = await review_member(generator, profile, member, period, usage)
        results.append(
            {
                "user_id": member.user_id,
                "name": member.name,
                "score": member.score,
                "rule_verdict": rule_verdict(member.score, _has_data(member)).value,
                "verdict": review.verdict.value,
                "explanation": review.explanation,
                "questions": review.questions,
                "source": source,
            }
        )

    summary: list[str] = []
    summary_source = "rules"
    if profile.effort is not Effort.LOW and results:
        reviews = "\n".join(f"- {r['name']} ({r['score']}/100, {r['verdict']}): {r['explanation']}" for r in results)
        try:
            team = await generate_structured(
                generator,
                TEAM_REVIEW.render(
                    period=period,
                    team_facts=", ".join(f"{k}={v}" for k, v in team_facts.items()),
                    reviews=truncate_to_tokens(reviews, profile.context_tokens // 2),
                ),
                TeamSummary,
                profile,
                usage,
            )
            summary, summary_source = team.bullets, "ai"
        except GenerationError as exc:
            logger.warning("team summary fell back to rules: %s", exc)
    if not summary:
        weak = [r["name"] for r in results if r["verdict"] == Verdict.WEAKLY.value]
        partly = [r["name"] for r in results if r["verdict"] == Verdict.PARTLY.value]
        summary = [f"{len(results)} people reviewed."]
        if weak:
            summary.append("Weakly supported reports: " + ", ".join(weak) + ".")
        if partly:
            summary.append("Partly supported reports: " + ", ".join(partly) + ".")
        if not weak and not partly:
            summary.append("Reported work matches recorded activity for everyone with reports.")

    return {
        "members": results,
        "summary": summary,
        "summary_source": summary_source,
        "usage": {"prompt_tokens": usage.prompt_tokens, "answer_tokens": usage.answer_tokens},
    }
