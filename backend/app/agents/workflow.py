"""Writer -> evaluator loop.

    draft = writer(source)
    repeat up to `max_rounds`:
        review = evaluator(source, draft)        # structured: score + notes
        stop if review.score >= `good_enough` or there are no notes
        draft = writer(source, draft, notes)

At low and medium effort `max_rounds` is 0, so this is a single call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.llm.effort import EffortProfile
from app.llm.generate import Generator, Usage
from app.llm.structured import generate_structured
from app.llm.tokens import truncate_to_tokens
from app.prompts import EVALUATE, REWRITE


class Review(BaseModel):
    score: int = Field(ge=1, le=10)
    notes: list[str] = Field(default_factory=list, max_length=6)


@dataclass
class LoopResult:
    text: str
    rounds: int
    scores: list[int] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def _add(total: Usage, part: Usage) -> None:
    total.prompt_tokens += part.prompt_tokens
    total.answer_tokens += part.answer_tokens


async def write_and_refine(
    generator: Generator,
    profile: EffortProfile,
    first_draft_messages: list[dict[str, str]],
    source: str,
    max_rounds: int | None = None,
    good_enough: int = 8,
) -> LoopResult:
    usage = Usage()
    result = await generator.complete(first_draft_messages, profile)
    _add(usage, result.usage)
    draft = result.text.strip()
    rounds = profile.refine_rounds if max_rounds is None else max_rounds
    scores: list[int] = []
    source_cut = truncate_to_tokens(source, profile.context_tokens // 2)
    done = 0
    for _ in range(rounds):
        review = await generate_structured(generator, EVALUATE.render(source=source_cut, draft=draft), Review, profile, usage)
        scores.append(review.score)
        if review.score >= good_enough or not review.notes:
            break
        notes = "\n".join(f"- {n}" for n in review.notes)
        rewritten = await generator.complete(REWRITE.render(source=source_cut, draft=draft, notes=notes), profile)
        _add(usage, rewritten.usage)
        draft = rewritten.text.strip() or draft
        done += 1
    return LoopResult(text=draft, rounds=done, scores=scores, usage=usage)
