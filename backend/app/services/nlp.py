"""Classification, sentiment and natural language inference, zero-shot on the local LLM.

Dedicated HuggingFace models (e.g. an NLI cross-encoder) would each add a few hundred MB of weights and a
PyTorch install. On a small machine the already-loaded chat model does these tasks well enough with a
constrained JSON answer, so no extra model is loaded.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.llm.effort import EffortProfile
from app.llm.generate import Generator, Usage
from app.llm.structured import generate_structured
from app.llm.tokens import truncate_to_tokens
from app.prompts import CLASSIFY, NLI, SENTIMENT


class Classification(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Sentiment(BaseModel):
    sentiment: SentimentLabel
    confidence: float = Field(ge=0, le=1)
    phrases: list[str] = Field(default_factory=list, max_length=3)


class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class Inference(BaseModel):
    label: NLILabel
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(max_length=400)


def _budget(profile: EffortProfile) -> int:
    return max(200, profile.context_tokens - profile.max_answer_tokens - 200)


async def classify(generator: Generator, profile: EffortProfile, text: str, labels: list[str], usage: Usage) -> Classification:
    result = await generate_structured(
        generator,
        CLASSIFY.render(labels=", ".join(labels), text=truncate_to_tokens(text, _budget(profile))),
        Classification,
        profile,
        usage,
    )
    if result.label not in labels:
        # Small models sometimes change the case or add words: snap to the closest given label.
        lowered = {label.lower(): label for label in labels}
        result.label = lowered.get(result.label.strip().lower(), labels[0])
        result.confidence = min(result.confidence, 0.5)
    return result


async def sentiment(generator: Generator, profile: EffortProfile, text: str, usage: Usage) -> Sentiment:
    return await generate_structured(
        generator, SENTIMENT.render(text=truncate_to_tokens(text, _budget(profile))), Sentiment, profile, usage
    )


async def infer(generator: Generator, profile: EffortProfile, claim: str, evidence: str, usage: Usage) -> Inference:
    budget = _budget(profile)
    return await generate_structured(
        generator,
        NLI.render(claim=truncate_to_tokens(claim, budget // 4), evidence=truncate_to_tokens(evidence, budget * 3 // 4)),
        Inference,
        profile,
        usage,
    )
