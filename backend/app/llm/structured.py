"""Structured output: JSON answers validated against a Pydantic model (its JSON Schema).

The schema is sent to the runtime (Ollama ``format`` / llama.cpp ``response_format``), which constrains the
output grammar. The answer is then validated; if it still fails, the model gets one repair attempt with the
validation error. The caller always receives a validated model instance or a ``GenerationError``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.effort import EffortProfile
from app.llm.generate import GenerationError, Generator, Usage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(text: str) -> object:
    """JSON from a model answer, tolerating code fences and text around one object."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


async def generate_structured(
    generator: Generator,
    messages: list[dict[str, str]],
    model: type[T],
    profile: EffortProfile,
    usage: Usage | None = None,
    repair_attempts: int = 1,
) -> T:
    schema = model.model_json_schema()
    attempt_messages = list(messages)
    last_error = ""
    for attempt in range(repair_attempts + 1):
        result = await generator.complete(attempt_messages, profile, json_schema=schema)
        if usage is not None:
            usage.prompt_tokens += result.usage.prompt_tokens
            usage.answer_tokens += result.usage.answer_tokens
        try:
            return model.model_validate(parse_json(result.text))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = str(exc)[:500]
            logger.info("structured answer invalid (attempt %d): %s", attempt + 1, last_error)
            attempt_messages = [
                *messages,
                {"role": "assistant", "content": result.text[:2000]},
                {
                    "role": "user",
                    "content": "That answer is not valid. Error: "
                    f"{last_error}\nAnswer again with JSON only that matches this schema:\n{json.dumps(schema)}",
                },
            ]
    raise GenerationError(f"the model did not return valid JSON: {last_error}")
