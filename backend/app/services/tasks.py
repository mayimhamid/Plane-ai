"""Task proposals from a description, as JSON Plane's task import already understands.

Nothing is created here. Plane shows the proposals in its import preview (with the problems of each row) and
only creates them when the person presses "Add": the assistant suggests, the person decides.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, Field, field_validator

from app.llm.effort import EffortProfile
from app.llm.generate import Generator, Usage
from app.llm.structured import generate_structured
from app.llm.tokens import truncate_to_tokens
from app.prompts import EXTRACT_TASKS

MAX_TASKS = 100


class TaskProposal(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = None
    planned_hours: float | None = Field(default=None, ge=0, le=1000)
    planned_days: float | None = Field(default=None, ge=0, le=365)
    due_date: str | None = None
    start_date: str | None = None
    priority: str | None = None
    assignees: list[str] | None = None
    module: str | None = None
    state: str | None = None
    labels: list[str] | None = None
    is_meeting: bool | None = None

    @field_validator("priority")
    @classmethod
    def _priority(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower()
        return v if v in {"urgent", "high", "medium", "low", "none"} else None

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        return v.strip()


class TaskProposals(BaseModel):
    tasks: list[TaskProposal] = Field(default_factory=list, max_length=MAX_TASKS)


async def propose_tasks(generator: Generator, profile: EffortProfile, prompt: str, usage: Usage) -> list[dict]:
    # Task lists can be long: give them more room than a usual answer, within the context window.
    roomy = replace(profile, max_answer_tokens=max(profile.max_answer_tokens, profile.context_tokens // 2))
    budget = roomy.context_tokens - roomy.max_answer_tokens - 200
    result = await generate_structured(
        generator, EXTRACT_TASKS.render(prompt=truncate_to_tokens(prompt, budget)), TaskProposals, roomy, usage
    )
    return [task.model_dump(exclude_none=True) for task in result.tasks if task.title]
