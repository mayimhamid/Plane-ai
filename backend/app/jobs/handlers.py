"""What each kind of job does, and the input it accepts (validated before the job is queued)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.ask import Section, answer_question
from app.agents.daily import DayPerson, review_day
from app.agents.team import MemberEvidence, analyze_team
from app.agents.workflow import write_and_refine
from app.llm.effort import EffortProfile
from app.llm.generate import Generator, Usage
from app.llm.tokens import input_budget, truncate_to_tokens
from app.prompts import SUMMARIZE
from app.services import documents as doc_service
from app.services.tasks import propose_tasks


@dataclass
class JobContext:
    generator: Generator
    profile: EffortProfile
    session: Session
    workspace_id: str
    progress: Callable[[str], None]


class SummarizeInput(BaseModel):
    kind: str = Field(default="text", max_length=40)
    title: str = Field(default="", max_length=300)
    text: str = Field(min_length=1, max_length=200_000)
    bullets: int = Field(default=6, ge=1, le=15)


class TasksInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)


class TeamInput(BaseModel):
    period: str = Field(max_length=60)
    team_facts: dict[str, float | int | str] = Field(default_factory=dict)
    members: list[MemberEvidence] = Field(max_length=200)


class DailyInput(BaseModel):
    day: str = Field(max_length=20)
    people: list[DayPerson] = Field(max_length=200)


class AskInput(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    period: str = Field(default="", max_length=60)
    sections: dict[str, Section] = Field(default_factory=dict)
    # Projects the asker may read: when given, matching work items and documents become the "docs" section.
    project_ids: list[str] | None = None


async def run_summarize(ctx: JobContext, data: SummarizeInput) -> dict:
    budget = input_budget(ctx.profile.context_tokens, ctx.profile.max_answer_tokens) - 150
    text = truncate_to_tokens(data.text, budget)
    result = await write_and_refine(
        ctx.generator,
        ctx.profile,
        SUMMARIZE.render(kind=data.kind, title=data.title, text=text, bullets=data.bullets),
        source=text,
    )
    return {
        "summary": result.text,
        "rounds": result.rounds,
        "scores": result.scores,
        "effort": ctx.profile.effort.value,
        "truncated": text != data.text,
        "cached": False,
    }


async def run_tasks(ctx: JobContext, data: TasksInput) -> dict:
    return {"tasks": await propose_tasks(ctx.generator, ctx.profile, data.prompt, Usage())}


async def run_team(ctx: JobContext, data: TeamInput) -> dict:
    ctx.progress(f"{len(data.members)} people")
    result = await analyze_team(ctx.generator, ctx.profile, data.members, data.period, data.team_facts)
    return {**result, "effort": ctx.profile.effort.value, "cached": False}


async def run_daily(ctx: JobContext, data: DailyInput) -> dict:
    result = await review_day(ctx.generator, ctx.profile, data.day, data.people, ctx.progress)
    return {**result, "effort": ctx.profile.effort.value}


async def run_ask(ctx: JobContext, data: AskInput) -> dict:
    sections = dict(data.sections)
    if data.project_ids is not None and "docs" not in sections:
        hits = await doc_service.search(
            ctx.session, ctx.generator, ctx.workspace_id, data.question, data.project_ids, limit=ctx.profile.retrieval_chunks
        )
        if hits:
            sections["docs"] = Section(
                title="Work items and documents", text="\n".join(f'- "{h.title}": {h.text}' for h in hits)
            )
    ctx.progress("reading the data")
    result = await answer_question(ctx.generator, ctx.profile, data.question, data.period, sections)
    return {**result, "question": data.question, "effort": ctx.profile.effort.value}


Handler = Callable[[JobContext, BaseModel], Awaitable[dict]]

HANDLERS: dict[str, tuple[type[BaseModel], Handler]] = {
    "summarize": (SummarizeInput, run_summarize),
    "tasks": (TasksInput, run_tasks),
    "team_review": (TeamInput, run_team),
    "daily_review": (DailyInput, run_daily),
    "ask": (AskInput, run_ask),
}
