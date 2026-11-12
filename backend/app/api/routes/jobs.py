"""Background jobs, admin ratings, and the training-data export.

Visibility: a job with audience ``user`` is readable only by the person who started it; ``managers`` jobs by any
caller Plane marks as a manager (``manager=true``) in the same workspace. Plane decides who is a manager.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_session, require_service_token, workspace_profile
from app.db.tables import Feedback, Job, Trace
from app.jobs.handlers import HANDLERS

router = APIRouter(prefix="/v1", tags=["jobs"], dependencies=[Depends(require_service_token)])


class JobIn(BaseModel):
    workspace_id: str
    user_id: str
    kind: str
    audience: str = Field(default="user", pattern="^(user|managers)$")
    key: str = Field(default="", max_length=200)
    input: dict


def job_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "key": job.key,
        "status": job.status,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
        "effort": job.effort,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def _visible(job: Job, workspace_id: str, user_id: str, manager: bool) -> bool:
    if job.workspace_id != workspace_id:
        return False
    return job.user_id == user_id or (job.audience == "managers" and manager)


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(body: JobIn, request: Request, session: Session = Depends(get_session)) -> dict:
    if body.kind not in HANDLERS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"unknown job kind: {body.kind}")
    model, _ = HANDLERS[body.kind]
    try:
        model.model_validate(body.input)
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=json.loads(exc.json())[:5]) from exc
    profile = workspace_profile(body.workspace_id, session, request.app.state.settings)
    job = Job(
        workspace_id=body.workspace_id,
        user_id=body.user_id,
        audience=body.audience,
        kind=body.kind,
        key=body.key,
        status="queued",
        progress="",
        input=body.input,
        error="",
        effort=profile.effort.value,
    )
    session.add(job)
    session.commit()
    request.app.state.jobs.submit(job.id)
    return job_dict(job)


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str, workspace_id: str, user_id: str, manager: bool = False, session: Session = Depends(get_session)
) -> dict:
    job = session.get(Job, job_id)
    if job is None or not _visible(job, workspace_id, user_id, manager):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="job not found")
    return job_dict(job)


@router.get("/jobs")
def list_jobs(
    workspace_id: str,
    user_id: str,
    kind: str,
    key: str | None = None,
    manager: bool = False,
    limit: int = 20,
    session: Session = Depends(get_session),
) -> list[dict]:
    stmt = select(Job).where(Job.workspace_id == workspace_id, Job.kind == kind)
    if key is not None:
        stmt = stmt.where(Job.key == key)
    stmt = stmt.order_by(Job.created_at.desc()).limit(min(max(limit, 1), 100) * 3)
    jobs = [j for j in session.scalars(stmt) if _visible(j, workspace_id, user_id, manager)]
    return [job_dict(j) for j in jobs[: min(max(limit, 1), 100)]]


class FeedbackIn(BaseModel):
    workspace_id: str
    user_id: str
    manager: bool = False
    item: str = Field(default="", max_length=100)
    rating: int = Field(ge=-1, le=1)
    correction: str = Field(default="", max_length=10_000)


@router.post("/jobs/{job_id}/feedback", status_code=status.HTTP_201_CREATED)
def add_feedback(job_id: str, body: FeedbackIn, session: Session = Depends(get_session)) -> dict:
    job = session.get(Job, job_id)
    if job is None or job.status != "done" or not _visible(job, body.workspace_id, body.user_id, body.manager):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="job not found")
    if body.rating == 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="rating must be 1 or -1")
    session.add(
        Feedback(
            job_id=job.id,
            workspace_id=body.workspace_id,
            user_id=body.user_id,
            item=body.item,
            rating=body.rating,
            correction=body.correction.strip(),
        )
    )
    session.commit()
    return {"ok": True}


def _trace_for(session: Session, job: Job, item: str) -> Trace | None:
    """The model call a rating is about: a person's call in a daily review, else the job's last call."""
    traces = list(session.scalars(select(Trace).where(Trace.job_id == job.id).order_by(Trace.position)))
    if not traces:
        return None
    if item and job.kind == "daily_review" and job.result:
        for person in job.result.get("people", []):
            if person.get("user_id") == item and person.get("trace") is not None:
                position = person["trace"]
                return next((t for t in traces if t.position == position), None)
        return None
    return traces[-1]


def training_examples(session: Session, workspace_id: str) -> Iterator[dict]:
    """Chat-format examples for fine-tuning (LoRA/QLoRA) or distillation: the prompt of a rated call, and as the
    target either the admin's correction or the model's own answer when it was rated good. Wrong answers with
    no correction are left out."""
    feedback = session.scalars(
        select(Feedback).where(Feedback.workspace_id == workspace_id).order_by(Feedback.created_at)
    )
    for fb in feedback:
        job = session.get(Job, fb.job_id)
        if job is None:
            continue
        trace = _trace_for(session, job, fb.item)
        if trace is None:
            continue
        target = fb.correction or (trace.output if fb.rating > 0 else "")
        if not target:
            continue
        yield {
            "messages": [*trace.messages, {"role": "assistant", "content": target}],
            "meta": {"kind": job.kind, "rating": fb.rating, "corrected": bool(fb.correction), "job": job.id},
        }


@router.get("/training/export")
def export_training(workspace_id: str, session: Session = Depends(get_session)) -> StreamingResponse:
    def lines() -> Iterator[str]:
        for example in training_examples(session, workspace_id):
            yield json.dumps(example, ensure_ascii=False) + "\n"

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="plane-ai-training.jsonl"'},
    )
