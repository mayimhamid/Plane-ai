"""The AI API Plane's backend calls. Every route needs the service token.

Plane decides who may do what (workspace roles, project membership) and passes the result: the workspace,
the user and the projects that user may read. This service never sees Plane's database.

Errors: 409 ``ai_disabled`` when the workspace turned AI off; 503 ``llm_unavailable`` when the runtime can't
answer. Search, index and settings work without the runtime.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.team import MemberEvidence, analyze_team
from app.agents.workflow import write_and_refine
from app.api.deps import get_session, require_service_token, workspace_profile
from app.llm.effort import Effort, profile_for
from app.llm.generate import GenerationError, Usage
from app.llm.tokens import input_budget, truncate_to_tokens
from app.prompts import SUMMARIZE, catalog
from app.services import chat as chat_service
from app.services import documents as doc_service
from app.services import nlp
from app.services import settings as ws_settings
from app.services.tasks import propose_tasks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["ai"], dependencies=[Depends(require_service_token)])


def _llm_unavailable(exc: Exception) -> HTTPException:
    logger.warning("generation failed: %s", exc)
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="llm_unavailable")


def _usage(u: Usage) -> dict[str, int]:
    return {"prompt_tokens": u.prompt_tokens, "answer_tokens": u.answer_tokens}


# ------------------------------------------------------------------ status and settings


@router.get("/status")
async def ai_status(request: Request) -> dict:
    settings = request.app.state.settings
    runtime = await request.app.state.llm.check()
    return {
        "runtime": settings.llm_runtime.value,
        "model": settings.llm_model_name,
        "llm": runtime.value,
        "embeddings": bool(settings.embed_model),
        "profiles": {
            e.value: profile_for(e, settings.llm_max_threads, settings.llm_context_length).as_dict() for e in Effort
        },
    }


@router.get("/prompts")
async def prompts() -> list[dict[str, object]]:
    return catalog()


class SettingsIn(BaseModel):
    enabled: bool
    effort: Effort
    user_id: str | None = None


@router.get("/workspaces/{workspace_id}/settings")
def get_ws_settings(workspace_id: str, request: Request, session: Session = Depends(get_session)) -> dict:
    settings = request.app.state.settings
    enabled, effort = ws_settings.get_setting(session, settings, workspace_id)
    return {"enabled": enabled, "effort": effort.value, "profile": ws_settings.profile(settings, effort).as_dict()}


@router.put("/workspaces/{workspace_id}/settings")
def put_ws_settings(
    workspace_id: str, body: SettingsIn, request: Request, session: Session = Depends(get_session)
) -> dict:
    settings = request.app.state.settings
    ws_settings.put_setting(session, workspace_id, enabled=body.enabled, effort=body.effort, user_id=body.user_id)
    return {
        "enabled": body.enabled,
        "effort": body.effort.value,
        "profile": ws_settings.profile(settings, body.effort).as_dict(),
    }


# ------------------------------------------------------------------ summarize


class SummarizeIn(BaseModel):
    workspace_id: str
    kind: str = Field(default="text", max_length=40)
    title: str = Field(default="", max_length=300)
    text: str = Field(min_length=1, max_length=200_000)
    bullets: int = Field(default=6, ge=1, le=15)


@router.post("/summarize")
async def summarize(body: SummarizeIn, request: Request, session: Session = Depends(get_session)) -> dict:
    settings = request.app.state.settings
    profile = workspace_profile(body.workspace_id, session, settings)
    cache = request.app.state.cache
    key = cache.key("summarize", body.model_dump(), profile.effort.value, SUMMARIZE.version)
    cached = cache.get(key)
    if cached is not None:
        return {**cached, "cached": True}
    budget = input_budget(profile.context_tokens, profile.max_answer_tokens) - 150
    text = truncate_to_tokens(body.text, budget)
    try:
        result = await write_and_refine(
            request.app.state.generator,
            profile,
            SUMMARIZE.render(kind=body.kind, title=body.title, text=text, bullets=body.bullets),
            source=text,
        )
    except GenerationError as exc:
        raise _llm_unavailable(exc) from exc
    data = {
        "summary": result.text,
        "rounds": result.rounds,
        "scores": result.scores,
        "effort": profile.effort.value,
        "truncated": text != body.text,
        "usage": _usage(result.usage),
    }
    cache.set(key, data)
    return {**data, "cached": False}


# ------------------------------------------------------------------ tasks


class TasksIn(BaseModel):
    workspace_id: str
    prompt: str = Field(min_length=1, max_length=100_000)


@router.post("/tasks/propose")
async def tasks_propose(body: TasksIn, request: Request, session: Session = Depends(get_session)) -> dict:
    profile = workspace_profile(body.workspace_id, session, request.app.state.settings)
    usage = Usage()
    try:
        tasks = await propose_tasks(request.app.state.generator, profile, body.prompt, usage)
    except GenerationError as exc:
        raise _llm_unavailable(exc) from exc
    return {"tasks": tasks, "effort": profile.effort.value, "usage": _usage(usage)}


# ------------------------------------------------------------------ chat


class ChatIn(BaseModel):
    workspace_id: str
    user_id: str
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=chat_service.MAX_MESSAGE_CHARS)
    project_ids: list[str] = Field(default_factory=list)
    use_workspace: bool = True


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(body: ChatIn, request: Request, session: Session = Depends(get_session)) -> StreamingResponse:
    """Server-sent events: `meta` (conversation and sources), `token` pieces, then `done` or `error`."""
    profile = workspace_profile(body.workspace_id, session, request.app.state.settings)
    generator = request.app.state.generator
    try:
        turn = await chat_service.prepare_turn(
            session,
            generator,
            profile,
            workspace_id=body.workspace_id,
            user_id=body.user_id,
            conversation_id=body.conversation_id,
            message=body.message,
            project_ids=body.project_ids,
            use_workspace=body.use_workspace,
        )
    except chat_service.ConversationNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found") from exc

    async def events() -> AsyncIterator[str]:
        sources = [
            {"n": n, "type": h.source_type, "id": h.source_id, "title": h.title, "url": h.url}
            for n, h in enumerate(turn.hits, start=1)
        ]
        yield _sse("meta", {"conversation_id": turn.conversation.id, "sources": sources, "effort": profile.effort.value})
        try:
            async for piece in chat_service.stream_answer(session, generator, profile, turn):
                yield _sse("token", {"text": piece})
        except GenerationError as exc:
            logger.warning("chat stream failed: %s", exc)
            yield _sse("error", {"detail": "llm_unavailable"})
            return
        yield _sse("done", {"conversation_id": turn.conversation.id})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/conversations")
def conversations(workspace_id: str, user_id: str, session: Session = Depends(get_session)) -> list[dict]:
    return [
        {"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()}
        for c in chat_service.list_conversations(session, workspace_id, user_id)
    ]


@router.get("/conversations/{conversation_id}")
def conversation(conversation_id: str, workspace_id: str, user_id: str, session: Session = Depends(get_session)) -> dict:
    try:
        conv = chat_service.get_conversation(session, workspace_id, user_id, conversation_id)
    except chat_service.ConversationNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found") from exc
    return {
        "id": conv.id,
        "title": conv.title,
        "summary": conv.summary,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in sorted(conv.messages, key=lambda m: m.position)
        ],
    }


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, workspace_id: str, user_id: str, session: Session = Depends(get_session)) -> None:
    try:
        conv = chat_service.get_conversation(session, workspace_id, user_id, conversation_id)
    except chat_service.ConversationNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found") from exc
    session.delete(conv)
    session.commit()


# ------------------------------------------------------------------ documents and search


class DocumentIn(BaseModel):
    source_type: str = Field(max_length=32)
    source_id: str = Field(max_length=64)
    title: str = Field(default="", max_length=300)
    text: str = Field(default="", max_length=200_000)
    project_id: str | None = None
    url: str = Field(default="", max_length=500)


class DocumentsIn(BaseModel):
    workspace_id: str
    documents: list[DocumentIn] = Field(max_length=500)


@router.post("/documents")
async def upsert_documents(body: DocumentsIn, request: Request, session: Session = Depends(get_session)) -> dict:
    docs = [doc_service.SourceDocument(**d.model_dump()) for d in body.documents]
    chunks = await doc_service.upsert_documents(session, request.app.state.generator, body.workspace_id, docs)
    return {"documents": len(docs), "chunks": chunks}


class DocumentsDeleteIn(BaseModel):
    workspace_id: str
    source_type: str
    source_ids: list[str] = Field(max_length=1000)


@router.post("/documents/delete")
def delete_documents(body: DocumentsDeleteIn, session: Session = Depends(get_session)) -> dict:
    doc_service.delete_documents(session, body.workspace_id, body.source_type, body.source_ids)
    return {"deleted": len(body.source_ids)}


class SearchIn(BaseModel):
    workspace_id: str
    query: str = Field(min_length=1, max_length=500)
    project_ids: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=50)


@router.post("/search")
async def search(body: SearchIn, request: Request, session: Session = Depends(get_session)) -> dict:
    hits = await doc_service.search(
        session, request.app.state.generator, body.workspace_id, body.query, body.project_ids, limit=body.limit
    )
    return {
        "results": [
            {
                "type": h.source_type,
                "id": h.source_id,
                "title": h.title,
                "url": h.url,
                "project_id": h.project_id,
                "snippet": h.text[:300],
                "score": h.score,
            }
            for h in hits
        ]
    }


# ------------------------------------------------------------------ NLP


class ClassifyIn(BaseModel):
    workspace_id: str
    text: str = Field(min_length=1, max_length=50_000)
    labels: list[str] = Field(min_length=2, max_length=20)


class TextIn(BaseModel):
    workspace_id: str
    text: str = Field(min_length=1, max_length=50_000)


class NLIIn(BaseModel):
    workspace_id: str
    claim: str = Field(min_length=1, max_length=5_000)
    evidence: str = Field(min_length=1, max_length=50_000)


@router.post("/classify")
async def classify(body: ClassifyIn, request: Request, session: Session = Depends(get_session)) -> dict:
    profile = workspace_profile(body.workspace_id, session, request.app.state.settings)
    usage = Usage()
    try:
        result = await nlp.classify(request.app.state.generator, profile, body.text, body.labels, usage)
    except GenerationError as exc:
        raise _llm_unavailable(exc) from exc
    return {**result.model_dump(mode="json"), "usage": _usage(usage)}


@router.post("/sentiment")
async def sentiment(body: TextIn, request: Request, session: Session = Depends(get_session)) -> dict:
    profile = workspace_profile(body.workspace_id, session, request.app.state.settings)
    usage = Usage()
    try:
        result = await nlp.sentiment(request.app.state.generator, profile, body.text, usage)
    except GenerationError as exc:
        raise _llm_unavailable(exc) from exc
    return {**result.model_dump(mode="json"), "usage": _usage(usage)}


@router.post("/nli")
async def nli(body: NLIIn, request: Request, session: Session = Depends(get_session)) -> dict:
    profile = workspace_profile(body.workspace_id, session, request.app.state.settings)
    usage = Usage()
    try:
        result = await nlp.infer(request.app.state.generator, profile, body.claim, body.evidence, usage)
    except GenerationError as exc:
        raise _llm_unavailable(exc) from exc
    return {**result.model_dump(mode="json"), "usage": _usage(usage)}


# ------------------------------------------------------------------ team analysis


class TeamIn(BaseModel):
    workspace_id: str
    period: str = Field(max_length=60)
    team_facts: dict[str, float | int | str] = Field(default_factory=dict)
    members: list[MemberEvidence] = Field(max_length=200)


@router.post("/analyze/team")
async def analyze(body: TeamIn, request: Request, session: Session = Depends(get_session)) -> dict:
    profile = workspace_profile(body.workspace_id, session, request.app.state.settings)
    cache = request.app.state.cache
    key = cache.key("team", body.model_dump(mode="json"), profile.effort.value)
    cached = cache.get(key)
    if cached is not None:
        return {**cached, "cached": True}
    data = await analyze_team(request.app.state.generator, profile, body.members, body.period, body.team_facts)
    data["effort"] = profile.effort.value
    cache.set(key, data)
    return {**data, "cached": False}
