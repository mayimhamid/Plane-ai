"""Plane AI's own tables (in its own database; Plane's database is never touched).

- ``ai_workspace_setting``: per Plane workspace, whether AI is on and at which effort.
- ``ai_conversation`` / ``ai_message``: chat history. This is the checkpointer: a conversation is rebuilt
  from here after a restart, including its running summary (memory).
- ``ai_document_chunk``: workspace content Plane sent for search and grounded answers (RAG), split in chunks.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class WorkspaceSetting(Base):
    __tablename__ = "ai_workspace_setting"

    workspace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    effort: Mapped[str] = mapped_column(String(10), default="medium")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    updated_by: Mapped[str | None] = mapped_column(String(64), nullable=True)


class Conversation(Base):
    __tablename__ = "ai_conversation"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(200), default="")
    # Running summary of the turns that no longer fit in the verbatim history.
    summary: Mapped[str] = mapped_column(Text, default="")
    # Messages already folded into `summary` (they stay stored, but are not sent again).
    summarized_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", order_by="Message.position", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "ai_message"
    __table_args__ = (UniqueConstraint("conversation_id", "position"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("ai_conversation.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    answer_tokens: Mapped[int] = mapped_column(Integer, default=0)
    prompt_version: Mapped[str] = mapped_column(String(40), default="")
    effort: Mapped[str] = mapped_column(String(10), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class DocumentChunk(Base):
    __tablename__ = "ai_document_chunk"
    __table_args__ = (Index("ix_ai_chunk_ws_source", "workspace_id", "source_type", "source_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(String(64), index=True)
    # Who may see it: the Plane project (None = workspace-wide, e.g. a Hub doc visible to all members).
    project_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_type: Mapped[str] = mapped_column(String(32))  # issue | doc | report ...
    source_id: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(300), default="")
    url: Mapped[str] = mapped_column(String(500), default="")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Job(Base):
    """Slow AI work done in the background (summaries, task proposals, reviews, answers).

    Plane starts a job and polls it, so no web request waits minutes on a busy CPU. Jobs survive a restart: the
    ones not finished are run again at start-up.
    """

    __tablename__ = "ai_job"
    __table_args__ = (Index("ix_ai_job_ws_kind_key", "workspace_id", "kind", "key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64))
    # user: only the person who asked may read it; managers: any manager of the workspace may.
    audience: Mapped[str] = mapped_column(String(16), default="user")
    kind: Mapped[str] = mapped_column(String(32))
    # Lets Plane find the latest result for the same question (e.g. the daily review of one day and team).
    key: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued | running | done | failed
    progress: Mapped[str] = mapped_column(String(200), default="")
    input: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str] = mapped_column(String(300), default="")
    effort: Mapped[str] = mapped_column(String(10), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Trace(Base):
    """One model call made by a job: the prompt and the answer. Rated traces become training data."""

    __tablename__ = "ai_trace"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("ai_job.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    messages: Mapped[list] = mapped_column(JSON)
    output: Mapped[str] = mapped_column(Text)


class Feedback(Base):
    """An admin's rating of a job's answer, with an optional corrected answer (the better target to learn)."""

    __tablename__ = "ai_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("ai_job.id", ondelete="CASCADE"), index=True)
    workspace_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64))
    # Which part was rated (e.g. a person id in a daily review); empty = the whole answer.
    item: Mapped[str] = mapped_column(String(100), default="")
    rating: Mapped[int] = mapped_column(Integer)  # 1 good, -1 wrong
    correction: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
