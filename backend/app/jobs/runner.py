"""Runs jobs in this process, one generation at a time (the generator's semaphore orders them).

A job is a row in ``ai_job``. ``submit`` starts it as an asyncio task; at start-up ``resume`` starts again every
job a restart interrupted, so nothing is silently lost. Every model call a job makes is stored as a trace.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from app.core.config import Settings
from app.db.tables import Job, Trace
from app.jobs.handlers import HANDLERS, JobContext
from app.llm.effort import Effort
from app.llm.generate import GenerationError, Generator, current_trace
from app.services import settings as ws_settings

logger = logging.getLogger(__name__)

KEEP_DAYS = 60


class JobRunner:
    def __init__(self, settings: Settings, session_factory, generator: Generator) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.generator = generator
        self._tasks: set[asyncio.Task] = set()

    def submit(self, job_id: str) -> asyncio.Task:
        task = asyncio.create_task(self._run(job_id), name=f"ai-job-{job_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def resume(self) -> int:
        """Start again the jobs a restart interrupted, and drop very old ones."""
        with self.session_factory() as session:
            cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)
            session.execute(delete(Job).where(Job.created_at < cutoff))
            pending = list(session.scalars(select(Job.id).where(Job.status.in_(("queued", "running")))))
            session.commit()
        for job_id in pending:
            self.submit(job_id)
        return len(pending)

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait a moment for running jobs at shutdown; the rest resume at the next start."""
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=timeout)
        for task in list(self._tasks):
            task.cancel()

    async def _run(self, job_id: str) -> None:
        session = self.session_factory()
        trace: list[dict] = []
        token = current_trace.set(trace)
        try:
            job = session.get(Job, job_id)
            if job is None or job.status in ("done", "failed"):
                return
            job.status = "running"
            session.commit()
            model, handler = HANDLERS[job.kind]
            data = model.model_validate(job.input)
            profile = ws_settings.profile(self.settings, Effort(job.effort or self.settings.default_effort))

            def progress(text: str) -> None:
                job.progress = text[:200]
                session.commit()

            ctx = JobContext(self.generator, profile, session, job.workspace_id, progress)
            try:
                job.result = await handler(ctx, data)
                job.status = "done"
            except GenerationError as exc:
                logger.warning("job %s (%s) failed: %s", job_id, job.kind, exc)
                job.status, job.error = "failed", "llm_unavailable"
            except Exception:  # never leave a job "running" forever
                logger.exception("job %s (%s) crashed", job_id, job.kind)
                job.status, job.error = "failed", "internal_error"
            job.progress = ""
            job.finished_at = datetime.now(timezone.utc)
            for position, call in enumerate(trace):
                session.add(Trace(job_id=job.id, position=position, messages=call["messages"], output=call["output"]))
            session.commit()
        except asyncio.CancelledError:
            # Shutdown: leave it queued so it runs again at the next start.
            session.rollback()
            raise
        finally:
            current_trace.reset(token)
            session.close()
