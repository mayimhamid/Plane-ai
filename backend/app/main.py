"""FastAPI application factory.

Run with ``python -m app`` (reads host/port from settings), or:
    uvicorn app.main:create_app --factory
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.api.routes import ai, health, jobs
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import Database, DatabaseProtocol
from app.llm.generate import Generator, create_generator
from app.llm.runtime import RuntimeClient, create_runtime_client
from app.jobs.runner import JobRunner
from app.services.cache import TTLCache

logger = logging.getLogger("app")


def create_app(
    settings: Settings | None = None,
    database_factory: Callable[[Settings], DatabaseProtocol] = Database,
    runtime_client_factory: Callable[[Settings], RuntimeClient] = create_runtime_client,
    generator_factory: Callable[[Settings], Generator] = create_generator,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "starting %s v%s (env=%s, llm_runtime=%s)",
            settings.app_name,
            __version__,
            settings.app_env.value,
            settings.llm_runtime.value,
        )
        app.state.db = database_factory(settings)
        try:
            await run_in_threadpool(app.state.db.ping)
            logger.info("database reachable")
        except Exception as exc:
            # Don't crash: /ready reports it, and the DB may come up later.
            logger.warning("database not reachable at startup: %s", exc.__class__.__name__)

        app.state.llm = runtime_client_factory(settings)
        app.state.generator = generator_factory(settings)
        app.state.jobs = JobRunner(settings, getattr(app.state.db, "session_factory", None), app.state.generator)
        try:
            resumed = app.state.jobs.resume()
            if resumed:
                logger.info("resumed %d unfinished jobs", resumed)
        except Exception as exc:  # the database may come up later; new jobs still work then
            logger.warning("could not resume jobs: %s", exc.__class__.__name__)
        llm_status = await app.state.llm.check()
        logger.info(
            "llm runtime %s at %s: %s (model %s)",
            settings.llm_runtime.value,
            settings.llm_base_url,
            llm_status.value,
            settings.llm_model_name,
        )
        try:
            yield
        finally:
            logger.info("shutting down")
            await app.state.jobs.drain()
            await app.state.llm.aclose()
            await app.state.generator.aclose()
            app.state.db.dispose()
            logger.info("shutdown complete")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
        # API docs only outside production
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.settings = settings
    app.state.cache = TTLCache(settings.cache_ttl_seconds)
    app.include_router(health.router)
    app.include_router(ai.router)
    app.include_router(jobs.router)
    return app
