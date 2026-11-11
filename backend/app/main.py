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
from app.api.routes import health
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import Database, DatabaseProtocol
from app.llm.runtime import RuntimeClient, create_runtime_client

logger = logging.getLogger("app")


def create_app(
    settings: Settings | None = None,
    database_factory: Callable[[Settings], DatabaseProtocol] = Database,
    runtime_client_factory: Callable[[Settings], RuntimeClient] = create_runtime_client,
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
            await app.state.llm.aclose()
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
    app.include_router(health.router)
    return app
