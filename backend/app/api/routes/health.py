"""Liveness and readiness probes.

- ``/health``: the process is up and serving requests. Never touches dependencies,
  so a slow database can't get a healthy container restarted.
- ``/ready``: dependencies are reachable; returns 503 otherwise.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request, Response, status
from starlette.concurrency import run_in_threadpool

from app import __version__

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

READY_CHECK_TIMEOUT_SECONDS = 5.0


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "service": "plane-ai",
        "version": __version__,
        "environment": settings.app_env.value,
    }


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict[str, object]:
    checks: dict[str, str] = {}

    try:
        await asyncio.wait_for(
            run_in_threadpool(request.app.state.db.ping), timeout=READY_CHECK_TIMEOUT_SECONDS
        )
        checks["database"] = "ok"
    except Exception as exc:  # any failure means "not ready"
        logger.warning("readiness check failed: database: %s", exc.__class__.__name__)
        checks["database"] = "unavailable"

    is_ready = all(v == "ok" for v in checks.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if is_ready else "not_ready", "checks": checks}
