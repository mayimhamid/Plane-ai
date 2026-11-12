"""Shared dependencies of the ``/v1`` routes."""

from __future__ import annotations

import hmac
from collections.abc import Iterator

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.llm.effort import Effort, EffortProfile
from app.services import settings as ws_settings


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def require_service_token(request: Request) -> None:
    """Only Plane's backend calls this service. It sends `Authorization: Bearer <PLANE_AI_SERVICE_TOKEN>`.

    Outside production an empty token turns the check off, so the API can be tried locally.
    """
    settings: Settings = request.app.state.settings
    expected = settings.service_token.get_secret_value()
    if not expected and not settings.is_production:
        return
    header = request.headers.get("authorization", "")
    given = header[7:] if header.lower().startswith("bearer ") else ""
    if not expected or not hmac.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid service token")


def get_session(request: Request) -> Iterator[Session]:
    session = request.app.state.db.session_factory()
    try:
        yield session
    finally:
        session.close()


def workspace_profile(
    workspace_id: str, session: Session, settings: Settings, override: str | None = None
) -> EffortProfile:
    """The effort profile of a workspace; 409 when the workspace turned AI off."""
    enabled, effort = ws_settings.get_setting(session, settings, workspace_id)
    if not enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="ai_disabled")
    if override:
        effort = Effort(override)
    return ws_settings.profile(settings, effort)

