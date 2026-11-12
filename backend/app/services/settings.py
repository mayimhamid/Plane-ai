"""Per-workspace AI switch and effort."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.tables import WorkspaceSetting
from app.llm.effort import Effort, EffortProfile, profile_for


def get_setting(session: Session, settings: Settings, workspace_id: str) -> tuple[bool, Effort]:
    row = session.get(WorkspaceSetting, workspace_id)
    if row is None:
        return True, Effort(settings.default_effort)
    return row.enabled, Effort(row.effort)


def put_setting(
    session: Session, workspace_id: str, *, enabled: bool, effort: Effort, user_id: str | None
) -> WorkspaceSetting:
    row = session.get(WorkspaceSetting, workspace_id)
    if row is None:
        row = WorkspaceSetting(workspace_id=workspace_id)
        session.add(row)
    row.enabled = enabled
    row.effort = effort.value
    row.updated_by = user_id
    session.commit()
    return row


def profile(settings: Settings, effort: Effort) -> EffortProfile:
    return profile_for(effort, settings.llm_max_threads, settings.llm_context_length)
