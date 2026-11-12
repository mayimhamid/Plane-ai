"""Background jobs, model-call traces and admin feedback.

Revision ID: 0002_ai_jobs
Revises: 0001_ai_core
Create Date: 2026-11-12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_ai_jobs"
down_revision = "0001_ai_core"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_job",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("workspace_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("audience", sa.String(16), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("progress", sa.String(200), nullable=False),
        sa.Column("input", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.String(300), nullable=False),
        sa.Column("effort", sa.String(10), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_ai_job"),
    )
    op.create_index("ix_ai_job_workspace_id", "ai_job", ["workspace_id"])
    op.create_index("ix_ai_job_ws_kind_key", "ai_job", ["workspace_id", "kind", "key"])
    op.create_table(
        "ai_trace",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("messages", sa.JSON(), nullable=False),
        sa.Column("output", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["ai_job.id"], name="fk_ai_trace_job_id_ai_job", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_ai_trace"),
    )
    op.create_index("ix_ai_trace_job_id", "ai_trace", ["job_id"])
    op.create_table(
        "ai_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("workspace_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("item", sa.String(100), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("correction", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["ai_job.id"], name="fk_ai_feedback_job_id_ai_job", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_ai_feedback"),
    )
    op.create_index("ix_ai_feedback_job_id", "ai_feedback", ["job_id"])
    op.create_index("ix_ai_feedback_workspace_id", "ai_feedback", ["workspace_id"])


def downgrade() -> None:
    op.drop_table("ai_feedback")
    op.drop_table("ai_trace")
    op.drop_table("ai_job")
