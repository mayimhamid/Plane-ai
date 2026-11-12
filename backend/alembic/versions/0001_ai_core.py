"""AI core tables: workspace settings, conversations and messages, document chunks.

Revision ID: 0001_ai_core
Revises:
Create Date: 2026-11-12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_ai_core"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_workspace_setting",
        sa.Column("workspace_id", sa.String(64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("effort", sa.String(10), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_by", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("workspace_id", name="pk_ai_workspace_setting"),
    )
    op.create_table(
        "ai_conversation",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("workspace_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("summarized_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_ai_conversation"),
    )
    op.create_index("ix_ai_conversation_workspace_id", "ai_conversation", ["workspace_id"])
    op.create_index("ix_ai_conversation_user_id", "ai_conversation", ["user_id"])
    op.create_table(
        "ai_message",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(36), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("answer_tokens", sa.Integer(), nullable=False),
        sa.Column("prompt_version", sa.String(40), nullable=False),
        sa.Column("effort", sa.String(10), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["ai_conversation.id"],
            name="fk_ai_message_conversation_id_ai_conversation",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_ai_message"),
        sa.UniqueConstraint("conversation_id", "position", name="uq_ai_message_conversation_id"),
    )
    op.create_index("ix_ai_message_conversation_id", "ai_message", ["conversation_id"])
    op.create_table(
        "ai_document_chunk",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("workspace_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=True),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("source_id", sa.String(64), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("url", sa.String(500), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_ai_document_chunk"),
    )
    op.create_index("ix_ai_document_chunk_workspace_id", "ai_document_chunk", ["workspace_id"])
    op.create_index("ix_ai_document_chunk_project_id", "ai_document_chunk", ["project_id"])
    op.create_index("ix_ai_chunk_ws_source", "ai_document_chunk", ["workspace_id", "source_type", "source_id"])


def downgrade() -> None:
    op.drop_table("ai_document_chunk")
    op.drop_table("ai_message")
    op.drop_table("ai_conversation")
    op.drop_table("ai_workspace_setting")
