"""Offline ingestion tables: pseudonym_mappings and ingestion_jobs (session 6)

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-21

Adapted from the teacher's ``0001_session6_initial``: only the two session 6 tables are created
here, because the RAG tables (documents, chunks) already exist from 0001-0003. Nothing existing is
touched, so the upgrade preserves all data; the downgrade drops only what this revision created.

* ``pseudonym_mappings``: salted HMAC of an original value -> stable pseudonym, per entity type.
  UNIQUE(entity_type, original_hash) is what makes concurrent pseudonymization converge.
* ``ingestion_jobs``: one row per offline ingestion run (pending/running/completed/failed).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pseudonym_mappings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("original_hash", sa.String(length=128), nullable=False),
        sa.Column("pseudonym", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("entity_type", "original_hash", name="uq_mappings_entity_hash"),
    )
    op.create_index(
        "idx_mappings_lookup", "pseudonym_mappings", ["entity_type", "original_hash"]
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column("job_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("documents_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error_message", sa.String(length=2048), nullable=True),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_index("idx_jobs_status", "ingestion_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("idx_jobs_status", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_index("idx_mappings_lookup", table_name="pseudonym_mappings")
    op.drop_table("pseudonym_mappings")
