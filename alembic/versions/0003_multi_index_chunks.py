"""Multi-index: chunks -> budget_chunks, plus transcript_chunks and technical_doc_chunks (session 10 live)

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-21

The heterogeneous corpus is split into one table per collection, so each one has its own metadata
schema, vector index and lifecycle (instead of a single ``chunks`` table where the dominant
document type floods the top-k). This revision:

1. Renames ``chunks`` to ``budget_chunks`` with its indexes, primary key, foreign key and id
   sequence. It is a pure rename: no row is copied or lost, and the STORED generated
   ``content_tsv`` column added in 0002 travels with the table.
2. Creates ``transcript_chunks`` and ``technical_doc_chunks`` with the SAME structural columns as
   ``budget_chunks`` (they share the ORM column mixin) and their own GIN indexes.

Text search configuration stays ``english`` everywhere (see the note in 0002). No vector index is
created here: HNSW comes in 0004.
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FTS_REGCONFIG = "english"
NEW_COLLECTIONS = ("transcript_chunks", "technical_doc_chunks")

# (old, new) names for the chunks -> budget_chunks table
_INDEX_RENAMES = (
    ("ix_chunks_document_id", "ix_budget_chunks_document_id"),
    ("ix_chunks_chunk_type", "ix_budget_chunks_chunk_type"),
    ("ix_chunks_metadata_gin", "ix_budget_chunks_metadata_gin"),
    ("ix_chunks_content_tsv", "ix_budget_chunks_content_tsv"),
)
_CONSTRAINT_RENAMES = (
    ("chunks_pkey", "budget_chunks_pkey"),
    ("chunks_document_id_fkey", "budget_chunks_document_id_fkey"),
)
_SEQUENCE_RENAME = ("chunks_id_seq", "budget_chunks_id_seq")


def _rename_constraint(table: str, old: str, new: str) -> None:
    # Guarded: a database created by hand may not have the default constraint names
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '{old}') THEN
                ALTER TABLE {table} RENAME CONSTRAINT {old} TO {new};
            END IF;
        END $$;
        """
    )


def _create_chunk_table(table: str) -> None:
    """Create one collection table identical in shape to ``budget_chunks``."""
    op.create_table(
        table,
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_type", sa.String(length=50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=1536), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # Raw SQL: Alembic has no first-class operation for a STORED generated tsvector column
    op.execute(
        f"ALTER TABLE {table} ADD COLUMN content_tsv tsvector "
        f"GENERATED ALWAYS AS (to_tsvector('{FTS_REGCONFIG}', content)) STORED"
    )
    op.create_index(f"ix_{table}_document_id", table, ["document_id"])
    op.create_index(f"ix_{table}_chunk_type", table, ["chunk_type"])
    op.create_index(f"ix_{table}_metadata_gin", table, ["metadata"], postgresql_using="gin")
    op.create_index(f"ix_{table}_content_tsv", table, ["content_tsv"], postgresql_using="gin")


def _drop_chunk_table(table: str) -> None:
    # Dropping the table drops its indexes; the explicit drops keep the downgrade symmetrical
    op.drop_index(f"ix_{table}_content_tsv", table_name=table, postgresql_using="gin")
    op.drop_index(f"ix_{table}_metadata_gin", table_name=table, postgresql_using="gin")
    op.drop_index(f"ix_{table}_chunk_type", table_name=table)
    op.drop_index(f"ix_{table}_document_id", table_name=table)
    op.drop_table(table)


def upgrade() -> None:
    op.rename_table("chunks", "budget_chunks")
    for old, new in _INDEX_RENAMES:
        op.execute(f"ALTER INDEX {old} RENAME TO {new}")
    for old, new in _CONSTRAINT_RENAMES:
        _rename_constraint("budget_chunks", old, new)
    op.execute(f"ALTER SEQUENCE IF EXISTS {_SEQUENCE_RENAME[0]} RENAME TO {_SEQUENCE_RENAME[1]}")

    for table in NEW_COLLECTIONS:
        _create_chunk_table(table)


def downgrade() -> None:
    # Only the two empty-by-design collections are dropped: budget data is never touched
    for table in NEW_COLLECTIONS:
        _drop_chunk_table(table)

    op.execute(f"ALTER SEQUENCE IF EXISTS {_SEQUENCE_RENAME[1]} RENAME TO {_SEQUENCE_RENAME[0]}")
    for old, new in _CONSTRAINT_RENAMES:
        _rename_constraint("budget_chunks", new, old)
    for old, new in _INDEX_RENAMES:
        op.execute(f"ALTER INDEX {new} RENAME TO {old}")
    op.rename_table("budget_chunks", "chunks")
