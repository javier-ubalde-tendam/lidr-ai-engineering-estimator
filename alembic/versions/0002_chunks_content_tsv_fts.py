"""Full-text search on chunks: generated content_tsv column + GIN index (session 10)

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-08

Lexical branch of the hybrid search. ``content_tsv`` is ``GENERATED ALWAYS ... STORED``:
Postgres recomputes it from ``content`` on every insert/update, so there is no trigger to
maintain and the existing rows are filled in by the ALTER TABLE itself.

Text search configuration: the exercise statement says the dataset is in Spanish, but
``data/budgets_sample.json`` is actually in ENGLISH ("Faceted search", "Order lifecycle"...).
``english`` stemming and stop-words are what improve lexical recall on this corpus. The config
name must stay identical here, in ``app.db.models.TEXT_SEARCH_CONFIG`` and in the store query,
otherwise the GIN index is not used and matching changes silently (a unit test checks it).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Raw SQL: Alembic has no first-class operation for a STORED generated tsvector column
    op.execute(
        "ALTER TABLE chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.create_index("ix_chunks_content_tsv", "chunks", ["content_tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_content_tsv", table_name="chunks", postgresql_using="gin")
    op.drop_column("chunks", "content_tsv")
