"""HNSW vector indexes on the three chunk collections (session 11 live)

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-21

Until now the collections ran on a sequential scan, which gets linearly slower as the corpus grows.
The retriever searches with ``cast(embedding, HALFVEC(1536)).cosine_distance(...)``, and an HNSW index
is only consulted when its operator class matches the query operator: a ``vector_cosine_ops`` index
would be silently ignored by that search. So the index is built on the SAME
``embedding::halfvec(1536)`` expression with ``halfvec_cosine_ops`` (half the index size too).

Once it exists HNSW is maintained incrementally: new chunks are indexed on insert, no rebuild.
``m=16`` / ``ef_construction=128`` mirror the session 8 live choices; ``IF NOT EXISTS`` keeps the
upgrade idempotent.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CHUNK_TABLES = ("budget_chunks", "transcript_chunks", "technical_doc_chunks")
EMBEDDING_DIMENSIONS = 1536
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 128


def _index_name(table: str) -> str:
    return f"ix_{table}_embedding_hnsw"


def upgrade() -> None:
    for table in CHUNK_TABLES:
        op.execute(
            f"CREATE INDEX IF NOT EXISTS {_index_name(table)} "
            f"ON {table} "
            f"USING hnsw ((embedding::halfvec({EMBEDDING_DIMENSIONS})) halfvec_cosine_ops) "
            f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        )


def downgrade() -> None:
    for table in CHUNK_TABLES:
        op.execute(f"DROP INDEX IF EXISTS {_index_name(table)}")
