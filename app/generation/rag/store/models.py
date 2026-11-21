from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, declared_attr, mapped_column

from app.config import EMBEDDING_DIMENSION
from app.foundation.persistence.models import Base

SOURCE_PATH_INDEX = "ix_documents_source_path"
# Text search configuration of the lexical branch (session 10). The corpus is in English. It must
# be identical in migration 0002, this model and ChunkSearchStore.search_lexical: with a different
# config Postgres cannot use the GIN index and the matching silently changes.
TEXT_SEARCH_CONFIG = "english"


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # Único (el ejemplo del enunciado es no único): garantiza el 409 con peticiones concurrentes
        Index(SOURCE_PATH_INDEX, "source_path", unique=True),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # `metadata` está reservado en declarative: atributo `metadata_` mapeado a la columna `metadata`
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )


class _ChunkColumns:
    """Columns and indexes shared by every chunk collection table (session 10 multi-index).

    A declarative mixin: SQLAlchemy copies each ``mapped_column`` onto the concrete table, so the
    three collections stay structurally identical without repeating the column list. What diverges
    between collections is the JSONB metadata schema, not the columns. Names must match migration
    0003 (``alembic check`` compares them).
    """

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Nullable por diseño del esquema (chunks futuros sin vector); la ingesta actual siempre lo rellena
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Read-only: Postgres computes it from `content` on every insert/update (no trigger to keep
    # in sync). deferred=True: loading a chunk entity does not fetch it unless asked
    content_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(f"to_tsvector('{TEXT_SEARCH_CONFIG}', content)", persisted=True),
        deferred=True,
    )

    @declared_attr.directive
    def __table_args__(cls) -> tuple[Index, ...]:
        name = cls.__tablename__  # type: ignore[attr-defined]
        return (
            Index(f"ix_{name}_document_id", "document_id"),
            Index(f"ix_{name}_chunk_type", "chunk_type"),
            Index(f"ix_{name}_metadata_gin", "metadata", postgresql_using="gin"),
            Index(f"ix_{name}_content_tsv", "content_tsv", postgresql_using="gin"),
        )


class BudgetChunkRow(_ChunkColumns, Base):
    """Chunks of historical budgets (the former ``chunks`` table)."""

    __tablename__ = "budget_chunks"


class TranscriptChunkRow(_ChunkColumns, Base):
    """Chunks of client meeting transcripts: metadata carries speakers / meeting_date."""

    __tablename__ = "transcript_chunks"


class TechnicalDocChunkRow(_ChunkColumns, Base):
    """Chunks of internal technical docs: metadata carries a version."""

    __tablename__ = "technical_doc_chunks"


# ORM-only compatibility aliases: ingestion, tests and the budget search still say "Chunk" and
# mean the budget collection (the teacher keeps `ChunkRow = BudgetChunkRow` the same way)
Chunk = BudgetChunkRow
ChunkRow = BudgetChunkRow


