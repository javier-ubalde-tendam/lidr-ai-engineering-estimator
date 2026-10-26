import pytest
from pgvector.sqlalchemy import Vector
from pydantic import ValidationError

from app.config import (
    EMBEDDING_DIMENSION,
    SUPPORTED_EMBEDDING_MODEL,
    DatabaseSettings,
    Settings,
)
from app.db.models import Base, Chunk, Document
from app.db.session import dispose_engine, get_engine


def test_tables_columns_and_nullability_match_the_spec() -> None:
    documents, chunks = Base.metadata.tables["documents"], Base.metadata.tables["chunks"]

    assert set(documents.columns.keys()) == {
        "id", "source_path", "document_type", "ingested_at", "metadata",
    }  # fmt: skip
    assert set(chunks.columns.keys()) == {
        "id", "document_id", "chunk_type", "content", "embedding", "metadata", "created_at",
    }  # fmt: skip
    assert chunks.c.embedding.nullable is True
    assert isinstance(chunks.c.embedding.type, Vector)
    assert chunks.c.embedding.type.dim == EMBEDDING_DIMENSION == 1536
    assert documents.c.document_type.type.length == 50
    assert chunks.c.chunk_type.type.length == 50
    assert not chunks.c.document_id.nullable
    (foreign_key,) = chunks.c.document_id.foreign_keys
    assert foreign_key.ondelete == "CASCADE"


def test_orm_uses_metadata_underscore_attribute_for_the_metadata_column() -> None:
    assert Document.metadata_.property.columns[0].name == "metadata"
    assert Chunk.metadata_.property.columns[0].name == "metadata"


def test_indexes_are_relational_only_and_source_path_is_unique() -> None:
    documents, chunks = Base.metadata.tables["documents"], Base.metadata.tables["chunks"]
    indexes = {index.name: index for index in [*documents.indexes, *chunks.indexes]}

    assert set(indexes) == {
        "ix_documents_source_path",
        "ix_chunks_document_id",
        "ix_chunks_chunk_type",
        "ix_chunks_metadata_gin",
    }
    assert indexes["ix_documents_source_path"].unique
    assert indexes["ix_chunks_metadata_gin"].dialect_options["postgresql"]["using"] == "gin"
    # Ningún índice vectorial (HNSW/IVFFlat): la búsqueda exacta es el baseline
    assert all("embedding" not in index.columns for index in indexes.values())


def test_embedding_model_other_than_the_contract_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no es compatible"):
        Settings(OPENAI_API_KEY="k", EMBEDDING_MODEL="text-embedding-3-large", _env_file=None)  # type: ignore[call-arg]
    assert (
        Settings(OPENAI_API_KEY="k", _env_file=None).EMBEDDING_MODEL  # type: ignore[call-arg]
        == SUPPORTED_EMBEDDING_MODEL
    )


def test_database_settings_do_not_require_llm_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/x")

    assert DatabaseSettings(_env_file=None).DATABASE_URL.endswith("@db:5432/x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.anyio
async def test_dispose_engine_without_engine_does_not_connect() -> None:
    get_engine.cache_clear()

    await dispose_engine()

    assert get_engine.cache_info().currsize == 0
