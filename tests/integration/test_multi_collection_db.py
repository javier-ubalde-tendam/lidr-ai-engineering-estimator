"""Multi-index store against real Postgres: one table per collection, same queries on each."""

import uuid
from datetime import date

import pytest
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from alembic import command
from app.config import EMBEDDING_DIMENSION
from app.generation.rag.retrieval.collections import Collection, HardFilters, spec_for
from app.generation.rag.retrieval.pipeline import retrieve
from app.generation.rag.schemas import EmbeddedChunk
from app.generation.rag.store.repository import ChunkSearchStore, DocumentRepository
from tests.integration.conftest import ROOT, run

pytestmark = pytest.mark.integration

VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)


def embedded(chunk_id: str, text: str, metadata: dict[str, str | int]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id, text=text, metadata=metadata, token_count=3, embedding=VECTOR
    )


@pytest.fixture
def seeded(session_factory: async_sessionmaker[AsyncSession]) -> async_sessionmaker[AsyncSession]:
    async def go() -> None:
        async with session_factory() as session, session.begin():
            repository = DocumentRepository(session)
            await repository.persist_document_with_chunks(
                source_path=f"budgets-{uuid.uuid4()}",
                document_type="historical_budget",
                doc_metadata={},
                embedded_chunks=[
                    embedded("B::1", "stripe payments budget", {"budget_id": "B1", "year": 2024})
                ],
            )
            for collection, texts in (
                (
                    Collection.TRANSCRIPT,
                    [("old call", "2022-01-10"), ("recent call", "2024-06-01")],
                ),
                (Collection.TECHNICAL_DOC, [("api spec", "v2")]),
            ):
                key = "meeting_date" if collection is Collection.TRANSCRIPT else "version"
                await repository.persist_document_with_chunks(
                    source_path=f"{collection.value}-{uuid.uuid4()}",
                    document_type=collection.value,
                    doc_metadata={},
                    embedded_chunks=[
                        embedded(f"{collection.value}::{index}", text, {key: value})
                        for index, (text, value) in enumerate(texts)
                    ],
                    collection=collection,
                )

    run(go())
    return session_factory


def contents(factory: async_sessionmaker[AsyncSession], collection: Collection, **kwargs) -> list[str]:  # type: ignore[no-untyped-def]
    async def go() -> list[str]:
        async with factory() as session:
            result = await retrieve(
                session,
                ChunkSearchStore(),
                query_embedding=VECTOR,
                query_text="call",
                collection=collection,
                search_mode="hybrid",
                **kwargs,
            )
        return sorted(chunk.content for chunk in result.chunks)

    return run(go())


def test_orm_and_migrations_agree_no_schema_drift(migrated_database: str) -> None:
    # Includes the HNSW indexes: they are excluded from autogenerate in alembic/env.py
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", migrated_database.replace("%", "%%"))

    command.check(config)  # raises CommandError when the models drift from the migrations


def test_each_collection_only_sees_its_own_table(seeded: async_sessionmaker[AsyncSession]) -> None:
    assert contents(seeded, Collection.BUDGET) == ["stripe payments budget"]
    assert contents(seeded, Collection.TRANSCRIPT) == ["old call", "recent call"]
    assert contents(seeded, Collection.TECHNICAL_DOC) == ["api spec"]


def test_persist_stamps_the_chunk_type_of_the_collection(
    seeded: async_sessionmaker[AsyncSession], sql
) -> None:  # type: ignore[no-untyped-def]
    assert sql("SELECT DISTINCT chunk_type FROM transcript_chunks")[0].chunk_type == "meeting_segment"
    assert sql("SELECT DISTINCT chunk_type FROM technical_doc_chunks")[0].chunk_type == "doc_section"


def test_hard_filter_clauses_apply_inside_the_query(
    seeded: async_sessionmaker[AsyncSession],
) -> None:
    clauses = spec_for(Collection.TRANSCRIPT).hard_filter_clauses(
        HardFilters(min_date=date(2024, 1, 1))
    )

    assert contents(seeded, Collection.TRANSCRIPT, extra_filters=clauses) == ["recent call"]


def test_retrieved_chunks_carry_provenance_and_date(
    seeded: async_sessionmaker[AsyncSession],
) -> None:
    async def go() -> list[tuple[str, date | None]]:
        async with seeded() as session:
            result = await retrieve(
                session,
                ChunkSearchStore(),
                query_embedding=VECTOR,
                query_text="call",
                collection=Collection.TRANSCRIPT,
            )
        return [(chunk.collection, chunk.document_date) for chunk in result.chunks]

    assert sorted(run(go()), key=str) == [
        ("transcript", date(2022, 1, 10)),
        ("transcript", date(2024, 6, 1)),
    ]
