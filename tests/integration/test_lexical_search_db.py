import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from alembic import command
from app.db.models import TEXT_SEARCH_CONFIG, Chunk, Document
from app.generation.rag.store import ChunkSearchStore
from tests.embedding_helpers import angle_vector
from tests.integration.conftest import ROOT, run

pytestmark = pytest.mark.integration

# A real query is a long project description: no single chunk contains ALL its lexemes
LONG_QUERY = (
    "Build an online store with a shopping cart, checkout, product reviews, "
    "loyalty programme and multi-currency invoicing"
)
CART_CHUNK = "Component: Cart and checkout service. Persistent cart, promotions and checkout."
OTHER_CHUNK = "Component: Telemedicine video sessions with appointment reminders."


def seed(factory: async_sessionmaker[AsyncSession], contents: list[str]) -> None:
    async def go() -> None:
        async with factory() as session, session.begin():
            document = Document(source_path=f"seed-{uuid.uuid4()}", document_type="t", metadata_={})
            session.add(document)
            await session.flush()
            session.add_all(
                Chunk(
                    document_id=document.id,
                    chunk_type="budget_component",
                    content=content,
                    embedding=None,
                    metadata_={"budget_id": f"BUD-{index}"},
                )
                for index, content in enumerate(contents)
            )

    run(go())


def lexical(
    factory: async_sessionmaker[AsyncSession], query: str, match_mode: Any = "or"
) -> list[Any]:
    async def go() -> list[Any]:
        async with factory() as session, session.begin():
            return await ChunkSearchStore().search_lexical(
                session, query_text=query, top_k=10, match_mode=match_mode
            )

    return run(go())


def test_migration_creates_generated_column_and_gin_index(sql: Callable[..., list[Any]]) -> None:
    (column,) = sql(
        "SELECT data_type, is_generated, generation_expression FROM information_schema.columns "
        "WHERE table_name = 'chunks' AND column_name = 'content_tsv'"
    )
    (index,) = sql("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_content_tsv'")

    assert column.data_type == "tsvector"
    assert column.is_generated == "ALWAYS"
    # Same config as the ORM model and the store query (otherwise the GIN index is bypassed)
    assert f"'{TEXT_SEARCH_CONFIG}'" in column.generation_expression
    assert "USING gin (content_tsv)" in index.indexdef


def test_column_is_filled_automatically_on_insert(
    session_factory: async_sessionmaker[AsyncSession], sql: Callable[..., list[Any]]
) -> None:
    seed(session_factory, [CART_CHUNK])

    (row,) = sql("SELECT content_tsv::text AS tsv FROM chunks")

    # english stemming: "checkout" -> 'checkout', "promotions" -> 'promot'
    assert "'checkout'" in row.tsv and "'promot'" in row.tsv


def test_or_mode_finds_rows_where_and_mode_finds_none(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seed(session_factory, [CART_CHUNK, OTHER_CHUNK])

    assert lexical(session_factory, LONG_QUERY, "and") == []
    rows = lexical(session_factory, LONG_QUERY, "or")
    assert [row.content for row in rows] == [CART_CHUNK]


def test_lexical_ranks_chunks_covering_more_terms_first(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seed(session_factory, ["Checkout page only.", CART_CHUNK])

    rows = lexical(session_factory, "shopping cart and checkout with promotions")

    assert [row.content for row in rows] == [CART_CHUNK, "Checkout page only."]
    assert rows[0].rank > rows[1].rank


def test_stop_words_only_query_returns_no_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seed(session_factory, [CART_CHUNK])

    assert lexical(session_factory, "and the of with") == []


def test_downgrade_and_upgrade_on_a_populated_table(
    migrated_database: str, sql: Callable[..., list[Any]]
) -> None:
    config = Config(str(Path(ROOT) / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", migrated_database.replace("%", "%%"))

    command.downgrade(config, "0001")
    try:
        assert (
            sql(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'chunks' AND column_name = 'content_tsv'"
            )
            == []
        )
        assert sql("SELECT 1 FROM pg_indexes WHERE indexname = 'ix_chunks_content_tsv'") == []
        # Rows that exist BEFORE the migration (like the 60 chunks of the dev DB)
        sql(
            "INSERT INTO documents (source_path, document_type) VALUES ('pre-0002', 't') "
            "RETURNING id"
        )
        sql(
            "INSERT INTO chunks (document_id, chunk_type, content) "
            "SELECT id, 'budget_component', :content FROM documents RETURNING id",
            content=CART_CHUNK,
        )
    finally:
        command.upgrade(config, "head")

    (row,) = sql("SELECT content_tsv IS NOT NULL AS filled FROM chunks")
    assert row.filled is True


def test_search_filtered_applies_the_distance_threshold(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async def go() -> tuple[list[Any], int]:
        async with session_factory() as session, session.begin():
            document = Document(source_path="vec", document_type="t", metadata_={})
            session.add(document)
            await session.flush()
            # Cosine distances to e0: 1 - cos(0.2) ~ 0.02 (near) and 1 - cos(1.4) ~ 0.83 (far)
            for name, angle in (("near", 0.2), ("far", 1.4)):
                session.add(
                    Chunk(
                        document_id=document.id,
                        chunk_type="c",
                        content=name,
                        embedding=angle_vector(angle),
                        metadata_={},
                    )
                )
            session.add(
                Chunk(
                    document_id=document.id,
                    chunk_type="c",
                    content="nov",
                    embedding=None,
                    metadata_={},
                )
            )
            await session.flush()
            return await ChunkSearchStore().search_filtered(
                session, query_vector=angle_vector(0.0), top_k=10, distance_threshold=0.6
            )

    rows, candidates_evaluated = run(go())

    assert [row.content for row in rows] == ["near"]
    assert candidates_evaluated == 2  # only chunks with an embedding
