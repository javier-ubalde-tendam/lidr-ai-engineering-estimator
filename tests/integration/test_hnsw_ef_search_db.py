"""``hnsw.ef_search`` against the real HNSW index: a LIMIT above 40 must return that many rows."""

import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.generation.rag.store.models import Chunk, Document
from app.generation.rag.store.repository import ChunkSearchStore, halfvec_cosine_distance
from tests.embedding_helpers import angle_vector
from tests.integration.conftest import run

pytestmark = pytest.mark.integration

ROWS = 150
LIMIT = 100  # above pgvector's default ef_search (40)


def seed_chunks(factory: async_sessionmaker[AsyncSession]) -> None:
    async def go() -> None:
        async with factory() as session, session.begin():
            document = Document(source_path=f"ef-{uuid.uuid4()}", document_type="t", metadata_={})
            session.add(document)
            await session.flush()
            session.add_all(
                Chunk(
                    document_id=document.id,
                    chunk_type="budget_component",
                    content=f"chunk {i}",
                    embedding=angle_vector(0.001 * i),
                    metadata_={},
                )
                for i in range(ROWS)
            )

    run(go())


def test_search_filtered_returns_the_full_limit_through_the_hnsw_index(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seed_chunks(session_factory)

    async def search_with_index(*, raise_ef_search: bool) -> int:
        async with session_factory() as session, session.begin():
            # Force the HNSW scan: the planner prefers a seq scan on such a small table
            await session.execute(text("SET LOCAL enable_seqscan = off"))
            if raise_ef_search:
                rows, _ = await ChunkSearchStore().search_filtered(
                    session, query_vector=angle_vector(0.0), top_k=LIMIT, distance_threshold=2.0
                )
                return len(rows)
            # Control: the same k-NN statement WITHOUT the store's SET LOCAL
            distance = halfvec_cosine_distance(Chunk, angle_vector(0.0))
            statement = select(Chunk.id).order_by(distance).limit(LIMIT)
            return len((await session.execute(statement)).all())

    assert run(search_with_index(raise_ef_search=True)) == LIMIT
    # Documents WHY the store raises it: the default ef_search truncates the scan
    assert run(search_with_index(raise_ef_search=False)) < LIMIT


def test_ef_search_does_not_leak_outside_the_transaction(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seed_chunks(session_factory)  # so the k-NN really runs distance functions

    async def current_ef_search() -> str:
        # SHOW only knows pgvector's setting once the extension is loaded in that backend (the
        # test pool opens a new connection per transaction), hence the distance call first
        async with session_factory() as session:
            async with session.begin():
                await ChunkSearchStore().search_filtered(
                    session, query_vector=angle_vector(0.0), top_k=500, distance_threshold=2.0
                )
                inside = await session.scalar(text("SHOW hnsw.ef_search"))
            async with session.begin():
                await session.execute(text("SELECT '[1,2]'::vector <=> '[1,2]'::vector"))
                outside = await session.scalar(text("SHOW hnsw.ef_search"))
        return f"{inside}/{outside}"

    assert run(current_ef_search()) == "500/40"
