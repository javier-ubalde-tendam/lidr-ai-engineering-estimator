import uuid
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import EMBEDDING_DIMENSION
from app.generation.rag.retrieval.pipeline import retrieve
from app.generation.rag.store.models import Chunk, Document
from app.generation.rag.store.repository import ChunkSearchStore
from tests.integration.conftest import run

pytestmark = pytest.mark.integration

QUERY = "cart checkout payments"
VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)


@pytest.fixture
def seeded(session_factory: async_sessionmaker[AsyncSession]) -> async_sessionmaker[AsyncSession]:
    async def go() -> None:
        async with session_factory() as session, session.begin():
            document = Document(source_path=f"seed-{uuid.uuid4()}", document_type="t", metadata_={})
            session.add(document)
            await session.flush()
            rows = [("finance", 2022), ("finance", 2024), ("ecommerce", 2024)]
            session.add_all(
                Chunk(
                    document_id=document.id,
                    chunk_type="budget_component",
                    content=f"{QUERY} {index}",
                    embedding=VECTOR,
                    metadata_={"budget_id": f"B{index}", "client_sector": sector, "year": year},
                )
                for index, (sector, year) in enumerate(rows)
            )

    run(go())
    return session_factory


def budgets(factory: async_sessionmaker[AsyncSession], mode: str, **filters: Any) -> list[str]:
    async def go() -> list[str]:
        async with factory() as session:
            result = await retrieve(
                session,
                ChunkSearchStore(),
                query_embedding=VECTOR,
                query_text=QUERY,
                search_mode=mode,  # type: ignore[arg-type]
                **filters,
            )
        return sorted(str(chunk.budget_id) for chunk in result.chunks)

    return run(go())


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, ["B0", "B1", "B2"]),
        ({"sectors": ["finance"]}, ["B0", "B1"]),
        ({"sectors": ["finance"], "project_year_min": 2023}, ["B1"]),
        ({"project_year_max": 2022}, ["B0"]),
        ({"chunk_types": ["nope"]}, []),
    ],
)
def test_metadata_filters_narrow_both_branches(
    seeded: async_sessionmaker[AsyncSession], mode: str, filters: dict[str, Any], expected: list[str]
) -> None:
    assert budgets(seeded, mode, **filters) == expected
