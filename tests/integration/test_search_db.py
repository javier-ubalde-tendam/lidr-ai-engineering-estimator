import math
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, Document
from tests.embedding_helpers import FakeOpenAIClient, angle_vector
from tests.integration.conftest import run

pytestmark = pytest.mark.integration


def seed(
    factory: async_sessionmaker[AsyncSession],
    rows: list[tuple[str, list[float] | None, dict[str, Any]]],
) -> None:
    """Inserta un documento con un chunk por fila: (content, embedding, metadata)."""

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
                    embedding=embedding,
                    metadata_=metadata,
                )
                for content, embedding, metadata in rows
            )

    run(go())


@pytest.fixture(autouse=True)
def query_is_e0(fake_openai: FakeOpenAIClient) -> None:
    # La query siempre se embebe como e0 (ángulo 0): distancia coseno de un chunk = 1 - cos(ángulo)
    fake_openai._vector_for = lambda _text, _index: angle_vector(0.0)


def search(api: TestClient, k: int | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"query": "anything"}
    if k is not None:
        body["k"] = k
    response = api.post("/search", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_ranking_is_ascending_by_cosine_distance(
    api: TestClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    angles = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 1.2, "e": 0.3}
    seed(session_factory, [(name, angle_vector(angle), {}) for name, angle in angles.items()])

    body = search(api, k=5)

    assert [r["content"] for r in body["results"]] == ["b", "e", "c", "a", "d"]
    distances = [r["distance"] for r in body["results"]]
    assert distances == sorted(distances)
    for result in body["results"]:
        assert result["distance"] == pytest.approx(1 - math.cos(angles[result["content"]]), abs=1e-5)


def test_k_limits_results_and_default_is_5(
    api: TestClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    seed(session_factory, [(f"c{i}", angle_vector(0.05 * (i + 1)), {}) for i in range(8)])

    assert [r["content"] for r in search(api, k=2)["results"]] == ["c0", "c1"]
    default = search(api)
    assert default["k"] == 5 and len(default["results"]) == 5


def test_fewer_chunks_than_k_returns_the_available_ones(
    api: TestClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    seed(session_factory, [("only", angle_vector(0.2), {}), ("two", angle_vector(0.4), {})])

    assert len(search(api, k=10)["results"]) == 2


def test_empty_corpus_returns_200_with_no_results(api: TestClient) -> None:
    body = search(api, k=5)

    assert body["results"] == []
    assert body["query"] == "anything" and body["k"] == 5
    assert isinstance(body["search_time_ms"], int)


def test_null_embeddings_are_excluded(
    api: TestClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    seed(
        session_factory,
        [("no vector", None, {}), ("near", angle_vector(0.2), {}), ("far", angle_vector(1.0), {})],
    )

    results = search(api, k=5)["results"]

    assert [r["content"] for r in results] == ["near", "far"]
    assert all(r["distance"] is not None for r in results)


def test_metadata_and_ids_are_serialized(
    api: TestClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    metadata = {"scope": "backend", "technologies": ["python", "fastapi"], "year": 2024}
    seed(session_factory, [("c", angle_vector(0.1), metadata)])

    (result,) = search(api)["results"]

    assert result["metadata"] == metadata
    assert result["chunk_type"] == "budget_component"
    assert isinstance(result["chunk_id"], int) and isinstance(result["document_id"], int)


def test_ranking_uses_cosine_not_euclidean_distance(
    api: TestClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    # "scaled" apunta más cerca de la query pero con norma 100: L2 lo pondría el último
    scaled = [100 * x for x in angle_vector(0.3)]
    seed(session_factory, [("plain", angle_vector(0.4), {}), ("scaled", scaled, {})])

    results = search(api)["results"]

    assert [r["content"] for r in results] == ["scaled", "plain"]
    assert results[0]["distance"] == pytest.approx(1 - math.cos(0.3), abs=1e-5)


def test_sql_orders_and_limits_in_postgres_with_bound_parameters(
    api: TestClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    seed(session_factory, [("c", angle_vector(0.1), {})])
    statements: list[tuple[str, Any]] = []

    @event.listens_for(session_factory.kw["bind"].sync_engine, "before_cursor_execute")
    def capture(_conn: Any, _cursor: Any, statement: str, parameters: Any, *_: Any) -> None:
        statements.append((statement, parameters))

    search(api, k=3)

    statement, _ = next(s for s in statements if "FROM chunks" in s[0] and "<=>" in s[0])
    assert "ORDER BY" in statement and "LIMIT" in statement
    assert "IS NOT NULL" in statement
    assert "[" not in statement  # el vector viaja como parámetro, no interpolado


def test_no_vector_index_exists(sql: Callable[..., list[Any]]) -> None:
    indexes = sql(
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks' AND "
        "(indexdef ILIKE '%hnsw%' OR indexdef ILIKE '%ivfflat%' OR indexdef ILIKE '%diskann%')"
    )

    assert indexes == []
