"""``hnsw.ef_search`` is raised inside the transaction so the HNSW scan can return ``LIMIT`` rows."""

from typing import Any

import pytest

from app.generation.rag.store.repository import (
    HNSW_DEFAULT_EF_SEARCH,
    HNSW_MAX_EF_SEARCH,
    ChunkSearchStore,
    hnsw_ef_search_for,
)

pytestmark = pytest.mark.anyio


class FakeResult:
    def all(self) -> list[Any]:
        return []


class RecordingSession:
    """Records the compiled statements in the order they are issued."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.params: list[dict[str, Any]] = []

    async def execute(self, statement: Any) -> FakeResult:
        compiled = statement.compile()
        self.statements.append(str(compiled))
        self.params.append(dict(compiled.params))
        return FakeResult()

    async def scalar(self, statement: Any) -> int:
        self.statements.append(str(statement.compile()))
        return 0


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (1, HNSW_DEFAULT_EF_SEARCH),  # never below pgvector's default
        (10, HNSW_DEFAULT_EF_SEARCH),
        (50, 50),  # the default recall_k that used to be truncated at 40
        (400, 400),
        (5000, HNSW_MAX_EF_SEARCH),  # pgvector rejects values above 1000
    ],
)
def test_ef_search_covers_the_limit_within_pgvector_bounds(limit: int, expected: int) -> None:
    assert hnsw_ef_search_for(limit) == expected


async def test_search_filtered_sets_ef_search_locally_before_the_knn_query() -> None:
    session = RecordingSession()

    await ChunkSearchStore().search_filtered(
        session,  # type: ignore[arg-type]
        query_vector=[0.1] * 1536,
        top_k=120,
        distance_threshold=0.5,
    )

    first = session.statements[0]
    assert "set_config" in first
    # is_local=true -> SET LOCAL semantics: the value dies with the transaction, no leak
    values = list(session.params[0].values())
    assert "hnsw.ef_search" in values and "120" in values and values[-1] is True
    assert "hnsw" not in " ".join(session.statements[1:])
    assert len(session.statements) == 3  # set_config, candidates count, k-NN
