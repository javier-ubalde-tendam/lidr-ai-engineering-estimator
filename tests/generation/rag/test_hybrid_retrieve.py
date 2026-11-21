"""retrieve() composition with a fake store, fake session and fake cross-encoder (no DB, no torch)."""

from types import SimpleNamespace
from typing import Any, Self, cast

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.pipeline import (
    NO_VECTOR_DISTANCE,
    resolve_search_options,
    retrieve,
)
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.store import ChunkSearchStore

pytestmark = pytest.mark.anyio


def row(chunk_id: int, distance: float | None = None, budget: str = "BUD-X") -> Any:
    # Mimics a SQLAlchemy Row: attribute access only
    return SimpleNamespace(
        id=chunk_id,
        chunk_type="budget_component",
        content=f"chunk {chunk_id}",
        metadata_={"budget_id": budget, "client_sector": "ecommerce", "year": 2024},
        distance=distance,
    )


class FakeSession:
    """Only what retrieve() uses: ``async with session.begin()``."""

    def begin(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class FakeStore(ChunkSearchStore):
    def __init__(
        self, vector: list[Any], lexical: list[Any] | None = None, fail: bool = False
    ) -> None:
        self._vector = vector
        self._lexical = lexical or []
        self._fail = fail
        self.vector_calls: list[dict[str, Any]] = []
        self.lexical_calls: list[dict[str, Any]] = []

    async def search_filtered(self, session: Any, **kwargs: Any) -> tuple[list[Any], int]:
        if self._fail:
            raise OperationalError("SELECT", {}, Exception("connection refused"))
        self.vector_calls.append(kwargs)
        return self._vector[: kwargs["top_k"]], 60

    async def search_lexical(self, session: Any, **kwargs: Any) -> list[Any]:
        self.lexical_calls.append(kwargs)
        return self._lexical[: kwargs["top_k"]]


class ScoreByIdModel:
    """Fake cross-encoder: score = value mapped from the chunk content."""

    def __init__(self, scores: dict[int, float]) -> None:
        self._scores = scores
        self.pairs_seen = 0

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs_seen += len(pairs)
        return [self._scores.get(int(doc.split()[-1]), 0.0) for _q, doc in pairs]


def fake_reranker(scores: dict[int, float]) -> tuple[CrossEncoderReranker, ScoreByIdModel]:
    reranker = CrossEncoderReranker("fake")
    model = ScoreByIdModel(scores)
    reranker._model = model
    return reranker, model


SESSION = cast(AsyncSession, FakeSession())
VECTOR = [row(i, distance=0.1 * i) for i in range(1, 21)]  # ids 1..20 ascending distance


async def run(store: FakeStore, **kwargs: Any) -> Any:
    params: dict[str, Any] = {"query_embedding": [0.0], "query_text": "query", "top_k": 3}
    return await retrieve(SESSION, store, **(params | kwargs))


async def test_config_a_vector_without_rerank_recalls_only_top_k() -> None:
    store = FakeStore(VECTOR)

    result = await run(store, search_mode="vector", rerank=False, recall_k=10)

    assert store.vector_calls[0]["top_k"] == 3
    assert store.lexical_calls == []
    assert [c.id for c in result.chunks] == [1, 2, 3]
    assert result.search_mode == "vector" and result.rerank is False
    assert result.candidates_evaluated == 60 and not result.low_confidence


async def test_config_b_hybrid_recalls_wide_and_fuses_both_branches() -> None:
    # Lexical favours 9 then 2: 2 is in both branches, 9 only lexical, 1 only vector top
    store = FakeStore(VECTOR, lexical=[row(9), row(2), row(99, budget="BUD-LEX")])

    result = await run(store, search_mode="hybrid", rerank=False, recall_k=10, rrf_k=60)

    assert store.vector_calls[0]["top_k"] == 10
    assert store.lexical_calls[0]["top_k"] == 10
    ids = [c.id for c in result.chunks]
    # 2 = 1/61 + 1/61 ; 9 = 1/68 (vector rank 8) + 1/60 ; 1 = 1/60 (vector only)
    assert ids == [2, 9, 1]
    assert result.lexical_hits == 3 and result.vector_hits == 10


async def test_hybrid_lexical_only_chunk_gets_the_sentinel_distance() -> None:
    store = FakeStore([row(1, distance=0.2)], lexical=[row(42)])

    result = await run(store, search_mode="hybrid", top_k=5)

    by_id = {c.id: c for c in result.chunks}
    assert by_id[42].distance == NO_VECTOR_DISTANCE
    assert by_id[1].distance == pytest.approx(0.2)


async def test_hybrid_chunk_in_both_branches_keeps_its_vector_distance() -> None:
    store = FakeStore([row(1, distance=0.2)], lexical=[row(1)])

    result = await run(store, search_mode="hybrid")

    assert result.chunks[0].distance == pytest.approx(0.2)


async def test_config_c_vector_with_rerank_reorders_and_cuts_to_top_n() -> None:
    store = FakeStore(VECTOR)
    reranker, model = fake_reranker({15: 0.9, 7: 0.8, 3: 0.7})

    result = await run(
        store, search_mode="vector", rerank=True, recall_k=20, rerank_top_n=2, reranker=reranker
    )

    assert store.vector_calls[0]["top_k"] == 20  # recall wide
    assert model.pairs_seen == 20  # the whole pool is rescored
    assert [c.id for c in result.chunks] == [15, 7]
    assert result.rerank is True


async def test_config_d_hybrid_with_rerank() -> None:
    store = FakeStore(VECTOR, lexical=[row(99)])
    reranker, model = fake_reranker({99: 5.0, 4: 1.0})

    result = await run(
        store, search_mode="hybrid", rerank=True, recall_k=10, rerank_top_n=2, reranker=reranker
    )

    # 11 fused candidates (10 vector + 1 lexical-only), pool capped at recall_k before reranking
    assert model.pairs_seen == 10
    assert [c.id for c in result.chunks] == [99, 4]
    assert result.search_mode == "hybrid" and result.rerank is True


async def test_rerank_off_never_builds_the_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise AssertionError("get_reranker must not be called")

    monkeypatch.setattr("app.dependencies.get_reranker", boom)

    await run(FakeStore(VECTOR), search_mode="hybrid", rerank=False)


async def test_rerank_on_without_injection_uses_the_app_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reranker, _ = fake_reranker({2: 1.0})
    monkeypatch.setattr("app.dependencies.get_reranker", lambda: reranker)

    result = await run(FakeStore(VECTOR), rerank=True, recall_k=5, rerank_top_n=1)

    assert [c.id for c in result.chunks] == [2]


async def test_soft_fail_when_nothing_is_retrieved() -> None:
    reranker, model = fake_reranker({})

    result = await run(FakeStore([]), search_mode="hybrid", rerank=True, reranker=reranker)

    assert result.chunks == [] and result.low_confidence is True
    assert model.pairs_seen == 0


async def test_database_failure_raises_retrieval_error() -> None:
    with pytest.raises(RetrievalError):
        await run(FakeStore(VECTOR, fail=True))


async def test_reranker_failure_raises_retrieval_error() -> None:
    reranker = CrossEncoderReranker("fake")
    reranker._model = SimpleNamespace(predict=lambda _pairs: 1 / 0)

    with pytest.raises(RetrievalError):
        await run(FakeStore(VECTOR), rerank=True, reranker=reranker)


async def test_lexical_match_mode_is_forwarded_to_the_store() -> None:
    store = FakeStore(VECTOR)

    await run(store, search_mode="hybrid", lexical_match_mode="and")

    assert store.lexical_calls[0]["match_mode"] == "and"


async def test_budget_id_and_metadata_are_mapped() -> None:
    result = await run(FakeStore([row(1, distance=0.1, budget="BUD-2024-005")]))

    (chunk,) = result.chunks
    assert chunk.budget_id == "BUD-2024-005"
    assert chunk.sector == "ecommerce" and chunk.project_year == 2024


# --- precedence: request field > environment > default ---


def test_precedence_request_beats_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_SEARCH_MODE", "hybrid")
    monkeypatch.setenv("RERANKER_ENABLED", "true")

    assert resolve_search_options("vector", False, Settings()) == ("vector", False)


def test_precedence_environment_beats_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_SEARCH_MODE", "hybrid")
    monkeypatch.setenv("RERANKER_ENABLED", "true")

    assert resolve_search_options(None, None, Settings()) == ("hybrid", True)


def test_precedence_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_SEARCH_MODE", "vector")
    monkeypatch.setenv("RERANKER_ENABLED", "false")

    assert resolve_search_options(None, None, Settings()) == ("vector", False)
    assert Settings.model_fields["RETRIEVAL_SEARCH_MODE"].default == "vector"
    assert Settings.model_fields["RERANKER_ENABLED"].default is False
