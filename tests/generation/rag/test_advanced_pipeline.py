"""Advanced pipeline: stage fusion helpers, StageConfig wiring and the async orchestration.

Every collaborator is a fake (store, session factory, embedder, LLM, models, reranker): no DB, no
network, no torch.
"""

from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import EMBEDDING_DIMENSION, Settings
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.advanced_pipeline import (
    StageConfig,
    _chunk_key,
    _fuse_collection,
    advanced_retrieve,
)
from app.generation.rag.retrieval.collections import Collection, HardFilters
from app.generation.rag.retrieval.pipeline import CollectionHits
from app.generation.rag.retrieval.query_transform import (
    DECOMPOSE,
    EXPAND,
    QueryTransformation,
    SubQuery,
)
from app.generation.rag.retrieval.router import RouteClassification
from app.generation.rag.schemas import RetrievedChunk
from app.generation.rag.store.repository import ChunkSearchStore

pytestmark = pytest.mark.anyio

TODAY = date(2024, 1, 1)
LONG_NO_RULES_QUERY = "what is the plan for onboarding and what else"  # no router vocabulary


def chunk(chunk_id: int, collection: str = "budget") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=f"{collection}-{chunk_id}",
        chunk_type="t",
        distance=0.2,
        collection=collection,
    )


def hits(vector_ids: list[int]) -> CollectionHits:
    return CollectionHits(
        collection=Collection.BUDGET,
        candidates={cid: chunk(cid) for cid in vector_ids},
        vector_ids=vector_ids,
        lexical_ids=[],
        candidates_evaluated=len(vector_ids),
    )


# --- pure helpers ---


def test_expansion_fuses_by_consensus_rrf() -> None:
    # id 3 is the only one both sub-queries surface: it must win
    fused = _fuse_collection(
        [hits([1, 2, 3]), hits([3, 4, 5])], technique=EXPAND, search_mode="vector", rrf_k=60
    )

    assert fused[0].id == 3


def test_decomposition_fuses_by_coverage_round_robin() -> None:
    fused = _fuse_collection(
        [hits([1, 2, 3]), hits([3, 4, 5])], technique=DECOMPOSE, search_mode="vector", rrf_k=60
    )

    assert [c.id for c in fused] == [1, 3, 2, 4, 5]


def test_chunk_key_separates_collections_with_the_same_id() -> None:
    assert _chunk_key(chunk(1, "budget")) != _chunk_key(chunk(1, "transcript"))


def test_stage_config_from_settings_applies_overrides() -> None:
    settings = Settings(_env_file=None, OPENAI_API_KEY="k")  # type: ignore[call-arg]

    stages = StageConfig.from_settings(
        settings,
        search_mode="hybrid",
        rerank=True,
        routing_enabled=False,  # explicit override beats the settings default
        temporal_decay_enabled=True,
        top_k=3,
    )

    assert stages.routing_enabled is False
    assert stages.query_transform_enabled is settings.QUERY_TRANSFORM_ENABLED  # from settings
    assert stages.temporal_decay_enabled is True
    assert (stages.search_mode, stages.rerank, stages.top_k) == ("hybrid", True, 3)
    assert stages.recall_k == settings.RETRIEVAL_RECALL_TOP_K
    assert stages.lexical_match_mode == settings.LEXICAL_MATCH_MODE
    # Models stay None: they are resolved through RuntimeModelConfig at call time
    assert stages.router_model is None and stages.query_transform_model is None


# --- orchestration with fakes ---


class FakeSessionFactory:
    def __call__(self) -> "FakeSessionFactory":
        return self

    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_: object) -> None:
        return None


class FakeStore(ChunkSearchStore):
    """Rows per (collection, first component of the query vector)."""

    def __init__(self, rows: dict[tuple[Collection, float], list[Any]], fail: bool = False):
        self._rows = rows
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

    async def search_filtered(self, session: Any, **kwargs: Any) -> tuple[list[Any], int]:
        if self._fail:
            raise RuntimeError("db down")
        self.calls.append(kwargs)
        rows = self._rows.get((kwargs["collection"], kwargs["query_vector"][0]), [])
        return rows[: kwargs["top_k"]], len(rows)

    async def search_lexical(self, session: Any, **kwargs: Any) -> list[Any]:
        return []


def row(chunk_id: int, **metadata: Any) -> Any:
    return SimpleNamespace(
        id=chunk_id, chunk_type="t", content=f"c{chunk_id}", metadata_=metadata, distance=0.1 * chunk_id
    )


class FakeEmbedder:
    """The first vector component identifies the sub-query: 0.0 unless the text says 'beta'."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.texts: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError("openai down")
        self.texts.append(text)
        return [1.0 if "beta" in text else 0.0] + [0.0] * (EMBEDDING_DIMENSION - 1)


class FakeLLM:
    def __init__(self, subqueries: list[SubQuery] | None = None, routed: list[Collection] | None = None):
        self.subqueries = subqueries or [SubQuery(topic="t", query="alpha")]
        self.routed = routed or [Collection.BUDGET]
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[Any, None]:
        self.calls.append(kwargs)
        if kwargs["response_model"] is QueryTransformation:
            return QueryTransformation(subqueries=self.subqueries), None
        return RouteClassification(targets=self.routed, reason="fake"), None


class FakeModels:
    def effective(self, key: str) -> str:
        return f"{key}-resolved"


class FakeReranker:
    """Logit per chunk label; records what it was asked to score."""

    def __init__(self, logits: dict[str, float], fail: bool = False) -> None:
        self.logits = logits
        self.fail = fail
        self.pool: list[str] = []

    def rerank_with_scores(
        self, query: str, candidates: list[RetrievedChunk], *, top_n: int
    ) -> list[tuple[RetrievedChunk, float]]:
        if self.fail:
            raise RuntimeError("torch exploded")
        self.pool = [c.content for c in candidates]
        scored = [(c, self.logits.get(c.content, 0.0)) for c in candidates]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:top_n]


def stages(**overrides: Any) -> StageConfig:
    base = {"routing_enabled": False, "query_transform_enabled": False, "search_mode": "vector",
            "rerank": False, "top_k": 5, "router_model": None, "query_transform_model": None}  # fmt: skip
    return StageConfig(**(base | overrides))


async def run(
    store: FakeStore,
    *,
    config: StageConfig,
    query: str = LONG_NO_RULES_QUERY,
    llm: FakeLLM | None = None,
    embedder: FakeEmbedder | None = None,
    reranker: FakeReranker | None = None,
    **kwargs: Any,
) -> Any:
    # Without routing the cascade falls back to ALL collections: pin budgets unless a test says so
    kwargs.setdefault("explicit_collections", None if config.routing_enabled else [Collection.BUDGET])
    return await advanced_retrieve(
        query_text=query,
        stages=config,
        reference_date=TODAY,
        llm=llm or FakeLLM(),  # type: ignore[arg-type]
        models=FakeModels(),  # type: ignore[arg-type]
        embedder=embedder or FakeEmbedder(),  # type: ignore[arg-type]
        session_factory=FakeSessionFactory(),  # type: ignore[arg-type]
        store=store,
        reranker=reranker,  # type: ignore[arg-type]
        **kwargs,
    )


async def test_collections_are_merged_round_robin_and_equal_ids_do_not_collide() -> None:
    store = FakeStore(
        {
            (Collection.BUDGET, 0.0): [row(1, budget_id="B1"), row(2, budget_id="B2")],
            (Collection.TRANSCRIPT, 0.0): [row(1, transcript_id="T1")],
        }
    )

    outcome = await run(
        store,
        config=stages(),
        explicit_collections=[Collection.BUDGET, Collection.TRANSCRIPT],
    )

    assert [(c.collection, c.id) for c in outcome.chunks] == [
        ("budget", 1),
        ("transcript", 1),  # same id as the budget chunk, kept thanks to the composite key
        ("budget", 2),
    ]
    assert outcome.cardinality == {"budget": 2, "transcript": 1}
    assert outcome.routing.level == "explicit" and outcome.technique == "direct"
    assert not outcome.low_confidence


async def test_without_rerank_the_relevance_score_is_positional() -> None:
    store = FakeStore({(Collection.BUDGET, 0.0): [row(1), row(2)]})

    outcome = await run(store, config=stages())

    assert [c.relevance_score for c in outcome.chunks] == [1.0, 0.5]


async def test_rerank_scores_go_through_sigmoid_and_decay_runs_last() -> None:
    store = FakeStore(
        {
            (Collection.BUDGET, 0.0): [
                row(1, year=2016),  # stale but the cross-encoder's favourite
                row(2, year=2024),
            ]
        }
    )
    reranker = FakeReranker({"c1": 2.0, "c2": 1.5})

    plain = await run(store, config=stages(rerank=True), reranker=reranker)
    decayed = await run(
        store, config=stages(rerank=True, temporal_decay_enabled=True), reranker=reranker
    )

    assert [c.id for c in plain.chunks] == [1, 2]
    assert 0.0 < plain.chunks[1].relevance_score < plain.chunks[0].relevance_score < 1.0
    assert [c.id for c in decayed.chunks] == [2, 1]  # recency overtakes the stale favourite
    assert decayed.chunks[1].relevance_score < plain.chunks[0].relevance_score


async def test_negative_logits_keep_their_order_under_decay() -> None:
    store = FakeStore({(Collection.BUDGET, 0.0): [row(1, year=2024), row(2, year=2024)]})
    reranker = FakeReranker({"c1": -1.0, "c2": -3.0})

    outcome = await run(
        store, config=stages(rerank=True, temporal_decay_enabled=True), reranker=reranker
    )

    assert [c.id for c in outcome.chunks] == [1, 2]
    assert all(c.relevance_score >= 0 for c in outcome.chunks)


async def test_the_final_cut_is_top_k_but_the_reranker_sees_the_whole_pool() -> None:
    store = FakeStore({(Collection.BUDGET, 0.0): [row(i) for i in range(1, 8)]})
    reranker = FakeReranker({})

    outcome = await run(store, config=stages(rerank=True, top_k=2, recall_k=10), reranker=reranker)

    assert len(outcome.chunks) == 2
    assert len(reranker.pool) == 7


async def test_decomposition_searches_each_subquery_and_covers_every_topic() -> None:
    llm = FakeLLM(subqueries=[SubQuery(topic="a", query="alpha"), SubQuery(topic="b", query="beta")])
    embedder = FakeEmbedder()
    store = FakeStore(
        {
            (Collection.BUDGET, 0.0): [row(1), row(2)],
            (Collection.BUDGET, 1.0): [row(3)],
        }
    )

    outcome = await run(
        store,
        config=stages(query_transform_enabled=True),
        query="oauth and a ledger, plus stripe checkout, and also recommendations for the catalog",
        llm=llm,
        embedder=embedder,
        explicit_collections=[Collection.BUDGET],
    )

    assert outcome.technique == "decompose"
    assert sorted(embedder.texts) == ["alpha", "beta"]
    assert [c.id for c in outcome.chunks] == [1, 3, 2]  # the niche topic gets its slot early
    assert [sq.topic for sq in outcome.subqueries] == ["a", "b"]


async def test_stage_models_come_from_runtime_config_unless_overridden() -> None:
    llm = FakeLLM(routed=[Collection.TRANSCRIPT])
    store = FakeStore({(Collection.TRANSCRIPT, 0.0): [row(1)]})

    await run(store, config=stages(routing_enabled=True), llm=llm)
    await run(store, config=stages(routing_enabled=True, router_model="pinned-model"), llm=llm)

    assert [call["model"] for call in llm.calls] == ["ROUTER_MODEL-resolved", "pinned-model"]


async def test_router_decision_decides_which_tables_are_searched() -> None:
    llm = FakeLLM(routed=[Collection.TRANSCRIPT])
    store = FakeStore({(Collection.TRANSCRIPT, 0.0): [row(5)]})

    outcome = await run(store, config=stages(routing_enabled=True), llm=llm)

    assert outcome.routing.level == "classifier"
    assert {call["collection"] for call in store.calls} == {Collection.TRANSCRIPT}
    assert outcome.chunks[0].collection == "transcript"


async def test_hard_filters_are_embedded_per_collection() -> None:
    store = FakeStore({})

    await run(
        store,
        config=stages(),
        explicit_collections=[Collection.BUDGET, Collection.TECHNICAL_DOC],
        hard_filters=HardFilters(sectors=("finance",), version="v2"),
    )

    extra = {call["collection"]: call["extra_filters"] for call in store.calls}
    assert len(extra[Collection.BUDGET]) == 1  # sector
    assert len(extra[Collection.TECHNICAL_DOC]) == 1  # version


async def test_empty_result_is_a_soft_fail_with_zero_cardinality() -> None:
    outcome = await run(FakeStore({}), config=stages())

    assert outcome.chunks == [] and outcome.low_confidence
    assert outcome.cardinality == {"budget": 0}


@pytest.mark.parametrize(
    "scenario", ["embedding", "database", "reranker"]
)  # each infrastructure failure surfaces as RetrievalError
async def test_failures_become_retrieval_error(scenario: str) -> None:
    store = FakeStore({(Collection.BUDGET, 0.0): [row(1)]}, fail=scenario == "database")
    embedder = FakeEmbedder(fail=scenario == "embedding")
    reranker = FakeReranker({}, fail=True) if scenario == "reranker" else None

    with pytest.raises(RetrievalError):
        await run(store, config=stages(rerank=reranker is not None), embedder=embedder, reranker=reranker)
