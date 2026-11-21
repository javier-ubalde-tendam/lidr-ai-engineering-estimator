"""Advanced multi-index retrieval pipeline (session 10).

Stages in the order "cheap and exclusive first, expensive and fine last, soft at the end":

    query transform -> routing -> hard filters -> hybrid search -> fusion
                    -> reranking -> temporal decay -> top-k

Every stage is governed by ``StageConfig``: the pipeline is the MAXIMUM path, not the mandatory
one, and each technique can be measured in isolation. The endpoint builds a ``StageConfig`` from
settings/runtime overrides; the evaluation harness builds named ``StageConfig`` values as data.

Fusion depends on the technique: expansion variants are fused by consensus (RRF), decomposition
sub-queries by coverage (round-robin). Across collections the merge is round-robin too, keyed by
``(collection, id)`` because chunk ids are only unique WITHIN a collection. Every chunk keeps its
provenance label.

Collaborators (LLM wrapper, runtime model config, embedder, session factory, store, reranker) are
parameters: generation code never imports the composition root.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.foundation.llm.runtime_config import RuntimeModelConfig
from app.foundation.llm.wrapper import LLMWrapper
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.errors import RetrievalError, ensure_valid_vectors
from app.generation.rag.retrieval.collections import Collection, HardFilters, spec_for
from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion, round_robin_merge
from app.generation.rag.retrieval.pipeline import CollectionHits, hybrid_search_one
from app.generation.rag.retrieval.query_transform import DECOMPOSE, SubQuery, transform_query
from app.generation.rag.retrieval.reranker import CrossEncoderReranker, default_reranker
from app.generation.rag.retrieval.router import RoutingDecision, route
from app.generation.rag.retrieval.temporal import apply_temporal_decay, sigmoid
from app.generation.rag.schemas import RetrievedChunk, SearchMode
from app.generation.rag.store.repository import ChunkSearchStore, LexicalMatchMode

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class StageConfig:
    """Which advanced stages run, plus their parameters. Pure data, no I/O.

    ``router_model`` / ``query_transform_model`` are optional overrides (the evaluation compares
    models as data); when ``None`` they are resolved through ``RuntimeModelConfig``.
    """

    routing_enabled: bool = True
    query_transform_enabled: bool = True
    search_mode: SearchMode = "hybrid"
    rerank: bool = True
    temporal_decay_enabled: bool = False
    top_k: int = 5
    recall_k: int = 50
    distance_threshold: float = 2.0  # permissive: this pipeline measures ranking, not the soft-fail
    rrf_k: int = 60
    lexical_match_mode: LexicalMatchMode = "or"
    half_life_days: int = 900
    max_subqueries: int = 4
    max_targets: int = 3
    router_model: str | None = None
    query_transform_model: str | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        search_mode: SearchMode,
        rerank: bool,
        routing_enabled: bool | None = None,
        query_transform_enabled: bool | None = None,
        temporal_decay_enabled: bool | None = None,
        top_k: int | None = None,
        distance_threshold: float | None = None,
    ) -> "StageConfig":
        """Effective config from the environment; an explicit (non-None) argument wins."""
        return cls(
            routing_enabled=(
                settings.RETRIEVAL_ROUTING_ENABLED if routing_enabled is None else routing_enabled
            ),
            query_transform_enabled=(
                settings.QUERY_TRANSFORM_ENABLED
                if query_transform_enabled is None
                else query_transform_enabled
            ),
            search_mode=search_mode,
            rerank=rerank,
            temporal_decay_enabled=(
                settings.TEMPORAL_DECAY_ENABLED
                if temporal_decay_enabled is None
                else temporal_decay_enabled
            ),
            top_k=top_k or settings.RERANK_TOP_N,
            recall_k=settings.RETRIEVAL_RECALL_TOP_K,
            distance_threshold=(
                settings.RETRIEVAL_DISTANCE_THRESHOLD
                if distance_threshold is None
                else distance_threshold
            ),
            rrf_k=settings.RRF_K,
            lexical_match_mode=settings.LEXICAL_MATCH_MODE,
            half_life_days=settings.TEMPORAL_DECAY_HALF_LIFE_DAYS,
            max_subqueries=settings.QUERY_MAX_SUBQUERIES,
            max_targets=settings.ROUTER_MAX_TARGETS,
        )


@dataclass
class AdvancedRetrievalOutcome:
    """Everything the endpoint surfaces: the results plus how they were obtained."""

    chunks: list[RetrievedChunk]
    routing: RoutingDecision
    technique: str
    subqueries: list[SubQuery]
    cardinality: dict[str, int] = field(default_factory=dict)
    low_confidence: bool = True


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, int]:
    """Cross-collection dedup identity (ids are unique only within a collection)."""
    return (chunk.collection, chunk.id)


def _fuse_collection(
    hits_per_subquery: list[CollectionHits], *, technique: str, search_mode: SearchMode, rrf_k: int
) -> list[RetrievedChunk]:
    """Fuse one collection's per-sub-query results into a single ordered list.

    Expansion/direct -> RRF (consensus across reformulations); decomposition -> round-robin
    (coverage across topics).
    """
    candidates: dict[int, RetrievedChunk] = {}
    for hits in hits_per_subquery:
        candidates.update(hits.candidates)

    ordered_per_subquery = [
        hits.fused_order(search_mode=search_mode, rrf_k=rrf_k) for hits in hits_per_subquery
    ]
    if technique == DECOMPOSE:
        return round_robin_merge(ordered_per_subquery, key=lambda chunk: chunk.id)

    fused = reciprocal_rank_fusion(
        [[chunk.id for chunk in ordered] for ordered in ordered_per_subquery], k=rrf_k
    )
    return [candidates[chunk_id] for chunk_id, _score in fused]


async def _stage_model(explicit: str | None, models: RuntimeModelConfig, key: str) -> str:
    """Explicit stage override, else the runtime override / environment default of ``key``."""
    if explicit:
        return explicit
    # RuntimeModelConfig reads Redis (blocking): keep it off the event loop
    return await asyncio.to_thread(models.effective, key)


async def advanced_retrieve(
    *,
    query_text: str,
    stages: StageConfig,
    reference_date: date,
    llm: LLMWrapper,
    models: RuntimeModelConfig,
    embedder: OpenAIEmbedder,
    session_factory: async_sessionmaker[AsyncSession],
    store: ChunkSearchStore,
    explicit_collections: list[Collection] | None = None,
    hard_filters: HardFilters | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> AdvancedRetrievalOutcome:
    """Run the full advanced pipeline and return the results plus the diagnostics.

    Raises ``RetrievalError`` when embedding, the database or the reranker fail.
    """
    hard_filters = hard_filters or HardFilters()

    # Stages 1+2: query transform and routing are independent, so they run concurrently
    router_model, transform_model = await asyncio.gather(
        _stage_model(stages.router_model, models, "ROUTER_MODEL"),
        _stage_model(stages.query_transform_model, models, "QUERY_TRANSFORM_MODEL"),
    )
    plan, routing = await asyncio.gather(
        transform_query(
            query_text,
            llm=llm,
            model=transform_model,
            enabled=stages.query_transform_enabled,
            max_subqueries=stages.max_subqueries,
        ),
        route(
            query_text,
            llm=llm,
            model=router_model,
            explicit=explicit_collections,
            rules_enabled=stages.routing_enabled,
            classifier_enabled=stages.routing_enabled,
            max_targets=stages.max_targets,
        ),
    )

    # Embed every sub-query once; the OpenAI client is synchronous, so each goes to a thread
    try:
        embeddings = await asyncio.gather(
            *(asyncio.to_thread(embedder.embed_one, sq.query) for sq in plan.subqueries)
        )
        ensure_valid_vectors(list(embeddings), expected_count=len(plan.subqueries))
    except Exception as exc:
        log.error("advanced_embed_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        raise RetrievalError("Failed to embed sub-queries") from exc

    # Stages 3+4: hard filters + hybrid search per (collection, sub-query). One session each, so
    # the fan-out is genuinely parallel (an AsyncSession cannot run concurrent statements)
    async def search(collection: Collection, sub_query: SubQuery, vector: list[float]) -> CollectionHits:
        async with session_factory() as session:
            return await hybrid_search_one(
                session,
                store,
                collection=collection,
                query_embedding=vector,
                query_text=sub_query.query,
                search_mode=stages.search_mode,
                recall_k=stages.recall_k,
                top_k=stages.top_k,
                distance_threshold=stages.distance_threshold,
                will_rerank=stages.rerank,
                lexical_match_mode=stages.lexical_match_mode,
                extra_filters=spec_for(collection).hard_filter_clauses(hard_filters),
            )

    try:
        flat_hits = await asyncio.gather(
            *(
                search(collection, sub_query, vector)
                for collection in routing.targets
                for sub_query, vector in zip(plan.subqueries, embeddings, strict=True)
            )
        )
    except Exception as exc:
        log.error("advanced_search_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        raise RetrievalError("Chunk store query failed") from exc

    # Regroup the flat results by collection (the sub-queries of a collection are contiguous)
    width = len(plan.subqueries)
    per_collection = {
        collection: list(flat_hits[index * width : (index + 1) * width])
        for index, collection in enumerate(routing.targets)
    }

    # Stage 5: fuse inside each collection, then round-robin across collections (coverage,
    # provenance preserved). The cardinality after the hard filters is the "silent empty" alarm
    cardinality: dict[str, int] = {}
    collection_orders: list[list[RetrievedChunk]] = []
    for collection, hits_list in per_collection.items():
        cardinality[collection.value] = hits_list[0].candidates_evaluated
        log.info(
            "advanced_collection_cardinality",
            collection=collection.value,
            candidates_evaluated=cardinality[collection.value],
        )
        ordered = _fuse_collection(
            hits_list, technique=plan.technique, search_mode=stages.search_mode, rrf_k=stages.rrf_k
        )
        if ordered:
            collection_orders.append(ordered)
    pool = round_robin_merge(collection_orders, key=_chunk_key)[: stages.recall_k]

    # Stage 6: cross-encoder scores, or a positional base score when reranking is off
    if stages.rerank and pool:
        ranker = reranker or default_reranker()
        try:
            reranked = await asyncio.to_thread(
                ranker.rerank_with_scores, query_text, pool, top_n=len(pool)
            )
        except Exception as exc:
            log.error("advanced_rerank_failed", error_type=type(exc).__name__, error=str(exc)[:200])
            raise RetrievalError("Reranking failed") from exc
        # Logits can be negative; decay is a multiplier, so the base must be non-negative
        scored = [(chunk, sigmoid(logit)) for chunk, logit in reranked]
    else:
        scored = [(chunk, 1.0 / (position + 1)) for position, chunk in enumerate(pool)]

    # Stage 7: temporal decay (soft, last), then the final cut
    if stages.temporal_decay_enabled:
        scored = apply_temporal_decay(
            scored, half_life_days=stages.half_life_days, reference_date=reference_date
        )
    final = [
        chunk.model_copy(update={"relevance_score": score}) for chunk, score in scored[: stages.top_k]
    ]

    log.info(
        "advanced_retrieve_done",
        technique=plan.technique,
        routing_level=routing.level,
        targets=[target.value for target in routing.targets],
        subqueries=len(plan.subqueries),
        pool=len(pool),
        results=len(final),
    )
    return AdvancedRetrievalOutcome(
        chunks=final,
        routing=routing,
        technique=plan.technique,
        subqueries=plan.subqueries,
        cardinality=cardinality,
        low_confidence=not final,
    )
