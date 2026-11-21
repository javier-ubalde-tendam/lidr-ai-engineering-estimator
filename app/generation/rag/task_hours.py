"""Per-task hours estimation by vector search.

The structure-only generation produces modules -> tasks WITHOUT hours. This module derives the
hours of each task from the historical task corpus (``chunk_type='historical_task'``, built by
``scripts/build_task_corpus.py``):

1. compose a search text from the task (module + name + description),
2. embed it and run a metadata-filtered search over the historical tasks,
3. combine the nearest neighbours' recorded ``estimated_hours`` into ONE number with a
   DISTANCE-WEIGHTED CONSENSUS and a reliability score that blends how close the neighbours are
   with how much they agree.

A task whose nearest neighbour is farther than the distance threshold gets NO hours
(``has_match=False``): the UI flags it and the human supplies the number. When the neighbours
contradict each other the spread is surfaced as an ``HourRange`` (``quality/synthesis.py``)
instead of being averaged away.

The consensus is a pure function; the collaborators (embedder, store, session factory, runtime
config) are injected, this package never reaches for the composition root.
"""

import asyncio
import statistics
from typing import cast

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.foundation.llm.runtime_config import RuntimeRetrievalConfig
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.errors import RetrievalError, ensure_valid_vectors
from app.generation.rag.quality.synthesis import synthesize_range
from app.generation.rag.retrieval.pipeline import retrieve
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import (
    SearchMode,
    TaskHoursEstimate,
    TaskHoursModuleInput,
    TaskHoursResult,
    TaskNeighbor,
)
from app.generation.rag.store.repository import ChunkSearchStore

log = structlog.get_logger()

HISTORICAL_TASK_CHUNK_TYPE = "historical_task"
# Floor on the inverse-distance weight: an exact match (distance ~ 0) must not get an infinite
# weight and silence every other neighbour
_WEIGHT_EPS = 1e-3
# One search per task, fanned out: bound it so a big wizard cannot exhaust the DB pool
_MAX_CONCURRENT_SEARCHES = 8


def compose_task_search_text(module: str, name: str, description: str | None) -> str:
    """Embeddable query text of one task.

    Mirrors the shape of the historical-task chunk text (module + component name + description)
    so query and corpus live in the same vocabulary space.
    """
    parts = []
    if module:
        parts.append(f"Module: {module}")
    parts.append(f"Task: {name}")
    if description:
        parts.append(description)
    return "\n".join(parts)


def consensus(neighbors: list[tuple[int, float]]) -> tuple[int, float, float]:
    """Distance-weighted consensus over ``(hours, distance)`` neighbours.

    Returns ``(hours, reliability, dispersion)``:

    * ``hours``: inverse-distance weighted mean of the neighbour hours, rounded.
    * ``reliability``: ``weighted_similarity * (1 - min(dispersion, 1))`` in ``[0, 1]``; high
      only when the neighbours are both CLOSE and in AGREEMENT.
    * ``dispersion``: coefficient of variation of the neighbour hours (std / mean), a transparent
      "how much do they disagree" number.

    Precondition: ``neighbors`` is not empty (the caller handles the no-match case).
    """
    weights = [1.0 / (_WEIGHT_EPS + distance) for _hours, distance in neighbors]
    total_weight = sum(weights)
    hours_values = [hours for hours, _distance in neighbors]

    weighted_hours = (
        sum(w * hours for w, (hours, _distance) in zip(weights, neighbors, strict=True))
        / total_weight
    )
    # Weighted mean of the per-neighbour similarity (1 - cosine distance, floored at 0)
    weighted_similarity = (
        sum(
            w * max(0.0, 1.0 - distance)
            for w, (_hours, distance) in zip(weights, neighbors, strict=True)
        )
        / total_weight
    )

    mean_hours = statistics.fmean(hours_values)
    if len(hours_values) > 1 and mean_hours > 0:
        dispersion = statistics.pstdev(hours_values) / mean_hours
    else:
        dispersion = 0.0

    reliability = weighted_similarity * (1.0 - min(dispersion, 1.0))
    reliability = max(0.0, min(1.0, reliability))
    return round(weighted_hours), round(reliability, 3), round(dispersion, 3)


class TaskHoursEstimator:
    """Estimates the hours of every task of a structure from the historical task corpus."""

    def __init__(
        self,
        *,
        embedder: OpenAIEmbedder,
        session_factory: async_sessionmaker[AsyncSession],
        store: ChunkSearchStore,
        retrieval_config: RuntimeRetrievalConfig,
        settings: Settings,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._session_factory = session_factory
        self._store = store
        self._retrieval_config = retrieval_config
        self._settings = settings
        self._reranker = reranker

    async def estimate_all(self, modules: list[TaskHoursModuleInput]) -> TaskHoursResult:
        """Estimate every task of every module; results keep the submission order.

        top-k, distance threshold, search mode, reranking and synthesis follow the runtime
        override (``PUT /api/v1/config/retrieval``) and fall back to the environment.
        """
        config = self._retrieval_config
        top_k = await asyncio.to_thread(config.effective_task_hours_top_k)
        threshold = await asyncio.to_thread(config.effective_task_hours_distance_threshold)
        search_mode = cast(SearchMode, await asyncio.to_thread(config.effective_search_mode))
        rerank = await asyncio.to_thread(config.effective_rerank)
        synthesis = await asyncio.to_thread(config.effective_synthesis)

        gate = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)

        async def bounded(module: str, name: str, description: str | None) -> TaskHoursEstimate:
            async with gate:
                return await self.estimate_one(
                    module,
                    name,
                    description,
                    top_k=top_k,
                    distance_threshold=threshold,
                    search_mode=search_mode,
                    rerank=rerank,
                    synthesis=synthesis,
                )

        estimates = await asyncio.gather(
            *(
                bounded(module.name, task.name, task.description)
                for module in modules
                for task in module.tasks
            )
        )
        matched = sum(1 for estimate in estimates if estimate.has_match)
        log.info(
            "task_hours_done",
            tasks=len(estimates),
            matched=matched,
            flagged=len(estimates) - matched,
            top_k=top_k,
            distance_threshold=threshold,
            search_mode=search_mode,
            rerank=rerank,
        )
        return TaskHoursResult(tasks=list(estimates))

    async def estimate_one(
        self,
        module: str,
        name: str,
        description: str | None,
        *,
        top_k: int,
        distance_threshold: float,
        search_mode: SearchMode = "vector",
        rerank: bool = False,
        synthesis: bool = False,
    ) -> TaskHoursEstimate:
        """Estimate the hours of one task.

        The search goes through the full retrieval pipeline (hybrid + optional cross-encoder),
        filtered to ``historical_task`` chunks: with reranking on, the neighbours feeding the
        consensus are the ones the cross-encoder judges most relevant, not just the closest in
        cosine space.
        """
        settings = self._settings
        search_text = compose_task_search_text(module, name, description)
        try:
            embedding = await asyncio.to_thread(self._embedder.embed_one, search_text)
            ensure_valid_vectors([embedding], expected_count=1)
        except Exception as exc:
            raise RetrievalError("Embedding the task search text failed.") from exc

        async with self._session_factory() as session:
            result = await retrieve(
                session,
                self._store,
                query_embedding=embedding,
                query_text=search_text,
                search_mode=search_mode,
                rerank=rerank,
                top_k=top_k,
                recall_k=max(settings.RETRIEVAL_RECALL_TOP_K, top_k),
                rerank_top_n=top_k,
                distance_threshold=distance_threshold,
                rrf_k=settings.RRF_K,
                lexical_match_mode=settings.LEXICAL_MATCH_MODE,
                chunk_types=[HISTORICAL_TASK_CHUNK_TYPE],
                reranker=self._reranker,
            )

        usable = [chunk for chunk in result.chunks if chunk.estimated_hours is not None]
        if not usable:
            return TaskHoursEstimate(module=module, task=name, has_match=False)

        neighbors = [
            TaskNeighbor(
                source_id=chunk.id,
                budget_id=chunk.budget_id,
                estimated_hours=int(chunk.estimated_hours or 0),
                distance=chunk.distance,
            )
            for chunk in usable
        ]
        hours, reliability, dispersion = consensus(
            [(neighbor.estimated_hours, neighbor.distance) for neighbor in neighbors]
        )
        # Contradicting neighbours: surface the spread as a range with a reason. The point
        # estimate still carries the consensus for the human to edit
        hours_range = (
            await synthesize_range(
                neighbors, dispersion, threshold=settings.SYNTHESIS_CONTRADICTION_THRESHOLD
            )
            if synthesis
            else None
        )
        return TaskHoursEstimate(
            module=module,
            task=name,
            estimated_hours=hours,
            reliability=reliability,
            has_match=True,
            dispersion=dispersion,
            neighbors=neighbors,
            hours_range=hours_range,
        )
