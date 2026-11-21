"""Single retrieval entrypoint composing the four session 10 configurations.

* ``search_mode="vector"``: k-NN with distance threshold (configuration A, the baseline).
* ``search_mode="hybrid"``: wide vector branch + lexical branch fused with RRF.
* ``rerank=True``: recall wide (``recall_k``) and let the cross-encoder keep ``rerank_top_n``.
  The recall stage must not lose the relevant chunk; the reranker's job is to float it up.

``hybrid_search_one`` is the single-collection recall primitive; ``retrieve`` wraps it for one
collection (budgets by default) and ``advanced_pipeline`` reuses it across collections and
sub-queries.
"""

import asyncio
import time
from typing import Any

import structlog
from sqlalchemy import ColumnElement, Row
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.foundation.llm.runtime_config import RuntimeRetrievalConfig
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.collections import Collection, spec_for
from app.generation.rag.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.generation.rag.retrieval.reranker import CrossEncoderReranker, default_reranker
from app.generation.rag.schemas import RetrievalResult, RetrievedChunk, SearchMode
from app.generation.rag.store.repository import ChunkSearchStore, LexicalMatchMode

logger = structlog.get_logger(__name__)

# Distance for a chunk found ONLY by the lexical branch: it never entered the vector ranking so
# it has no cosine distance. 1.0 = orthogonal ("far"). Display-only: fusion and reranking ignore it
NO_VECTOR_DISTANCE = 1.0


def resolve_search_options(
    search_mode: SearchMode | None,
    rerank: bool | None,
    settings: Settings,
    runtime: RuntimeRetrievalConfig | None = None,
) -> tuple[SearchMode, bool]:
    """Precedence: request field -> runtime override (Redis) -> environment (Settings).

    ``runtime.effective_*`` already falls back to Settings (and degrades to it if Redis is down).
    It does blocking I/O: async callers run this function in a thread.
    """
    if runtime is None:
        resolved_mode = search_mode if search_mode is not None else settings.RETRIEVAL_SEARCH_MODE
        resolved_rerank = rerank if rerank is not None else settings.RERANKER_ENABLED
        return resolved_mode, resolved_rerank
    resolved_mode = search_mode if search_mode is not None else runtime.effective_search_mode()
    resolved_rerank = rerank if rerank is not None else runtime.effective_rerank()
    return resolved_mode, resolved_rerank


class CollectionHits:
    """Recall output for one (collection, query): candidate pool + the per-branch rankings.

    ``candidates`` maps chunk id -> chunk (ids are unique WITHIN a collection). ``vector_ids`` and
    ``lexical_ids`` are the branch orderings the caller fuses: RRF for hybrid, or across
    sub-queries (see ``advanced_pipeline``).
    """

    def __init__(
        self,
        *,
        collection: Collection,
        candidates: dict[int, RetrievedChunk],
        vector_ids: list[int],
        lexical_ids: list[int],
        candidates_evaluated: int,
    ) -> None:
        self.collection = collection
        self.candidates = candidates
        self.vector_ids = vector_ids
        self.lexical_ids = lexical_ids
        self.candidates_evaluated = candidates_evaluated

    def fused_order(self, *, search_mode: SearchMode, rrf_k: int) -> list[RetrievedChunk]:
        """Chunks best first: RRF of both branches for hybrid, vector distance otherwise."""
        if search_mode == "hybrid":
            fused = reciprocal_rank_fusion([self.vector_ids, self.lexical_ids], k=rrf_k)
            return [self.candidates[chunk_id] for chunk_id, _score in fused]
        return [self.candidates[chunk_id] for chunk_id in self.vector_ids]


async def hybrid_search_one(
    session: AsyncSession,
    store: ChunkSearchStore,
    *,
    collection: Collection,
    query_embedding: list[float],
    query_text: str,
    search_mode: SearchMode,
    recall_k: int,
    top_k: int,
    distance_threshold: float,
    will_rerank: bool,
    lexical_match_mode: LexicalMatchMode = "or",
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    extra_filters: list[ColumnElement[bool]] | None = None,
) -> CollectionHits:
    """Run the vector (and, for hybrid, lexical) branches over ONE collection.

    Recall wide whenever a later stage (fusion or rerank) will re-sort; recall exactly ``top_k``
    only for the plain vector path, where the order is final. It does not open a transaction or
    catch errors: the caller owns both. The metadata filters apply to both branches.
    """
    spec = spec_for(collection)
    vector_limit = recall_k if (will_rerank or search_mode == "hybrid") else top_k
    filters: dict[str, Any] = {
        "collection": collection,
        "sectors": sectors,
        "project_year_min": project_year_min,
        "project_year_max": project_year_max,
        "chunk_types": chunk_types,
        "extra_filters": extra_filters,
    }

    vector_rows, candidates_evaluated = await store.search_filtered(
        session,
        query_vector=query_embedding,
        top_k=vector_limit,
        distance_threshold=distance_threshold,
        **filters,
    )
    lexical_rows: list[Row[Any]] = []
    if search_mode == "hybrid":
        lexical_rows = await store.search_lexical(
            session,
            query_text=query_text,
            top_k=recall_k,
            match_mode=lexical_match_mode,
            **filters,
        )

    # Candidate pool by id; when a chunk is in both branches its real vector distance wins
    candidates = {row.id: spec.to_chunk(row, distance=float(row.distance)) for row in vector_rows}
    for row in lexical_rows:
        candidates.setdefault(row.id, spec.to_chunk(row, distance=NO_VECTOR_DISTANCE))

    return CollectionHits(
        collection=collection,
        candidates=candidates,
        vector_ids=[row.id for row in vector_rows],
        lexical_ids=[row.id for row in lexical_rows],
        candidates_evaluated=candidates_evaluated,
    )


async def retrieve(
    session: AsyncSession,
    store: ChunkSearchStore,
    *,
    query_embedding: list[float],
    query_text: str,
    search_mode: SearchMode = "vector",
    rerank: bool = False,
    top_k: int = 10,
    recall_k: int = 50,
    rerank_top_n: int = 5,
    distance_threshold: float = 0.6,
    rrf_k: int = DEFAULT_RRF_K,
    lexical_match_mode: LexicalMatchMode = "or",
    collection: Collection = Collection.BUDGET,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    extra_filters: list[ColumnElement[bool]] | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> RetrievalResult:
    """Retrieve chunks of ONE collection (default: budgets), best first.

    ``low_confidence`` (soft-fail) = nothing retrieved. The optional metadata filters apply to
    both branches (vector and lexical). ``reranker`` is injectable for tests; when ``None`` and
    ``rerank`` is on, the app singleton is used. Raises ``RetrievalError`` if the database or the
    reranker fails.
    """
    started = time.perf_counter()

    try:
        async with session.begin():
            hits = await hybrid_search_one(
                session,
                store,
                collection=collection,
                query_embedding=query_embedding,
                query_text=query_text,
                search_mode=search_mode,
                recall_k=recall_k,
                top_k=top_k,
                distance_threshold=distance_threshold,
                will_rerank=rerank,
                lexical_match_mode=lexical_match_mode,
                sectors=sectors,
                project_year_min=project_year_min,
                project_year_max=project_year_max,
                chunk_types=chunk_types,
                extra_filters=extra_filters,
            )
    except (SQLAlchemyError, OSError) as exc:
        logger.error("rag_retrieve_db_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        raise RetrievalError("Chunk store query failed") from exc

    ordered = hits.fused_order(search_mode=search_mode, rrf_k=rrf_k)

    if rerank and ordered:
        if reranker is None:
            reranker = default_reranker()
        try:
            # The cross-encoder is CPU-bound and synchronous: keep it off the event loop
            final = await asyncio.to_thread(
                reranker.rerank, query_text, ordered[:recall_k], top_n=rerank_top_n
            )
        except Exception as exc:  # model load/inference errors vary by backend
            logger.error("rag_rerank_failed", error_type=type(exc).__name__, error=str(exc)[:200])
            raise RetrievalError("Reranking failed") from exc
    else:
        final = ordered[:top_k]

    search_time_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "rag_retrieve_done",
        collection=collection.value,
        search_mode=search_mode,
        rerank=rerank,
        vector_hits=len(hits.vector_ids),
        lexical_hits=len(hits.lexical_ids),
        results=len(final),
        candidates_evaluated=hits.candidates_evaluated,
        search_time_ms=search_time_ms,
    )
    return RetrievalResult(
        chunks=final,
        low_confidence=not final,
        candidates_evaluated=hits.candidates_evaluated,
        search_mode=search_mode,
        rerank=rerank,
        search_time_ms=search_time_ms,
        vector_hits=len(hits.vector_ids),
        lexical_hits=len(hits.lexical_ids),
    )
