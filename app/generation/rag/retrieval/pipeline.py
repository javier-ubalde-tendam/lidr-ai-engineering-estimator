"""Single retrieval entrypoint composing the four session 10 configurations.

* ``search_mode="vector"``: k-NN with distance threshold (configuration A, the baseline).
* ``search_mode="hybrid"``: wide vector branch + lexical branch fused with RRF.
* ``rerank=True``: recall wide (``recall_k``) and let the cross-encoder keep ``rerank_top_n``.
  The recall stage must not lose the relevant chunk; the reranker's job is to float it up.
"""

import asyncio
import time
from typing import Any

import structlog
from sqlalchemy import Row
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import RetrievalResult, RetrievedChunk, SearchMode
from app.generation.rag.store import ChunkSearchStore, LexicalMatchMode

logger = structlog.get_logger(__name__)

# Distance for a chunk found ONLY by the lexical branch: it never entered the vector ranking so
# it has no cosine distance. 1.0 = orthogonal ("far"). Display-only: fusion and reranking ignore it
NO_VECTOR_DISTANCE = 1.0


def resolve_search_options(
    search_mode: SearchMode | None, rerank: bool | None, settings: Settings
) -> tuple[SearchMode, bool]:
    """Precedence: request field -> environment (Settings) -> Settings defaults."""
    resolved_mode = search_mode if search_mode is not None else settings.RETRIEVAL_SEARCH_MODE
    resolved_rerank = rerank if rerank is not None else settings.RERANKER_ENABLED
    return resolved_mode, resolved_rerank


def _to_chunk(row: Row[Any], *, distance: float) -> RetrievedChunk:
    metadata = row.metadata_
    budget_id = metadata.get("budget_id")
    return RetrievedChunk(
        id=row.id,
        content=row.content,
        sector=str(metadata.get("client_sector", "unknown")),
        project_year=int(metadata.get("year", 0)),
        chunk_type=row.chunk_type,
        distance=distance,
        budget_id=str(budget_id) if budget_id is not None else None,
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
    reranker: CrossEncoderReranker | None = None,
) -> RetrievalResult:
    """Retrieve chunks best first. ``low_confidence`` (soft-fail) = nothing retrieved.

    ``reranker`` is injectable for tests; when ``None`` and ``rerank`` is on, the app singleton
    is used. Raises ``RetrievalError`` if the database or the reranker fails.
    """
    started = time.perf_counter()
    # Recall wide only when a later stage (fusion or rerank) re-sorts; otherwise top_k is final
    vector_limit = recall_k if (rerank or search_mode == "hybrid") else top_k

    try:
        async with session.begin():
            vector_rows, candidates_evaluated = await store.search_filtered(
                session,
                query_vector=query_embedding,
                top_k=vector_limit,
                distance_threshold=distance_threshold,
            )
            lexical_rows: list[Row[Any]] = []
            if search_mode == "hybrid":
                lexical_rows = await store.search_lexical(
                    session, query_text=query_text, top_k=recall_k, match_mode=lexical_match_mode
                )
    except (SQLAlchemyError, OSError) as exc:
        logger.error("rag_retrieve_db_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        raise RetrievalError("Chunk store query failed") from exc

    # Candidate pool by id; when a chunk is in both branches its real vector distance wins
    candidates = {row.id: _to_chunk(row, distance=float(row.distance)) for row in vector_rows}
    for row in lexical_rows:
        candidates.setdefault(row.id, _to_chunk(row, distance=NO_VECTOR_DISTANCE))

    if search_mode == "hybrid":
        fused = reciprocal_rank_fusion(
            [[row.id for row in vector_rows], [row.id for row in lexical_rows]], k=rrf_k
        )
        ordered = [candidates[chunk_id] for chunk_id, _score in fused]
    else:
        ordered = [candidates[row.id] for row in vector_rows]

    if rerank and ordered:
        if reranker is None:
            # Imported here: app.dependencies pulls the whole app wiring (OpenAI client, ...)
            from app.dependencies import get_reranker

            reranker = get_reranker()
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
        search_mode=search_mode,
        rerank=rerank,
        vector_hits=len(vector_rows),
        lexical_hits=len(lexical_rows),
        results=len(final),
        candidates_evaluated=candidates_evaluated,
        search_time_ms=search_time_ms,
    )
    return RetrievalResult(
        chunks=final,
        low_confidence=not final,
        candidates_evaluated=candidates_evaluated,
        search_mode=search_mode,
        rerank=rerank,
        search_time_ms=search_time_ms,
        vector_hits=len(vector_rows),
        lexical_hits=len(lexical_rows),
    )
