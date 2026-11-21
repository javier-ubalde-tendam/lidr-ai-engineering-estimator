import asyncio

import openai
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.rate_limiting import limiter
from app.api.security import require_retrieval_key
from app.config import Settings, get_settings
from app.dependencies import (
    get_chunk_store,
    get_embedder,
    get_reranker,
    get_runtime_retrieval_config,
)
from app.foundation.llm.runtime_config import RuntimeRetrievalConfig
from app.foundation.persistence.database import get_db_session
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.errors import InvalidEmbeddingError, RetrievalError, ensure_valid_vectors
from app.generation.rag.retrieval.pipeline import resolve_search_options, retrieve
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import RetrievalRequest, RetrievalResult
from app.generation.rag.store.repository import ChunkSearchStore

router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])
logger = structlog.get_logger(__name__)


def _bad_gateway(event: str, exc: Exception, detail: str) -> HTTPException:
    # Log only the error type and a truncated message; the client gets a generic message
    logger.error(event, error_type=type(exc).__name__, error=str(exc)[:200])
    return HTTPException(status_code=502, detail=detail)


@router.post(
    "/search",
    response_model=RetrievalResult,
    dependencies=[Depends(require_retrieval_key)],
)
@limiter.limit("120/minute")
async def search(
    request: Request,  # slowapi needs the Starlette request under this exact name
    payload: RetrievalRequest,
    session: AsyncSession = Depends(get_db_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
    store: ChunkSearchStore = Depends(get_chunk_store),
    # Constructing the reranker is cheap: the model only loads on the first rerank
    reranker: CrossEncoderReranker = Depends(get_reranker),
    settings: Settings = Depends(get_settings),
    runtime: RuntimeRetrievalConfig = Depends(get_runtime_retrieval_config),
) -> RetrievalResult:
    """Vector or hybrid search, optionally reranked. Empty result = 200 with low_confidence.

    Requires the ``X-API-Key`` header (``RETRIEVAL_API_KEY``); limited to 120 requests/minute
    per key. ``search_mode``/``rerank`` precedence: request field -> runtime override
    (``PUT /api/v1/config/retrieval``) -> environment. Optional metadata filters (sectors,
    project years, chunk types) narrow the candidates; ``distance_threshold`` is the relevance
    floor.
    """
    # The runtime store does blocking Redis I/O: keep it off the event loop
    search_mode, rerank = await asyncio.to_thread(
        resolve_search_options, payload.search_mode, payload.rerank, settings, runtime
    )
    try:
        query_vector = await asyncio.to_thread(embedder.embed_one, payload.query_text)
        ensure_valid_vectors([query_vector], expected_count=1)
    except (openai.OpenAIError, InvalidEmbeddingError) as exc:
        raise _bad_gateway("retrieval_embedding_failed", exc, "Embedding service error") from exc

    try:
        return await retrieve(
            session,
            store,
            query_embedding=query_vector,
            query_text=payload.query_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=payload.top_k,
            recall_k=settings.RETRIEVAL_RECALL_TOP_K,
            rerank_top_n=settings.RERANK_TOP_N,
            distance_threshold=payload.distance_threshold,
            rrf_k=settings.RRF_K,
            lexical_match_mode=settings.LEXICAL_MATCH_MODE,
            sectors=payload.sectors,
            project_year_min=payload.project_year_min,
            project_year_max=payload.project_year_max,
            chunk_types=payload.chunk_types,
            reranker=reranker,
        )
    except RetrievalError as exc:
        raise _bad_gateway("retrieval_failed", exc, "Retrieval service error") from exc
