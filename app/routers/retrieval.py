import asyncio

import openai
import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.session import get_db_session
from app.dependencies import get_chunk_store, get_embedder, get_reranker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.errors import InvalidEmbeddingError, ensure_valid_vectors
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.pipeline import resolve_search_options, retrieve
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import RetrievalRequest, RetrievalResult
from app.generation.rag.store import ChunkSearchStore

router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])
logger = structlog.get_logger(__name__)


def _bad_gateway(event: str, exc: Exception, detail: str) -> HTTPException:
    # Log only the error type and a truncated message; the client gets a generic message
    logger.error(event, error_type=type(exc).__name__, error=str(exc)[:200])
    return HTTPException(status_code=502, detail=detail)


@router.post("/search", response_model=RetrievalResult)
async def search(
    request: RetrievalRequest,
    session: AsyncSession = Depends(get_db_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
    store: ChunkSearchStore = Depends(get_chunk_store),
    # Constructing the reranker is cheap: the model only loads on the first rerank
    reranker: CrossEncoderReranker = Depends(get_reranker),
    settings: Settings = Depends(get_settings),
) -> RetrievalResult:
    """Vector or hybrid search, optionally reranked. Empty result = 200 with low_confidence."""
    search_mode, rerank = resolve_search_options(request.search_mode, request.rerank, settings)
    try:
        query_vector = await asyncio.to_thread(embedder.embed_one, request.query_text)
        ensure_valid_vectors([query_vector], expected_count=1)
    except (openai.OpenAIError, InvalidEmbeddingError) as exc:
        raise _bad_gateway("retrieval_embedding_failed", exc, "Embedding service error") from exc

    try:
        return await retrieve(
            session,
            store,
            query_embedding=query_vector,
            query_text=request.query_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=request.top_k,
            recall_k=settings.RETRIEVAL_RECALL_TOP_K,
            rerank_top_n=settings.RERANK_TOP_N,
            distance_threshold=request.distance_threshold,
            rrf_k=settings.RRF_K,
            lexical_match_mode=settings.LEXICAL_MATCH_MODE,
            reranker=reranker,
        )
    except RetrievalError as exc:
        raise _bad_gateway("retrieval_failed", exc, "Retrieval service error") from exc
