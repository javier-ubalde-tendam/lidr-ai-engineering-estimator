from collections.abc import Callable

import openai
import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.errors import service_error
from app.dependencies import (
    ALL_STRATEGIES,
    get_chunker_builder,
    get_embedder,
    get_ingestion_service,
)
from app.generation.rag.analysis.comparison import (
    ChunkingComparator,
    CompareRequest,
    CompareResponse,
)
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.errors import DocumentAlreadyIngestedError, InvalidEmbeddingError
from app.generation.rag.ingest_service import IngestionService
from app.generation.rag.schemas import (
    DuplicateDocumentResponse,
    IngestRequest,
    IngestResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={409: {"model": DuplicateDocumentResponse, "description": "Document already ingested"}},
)
async def ingest(
    request: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse | JSONResponse:
    try:
        return await service.ingest(request)
    except DocumentAlreadyIngestedError as exc:
        # JSONResponse (no HTTPException): `document_id` debe quedar en el primer nivel, junto a `detail`
        return JSONResponse(
            status_code=409,
            content={"detail": "Document already ingested", "document_id": exc.document_id},
        )
    except (openai.OpenAIError, InvalidEmbeddingError) as exc:
        raise service_error("embedding_ingest_failed", exc, "Embedding service error") from exc
    except (SQLAlchemyError, OSError) as exc:
        raise service_error("database_ingest_failed", exc, "Database error") from exc


# Sync `def` on purpose: chunkers and the embedder are blocking (LLM / embeddings clients), so
# FastAPI runs the handler in its thread pool instead of stalling the event loop.
@router.post("/compare", response_model=CompareResponse)
def compare(
    request: CompareRequest,
    embedder: OpenAIEmbedder = Depends(get_embedder),
    build_chunkers: Callable[[list[str]], list[Chunker]] = Depends(get_chunker_builder),
) -> CompareResponse:
    """Run several chunking strategies over the same budgets and compare them.

    Returns per-strategy corpus stats and, when queries are given, the top-k chunks each
    strategy retrieves. Nothing is persisted. The semantic, propositional and
    contextual_retrieval strategies call paid APIs (see ``ingestion_cost_usd``).
    """
    names = request.strategies or ALL_STRATEGIES
    try:
        chunkers = build_chunkers(names)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {exc.args[0]}") from exc
    except RuntimeError as exc:
        # A strategy needs a provider key that is not configured (the message names the key only)
        logger.error("embeddings_compare_failed", reason="missing_api_key", error=str(exc)[:200])
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        "embeddings_compare_received",
        total_budgets=len(request.budgets),
        strategies=names,
        n_queries=len(request.queries),
    )
    comparator = ChunkingComparator(chunkers, embedder)
    try:
        stats = comparator.compute_stats(request.budgets)
        queries = comparator.run_queries(request.budgets, request.queries, request.top_k)
    except Exception as exc:
        raise service_error(
            "embeddings_compare_failed", exc, "Failed to run chunking comparison"
        ) from exc
    return CompareResponse(stats_per_strategy=stats, queries_per_strategy=queries)
