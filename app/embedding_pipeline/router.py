import openai
import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.dependencies import get_ingestion_service, get_retriever
from app.embedding_pipeline.errors import DocumentAlreadyIngestedError, InvalidEmbeddingError
from app.embedding_pipeline.ingestion import IngestionService
from app.embedding_pipeline.retriever import Retriever
from app.embedding_pipeline.schemas import (
    DuplicateDocumentResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])
search_router = APIRouter(tags=["search"])
logger = structlog.get_logger(__name__)


def _service_error(event: str, exc: Exception, detail: str) -> HTTPException:
    # Al log solo tipo y mensaje truncado (nunca cuerpos ni claves); al cliente un mensaje genérico
    logger.error(event, error_type=type(exc).__name__, error=str(exc)[:200])
    return HTTPException(status_code=500, detail=detail)


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
        raise _service_error("embedding_ingest_failed", exc, "Embedding service error") from exc
    except (SQLAlchemyError, OSError) as exc:
        raise _service_error("database_ingest_failed", exc, "Database error") from exc


@search_router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    try:
        return await retriever.search(request.query, request.k)
    except (openai.OpenAIError, InvalidEmbeddingError) as exc:
        raise _service_error("embedding_search_failed", exc, "Embedding service error") from exc
    except (SQLAlchemyError, OSError) as exc:
        raise _service_error("database_search_failed", exc, "Database error") from exc
