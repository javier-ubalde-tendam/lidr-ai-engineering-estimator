import openai
import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

router = APIRouter(prefix="/embeddings", tags=["embeddings"])
logger = structlog.get_logger(__name__)


# Endpoint síncrono (`def`, no `async def`): el cliente OpenAI es bloqueante. FastAPI ejecuta
# los `def` en un threadpool, así que no bloquea el event loop; con `async def` sí lo haría.
@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> IngestResponse:
    chunks = chunker.chunk(request.budgets)
    try:
        embedded = embedder.embed_many(chunks)
    except openai.OpenAIError as exc:
        # Se loguea el detalle (truncado) pero al cliente solo le llega un mensaje genérico
        logger.error("embedding_ingest_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        raise HTTPException(status_code=500, detail="Embedding service error") from exc

    total_tokens = sum(chunk.token_count for chunk in embedded)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(total_tokens),
    )
    return IngestResponse(chunks=embedded, stats=stats)
