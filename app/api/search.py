import openai
from fastapi import APIRouter, Depends
from sqlalchemy.exc import SQLAlchemyError

from app.api.errors import service_error
from app.dependencies import get_retriever
from app.generation.rag.errors import InvalidEmbeddingError
from app.generation.rag.retriever import Retriever
from app.generation.rag.schemas import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    try:
        return await retriever.search(request.query, request.k)
    except (openai.OpenAIError, InvalidEmbeddingError) as exc:
        raise service_error("embedding_search_failed", exc, "Embedding service error") from exc
    except (SQLAlchemyError, OSError) as exc:
        raise service_error("database_search_failed", exc, "Database error") from exc
