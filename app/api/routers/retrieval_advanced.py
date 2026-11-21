"""``POST /v1/retrieval/advanced-search``: multi-index advanced retrieval (session 10 live).

Adds to ``/v1/retrieval/search``: query transform (expansion / decomposition), cascade routing
across the three collections, hard metadata filters, differentiated fusion, reranking and temporal
decay; every stage is independently switchable. ``/v1/retrieval/search`` stays untouched.

Thin transport: auth in ``require_retrieval_key`` (401), rate limit in ``@limiter.limit`` (429),
validation in the request model (422). The response surfaces not only the chunks but HOW they were
obtained (routing level and reason, technique, sub-queries, per-collection cardinality).
"""

import asyncio
from datetime import UTC, date, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.rate_limiting import limiter
from app.api.security import require_retrieval_key
from app.config import Settings, get_settings
from app.dependencies import (
    get_async_session_factory,
    get_chunk_store,
    get_embedder,
    get_llm_wrapper,
    get_reranker,
    get_runtime_config,
    get_runtime_retrieval_config,
)
from app.foundation.llm.runtime_config import RuntimeModelConfig, RuntimeRetrievalConfig
from app.foundation.llm.wrapper import LLMWrapper
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.advanced_pipeline import StageConfig, advanced_retrieve
from app.generation.rag.retrieval.collections import Collection, HardFilters
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import RetrievedChunk, SearchMode, Sector
from app.generation.rag.store.repository import ChunkSearchStore

router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])
logger = structlog.get_logger(__name__)


class AdvancedRetrievalRequest(BaseModel):
    # extra="forbid": an unknown field is a client bug, not something to ignore silently
    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=10, max_length=2000)
    # Level-0 routing: naming the collections means the router classifies nothing
    collections: list[Collection] | None = Field(
        default=None, description="Explicit collections to search (skips routing)"
    )
    top_k: int = Field(default=5, ge=1, le=30)
    # Hard metadata filters, embedded in the search query (each collection uses what it knows)
    technologies: list[str] | None = Field(default=None, description="Budgets: main technology")
    sectors: list[Sector] | None = Field(default=None, description="Budgets: client sector")
    min_date: date | None = Field(
        default=None, description="Exclude chunks older than this (budgets: by year)"
    )
    doc_version: str | None = Field(default=None, description="Technical docs: version")
    # Per-request stage overrides: None = runtime override (PUT /api/v1/config/retrieval) -> .env
    search_mode: SearchMode | None = None
    rerank: bool | None = None
    routing_enabled: bool | None = None
    query_transform_enabled: bool | None = None
    temporal_decay_enabled: bool | None = None


class RoutingInfo(BaseModel):
    level: str = Field(description="explicit | rules | classifier | fallback")
    targets: list[str]
    reason: str


class SubQueryInfo(BaseModel):
    topic: str
    query: str


class AdvancedRetrievalResult(BaseModel):
    """Results plus the diagnostics that explain how they were retrieved."""

    chunks: list[RetrievedChunk]
    low_confidence: bool
    routing: RoutingInfo
    technique: str = Field(description="direct | expand | decompose")
    subqueries: list[SubQueryInfo]
    cardinality: dict[str, int] = Field(
        description="Candidates per collection after the hard filters (0 = silent empty)"
    )


def resolve_stages(
    payload: AdvancedRetrievalRequest, settings: Settings, runtime: RuntimeRetrievalConfig
) -> StageConfig:
    """Precedence for every toggle: request field -> runtime override (Redis) -> environment.

    Blocking (Redis): async callers run it in a thread.
    """

    def pick(requested: bool | None, runtime_value: bool) -> bool:
        return requested if requested is not None else runtime_value

    return StageConfig.from_settings(
        settings,
        search_mode=payload.search_mode or runtime.effective_search_mode(),  # type: ignore[arg-type]
        rerank=pick(payload.rerank, runtime.effective_rerank()),
        routing_enabled=pick(payload.routing_enabled, runtime.effective_routing()),
        query_transform_enabled=pick(
            payload.query_transform_enabled, runtime.effective_query_transform()
        ),
        temporal_decay_enabled=pick(
            payload.temporal_decay_enabled, runtime.effective_temporal_decay()
        ),
        top_k=payload.top_k,
    )


@router.post(
    "/advanced-search",
    response_model=AdvancedRetrievalResult,
    dependencies=[Depends(require_retrieval_key)],
)
@limiter.limit("120/minute")
async def advanced_search(
    request: Request,  # slowapi needs the Starlette request under this exact name
    payload: AdvancedRetrievalRequest,
    settings: Settings = Depends(get_settings),
    runtime: RuntimeRetrievalConfig = Depends(get_runtime_retrieval_config),
    models: RuntimeModelConfig = Depends(get_runtime_config),
    llm: LLMWrapper = Depends(get_llm_wrapper),
    embedder: OpenAIEmbedder = Depends(get_embedder),
    store: ChunkSearchStore = Depends(get_chunk_store),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_async_session_factory),
    reranker: CrossEncoderReranker = Depends(get_reranker),
) -> AdvancedRetrievalResult:
    """Run the advanced multi-index pipeline. Requires ``X-API-Key`` (``RETRIEVAL_API_KEY``)."""
    stages = await asyncio.to_thread(resolve_stages, payload, settings, runtime)
    hard_filters = HardFilters(
        technologies=tuple(payload.technologies or ()),
        sectors=tuple(payload.sectors or ()),
        min_date=payload.min_date,
        version=payload.doc_version,
    )
    try:
        outcome = await advanced_retrieve(
            query_text=payload.query_text,
            stages=stages,
            reference_date=datetime.now(UTC).date(),
            llm=llm,
            models=models,
            embedder=embedder,
            session_factory=session_factory,
            store=store,
            explicit_collections=payload.collections,
            hard_filters=hard_filters,
            reranker=reranker,
        )
    except RetrievalError as exc:
        # Log only the error type and a truncated message; the client gets a generic message
        logger.error("advanced_retrieval_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        raise HTTPException(status_code=502, detail="Retrieval service error") from exc

    return AdvancedRetrievalResult(
        chunks=outcome.chunks,
        low_confidence=outcome.low_confidence,
        routing=RoutingInfo(
            level=outcome.routing.level,
            targets=[target.value for target in outcome.routing.targets],
            reason=outcome.routing.reason,
        ),
        technique=outcome.technique,
        subqueries=[SubQueryInfo(topic=sq.topic, query=sq.query) for sq in outcome.subqueries],
        cardinality=outcome.cardinality,
    )
