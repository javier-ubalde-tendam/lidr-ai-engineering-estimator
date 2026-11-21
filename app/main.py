from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request, Response
from slowapi.errors import RateLimitExceeded

from app.api import config as config_api
from app.api.embeddings import router as embeddings_router
from app.api.estimations import router as estimations_router
from app.api.ingestion import router as ingestion_router
from app.api.ingestion import warm_up_catalog
from app.api.rate_limiting import limiter, rate_limit_exceeded_handler
from app.api.routers.corpus_index import router as corpus_index_router
from app.api.routers.estimate import router as estimate_router
from app.api.routers.estimate_stages import router as estimate_stages_router
from app.api.routers.estimate_tasks import router as estimate_tasks_router
from app.api.routers.retrieval import router as retrieval_router
from app.api.routers.retrieval_advanced import router as retrieval_advanced_router
from app.api.search import router as search_router
from app.api.sessions import router as sessions_router
from app.config import get_settings
from app.foundation.persistence.database import dispose_engine
from app.logging_config import configure_logging

configure_logging()

# Only well-formed ids are echoed back / logged; anything else gets a fresh UUID
_MAX_REQUEST_ID_LENGTH = 128


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Sin conexiones al arrancar (el pool es perezoso); al parar se cierra el engine si se usó
    warm_up_catalog()
    yield
    await dispose_engine()


app = FastAPI(
    title="LiDR AI Engineering Estimator - Session 08 (pgvector)",
    description="AI-powered project estimation tool that generates effort and cost "
    "estimations from meeting transcriptions using LLMs.",
    version="0.1.0",
    lifespan=lifespan,
)

# The @limiter.limit decorators on the routes read app.state.limiter; the handler turns a
# breach into a JSON 429 with Retry-After
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


@app.middleware("http")
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Give each request a correlation id: bound to the structlog context (so every log line
    of the request carries it) and echoed on the ``X-Request-ID`` response header."""
    incoming = request.headers.get("X-Request-ID", "")
    request_id = incoming if _is_valid_request_id(incoming) else str(uuid4())
    request.state.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.unbind_contextvars("request_id")
    response.headers["X-Request-ID"] = request_id
    return response


def _is_valid_request_id(value: str) -> bool:
    return 0 < len(value) <= _MAX_REQUEST_ID_LENGTH and value.isascii() and value.isprintable()


app.include_router(estimations_router, prefix="/api/v1")
# Sin prefijo /api/v1 (a petición del enunciado): endpoints de sesión conversacional
app.include_router(sessions_router)
# Ingesta en PostgreSQL (sesión 08): POST /embeddings/ingest y POST /search, también sin /api/v1
app.include_router(embeddings_router)
app.include_router(search_router)
# Session 10: hybrid search + reranking (POST /v1/retrieval/search). /search is left untouched
app.include_router(retrieval_router)
# Session 10 live: multi-index advanced retrieval (POST /v1/retrieval/advanced-search)
app.include_router(retrieval_advanced_router)
# Session 9: transcript -> grounded estimate (POST /v1/estimate/from-transcript)
app.include_router(estimate_router)
# Didactic extras (8c): the pipeline one stage at a time, per-task hours, incremental corpus growth
app.include_router(estimate_stages_router)
app.include_router(estimate_tasks_router)
app.include_router(corpus_index_router)
# Session 6: offline ingestion of the catalog's included sources (POST /api/v1/ingestion/runs)
app.include_router(ingestion_router)
# Runtime model / retrieval-stage overrides (Redis): GET/PUT /api/v1/config/{models,retrieval}
app.include_router(config_api.router)


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok!", "environment": settings.APP_ENV, "llm": settings.LLM_PROVIDER}
