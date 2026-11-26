"""Startup wiring of the estimation graph (Session 13).

``main.py`` may only import ``api``, ``foundation`` and ``config``, so the real dependencies of the
graph are assembled here (``api`` may import ``dependencies`` and ``domain``) and the compiled graph
is left in ``app.state.graph``. A failure to open the checkpointer is NOT fatal: the service starts
with ``app.state.graph = None`` and ``POST /v1/estimate/graph`` answers 503.
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import structlog
from fastapi import FastAPI

from app.config import get_settings
from app.dependencies import (
    get_chunk_store,
    get_embedder,
    get_llm_wrapper,
    get_reranker,
    get_transcript_estimator,
)
from app.domain.estimation_service import build_budget_search_backend
from app.domain.graph import GraphDependencies, build_graph
from app.foundation.observability import configure_logfire
from app.foundation.persistence.checkpointer import open_checkpointer

log = structlog.get_logger()


def build_graph_dependencies() -> GraphDependencies:
    """Real collaborators: the shared LLM wrapper and the S9/S10 retrieval as the search backend."""
    settings = get_settings()
    # Explicit arguments: the factory's Depends(...) defaults only resolve inside a request
    estimator = get_transcript_estimator(
        embedder=get_embedder(), store=get_chunk_store(), reranker=get_reranker()
    )
    return GraphDependencies(
        llm_wrapper=get_llm_wrapper(),
        search_backend=build_budget_search_backend(estimator, settings),
        settings=settings,
    )


@asynccontextmanager
async def graph_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure Logfire, open the checkpointer and expose the compiled graph on ``app.state``."""
    configure_logfire(app)
    app.state.graph = None
    async with AsyncExitStack() as stack:
        if get_settings().GRAPH_ENABLED:
            try:
                checkpointer = await stack.enter_async_context(open_checkpointer())
                app.state.graph = build_graph(checkpointer, build_graph_dependencies())
                log.info("graph_ready")
            except Exception as exc:  # noqa: BLE001 - the rest of the API must keep working
                app.state.graph = None
                log.warning(
                    "graph_unavailable", error_type=type(exc).__name__, error=str(exc)[:200]
                )
        else:
            log.info("graph_disabled")
        yield
    app.state.graph = None
