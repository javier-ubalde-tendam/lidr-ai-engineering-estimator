from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db.session import dispose_engine
from app.embedding_pipeline.router import router as embeddings_router
from app.embedding_pipeline.router import search_router
from app.logging_config import configure_logging
from app.routers.estimations import router as estimations_router
from app.routers.sessions import router as sessions_router

configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Sin conexiones al arrancar (el pool es perezoso); al parar se cierra el engine si se usó
    yield
    await dispose_engine()


app = FastAPI(
    title="LiDR AI Engineering Estimator - Session 08 (pgvector)",
    description="AI-powered project estimation tool that generates effort and cost "
    "estimations from meeting transcriptions using LLMs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(estimations_router, prefix="/api/v1")
# Sin prefijo /api/v1 (a petición del enunciado): endpoints de sesión conversacional
app.include_router(sessions_router)
# Ingesta en PostgreSQL (sesión 08): POST /embeddings/ingest y POST /search, también sin /api/v1
app.include_router(embeddings_router)
app.include_router(search_router)


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok!", "environment": settings.APP_ENV, "llm": settings.LLM_PROVIDER}
