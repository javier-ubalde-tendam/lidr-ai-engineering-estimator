from fastapi import FastAPI

from app.config import get_settings
from app.embedding_pipeline.router import router as embeddings_router
from app.logging_config import configure_logging
from app.routers.estimations import router as estimations_router
from app.routers.sessions import router as sessions_router

configure_logging()

app = FastAPI(
    title="LiDR AI Engineering Estimator - Session 05 (memory + attachments)",
    description="AI-powered project estimation tool that generates effort and cost "
    "estimations from meeting transcriptions using LLMs.",
    version="0.1.0",
)

app.include_router(estimations_router, prefix="/api/v1")
# Sin prefijo /api/v1 (a petición del enunciado): endpoints de sesión conversacional
app.include_router(sessions_router)
# Pipeline de embeddings (sesión 07): POST /embeddings/ingest, también sin prefijo /api/v1
app.include_router(embeddings_router)


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok!", "environment": settings.APP_ENV, "llm": settings.LLM_PROVIDER}
