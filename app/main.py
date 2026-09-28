from fastapi import FastAPI
from app.config import get_settings
from app.logging_config import configure_logging
from app.routers.estimations import router as estimations_router

configure_logging()

app = FastAPI(
    title="LiDR AI Engineering Estimator - Session 02 (CAG)",
    description="AI-powered project estimation tool that generates effort and cost "
    "estimations from meeting transcriptions using LLMs.",
    version="0.1.0",
)

app.include_router(estimations_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok!", "environment": settings.APP_ENV, "llm": settings.LLM_PROVIDER}
