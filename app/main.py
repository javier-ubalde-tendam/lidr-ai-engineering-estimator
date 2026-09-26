from fastapi import FastAPI

from app.routers.estimations import router as estimations_router

app = FastAPI(
    title="LiDR AI Engineering Estimator - Session 02 (CAG)",
    description="AI-powered project estimation tool that generates effort and cost "
    "estimations from meeting transcriptions using LLMs.",
    version="0.1.0",
)

app.include_router(estimations_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
