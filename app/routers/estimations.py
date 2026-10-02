import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.llm_service import estimate_project, estimate_project_stream

router = APIRouter(tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(request: EstimationRequest) -> EstimationResponse:
    settings = get_settings()
    # Por ahora solo usamos description; project_type/detail_level/output_format se usarán con los templates Jinja2
    estimation, tokens_input, tokens_output, model = estimate_project(request.description)
    return EstimationResponse(
        estimation=estimation,
        prompt_version="v0",  # Valor fijo hasta que existan los prompts versionados
        model=model,
        provider=settings.LLM_PROVIDER,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
    )


def _format_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/estimate/stream")
def create_estimation_stream(request: EstimationRequest) -> StreamingResponse:
    stats: dict = {}

    def event_generator():
        for chunk in estimate_project_stream(request.description, stats):
            yield _format_sse("token", {"content": chunk})
        yield _format_sse("metrics", stats)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )