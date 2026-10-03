import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.llm_service import EstimationFailedError, estimate_project, estimate_project_stream

router = APIRouter(tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(request: EstimationRequest) -> EstimationResponse:
    settings = get_settings()
    try:
        result, *_ = estimate_project(request)
    except EstimationFailedError as exc:
        # 502: el fallo es del proveedor LLM; el detalle interno no se expone al cliente
        raise HTTPException(status_code=502, detail="The LLM did not return a valid estimation") from exc
    return EstimationResponse(result=result, prompt_version=settings.PROMPT_VERSION)


def _format_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/estimate/stream")
def create_estimation_stream(request: EstimationRequest) -> StreamingResponse:
    stats: dict = {}
    prompt_version = get_settings().PROMPT_VERSION

    def event_generator():
        for chunk in estimate_project_stream(request, stats):
            yield _format_sse("token", {"content": chunk})
        yield _format_sse("metrics", {**stats, "prompt_version": prompt_version})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )