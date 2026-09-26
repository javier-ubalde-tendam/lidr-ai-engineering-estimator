from fastapi import APIRouter

from app.config import get_settings
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.llm_service import estimate_project

router = APIRouter(tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(request: EstimationRequest) -> EstimationResponse:
    settings = get_settings()
    estimation, tokens_input, tokens_output = estimate_project(request.transcription)
    return EstimationResponse(
        estimation=estimation,
        model=settings.LLM_MODEL,
        provider=settings.LLM_PROVIDER,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
    )
