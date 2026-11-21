import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.dependencies import get_estimation_service
from app.domain.estimation_service import EstimationFailedError, EstimationService
from app.domain.schemas.estimation import EstimationRequest, EstimationResponse
from app.foundation.guardrails.input import InputGuardrailViolation

router = APIRouter(tags=["estimations"])


def _guardrail_violation_detail(exc: InputGuardrailViolation) -> dict:
    return {"reason": exc.reason, "message": exc.message}


@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(
    request: EstimationRequest, service: EstimationService = Depends(get_estimation_service)
) -> EstimationResponse:
    try:
        return service.estimate(request)
    except InputGuardrailViolation as exc:
        raise HTTPException(status_code=400, detail=_guardrail_violation_detail(exc)) from exc
    except EstimationFailedError as exc:
        # 502: el fallo es del proveedor LLM; el detalle interno no se expone al cliente
        raise HTTPException(status_code=502, detail="The LLM did not return a valid estimation") from exc


def _format_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/estimate/stream")
def create_estimation_stream(
    request: EstimationRequest, service: EstimationService = Depends(get_estimation_service)
) -> StreamingResponse:
    # Se valida aquí (fuera del generador) para poder devolver un 400 real: una vez que
    # StreamingResponse empieza a iterar, las cabeceras (200) ya se enviaron y no se pueden cambiar
    try:
        service.check_request(request)
    except InputGuardrailViolation as exc:
        raise HTTPException(status_code=400, detail=_guardrail_violation_detail(exc)) from exc

    stats: dict = {}
    prompt_version = service.settings.PROMPT_VERSION

    def event_generator():
        try:
            for event, result in service.estimate_stream(request, stats):
                yield _format_sse(event, result.model_dump(mode="json"))
        except EstimationFailedError:
            # Ya se enviaron cabeceras 200: el fallo se comunica como evento, y sin exponer detalle interno
            yield _format_sse("error", {"detail": "The LLM did not return a valid estimation"})
        yield _format_sse("metrics", {**stats, "prompt_version": prompt_version})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
