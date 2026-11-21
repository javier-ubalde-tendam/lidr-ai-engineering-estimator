"""``POST /v1/estimate/from-transcript``: transcript -> grounded, citation-backed estimate.

The strictest endpoint of the service: a tight rate limit (10/min) and idempotency, because each
call can drive a multi-step LLM pipeline.

Thin transport: validation in ``EstimateRequest`` (422), auth in ``require_estimate_key`` (401),
rate limiting in the decorator (429). Any pipeline failure becomes a 502 (the request id is on
the ``X-Request-ID`` response header). "Nothing relevant retrieved" is NOT a failure: it is a
200 with ``confidence="insufficient"``.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.dependencies import get_transcript_estimator
from app.generation.rag.errors import RagError
from app.generation.rag.estimator import TranscriptEstimator
from app.generation.rag.schemas import Estimate, EstimateRequest

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate", tags=["estimate"])


@router.post(
    "/from-transcript",
    response_model=Estimate,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def from_transcript(
    request: Request,  # slowapi needs the Starlette request under this exact name
    payload: EstimateRequest,
    estimator: TranscriptEstimator = Depends(get_transcript_estimator),
) -> Estimate:
    """Produce a grounded estimate from a raw transcript (idempotent on ``idempotency_key``)."""
    try:
        return await estimator.estimate_from_transcript(
            payload.transcript, idempotency_key=payload.idempotency_key
        )
    except RagError as exc:
        log.error("estimate_failed", error_type=type(exc).__name__, error=str(exc)[:300])
        raise HTTPException(status_code=502, detail="Failed to produce an estimate.") from exc
