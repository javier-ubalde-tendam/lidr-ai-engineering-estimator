"""``POST /v1/estimate/agent``: transcript -> estimate by a tool-calling agent (Session 12).

The most expensive endpoint of the service (one run = several paid model turns), hence the
strictest rate limit (5/min). Thin transport: validation in ``AgentEstimateRequest`` (422), auth in
``require_estimate_key`` (401), rate limit in the decorator (429). The agent loop and its tools
live in ``generation/agentic``; the conductor wires them.

Status mapping: the agent's own outcome (``done`` / ``max_steps_exceeded`` / ``error``) is business
information and always travels in a 200 body together with the trace; transport failures are
HTTP errors: no OpenAI key -> 503, failing OpenAI call -> 502.
"""

from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.config import get_settings
from app.dependencies import get_agent_estimation_service, get_async_openai_client
from app.domain.estimation_service import AgentUnavailableError, EstimationService
from app.generation.agentic.agent_loop import AgentLoopError
from app.generation.agentic.agent_schemas import AgentResult

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate", tags=["estimate-agent"])


def _agent_models() -> list[str]:
    # Only OpenAI reasoning models: the agent uses the Responses API and the `reasoning` parameter
    return [model for model in get_settings().AVAILABLE_MODELS if model.startswith("gpt-5")]


class AgentEstimateRequest(BaseModel):
    """Payload of ``POST /v1/estimate/agent``; the optional fields override the AGENT_* settings."""

    transcript: str = Field(min_length=100, max_length=50_000)
    model: str | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    max_iterations: int | None = Field(default=None, ge=1, le=20)

    @field_validator("model")
    @classmethod
    def model_is_allowed(cls, value: str | None) -> str | None:
        allowed = _agent_models()
        if value is not None and value not in allowed:
            raise ValueError(f"model must be one of {allowed}")
        return value


async def require_agent_client(
    client: AsyncOpenAI | None = Depends(get_async_openai_client),
) -> None:
    """503 when there is no OPENAI_API_KEY: the agent needs OpenAI specifically."""
    if client is None:
        raise HTTPException(
            status_code=503, detail="The agent requires OPENAI_API_KEY to be configured."
        )


@router.post(
    "/agent",
    response_model=AgentResult,
    dependencies=[Depends(require_estimate_key), Depends(require_agent_client)],
)
@limiter.limit("5/minute")
async def estimate_agent(
    request: Request,  # slowapi needs the Starlette request under this exact name
    payload: AgentEstimateRequest,
    service: EstimationService = Depends(get_agent_estimation_service),
) -> AgentResult:
    """Run the estimation agent and return the estimate together with its reasoning trace."""
    try:
        return await service.estimate_with_agent(
            payload.transcript,
            model=payload.model,
            reasoning_effort=payload.reasoning_effort,
            max_iterations=payload.max_iterations,
        )
    except AgentUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentLoopError as exc:
        log.error("agent_failed", steps_completed=len(exc.trace), error=str(exc)[:300])
        raise HTTPException(status_code=502, detail="The agent failed to call OpenAI.") from exc
