"""DTOs de la traza del Actor-Critic-Boss (van en la response del endpoint)."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.estimation import ConversationalEstimationResponse
from app.services.llm_wrapper import LLMCallMeta

BossDecision = Literal["accept", "iterate", "synthesize"]


class ACBIteration(BaseModel):
    iteration: int = Field(ge=0)
    decision_after: BossDecision
    critic_verdict: str
    critic_confidence: int = Field(ge=0, le=100)
    issue_summary: list[str] = Field(default_factory=list)


class BossTrace(BaseModel):
    iterations: list[ACBIteration] = Field(default_factory=list)
    final_decision: BossDecision
    iterations_run: int = Field(ge=0)
    # Agregado de todas las llamadas (actor + critic) para la observabilidad de coste y latencia
    llm_usage: LLMCallMeta | None = None


class ACBResponse(ConversationalEstimationResponse):
    acb: BossTrace
