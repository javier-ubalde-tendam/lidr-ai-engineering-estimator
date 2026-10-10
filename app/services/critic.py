"""Critic: auditoría independiente de una estimación, sin estado y sin decidir qué hacer con ella.

Es fail-open: si falla (red, validación, proveedor) devuelve un accept sin confianza para que
el ACB nunca empeore la disponibilidad respecto a usar solo el actor.
"""

import structlog

from app.config import get_settings
from app.prompts.loader import render_critic_prompt
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import EstimationResult
from app.services.llm_wrapper import LLMCallMeta, call_structured
from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier

logger = structlog.get_logger(__name__)


def review(
    transcript: str,
    metadata: ProjectMetadata,
    tier: Tier,
    estimation: EstimationResult,
    **log_context: object,
) -> tuple[CriticFeedback, LLMCallMeta | None]:
    settings = get_settings()
    system_prompt, user_prompt = render_critic_prompt(
        transcript=transcript, metadata=metadata, tier=tier.value, result=estimation
    )
    try:
        feedback, meta = call_structured(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            CriticFeedback,
            settings.CRITIC_MODEL,
            max_retries=3,
            purpose="critic",
            **log_context,
        )
    except Exception as exc:  # noqa: BLE001 - fail-open: cualquier fallo del critic equivale a "no objeciones"
        logger.warning("critic_failed_fallback_accept", error_type=type(exc).__name__, error=str(exc))
        return CriticFeedback(verdict="accept", issues=[], confidence_in_review=0), None

    logger.info(
        "critic_completed",
        verdict=feedback.verdict,
        issues=len(feedback.issues),
        confidence_in_review=feedback.confidence_in_review,
    )
    return feedback, meta
