import structlog

from app.schemas.estimation import (
    LOW_CONFIDENCE_THRESHOLD,
    OUT_OF_SCOPE_PREFIX,
    EstimationResult,
    Phase,
)

logger = structlog.get_logger(__name__)

# Deja sitio para el prefijo + resto del summary (max_length=1200) sin acercarse al límite
_REASONING_EXCERPT_MAX_CHARS = 400


def enforce_scope_response(result: EstimationResult) -> EstimationResult:
    """Red de seguridad por si el model_validator de baja confianza no llegó a disparar.

    Política filter (nunca lanza): reescribe el resultado como "out of scope" cuando
    confidence_pct < LOW_CONFIDENCE_THRESHOLD pero el summary no lo refleja todavía.
    """
    already_out_of_scope = result.summary.startswith(OUT_OF_SCOPE_PREFIX)
    if result.confidence_pct >= LOW_CONFIDENCE_THRESHOLD or already_out_of_scope:
        return result

    logger.warning("scope_guardrail_triggered", confidence_pct=result.confidence_pct)
    excerpt = result.summary[:_REASONING_EXCERPT_MAX_CHARS]
    return EstimationResult(
        summary=f"{OUT_OF_SCOPE_PREFIX} {excerpt}",
        confidence_pct=result.confidence_pct,
        phases=[
            Phase(
                name="Not estimated",
                duration_weeks=1,
                cost_eur=0,
                summary="The description did not provide enough detail for a reliable estimation.",
            )
        ],
        total_duration_weeks=1,
        total_cost_eur=0,
    )
