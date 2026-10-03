from app.guardrails.output import enforce_scope_response
from app.schemas.estimation import (
    LOW_CONFIDENCE_THRESHOLD,
    OUT_OF_SCOPE_PREFIX,
    EstimationResult,
    Phase,
)

GOOD_PHASES = [
    Phase(name="Design", duration_weeks=2, cost_eur=4000, summary="Design the platform."),
    Phase(name="Build", duration_weeks=3, cost_eur=6000, summary="Build the platform."),
]


def make_result(summary: str, confidence_pct: int) -> EstimationResult:
    # model_construct salta los model_validator: simula el caso en que no llegaron a disparar
    # (p.ej. confidence_pct justo en el límite, o un threshold relajado en el futuro)
    return EstimationResult.model_construct(
        summary=summary,
        confidence_pct=confidence_pct,
        phases=GOOD_PHASES,
        total_duration_weeks=5,
        total_cost_eur=10000,
    )


def test_low_confidence_without_prefix_is_rewritten_as_out_of_scope():
    original = make_result("A vague summary that lacks real detail.", LOW_CONFIDENCE_THRESHOLD - 1)

    result = enforce_scope_response(original)

    assert result.summary.startswith(OUT_OF_SCOPE_PREFIX)
    assert "vague summary" in result.summary
    assert result.confidence_pct == LOW_CONFIDENCE_THRESHOLD - 1
    assert result.phases == [
        Phase(
            name="Not estimated",
            duration_weeks=1,
            cost_eur=0,
            summary="The description did not provide enough detail for a reliable estimation.",
        )
    ]
    assert result.total_cost_eur == 0
    assert result.total_duration_weeks == 1


def test_already_out_of_scope_result_is_left_untouched():
    original = make_result(f"{OUT_OF_SCOPE_PREFIX} Too vague to size reliably.", LOW_CONFIDENCE_THRESHOLD - 1)

    result = enforce_scope_response(original)

    assert result == original


def test_high_confidence_result_is_left_untouched():
    original = EstimationResult(
        summary="Clear and well scoped project summary.",
        confidence_pct=80,
        phases=GOOD_PHASES,
        total_duration_weeks=5,
        total_cost_eur=10000,
    )

    result = enforce_scope_response(original)

    assert result == original
