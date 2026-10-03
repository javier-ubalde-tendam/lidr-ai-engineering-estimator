import pytest
from pydantic import ValidationError

from app.schemas.estimation import EstimationRequest, EstimationResult, ProjectType

VALID_FIELDS = {
    "description": "Aplicación web SaaS de facturación para pequeños negocios",
    "project_type": "web_saas",
    "detail_level": "summary",
    "output_format": "narrative",
}

def test_description_too_short_is_rejected():
    # Se pasan strings: Pydantic los convierte a Enum. pytest.raises equivale a assertThrows
    with pytest.raises(ValidationError):
        EstimationRequest(
            description="corta",
            project_type="web_saas",
            detail_level="summary",
            output_format="narrative",
        )


def test_valid_request_is_accepted():
    request = EstimationRequest(**VALID_FIELDS)  # ** desempaqueta el dict como argumentos con nombre
    assert request.project_type == ProjectType.WEB_SAAS


def test_unknown_project_type_is_rejected():
    with pytest.raises(ValidationError):
        EstimationRequest(**{**VALID_FIELDS, "project_type": "no_existe"})


def test_description_too_long_is_rejected():
    with pytest.raises(ValidationError):
        EstimationRequest(**{**VALID_FIELDS, "description": "x" * 10001})


def make_phase(**overrides) -> dict:
    fields = {
        "name": "Implementation",
        "duration_weeks": 3,
        "cost_eur": 6000,
        "summary": "Build the core features.",
    }
    return {**fields, **overrides}


def make_result(**overrides) -> dict:
    # Dict como el JSON del LLM: model_validate lo valida igual que lo hará Instructor
    fields = {
        "summary": "Web platform with billing and reporting.",
        "confidence_pct": 80,
        "phases": [make_phase(name="Design", duration_weeks=2, cost_eur=4000), make_phase()],
        "total_duration_weeks": 5,
        "total_cost_eur": 10000,
    }
    return {**fields, **overrides}


def test_valid_result_is_accepted():
    result = EstimationResult.model_validate(make_result())

    assert result.total_cost_eur == 10000
    assert len(result.phases) == 2


def test_totals_mismatch_does_not_reject_the_result_but_is_reported():
    result = EstimationResult.model_validate(make_result(total_cost_eur=10500))

    message = result.totals_mismatch()

    assert message == "total_cost_eur is 10500 but the phases sum to 10000"


def test_totals_mismatch_is_none_when_totals_match():
    assert EstimationResult.model_validate(make_result()).totals_mismatch() is None


def test_low_confidence_without_out_of_scope_prefix_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        EstimationResult.model_validate(make_result(confidence_pct=20))

    assert "Out of scope:" in exc_info.value.errors()[0]["msg"]


def test_low_confidence_with_out_of_scope_prefix_is_accepted():
    summary = "Out of scope: the description is too vague to size."

    result = EstimationResult.model_validate(make_result(confidence_pct=20, summary=summary))

    assert result.confidence_pct == 20


def test_confidence_29_still_requires_out_of_scope_prefix():
    with pytest.raises(ValidationError):
        EstimationResult.model_validate(make_result(confidence_pct=29))


def test_confidence_30_does_not_require_out_of_scope_prefix():
    # El umbral es "< 30": 30 ya es una estimación normal
    EstimationResult.model_validate(make_result(confidence_pct=30))


def test_summary_shorter_than_10_chars_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        EstimationResult.model_validate(make_result(summary="too short"))  # 9 caracteres

    # Se comprueba el motivo para que el test no pase por otro error distinto
    error = exc_info.value.errors()[0]
    assert error["loc"] == ("summary",)
    assert error["type"] == "string_too_short"


def test_more_than_8_phases_is_rejected():
    phases = [make_phase(cost_eur=1000) for _ in range(9)]
    # Totales coherentes con las fases: así el único fallo posible es el límite de fases
    data = make_result(phases=phases, total_cost_eur=9000, total_duration_weeks=27)

    with pytest.raises(ValidationError) as exc_info:
        EstimationResult.model_validate(data)

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("phases",)
    assert error["type"] == "too_long"