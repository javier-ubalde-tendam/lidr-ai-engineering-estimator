import json

import instructor
import litellm
import pytest

from app.schemas.estimation import EstimationRequest
from app.services import llm_service
from app.services.llm_service import EstimationFailedError, estimate_project

GOOD = {
    "summary": "Web platform with billing and reporting.",
    "confidence_pct": 80,
    "phases": [
        {"name": "Design", "duration_weeks": 2, "cost_eur": 4000, "summary": "Design the platform."},
        {"name": "Build", "duration_weeks": 3, "cost_eur": 6000, "summary": "Build the platform."},
    ],
    "total_duration_weeks": 5,
    "total_cost_eur": 10000,
}
BAD = {**GOOD, "total_cost_eur": 10500}  # no cuadra con la suma de las fases

REQUEST = EstimationRequest(
    description="Plataforma web de facturación para pequeños negocios",
    project_type="web_saas",
    detail_level="summary",
    output_format="narrative",
)


def tool_call_response(payload: dict) -> litellm.ModelResponse:
    # Respuesta con la forma que devuelve el LLM en modo tools: el JSON va en tool_calls, no en content
    return litellm.ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "EstimationResult", "arguments": json.dumps(payload)},
                        }
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        model="fake",
    )


def use_fake_llm(monkeypatch, payloads: list[dict] | None = None, error: Exception | None = None):
    """Sustituye el cliente de Instructor por uno real que habla con un LLM falso; devuelve las llamadas."""
    calls: list[list[dict]] = []

    def fake_completion(*args, **kwargs):
        calls.append(kwargs["messages"])
        if error:
            raise error
        return tool_call_response(payloads[min(len(calls) - 1, len(payloads) - 1)])

    monkeypatch.setattr(llm_service, "_structured_client", instructor.from_litellm(fake_completion))
    return calls


def test_estimate_project_returns_validated_result_and_tokens(monkeypatch):
    use_fake_llm(monkeypatch, [GOOD])

    result, tokens_input, tokens_output, _ = estimate_project(REQUEST)

    assert result.total_cost_eur == 10000
    assert (tokens_input, tokens_output) == (100, 50)


def test_invalid_output_is_retried_with_the_validation_error(monkeypatch):
    calls = use_fake_llm(monkeypatch, [BAD, GOOD])

    result, tokens_input, tokens_output, _ = estimate_project(REQUEST)

    assert result.total_cost_eur == 10000
    assert len(calls) == 2
    # El reintento incluye el mensaje del validator para que el modelo pueda corregirse
    assert "total_cost_eur is 10500 but the phases sum to 10000" in calls[1][-1]["content"]
    # Los tokens son la suma de ambos intentos
    assert (tokens_input, tokens_output) == (200, 100)


def test_estimate_project_fails_after_exhausting_retries(monkeypatch):
    calls = use_fake_llm(monkeypatch, [BAD])

    with pytest.raises(EstimationFailedError):
        estimate_project(REQUEST)

    assert len(calls) == 1 + llm_service.settings.STRUCTURED_OUTPUT_MAX_RETRIES


def test_provider_errors_are_wrapped_and_not_retried(monkeypatch):
    auth_error = litellm.AuthenticationError(message="bad key", llm_provider="openai", model="fake")
    calls = use_fake_llm(monkeypatch, error=auth_error)

    with pytest.raises(EstimationFailedError):
        estimate_project(REQUEST)

    assert len(calls) == 1
