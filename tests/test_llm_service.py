import json

import instructor
import litellm
import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from app.cache import build_cache_key
from app.guardrails.input import InputGuardrailViolation
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest, EstimationResult
from app.services import llm_service, llm_wrapper
from app.services.llm_service import (
    EstimationFailedError,
    estimate_project,
    estimate_project_stream_structured,
)


@pytest.fixture(autouse=True)
def bypass_input_guardrail(monkeypatch):
    # El guardrail de input se prueba aparte (tests/test_guardrails_input.py); aquí se neutraliza
    # para que estos tests no dependan de la Moderation API real
    monkeypatch.setattr(llm_service, "check_input", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def bypass_semantic_cache(monkeypatch):
    # El cache semántico se prueba aparte (tests/test_semantic_cache.py); aquí se neutraliza
    # para no depender de Redis-stack ni de la API de embeddings real
    monkeypatch.setattr(llm_service, "semantic_cache_lookup", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_service, "semantic_cache_store", lambda *args, **kwargs: None)

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
TOTALS_MISMATCH = {**GOOD, "total_cost_eur": 10500}  # es válido, pero no cuadra con la suma de las fases
LOW_CONFIDENCE_WITHOUT_PREFIX = {**GOOD, "confidence_pct": 10}  # exige summary "Out of scope:..."

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

    monkeypatch.setattr(llm_wrapper, "_structured_client", instructor.from_litellm(fake_completion))
    return calls


def test_estimate_project_returns_validated_result_and_tokens(monkeypatch):
    use_fake_llm(monkeypatch, [GOOD])

    result, tokens_input, tokens_output, _ = estimate_project(REQUEST)

    assert result.total_cost_eur == 10000
    assert (tokens_input, tokens_output) == (100, 50)


def test_invalid_output_is_retried_with_the_validation_error(monkeypatch):
    calls = use_fake_llm(monkeypatch, [LOW_CONFIDENCE_WITHOUT_PREFIX, GOOD])

    result, tokens_input, tokens_output, _ = estimate_project(REQUEST)

    assert result.total_cost_eur == 10000
    assert len(calls) == 2
    # El reintento incluye el mensaje del validator para que el modelo pueda corregirse
    assert "requires summary to start with 'Out of scope:'" in calls[1][-1]["content"]
    # Los tokens son la suma de ambos intentos
    assert (tokens_input, tokens_output) == (200, 100)


def test_estimate_project_fails_after_exhausting_retries(monkeypatch):
    calls = use_fake_llm(monkeypatch, [LOW_CONFIDENCE_WITHOUT_PREFIX])

    with pytest.raises(EstimationFailedError):
        estimate_project(REQUEST)

    assert len(calls) == 1 + llm_service.settings.STRUCTURED_OUTPUT_MAX_RETRIES


def test_estimate_project_accepts_totals_mismatch_without_retrying(monkeypatch):
    calls = use_fake_llm(monkeypatch, [TOTALS_MISMATCH])

    result, *_ = estimate_project(REQUEST)

    assert result.total_cost_eur == 10500
    assert len(calls) == 1


def test_provider_errors_are_wrapped_and_not_retried(monkeypatch):
    auth_error = litellm.AuthenticationError(message="bad key", llm_provider="openai", model="fake")
    calls = use_fake_llm(monkeypatch, error=auth_error)

    with pytest.raises(EstimationFailedError):
        estimate_project(REQUEST)

    assert len(calls) == 1


def test_estimate_project_never_calls_the_llm_when_the_input_guardrail_fails(monkeypatch):
    calls = use_fake_llm(monkeypatch, [GOOD])

    def failing_check_input(description, *, openai_client=None):
        raise InputGuardrailViolation(reason="pii", message="blocked")

    monkeypatch.setattr(llm_service, "check_input", failing_check_input)

    with pytest.raises(InputGuardrailViolation):
        estimate_project(REQUEST)

    assert calls == []


def test_estimate_project_runs_the_output_scope_guardrail(monkeypatch):
    use_fake_llm(monkeypatch, [GOOD])
    received = []
    monkeypatch.setattr(llm_service, "enforce_scope_response", lambda result: received.append(result) or result)

    result, *_ = estimate_project(REQUEST)

    assert received == [result]


# --- Streaming estructurado ---


def tool_call_stream(raw_json: str, size: int = 40) -> list[ModelResponseStream]:
    # Forma real de un stream con tool calling: fragmentos JSON, un chunk de cierre y otro solo con usage
    def chunk(choices, **extra):
        return ModelResponseStream(id="c", model="gpt-4o-mini", choices=choices, **extra)

    fragments = [
        chunk(
            [
                StreamingChoices(
                    index=0,
                    finish_reason=None,
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=0,
                                type="function",
                                id="call_1",
                                function=Function(name="EstimationResult", arguments=raw_json[i : i + size]),
                            )
                        ]
                    ),
                )
            ]
        )
        for i in range(0, len(raw_json), size)
    ]
    closing = chunk([StreamingChoices(index=0, finish_reason="tool_calls", delta=Delta())])
    usage = chunk([], usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150))
    return [*fragments, closing, usage]


def use_fake_stream(monkeypatch, payloads: list[dict | str]) -> list[dict]:
    """Sustituye litellm.completion por un stream falso (uno por llamada); devuelve los kwargs recibidos."""
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        payload = payloads[min(len(calls) - 1, len(payloads) - 1)]
        return iter(tool_call_stream(payload if isinstance(payload, str) else json.dumps(payload)))

    # _structured_client conserva la referencia original, así que el camino bloqueante no se ve afectado
    monkeypatch.setattr(llm_service.litellm, "completion", fake_completion)
    return calls


@pytest.fixture
def fake_cache(monkeypatch) -> dict[str, str]:
    # Evita tocar el Redis real: los tests no deben leer ni escribir en la caché de desarrollo
    store: dict[str, str] = {}
    monkeypatch.setattr(llm_service, "get_cached_estimation", store.get)
    monkeypatch.setattr(llm_service, "set_cached_estimation", store.__setitem__)
    return store


def run_stream(stats: dict | None = None):
    stats = {} if stats is None else stats
    return list(estimate_project_stream_structured(REQUEST, stats)), stats


def filled_fields(result: EstimationResult) -> int:
    return sum(value is not None for value in result.model_dump().values())


def test_stream_emits_growing_partials_and_a_validated_complete(monkeypatch, fake_cache):
    calls = use_fake_stream(monkeypatch, [GOOD])

    events, stats = run_stream()

    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "complete"
    assert set(kinds[:-1]) == {"partial"}
    # Los campos se van completando: el primer parcial va casi vacío y ninguno retrocede
    counts = [filled_fields(result) for kind, result in events if kind == "partial"]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]
    assert events[0][1].total_cost_eur is None
    # El complete es un EstimationResult real que ha pasado la validación y no genera avisos
    assert events[-1][1] == EstimationResult.model_validate(GOOD)
    assert "validation_warnings" not in stats
    assert calls[0]["stream"] is True
    assert calls[0]["tool_choice"]["function"]["name"] == "EstimationResult"
    assert stats["cache_hit"] is False
    assert (stats["provider_used"], stats["fallback_used"]) == ("openai", False)
    assert (stats["tokens_input"], stats["tokens_output"]) == (100, 50)
    # Solo se cachea el resultado validado, serializado como JSON
    assert [EstimationResult.model_validate_json(v) for v in fake_cache.values()] == [events[-1][1]]


def test_stream_cache_hit_replays_the_cached_result_without_calling_the_llm(monkeypatch, fake_cache):
    cached = EstimationResult.model_validate(GOOD)
    fake_cache[build_cache_key(*render_estimation_prompt(REQUEST))] = cached.model_dump_json()
    calls = use_fake_stream(monkeypatch, [GOOD])

    events, stats = run_stream()

    assert calls == []
    kinds = [kind for kind, _ in events]
    assert kinds.count("partial") > 1  # se trocea el JSON cacheado como si llegara en streaming
    assert kinds[-1] == "complete"
    assert events[-1][1] == cached
    assert stats == {"cache_hit": True, "exact_cache_hit": True, "semantic_cache_hit": False}


def test_stream_ignores_stale_cache_entries_that_are_not_valid_results(monkeypatch, fake_cache):
    key = build_cache_key(*render_estimation_prompt(REQUEST))
    fake_cache[key] = "texto libre del streaming anterior"
    calls = use_fake_stream(monkeypatch, [GOOD])

    events, stats = run_stream()

    assert len(calls) == 1
    assert stats["cache_hit"] is False
    assert EstimationResult.model_validate_json(fake_cache[key]) == events[-1][1]


def test_stream_totals_mismatch_is_accepted_flagged_and_not_cached(monkeypatch, fake_cache):
    use_fake_stream(monkeypatch, [TOTALS_MISMATCH])

    events, stats = run_stream()

    # Sin reintento bloqueante: el resultado llega como complete y el desajuste viaja en stats (evento metrics)
    assert events[-1][0] == "complete"
    assert events[-1][1].total_cost_eur == 10500
    assert "blocking_fallback_used" not in stats
    assert stats["validation_warnings"] == [
        "totals_match_phases: total_cost_eur is 10500 but the phases sum to 10000"
    ]
    # No se cachea, para que el usuario pueda volver a pedir la estimación y obtener una coherente
    assert fake_cache == {}


def test_stream_invalid_final_result_falls_back_to_a_blocking_retry(monkeypatch, fake_cache):
    use_fake_stream(monkeypatch, [LOW_CONFIDENCE_WITHOUT_PREFIX])
    blocking_calls = use_fake_llm(monkeypatch, [GOOD])

    events, stats = run_stream()

    # El resultado inválido nunca llega como complete: el único complete es el del reintento bloqueante
    # (se compara model_dump(): Instructor adjunta _raw_response a la instancia y rompe el ==)
    completes = [result.model_dump() for kind, result in events if kind == "complete"]
    assert completes == [GOOD]
    assert len(blocking_calls) == 1
    assert stats["blocking_fallback_used"] is True
    # Tokens del intento en streaming + los del reintento
    assert (stats["tokens_input"], stats["tokens_output"]) == (200, 100)
    assert [json.loads(v) for v in fake_cache.values()] == [GOOD]


def test_stream_truncated_json_is_not_accepted_as_complete(monkeypatch, fake_cache):
    # Instructor no valida un JSON sin cerrar (p. ej. por max_tokens): lo debe cazar la validación final
    use_fake_stream(monkeypatch, [json.dumps(GOOD)[:-30]])
    use_fake_llm(monkeypatch, [GOOD])

    events, stats = run_stream()

    assert [result.model_dump() for kind, result in events if kind == "complete"] == [GOOD]
    assert stats["blocking_fallback_used"] is True


def test_stream_runs_the_output_scope_guardrail_on_the_happy_path(monkeypatch, fake_cache):
    use_fake_stream(monkeypatch, [GOOD])
    received = []
    monkeypatch.setattr(llm_service, "enforce_scope_response", lambda result: received.append(result) or result)

    events, _ = run_stream()

    assert received == [events[-1][1]]  # se llama exactamente una vez, con el resultado final


def test_stream_does_not_double_apply_the_output_scope_guardrail_on_blocking_fallback(monkeypatch, fake_cache):
    # estimate_project() ya aplica el guardrail internamente: el camino de fallback no debe repetirlo
    use_fake_stream(monkeypatch, [json.dumps(GOOD)[:-30]])  # JSON truncado -> dispara el fallback bloqueante
    use_fake_llm(monkeypatch, [GOOD])
    received = []
    monkeypatch.setattr(llm_service, "enforce_scope_response", lambda result: received.append(result) or result)

    events, stats = run_stream()

    assert stats["blocking_fallback_used"] is True
    assert received == [events[-1][1]]  # una sola vez en total, no cero ni dos


def test_stream_fails_without_caching_when_the_blocking_retry_also_fails(monkeypatch, fake_cache):
    use_fake_stream(monkeypatch, [LOW_CONFIDENCE_WITHOUT_PREFIX])
    use_fake_llm(monkeypatch, [LOW_CONFIDENCE_WITHOUT_PREFIX])
    events = []

    with pytest.raises(EstimationFailedError):
        for event in estimate_project_stream_structured(REQUEST, {}):
            events.append(event)

    assert {kind for kind, _ in events} == {"partial"}
    assert fake_cache == {}


# --- Cache semántico (wiring; la lógica de threshold/log_only vive en test_semantic_cache.py) ---


def test_stream_serves_a_semantic_cache_hit_without_calling_the_llm(monkeypatch, fake_cache):
    semantic_hit = EstimationResult.model_validate(GOOD)
    calls = use_fake_stream(monkeypatch, [GOOD])  # si se llamara al LLM, esto delataría la llamada
    monkeypatch.setattr(llm_service, "semantic_cache_lookup", lambda request, prompt_version, openai_client: semantic_hit)

    events, stats = run_stream()

    assert calls == []
    assert (stats["exact_cache_hit"], stats["semantic_cache_hit"], stats["cache_hit"]) == (False, True, True)
    assert events[-1] == ("complete", semantic_hit)


def test_stream_promotes_a_semantic_cache_hit_to_the_exact_match_cache(monkeypatch, fake_cache):
    # Repetir el mismo texto literal tras un hit semántico debe entrar ya por exact-match,
    # sin volver a llamar a la API de embeddings
    semantic_hit = EstimationResult.model_validate(GOOD)
    monkeypatch.setattr(llm_service, "semantic_cache_lookup", lambda request, prompt_version, openai_client: semantic_hit)

    run_stream()  # primera vez: solo hay hit semántico

    key = build_cache_key(*render_estimation_prompt(REQUEST))
    assert EstimationResult.model_validate_json(fake_cache[key]) == semantic_hit


def test_stream_falls_back_to_the_llm_when_the_semantic_cache_misses(monkeypatch, fake_cache):
    calls = use_fake_stream(monkeypatch, [GOOD])
    monkeypatch.setattr(llm_service, "semantic_cache_lookup", lambda request, prompt_version, openai_client: None)

    events, stats = run_stream()

    assert len(calls) == 1  # no hubo hit (miss real o log-only): sigue el flujo normal
    assert stats["semantic_cache_hit"] is False
    assert events[-1][1] == EstimationResult.model_validate(GOOD)


def test_stream_stores_in_the_semantic_cache_after_a_clean_result(monkeypatch, fake_cache):
    use_fake_stream(monkeypatch, [GOOD])
    stored = []
    monkeypatch.setattr(llm_service, "semantic_cache_lookup", lambda request, prompt_version, openai_client: None)
    monkeypatch.setattr(
        llm_service,
        "semantic_cache_store",
        lambda request, result, prompt_version, openai_client: stored.append(result),
    )

    events, _ = run_stream()

    assert stored == [events[-1][1]]


def test_stream_does_not_store_in_the_semantic_cache_when_totals_mismatch(monkeypatch, fake_cache):
    # Misma condición de escritura que el exact-match: un resultado con avisos no se cachea
    use_fake_stream(monkeypatch, [TOTALS_MISMATCH])
    stored = []
    monkeypatch.setattr(llm_service, "semantic_cache_lookup", lambda request, prompt_version, openai_client: None)
    monkeypatch.setattr(
        llm_service,
        "semantic_cache_store",
        lambda request, result, prompt_version, openai_client: stored.append(result),
    )

    run_stream()

    assert stored == []


def test_stream_falls_back_to_the_other_provider_when_the_primary_fails(monkeypatch, fake_cache):
    auth_error = litellm.AuthenticationError(message="bad key", llm_provider="openai", model="fake")
    models_called: list[str] = []

    def fake_completion(**kwargs):
        models_called.append(kwargs["model"])
        if not kwargs["model"].startswith("anthropic/"):
            raise auth_error
        return iter(tool_call_stream(json.dumps(GOOD)))

    monkeypatch.setattr(llm_service.litellm, "completion", fake_completion)

    events, stats = run_stream()

    assert events[-1][0] == "complete"
    assert (stats["provider_used"], stats["fallback_used"]) == ("anthropic", True)
    assert len(models_called) == 2  # el fallo de autenticación no se reintenta: salta directo al secundario


def test_stream_raises_when_every_provider_fails(monkeypatch, fake_cache):
    auth_error = litellm.AuthenticationError(message="bad key", llm_provider="openai", model="fake")

    def failing_completion(**kwargs):
        raise auth_error

    monkeypatch.setattr(llm_service.litellm, "completion", failing_completion)

    with pytest.raises(EstimationFailedError):
        run_stream()

    assert fake_cache == {}
