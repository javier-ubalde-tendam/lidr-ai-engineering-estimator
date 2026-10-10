import time
from types import SimpleNamespace

import pytest
from instructor.core.exceptions import InstructorRetryException

from app.schemas.critic import CriticFeedback
from app.services.llm_wrapper import (
    MODEL_COSTS,
    LLMCallMeta,
    build_call_meta,
    call_structured,
    estimate_cost_usd,
    normalise_model_name,
    provider_from_model,
    sum_call_metas,
)
from tests.sessions_helpers import ACCEPT_CRITIC_PAYLOAD, use_fake_llm


def test_estimate_cost_usd_uses_the_price_table_per_million_tokens():
    expected = (1_000_000 * MODEL_COSTS["gpt-4o-mini"]["input"] + 500_000 * MODEL_COSTS["gpt-4o-mini"]["output"]) / 1e6

    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 500_000) == pytest.approx(expected)


def test_estimate_cost_usd_accepts_the_litellm_provider_prefix():
    assert estimate_cost_usd("anthropic/claude-haiku-4-5", 1000, 1000) == estimate_cost_usd("claude-haiku-4-5", 1000, 1000)
    assert estimate_cost_usd("claude-haiku-4-5", 1000, 1000) > 0


def test_estimate_cost_usd_is_zero_for_unknown_models():
    assert estimate_cost_usd("some-future-model", 1000, 1000) == 0.0


def test_normalise_model_name_strips_provider_prefixes():
    assert normalise_model_name("anthropic/claude-haiku-4-5") == "claude-haiku-4-5"
    assert normalise_model_name("gpt-4o-mini") == "gpt-4o-mini"


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("gpt-4o-mini", "openai"),
        ("o3-mini", "openai"),
        ("claude-haiku-4-5", "anthropic"),
        ("anthropic/claude-sonnet-4-5", "anthropic"),
        ("mistral-large", "unknown"),
    ],
)
def test_provider_from_model(model, provider):
    assert provider_from_model(model) == provider


def test_build_call_meta_reads_usage_and_computes_cost_and_latency():
    completion = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2000, completion_tokens=500))
    started_at = time.perf_counter() - 0.25

    meta = build_call_meta(completion, "anthropic/claude-haiku-4-5", started_at)

    assert (meta.tokens_in, meta.tokens_out) == (2000, 500)
    assert meta.model == "claude-haiku-4-5"
    assert meta.provider == "anthropic"
    assert meta.cost_usd == estimate_cost_usd("claude-haiku-4-5", 2000, 500)
    assert meta.latency_ms >= 250


@pytest.mark.parametrize("completion", [SimpleNamespace(usage=None), SimpleNamespace(), object()])
def test_build_call_meta_tolerates_a_missing_usage(completion):
    meta = build_call_meta(completion, "gpt-4o-mini", time.perf_counter())

    assert (meta.tokens_in, meta.tokens_out, meta.cost_usd) == (0, 0, 0.0)


def _meta(**overrides) -> LLMCallMeta:
    fields = {"latency_ms": 100, "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.001, "model": "gpt-4o-mini", "provider": "openai"}
    return LLMCallMeta(**{**fields, **overrides})


def test_sum_call_metas_adds_the_numeric_fields():
    total = sum_call_metas([_meta(), _meta(latency_ms=50, tokens_in=1, tokens_out=2, cost_usd=0.002)])

    assert (total.latency_ms, total.tokens_in, total.tokens_out) == (150, 11, 7)
    assert total.cost_usd == pytest.approx(0.003)
    assert (total.model, total.provider) == ("gpt-4o-mini", "openai")


def test_sum_call_metas_lists_every_model_and_provider_once():
    total = sum_call_metas([_meta(), _meta(model="claude-haiku-4-5", provider="anthropic"), _meta()])

    assert total.model == "gpt-4o-mini+claude-haiku-4-5"
    assert total.provider == "openai+anthropic"


def test_sum_call_metas_of_nothing_is_zero():
    assert sum_call_metas([]).tokens_in == 0


def test_call_structured_returns_the_meta_and_logs_the_six_fields(monkeypatch):
    from structlog.testing import capture_logs

    use_fake_llm(monkeypatch)

    with capture_logs() as logs:
        feedback, meta = call_structured(
            [{"role": "user", "content": "review"}], CriticFeedback, "gpt-4o-mini", max_retries=1, purpose="critic"
        )

    assert feedback.verdict == ACCEPT_CRITIC_PAYLOAD["verdict"]
    assert (meta.tokens_in, meta.tokens_out) == (10, 5)
    completed = [log for log in logs if log["event"] == "llm_call_completed"]
    assert len(completed) == 1
    assert {"latency_ms", "tokens_in", "tokens_out", "cost_usd", "model", "provider", "purpose"} <= completed[0].keys()
    assert completed[0]["purpose"] == "critic"


def test_call_structured_logs_failures_with_the_six_fields_and_reraises(monkeypatch):
    from structlog.testing import capture_logs

    fake = use_fake_llm(monkeypatch)
    fake.payloads["CriticFeedback"] = [RuntimeError("provider down")]

    with capture_logs() as logs, pytest.raises(InstructorRetryException):
        call_structured(
            [{"role": "user", "content": "review"}], CriticFeedback, "gpt-4o-mini", max_retries=1, purpose="critic"
        )

    failed = [log for log in logs if log["event"] == "llm_call_failed"]
    assert len(failed) == 1
    assert {"latency_ms", "tokens_in", "tokens_out", "cost_usd", "model", "provider", "purpose"} <= failed[0].keys()
