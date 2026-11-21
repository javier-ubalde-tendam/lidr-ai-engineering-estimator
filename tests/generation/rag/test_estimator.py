"""TranscriptEstimator: the S9 pipeline with every collaborator faked."""

import pytest

from app.config import get_settings
from app.generation.rag.errors import GenerationError, MalformedEstimateError, RetrievalError
from app.generation.rag.schemas import Estimate, EstimationQuery
from tests.estimator_helpers import (
    TRANSCRIPT,
    FakeEmbedder,
    ScriptedLLM,
    good_estimate,
    make_estimator,
    patch_retrieve,
    result_with,
)
from tests.runtime_helpers import FakeRuntimeModelConfig, FakeRuntimeRetrievalConfig

pytestmark = pytest.mark.anyio


async def test_happy_path_runs_every_stage_and_returns_the_grounded_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = ScriptedLLM(good_estimate("1"))
    embedder = FakeEmbedder()
    spy = patch_retrieve(monkeypatch, result_with(1, 2))

    estimate = await make_estimator(llm, embedder=embedder).estimate_from_transcript(TRANSCRIPT)

    assert estimate.confidence == "high" and estimate.total_engineer_days == 10
    # The embedded text is the composed brief, not the raw transcript
    assert embedder.texts == ["payments marketplace for finance"]
    call = spy.calls[0]
    assert call["query_text"] == "payments marketplace for finance"
    assert call["sectors"] == ["finance"]
    # Generation receives the <source> block with the retrieved ids
    user_message = llm.generation_calls[0]["user_message"]
    assert '<source id="1"' in user_message and '<source id="2"' in user_message


async def test_models_and_reasoning_come_from_the_runtime_config_and_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = ScriptedLLM(good_estimate("1"))
    models = FakeRuntimeModelConfig({"REFORMULATION_MODEL": "gpt-4o-mini", "GENERATION_MODEL": "gpt-4o"})
    patch_retrieve(monkeypatch, result_with(1))

    # The gate (on by default) adds a third LLM call: this test is about the first two
    retrieval_config = FakeRuntimeRetrievalConfig(hallucination_gate=False)

    await make_estimator(
        llm, models=models, retrieval_config=retrieval_config
    ).estimate_from_transcript(TRANSCRIPT)

    reformulation, generation = llm.calls
    settings = get_settings()
    assert reformulation["model"] == "gpt-4o-mini"
    assert generation["model"] == "gpt-4o"
    # Reasoning tokens count against max_tokens: the generous ceiling must reach the wrapper
    assert generation["max_tokens"] == settings.GENERATION_MAX_TOKENS
    assert generation["reasoning_effort"] == settings.GENERATION_REASONING_EFFORT


async def test_retrieval_mode_and_rerank_follow_the_runtime_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = patch_retrieve(monkeypatch, result_with(1))
    retrieval_config = FakeRuntimeRetrievalConfig(search_mode="hybrid", rerank=True)

    await make_estimator(
        ScriptedLLM(good_estimate("1")), retrieval_config=retrieval_config
    ).estimate_from_transcript(TRANSCRIPT)

    assert (spy.calls[0]["search_mode"], spy.calls[0]["rerank"]) == ("hybrid", True)


async def test_unknown_sector_does_not_filter_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    brief = EstimationQuery(function="crm", sector="Space Tourism")
    llm = ScriptedLLM(good_estimate("1"), brief=brief)
    spy = patch_retrieve(monkeypatch, result_with(1))

    await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)

    assert spy.calls[0]["sectors"] is None


async def test_soft_fail_returns_insufficient_without_generating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = ScriptedLLM()
    patch_retrieve(monkeypatch, result_with())

    estimate = await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)

    assert estimate.confidence == "insufficient"
    assert estimate.total_engineer_days is None and estimate.modules == []
    assert estimate.insufficient_context_explanation
    assert llm.generation_calls == []


async def test_invented_chunk_ids_trigger_one_corrective_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = ScriptedLLM(good_estimate("999"), good_estimate("1"))
    patch_retrieve(monkeypatch, result_with(1))

    estimate = await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)

    assert estimate.confidence == "high"
    assert len(llm.generation_calls) == 2
    retry_message = llm.generation_calls[1]["user_message"]
    assert "<correction>" in retry_message and "999" in retry_message


async def test_citations_still_invented_after_the_retry_downgrade_to_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = ScriptedLLM(good_estimate("999"), good_estimate("888"))
    patch_retrieve(monkeypatch, result_with(1))

    estimate = await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)

    assert estimate.confidence == "low"
    assert len(llm.generation_calls) == 2  # exactly one retry, never a loop


async def test_incoherent_insufficient_estimate_is_repaired_once_then_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoherent = Estimate(
        confidence="insufficient", reasoning="r", total_engineer_days=5,
        insufficient_context_explanation="no data",
    )
    patch_retrieve(monkeypatch, result_with(1))

    repaired = await make_estimator(ScriptedLLM(incoherent, good_estimate("1"))).estimate_from_transcript(
        TRANSCRIPT
    )
    assert repaired.confidence == "high"

    with pytest.raises(MalformedEstimateError):
        await make_estimator(ScriptedLLM(incoherent, incoherent)).estimate_from_transcript(TRANSCRIPT)


async def test_llm_failure_during_generation_is_a_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = ScriptedLLM()  # no scripted estimate: the generation call raises IndexError
    patch_retrieve(monkeypatch, result_with(1))

    with pytest.raises(GenerationError):
        await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)


async def test_retrieval_errors_propagate_for_the_router_to_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_retrieve(monkeypatch, RetrievalError("db down"))

    with pytest.raises(RetrievalError):
        await make_estimator(ScriptedLLM()).estimate_from_transcript(TRANSCRIPT)


async def test_embedding_failure_is_a_retrieval_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEmbedder(FakeEmbedder):
        def embed_one(self, text: str) -> list[float]:
            raise ConnectionError("provider down")

    patch_retrieve(monkeypatch, result_with(1))

    with pytest.raises(RetrievalError):
        await make_estimator(ScriptedLLM(), embedder=BrokenEmbedder()).estimate_from_transcript(
            TRANSCRIPT
        )


async def test_token_budget_drops_chunks_that_do_not_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_CONTEXT_TOKENS", "300")  # CharEncoder: 1 char = 1 token
    llm = ScriptedLLM(good_estimate("1"))
    patch_retrieve(monkeypatch, result_with(1, 2, 3))

    await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)

    user_message = llm.generation_calls[0]["user_message"]
    assert '<source id="1"' in user_message and '<source id="3"' not in user_message
