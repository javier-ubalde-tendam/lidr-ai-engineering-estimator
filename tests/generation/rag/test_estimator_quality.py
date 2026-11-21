"""Optional Session 11 stages inside ``TranscriptEstimator``: augmentation and hallucination gate.

Covers the defaults (both on, as the Settings ship), the off switch, the runtime-override
precedence (Redis override -> Settings) and what each stage does to the estimate / the context.
"""

import re

import fakeredis
import pytest

from app.config import Settings
from app.foundation.llm.runtime_config import RuntimeRetrievalConfig
from app.generation.rag.quality.hallucination import JudgePanel
from app.generation.rag.schemas import (
    Estimate,
    LineVerdict,
    RetrievalResult,
    RetrievedChunk,
    SourceReference,
    TaskItem,
    WorkModule,
)
from tests.estimator_helpers import (
    TRANSCRIPT,
    ScriptedLLM,
    good_estimate,
    make_estimator,
    patch_retrieve,
)
from tests.runtime_helpers import FakeRuntimeModelConfig, FakeRuntimeRetrievalConfig

pytestmark = pytest.mark.anyio

FILLER = "Intro prose without any figure"


def chunk_with_filler(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=f"{FILLER}\nPAY-{chunk_id} :: Stripe Connect - 120 h\nclosing filler",
        sector="finance",
        project_year=2023,
        chunk_type="budget_component",
        distance=0.2,
        budget_id="BUD-2023-001",
    )


def retrieval_of(*chunk_ids: int) -> RetrievalResult:
    return RetrievalResult(
        chunks=[chunk_with_filler(i) for i in chunk_ids],
        low_confidence=False,
        candidates_evaluated=60,
    )


def estimate_of(days: int, *, confidence: str = "high") -> Estimate:
    """One grounded line citing chunk 1 (which records 120 h = 15 engineer-days)."""
    return Estimate(
        total_engineer_days=days,
        duration_weeks=4,
        modules=[
            WorkModule(
                name="Payments",
                tasks=[
                    TaskItem(
                        name="Checkout",
                        engineer_days=days,
                        grounded=True,
                        sources=[
                            SourceReference(chunk_id="1", document_id="BUD-2023-001", evidence="120 h")
                        ],
                    )
                ],
            )
        ],
        confidence=confidence,  # type: ignore[arg-type]
        reasoning="derived from the sources",
    )


def source_ids(user_message: str) -> list[int]:
    return [int(found) for found in re.findall(r'<source id="(\d+)"', user_message)]


def real_runtime(settings: Settings, **overrides: str) -> RuntimeRetrievalConfig:
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    for key, value in overrides.items():
        redis_client.hset("estimator:runtime_retrieval", key, value)
    return RuntimeRetrievalConfig(redis_client, settings)


def make_settings(**overrides: object) -> Settings:
    return Settings(OPENAI_API_KEY="sk-test", _env_file=None, **overrides)  # type: ignore[arg-type]


async def test_both_stages_are_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_retrieve(monkeypatch, retrieval_of(1, 2, 3))
    llm = ScriptedLLM(estimate_of(10))
    # Real runtime object with nothing overridden: the Settings defaults decide
    settings = make_settings()
    estimator = make_estimator(llm, retrieval_config=real_runtime(settings), settings=settings)  # type: ignore[arg-type]

    estimate = await estimator.estimate_from_transcript(TRANSCRIPT)

    user_message = llm.generation_calls[0]["user_message"]
    assert FILLER not in user_message  # compressed
    assert source_ids(user_message) == [1, 3, 2]  # edge-loaded: best first, 2nd best last
    assert len(llm.judge_calls) == 1
    assert estimate.confidence == "high"


async def test_both_stages_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_retrieve(monkeypatch, retrieval_of(1, 2, 3))
    llm = ScriptedLLM(estimate_of(90))  # would be degraded if the gate ran
    retrieval_config = FakeRuntimeRetrievalConfig(augmentation=False, hallucination_gate=False)

    estimate = await make_estimator(llm, retrieval_config=retrieval_config).estimate_from_transcript(
        TRANSCRIPT
    )

    user_message = llm.generation_calls[0]["user_message"]
    assert user_message.count(FILLER) == 3 and source_ids(user_message) == [1, 2, 3]
    assert llm.judge_calls == []
    assert estimate.confidence == "high"


async def test_augmentation_passes_follow_their_own_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_retrieve(monkeypatch, retrieval_of(1, 2, 3))
    llm = ScriptedLLM(estimate_of(10))
    settings = make_settings(AUGMENTATION_COMPRESS=False, AUGMENTATION_REORDER=True)
    estimator = make_estimator(
        llm,
        retrieval_config=FakeRuntimeRetrievalConfig(augmentation=True, hallucination_gate=False),
        settings=settings,
    )

    await estimator.estimate_from_transcript(TRANSCRIPT)

    user_message = llm.generation_calls[0]["user_message"]
    assert user_message.count(FILLER) == 3  # not compressed
    assert source_ids(user_message) == [1, 3, 2]  # but reordered


@pytest.mark.parametrize(
    ("settings_flag", "override", "expected_on"),
    [
        (True, None, True),
        (False, None, False),
        (True, "false", False),  # runtime override beats the Settings default...
        (False, "true", True),  # ...in both directions
    ],
)
async def test_runtime_override_takes_precedence_over_settings(
    monkeypatch: pytest.MonkeyPatch, settings_flag: bool, override: str | None, expected_on: bool
) -> None:
    patch_retrieve(monkeypatch, retrieval_of(1, 2))
    llm = ScriptedLLM(estimate_of(10), estimate_of(10))
    settings = make_settings(AUGMENTATION_ENABLED=settings_flag, HALLUCINATION_GATE_ENABLED=settings_flag)
    overrides = (
        {"AUGMENTATION_ENABLED": override, "HALLUCINATION_GATE_ENABLED": override}
        if override
        else {}
    )
    estimator = make_estimator(
        llm, retrieval_config=real_runtime(settings, **overrides), settings=settings  # type: ignore[arg-type]
    )

    await estimator.estimate_from_transcript(TRANSCRIPT)

    compressed = FILLER not in llm.generation_calls[0]["user_message"]
    assert compressed is expected_on
    assert bool(llm.judge_calls) is expected_on


async def test_an_inflated_line_degrades_a_confident_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_retrieve(monkeypatch, retrieval_of(1, 2))
    llm = ScriptedLLM(estimate_of(90))  # claims 90d; the cited chunk records 120 h = 15d

    estimate = await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)

    assert estimate.confidence == "low"
    assert estimate.total_engineer_days == 90  # the numbers are kept, only the confidence drops


async def test_a_judge_rejection_degrades_the_estimate_even_when_the_numbers_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_retrieve(monkeypatch, retrieval_of(1, 2))
    verdict = LineVerdict(module="Payments", component="Checkout", entailed=False, reason="no")
    llm = ScriptedLLM(estimate_of(10), judge=JudgePanel(verdicts=[verdict]))

    estimate = await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)

    assert estimate.confidence == "low"


async def test_the_gate_never_raises_the_confidence_and_keeps_a_grounded_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_retrieve(monkeypatch, retrieval_of(1, 2))
    llm = ScriptedLLM(estimate_of(90, confidence="low"), estimate_of(10, confidence="medium"))
    estimator = make_estimator(llm)

    already_low = await estimator.estimate_from_transcript(TRANSCRIPT)
    grounded = await estimator.estimate_from_transcript(TRANSCRIPT)

    assert already_low.confidence == "low"
    assert grounded.confidence == "medium"


async def test_a_failing_judge_does_not_break_the_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_retrieve(monkeypatch, retrieval_of(1, 2))
    llm = ScriptedLLM(estimate_of(10), judge=RuntimeError("provider down"))

    estimate = await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)

    assert estimate.confidence == "high" and len(llm.judge_calls) == 1


async def test_the_judge_model_comes_from_the_runtime_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_retrieve(monkeypatch, retrieval_of(1, 2))
    llm = ScriptedLLM(estimate_of(10))
    models = FakeRuntimeModelConfig({"HALLUCINATION_JUDGE_MODEL": "gpt-4o"})

    await make_estimator(llm, models=models).estimate_from_transcript(TRANSCRIPT)

    judge_call = llm.judge_calls[0]
    assert judge_call["model"] == "gpt-4o"
    assert judge_call["max_tokens"] == make_settings().GENERATION_MAX_TOKENS


async def test_the_gate_runs_on_the_corrected_estimate_after_a_citation_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_retrieve(monkeypatch, retrieval_of(1, 2))
    llm = ScriptedLLM(good_estimate("999"), good_estimate("1"))  # first cites a missing chunk

    estimate = await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)

    assert estimate.confidence == "high"
    assert len(llm.generation_calls) == 2 and len(llm.judge_calls) == 1


async def test_neither_stage_runs_on_a_retrieval_soft_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_retrieve(
        monkeypatch,
        RetrievalResult(chunks=[], low_confidence=True, candidates_evaluated=0),
    )
    llm = ScriptedLLM()

    estimate = await make_estimator(llm).estimate_from_transcript(TRANSCRIPT)

    assert estimate.confidence == "insufficient"
    assert llm.generation_calls == [] and llm.judge_calls == []
