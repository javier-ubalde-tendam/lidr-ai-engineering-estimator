"""Two-stage hours synthesis: range on contradiction, deterministic floor, optional LLM wording."""

import statistics
from typing import Any

import pytest

from app.foundation.llm.wrapper import LLMCallMeta
from app.generation.rag.quality.synthesis import is_contradiction, synthesize_range
from app.generation.rag.schemas import TaskNeighbor

pytestmark = pytest.mark.anyio

META = LLMCallMeta(latency_ms=1, tokens_in=1, tokens_out=1, cost_usd=0.0, model="m", provider="p")


class FakeReasonLLM:
    """Stands in for ``LLMWrapper``: answers with the pydantic model built from ``reason``."""

    def __init__(self, reason: str | Exception) -> None:
        self._reason = reason
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[Any, LLMCallMeta]:
        self.calls.append(kwargs)
        if isinstance(self._reason, Exception):
            raise self._reason
        return kwargs["response_model"](reason=self._reason), META


def make_neighbors(hours: list[int]) -> list[TaskNeighbor]:
    return [
        TaskNeighbor(source_id=i, budget_id=f"TASK-{i}", estimated_hours=h, distance=0.2 + i * 0.02)
        for i, h in enumerate(hours)
    ]


def dispersion_of(hours: list[int]) -> float:
    """Coefficient of variation, the dispersion the (future) hours pipeline feeds in."""
    return statistics.pstdev(hours) / statistics.fmean(hours)


def test_is_contradiction_compares_against_the_threshold() -> None:
    assert is_contradiction(0.4, 0.35) is True
    assert is_contradiction(0.35, 0.35) is False
    assert is_contradiction(None, 0.35) is False


async def test_range_is_emitted_on_contradiction_without_calling_an_llm() -> None:
    hours = [90, 40, 40]

    hour_range = await synthesize_range(make_neighbors(hours), dispersion_of(hours), threshold=0.35)

    assert hour_range is not None
    assert (hour_range.low, hour_range.high) == (40, 90)
    assert "40h" in hour_range.reason and "90h" in hour_range.reason


async def test_no_range_when_the_sources_agree_or_there_are_no_neighbors() -> None:
    hours = [50, 52, 48]

    assert await synthesize_range(make_neighbors(hours), dispersion_of(hours), threshold=0.35) is None
    assert await synthesize_range([], 0.9, threshold=0.35) is None
    assert await synthesize_range(make_neighbors([40, 90]), None, threshold=0.35) is None


async def test_the_llm_only_rephrases_the_reason_never_the_numbers() -> None:
    llm = FakeReasonLLM("one analog included the payment gateway integration")
    hours = [40, 90]

    hour_range = await synthesize_range(
        make_neighbors(hours), dispersion_of(hours), threshold=0.35, llm=llm, model="cheap"  # type: ignore[arg-type]
    )

    assert hour_range is not None
    assert (hour_range.low, hour_range.high) == (40, 90)
    assert hour_range.reason == "one analog included the payment gateway integration"
    assert llm.calls[0]["model"] == "cheap"


async def test_llm_failure_or_missing_model_falls_back_to_the_deterministic_reason() -> None:
    hours = [40, 90]
    neighbors, dispersion = make_neighbors(hours), dispersion_of(hours)
    failing = FakeReasonLLM(RuntimeError("provider down"))
    unused = FakeReasonLLM("never used")

    after_failure = await synthesize_range(
        neighbors, dispersion, threshold=0.35, llm=failing, model="cheap"  # type: ignore[arg-type]
    )
    without_model = await synthesize_range(
        neighbors, dispersion, threshold=0.35, llm=unused, model=None  # type: ignore[arg-type]
    )

    assert after_failure is not None and "disagree" in after_failure.reason
    assert without_model == after_failure
    assert unused.calls == []


async def test_the_llm_is_not_called_when_there_is_no_contradiction() -> None:
    llm = FakeReasonLLM("never used")
    hours = [50, 52]

    result = await synthesize_range(
        make_neighbors(hours), dispersion_of(hours), threshold=0.35, llm=llm, model="cheap"  # type: ignore[arg-type]
    )

    assert result is None and llm.calls == []
