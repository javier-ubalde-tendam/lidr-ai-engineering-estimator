"""Query understanding with a fake LLM wrapper (no network)."""

from typing import Any

import pytest

from app.foundation.llm.wrapper import LLMCallMeta
from app.generation.rag.errors import ReformulationError
from app.generation.rag.query_reformulator import compose_search_text, reformulate_query
from app.generation.rag.schemas import EstimationQuery

pytestmark = pytest.mark.anyio

META = LLMCallMeta(latency_ms=1, tokens_in=1, tokens_out=1, cost_usd=0.0, model="m", provider="p")


class FakeLLM:
    """Replays one outcome per call; an Exception outcome is raised."""

    def __init__(self, *outcomes: Any) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[Any, LLMCallMeta]:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, META


async def reformulate(llm: FakeLLM) -> EstimationQuery:
    return await reformulate_query("a long transcript", llm=llm, model="gpt-5-mini", max_tokens=999)  # type: ignore[arg-type]


async def test_structured_extraction_is_the_primary_path() -> None:
    brief = EstimationQuery(function="B2B marketplace", technologies=["Stripe"], sector="finance")
    llm = FakeLLM(brief)

    assert await reformulate(llm) == brief
    call = llm.calls[0]
    assert call["response_model"] is EstimationQuery
    assert (call["model"], call["max_tokens"]) == ("gpt-5-mini", 999)
    assert call["user_message"] == "a long transcript"


async def test_falls_back_to_a_free_text_rewrite_when_extraction_fails() -> None:
    llm = FakeLLM(RuntimeError("bad json"), type("R", (), {"query": " marketplace with payments "})())

    query = await reformulate(llm)

    assert query == EstimationQuery(function="marketplace with payments")
    assert len(llm.calls) == 2


async def test_raises_when_both_paths_fail() -> None:
    with pytest.raises(ReformulationError):
        await reformulate(FakeLLM(RuntimeError("one"), RuntimeError("two")))


async def test_raises_when_the_fallback_returns_an_empty_query() -> None:
    with pytest.raises(ReformulationError):
        await reformulate(FakeLLM(RuntimeError("one"), type("R", (), {"query": "   "})()))


def test_compose_search_text_joins_the_grounded_fields() -> None:
    query = EstimationQuery(
        function="B2B payments marketplace",
        technologies=["Stripe Connect", "KYC"],
        sector="healthcare",
        country="Germany",
        regulations=["BaFin"],
    )

    assert compose_search_text(query) == (
        "B2B payments marketplace with Stripe Connect, KYC for healthcare in Germany BaFin-compliant"
    )


def test_compose_search_text_drops_empty_fields() -> None:
    assert compose_search_text(EstimationQuery(function="  Internal CRM ")) == "Internal CRM"
