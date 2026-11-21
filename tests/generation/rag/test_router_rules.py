"""Cascade router: explicit -> rules -> classifier -> fallback. The LLM is a fake (no API)."""

from typing import Any

import pytest

from app.generation.rag.retrieval.collections import ALL_COLLECTIONS, Collection
from app.generation.rag.retrieval.router import RouteClassification, route

pytestmark = pytest.mark.anyio


class FakeLLM:
    """Stands in for ``LLMWrapper``: returns a canned classification and counts the calls."""

    def __init__(self, classification: RouteClassification | None = None, fail: bool = False):
        self._classification = classification
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[RouteClassification | None, None]:
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError("api down")
        return self._classification, None


def classification(*targets: Collection, reason: str = "x") -> RouteClassification:
    return RouteClassification(targets=list(targets), reason=reason)


async def test_explicit_collections_short_circuit_the_cascade() -> None:
    llm = FakeLLM(classification(Collection.BUDGET))

    decision = await route("anything", llm=llm, model="m", explicit=[Collection.TRANSCRIPT])  # type: ignore[arg-type]

    assert (decision.level, decision.targets) == ("explicit", [Collection.TRANSCRIPT])
    assert llm.calls == []  # nothing was classified


async def test_rules_win_before_the_classifier() -> None:
    llm = FakeLLM(classification(Collection.BUDGET))

    decision = await route("what did the client say in the meeting", llm=llm, model="m")  # type: ignore[arg-type]

    assert (decision.level, decision.targets) == ("rules", [Collection.TRANSCRIPT])
    assert llm.calls == []


async def test_classifier_runs_when_no_rule_fires_and_uses_the_given_model() -> None:
    llm = FakeLLM(classification(Collection.BUDGET, Collection.TRANSCRIPT, reason="spans both"))

    decision = await route("a query with no routing vocabulary at all", llm=llm, model="router-m")  # type: ignore[arg-type]

    assert decision.level == "classifier"
    assert decision.targets == [Collection.BUDGET, Collection.TRANSCRIPT]
    assert decision.reason == "spans both"
    assert len(llm.calls) == 1 and llm.calls[0]["model"] == "router-m"


async def test_fallback_searches_everything_when_rules_and_classifier_are_off() -> None:
    llm = FakeLLM()

    decision = await route(
        "xyzzy", llm=llm, model="m", rules_enabled=False, classifier_enabled=False  # type: ignore[arg-type]
    )

    assert decision.level == "fallback"
    assert decision.targets == list(ALL_COLLECTIONS)
    assert llm.calls == []


async def test_classifier_failure_degrades_to_fallback() -> None:
    decision = await route(
        "a query with no routing vocabulary at all", llm=FakeLLM(fail=True), model="m"  # type: ignore[arg-type]
    )

    assert decision.level == "fallback"


async def test_targets_are_deduplicated_and_capped() -> None:
    llm = FakeLLM(classification(Collection.BUDGET, Collection.BUDGET, Collection.TRANSCRIPT))

    decision = await route(
        "no vocabulary", llm=llm, model="m", rules_enabled=False, max_targets=1  # type: ignore[arg-type]
    )

    assert decision.targets == [Collection.BUDGET]


async def test_explicit_targets_are_capped_too() -> None:
    decision = await route(
        "q",
        llm=FakeLLM(),  # type: ignore[arg-type]
        model="m",
        explicit=[Collection.BUDGET, Collection.TRANSCRIPT, Collection.TECHNICAL_DOC],
        max_targets=2,
    )

    assert decision.targets == [Collection.BUDGET, Collection.TRANSCRIPT]
