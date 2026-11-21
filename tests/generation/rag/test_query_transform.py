"""Query expansion / decomposition: technique heuristic, sub-query cap and degrade-to-direct."""

from typing import Any

import pytest

from app.generation.rag.retrieval.query_transform import (
    DECOMPOSE,
    DIRECT,
    EXPAND,
    QueryTransformation,
    SubQuery,
    choose_technique,
    transform_query,
)

pytestmark = pytest.mark.anyio

MULTI_TOPIC = "oauth and a ledger, plus stripe checkout, and also recommendations for the catalog"


class FakeLLM:
    def __init__(self, transformation: QueryTransformation | None = None, fail: bool = False):
        self._transformation = transformation
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[QueryTransformation | None, None]:
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError("api down")
        return self._transformation, None


def sub_queries(count: int) -> QueryTransformation:
    return QueryTransformation(
        subqueries=[SubQuery(topic=f"t{i}", query=f"q{i}") for i in range(count)]
    )


def test_choose_technique_by_shape() -> None:
    assert choose_technique("stripe checkout") == DIRECT  # too short
    assert choose_technique("secure mobile banking authentication backend service") == EXPAND
    assert choose_technique(MULTI_TOPIC) == DECOMPOSE


def test_the_schema_caps_sub_queries_at_four() -> None:
    with pytest.raises(ValueError):
        sub_queries(5)


async def test_disabled_returns_a_single_direct_query_without_calling_the_llm() -> None:
    llm = FakeLLM()

    plan = await transform_query(MULTI_TOPIC, llm=llm, model="m", enabled=False)  # type: ignore[arg-type]

    assert plan.technique == DIRECT
    assert [sq.query for sq in plan.subqueries] == [MULTI_TOPIC]
    assert llm.calls == []


async def test_short_query_passes_through_without_llm() -> None:
    llm = FakeLLM()

    plan = await transform_query("stripe checkout", llm=llm, model="m")  # type: ignore[arg-type]

    assert plan.technique == DIRECT
    assert llm.calls == []


async def test_decomposition_is_capped_by_max_subqueries_and_uses_the_model() -> None:
    llm = FakeLLM(sub_queries(4))

    plan = await transform_query(MULTI_TOPIC, llm=llm, model="qt-model", max_subqueries=2)  # type: ignore[arg-type]

    assert plan.technique == DECOMPOSE
    assert [sq.query for sq in plan.subqueries] == ["q0", "q1"]
    assert len(llm.calls) == 1 and llm.calls[0]["model"] == "qt-model"


async def test_expansion_prompt_differs_from_decomposition_prompt() -> None:
    expand_llm, decompose_llm = FakeLLM(sub_queries(2)), FakeLLM(sub_queries(2))

    await transform_query(
        "secure mobile banking authentication backend service", llm=expand_llm, model="m"  # type: ignore[arg-type]
    )
    await transform_query(MULTI_TOPIC, llm=decompose_llm, model="m")  # type: ignore[arg-type]

    assert expand_llm.calls[0]["system_prompt"] != decompose_llm.calls[0]["system_prompt"]


async def test_generation_failure_degrades_to_direct() -> None:
    query = "secure mobile banking authentication backend service"

    plan = await transform_query(query, llm=FakeLLM(fail=True), model="m")  # type: ignore[arg-type]

    assert plan.technique == DIRECT
    assert [sq.query for sq in plan.subqueries] == [query]
