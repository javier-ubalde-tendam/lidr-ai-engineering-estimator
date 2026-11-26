"""End-to-end behaviour of the estimation graph with fakes (no network, no Postgres, no keys)."""

import copy
import operator
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.domain.graph import EstimationState, build_graph
from app.domain.graph.nodes import search_budgets
from tests.domain.graph.helpers import (
    TRANSCRIPT,
    ScriptedLLM,
    ScriptedSearch,
    estimate_of,
    make_deps,
)

pytestmark = pytest.mark.anyio

COMPONENTS = [("Payments", "backend"), ("Reporting", "data")]
HOURS = {"Payments": [100.0, 120.0], "Reporting": [40.0]}
# Medians: Payments 110, Reporting 40 -> inside the range of their references
CLEAN = estimate_of(("Payments", 110.0, ["B-Pay-0", "B-Pay-1"]), ("Reporting", 40.0, ["B-Rep-0"]))
ORDER = [
    "extract_requirements",
    "classify_components",
    "search_budgets",
    "generate_estimate",
    "validate_and_consolidate",
]


async def run_graph(
    estimate: Any = CLEAN, search: ScriptedSearch | None = None, thread: str = "t-1"
) -> tuple[Any, list[str], Any]:
    graph = build_graph(
        MemorySaver(), make_deps(ScriptedLLM(estimate, COMPONENTS), search or ScriptedSearch(HOURS))
    )
    config: Any = {"configurable": {"thread_id": thread}}
    visited: list[str] = []
    async for update in graph.astream({"transcript": TRANSCRIPT}, config, stream_mode="updates"):
        visited += list(update)
    return graph, visited, config


async def test_accumulator_fields_concatenate_partial_updates() -> None:
    class State(TypedDict, total=False):
        errors: Annotated[list[str], operator.add]
        plain: str

    async def first(_: State) -> dict[str, Any]:
        return {"errors": ["a"], "plain": "one"}

    async def second(_: State) -> dict[str, Any]:
        return {"errors": ["b"], "plain": "two"}

    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    result = await builder.compile().ainvoke({})

    assert result["errors"] == ["a", "b"]  # accumulated, not overwritten
    assert result["plain"] == "two"  # last writer wins


async def test_a_node_does_not_mutate_its_input_state() -> None:
    state: EstimationState = {
        "components": [{"name": "Payments", "category": "backend"}],
        "budget_matches": [],
        "errors": [],
    }
    before = copy.deepcopy(state)
    deps = make_deps(ScriptedLLM(CLEAN, COMPONENTS), ScriptedSearch(HOURS))

    update = await search_budgets(state, deps)

    assert state == before
    assert len(update["budget_matches"]) == 2


async def test_clean_run_is_validated_without_review_and_keeps_node_order() -> None:
    search = ScriptedSearch(HOURS)
    graph, visited, config = await run_graph(search=search)

    assert visited == ORDER  # flag_for_review is never visited
    values = (await graph.aget_state(config)).values
    assert values["status"] == "validated"
    assert values["errors"] == []
    assert [(m["component"], m["amount"]) for m in values["budget_matches"]] == [
        ("Payments", 100.0),
        ("Payments", 120.0),
        ("Reporting", 40.0),
    ]
    # Sequential, one search per component
    assert search.queries == ["Payments (backend)", "Reporting (data)"]
    assert values["estimate"]["total_hours"] == 150.0


async def test_out_of_range_estimate_goes_through_flag_for_review() -> None:
    too_high = estimate_of(("Payments", 400.0, ["B-Pay-0"]), ("Reporting", 40.0, ["B-Rep-0"]))
    graph, visited, config = await run_graph(estimate=too_high)

    assert visited == [*ORDER, "flag_for_review"]
    values = (await graph.aget_state(config)).values
    assert values["status"] == "needs_review"
    assert any("out_of_range" in error for error in values["errors"])
    assert any("flagged for human review" in error for error in values["errors"])


async def test_retrieval_failure_on_one_component_is_soft() -> None:
    estimate = estimate_of(("Payments", 0.0, []), ("Reporting", 40.0, ["B-Rep-0"]))
    graph, visited, config = await run_graph(
        estimate=estimate, search=ScriptedSearch(HOURS, failing=("Payments",))
    )

    values = (await graph.aget_state(config)).values
    assert visited == ORDER  # the graph finished
    assert any("search_budgets failed for 'Payments'" in e for e in values["errors"])
    assert [m["component"] for m in values["budget_matches"]] == ["Reporting"]
    # An unbudgeted component is only a warning: it does not force a review
    assert values["status"] == "validated"


async def test_same_thread_id_leaves_a_recoverable_state() -> None:
    graph, _, config = await run_graph(thread="run-42")

    snapshot = await graph.aget_state(config)
    assert snapshot.values["status"] == "validated"
    assert snapshot.values["transcript"] == TRANSCRIPT

    other: Any = {"configurable": {"thread_id": "another"}}
    assert (await graph.aget_state(other)).values == {}


async def test_reasoning_kwargs_only_for_gpt5_models() -> None:
    llm = ScriptedLLM(CLEAN, COMPONENTS)
    graph = build_graph(None, make_deps(llm, ScriptedSearch(HOURS)))
    await graph.ainvoke({"transcript": TRANSCRIPT})

    by_purpose = {call["purpose"]: call for call in llm.calls}
    assert "reasoning_effort" not in by_purpose["graph_extract_requirements"]  # gpt-4o-mini
    assert by_purpose["graph_generate_estimate"]["reasoning_effort"] == "low"  # gpt-5-mini
    assert by_purpose["graph_generate_estimate"]["max_tokens"] > 0
