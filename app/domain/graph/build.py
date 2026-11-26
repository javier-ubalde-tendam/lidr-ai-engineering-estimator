"""Wiring of the estimation graph (nodes + edges). No business logic lives here."""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.domain.graph import nodes
from app.domain.graph.state import EstimationState


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None, deps: nodes.GraphDependencies
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the graph; ``checkpointer=None`` gives a graph without persistence (tests)."""

    # Small closures bind ``deps`` (LangGraph inspects node signatures, so no functools.partial)
    async def _extract(state: EstimationState) -> dict[str, Any]:
        return await nodes.extract_requirements(state, deps)

    async def _classify(state: EstimationState) -> dict[str, Any]:
        return await nodes.classify_components(state, deps)

    async def _search(state: EstimationState) -> dict[str, Any]:
        return await nodes.search_budgets(state, deps)

    async def _generate(state: EstimationState) -> dict[str, Any]:
        return await nodes.generate_estimate(state, deps)

    async def _validate(state: EstimationState) -> dict[str, Any]:
        return await nodes.validate_and_consolidate(state, deps)

    async def _flag(state: EstimationState) -> dict[str, Any]:
        return await nodes.flag_for_review(state, deps)

    builder = StateGraph(EstimationState)
    builder.add_node("extract_requirements", _extract)
    builder.add_node("classify_components", _classify)
    builder.add_node("search_budgets", _search)
    builder.add_node("generate_estimate", _generate)
    builder.add_node("validate_and_consolidate", _validate)
    builder.add_node("flag_for_review", _flag)

    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "classify_components")
    builder.add_edge("classify_components", "search_budgets")
    builder.add_edge("search_budgets", "generate_estimate")
    builder.add_edge("generate_estimate", "validate_and_consolidate")
    # Level 3: the only branch. "validated" ends the run, anything else is flagged first
    builder.add_conditional_edges(
        "validate_and_consolidate",
        nodes.route_after_validation,
        {"validated": END, "needs_review": "flag_for_review"},
    )
    builder.add_edge("flag_for_review", END)
    return builder.compile(checkpointer=checkpointer)
