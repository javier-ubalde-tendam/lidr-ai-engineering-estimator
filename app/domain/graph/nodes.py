"""Nodes of the estimation graph.

Each node is a module-level ``async def node(state, deps) -> dict``: it reads the state, returns
a partial update and never mutates its input. Collaborators arrive through ``GraphDependencies``
(no service locator); ``build.py`` binds ``deps`` to each node. Every body runs inside a Logfire
span (a no-op until ``configure_logfire`` runs), which is what shows one span per node in a trace.
"""

import asyncio
import statistics
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import logfire
from pydantic import BaseModel

from app.config import Settings
from app.domain.graph.schemas import ComponentClassification, RequirementsExtraction
from app.domain.graph.state import BudgetMatch, Component, EstimationState
from app.foundation.llm.wrapper import LLMWrapper
from app.generation.agentic.agent_schemas import (
    MIN_QUERY_LENGTH,
    AgentEstimate,
    SearchBudgetsArgs,
)
from app.generation.agentic.agent_tools import SearchBackend, validate_estimate

T = TypeVar("T", bound=BaseModel)

# Reasoning models (gpt-5*) spend reasoning tokens from the max_tokens budget: give them room
REASONING_MODEL_PREFIX = "gpt-5"
REASONING_EFFORT = "low"
REASONING_MAX_TOKENS = 16000

STATUS_VALIDATED = "validated"
STATUS_NEEDS_REVIEW = "needs_review"

EXTRACT_SYSTEM_PROMPT = (
    "You extract project requirements from a meeting transcript. Return atomic, self-contained "
    "requirements, one capability or constraint each. Ignore small talk, greetings, scheduling "
    "and anything that is not a requirement. Never invent requirements that the transcript does "
    "not state."
)

CLASSIFY_SYSTEM_PROMPT = (
    "You group software project requirements into buildable components and give each one a "
    "broad category (backend, frontend, integration, data, infrastructure, security, ...). "
    "Every requirement must belong to some component; merge requirements that are built "
    "together, and do not create components that no requirement asks for."
)

GENERATE_SYSTEM_PROMPT = (
    "You consolidate a software project estimate IN ENGINEER-HOURS. For each component you get "
    "the historical tasks retrieved for it (budget_id and hours). Anchor the component's hours "
    "on the MEDIAN of its historical hours and cite the budget_ids you relied on in "
    "reference_budget_ids. A component with no historical reference gets 0 hours: never guess "
    "a number, and say so in `notes` (the total is then a lower bound). total_hours must equal "
    "the sum of the component hours."
)


@dataclass(frozen=True)
class GraphDependencies:
    """Collaborators injected into the nodes (built by the composition root, faked in tests)."""

    llm_wrapper: LLMWrapper
    search_backend: SearchBackend
    settings: Settings


def _reasoning_kwargs(model: str) -> dict[str, Any]:
    # Non-reasoning models reject reasoning_effort, so it is only sent for gpt-5*
    if model.startswith(REASONING_MODEL_PREFIX):
        return {"reasoning_effort": REASONING_EFFORT, "max_tokens": REASONING_MAX_TOKENS}
    return {}


async def _structured(
    deps: GraphDependencies,
    *,
    system_prompt: str,
    user_message: str,
    response_model: type[T],
    model: str,
    purpose: str,
) -> T:
    # The wrapper is synchronous (Instructor + LiteLLM): keep the event loop free
    result, _meta = await asyncio.to_thread(
        deps.llm_wrapper.complete_structured,
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=response_model,
        model=model,
        purpose=purpose,
        **_reasoning_kwargs(model),
    )
    return result


async def extract_requirements(state: EstimationState, deps: GraphDependencies) -> dict[str, Any]:
    with logfire.span("node: extract_requirements", estimation_id=state.get("estimation_id")):
        extraction = await _structured(
            deps,
            system_prompt=EXTRACT_SYSTEM_PROMPT,
            user_message=state["transcript"],
            response_model=RequirementsExtraction,
            model=deps.settings.GRAPH_EXTRACTION_MODEL,
            purpose="graph_extract_requirements",
        )
        return {"requirements": extraction.requirements}


async def classify_components(state: EstimationState, deps: GraphDependencies) -> dict[str, Any]:
    with logfire.span("node: classify_components", estimation_id=state.get("estimation_id")):
        requirements = state.get("requirements", [])
        if not requirements:
            return {
                "components": [],
                "errors": ["classify_components: no requirements to classify"],
            }
        classification = await _structured(
            deps,
            system_prompt=CLASSIFY_SYSTEM_PROMPT,
            user_message="\n".join(f"- {requirement}" for requirement in requirements),
            response_model=ComponentClassification,
            model=deps.settings.GRAPH_EXTRACTION_MODEL,
            purpose="graph_classify_components",
        )
        components: list[Component] = [
            {"name": item.name, "category": item.category} for item in classification.components
        ]
        return {"components": components}


def _component_query(component: Component) -> str:
    query = f"{component['name']} ({component['category']})"
    # SearchQuery requires >= MIN_QUERY_LENGTH characters: pad very short names
    if len(query.strip()) < MIN_QUERY_LENGTH:
        query = f"{query} software component"
    return query


async def search_budgets(state: EstimationState, deps: GraphDependencies) -> dict[str, Any]:
    with logfire.span("node: search_budgets", estimation_id=state.get("estimation_id")):
        matches: list[BudgetMatch] = []
        errors: list[str] = []
        # One component after another on purpose: parallel fan-out (Send) is out of scope here
        for component in state.get("components", []):
            try:
                items = await deps.search_backend(
                    SearchBudgetsArgs(query=_component_query(component), filters=None)
                )
            except Exception as exc:  # noqa: BLE001 - soft failure: report it and keep going
                errors.append(
                    f"search_budgets failed for '{component['name']}': "
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                )
                continue
            for item in items:
                # An item without hours cannot anchor an estimate
                if item.get("estimated_hours") is None:
                    continue
                matches.append(
                    {
                        "component": component["name"],
                        "reference_budget_id": item.get("budget_id"),
                        "amount": float(item["estimated_hours"]),
                        "distance": float(item.get("distance", 1.0)),
                    }
                )
        # Only what is new: the reducers concatenate it with the previous state
        return {"budget_matches": matches, "errors": errors}


def _generation_message(state: EstimationState) -> str:
    matches = state.get("budget_matches", [])
    lines: list[str] = []
    for component in state.get("components", []):
        own = [match for match in matches if match["component"] == component["name"]]
        lines.append(f"## {component['name']} ({component['category']})")
        if not own:
            lines.append("No historical reference found.")
            continue
        median = statistics.median(match["amount"] for match in own)
        lines.append(f"Median historical hours: {median:g}")
        lines.extend(
            f"- budget_id={match['reference_budget_id']} hours={match['amount']:g} "
            f"distance={match['distance']:.2f}"
            for match in own
        )
    return "\n".join(lines)


async def generate_estimate(state: EstimationState, deps: GraphDependencies) -> dict[str, Any]:
    with logfire.span("node: generate_estimate", estimation_id=state.get("estimation_id")):
        if not state.get("components"):
            return {"estimate": None, "errors": ["generate_estimate: no components to estimate"]}
        result = await _structured(
            deps,
            system_prompt=GENERATE_SYSTEM_PROMPT,
            user_message=_generation_message(state),
            response_model=AgentEstimate,
            model=deps.settings.GRAPH_GENERATION_MODEL,
            purpose="graph_generate_estimate",
        )
        return {"estimate": result.model_dump()}


def _validation_args(state: EstimationState, estimate: dict[str, Any]) -> dict[str, Any]:
    """Arguments of ``validate_estimate``: one "task" per component, with its references."""
    matches = state.get("budget_matches", [])
    components: list[dict[str, Any]] = []
    for component in estimate["components"]:
        cited = set(component["reference_budget_ids"])
        own = [match for match in matches if match["component"] == component["name"]]
        if not own:
            # The model may have renamed the component: fall back to the ids it cites
            own = [match for match in matches if match["reference_budget_id"] in cited]
        references = [
            {
                "budget_id": match["reference_budget_id"] or "unknown",
                "estimated_hours": match["amount"],
            }
            for match in own
        ]
        components.append(
            {
                "name": component["name"],
                "tasks": [
                    {
                        "name": component["name"],
                        "estimated_hours": component["hours"],
                        "references": references,
                    }
                ],
            }
        )
    return {"components": components, "total_hours": estimate["total_hours"]}


async def validate_and_consolidate(
    state: EstimationState, deps: GraphDependencies
) -> dict[str, Any]:
    with logfire.span(
        "node: validate_and_consolidate", estimation_id=state.get("estimation_id")
    ) as span:
        estimate = state.get("estimate")
        if not estimate:
            return {"status": STATUS_NEEDS_REVIEW, "errors": ["no_estimate: nothing to validate"]}
        # Same deterministic checks as the S12 agent: not re-implemented here
        report = await validate_estimate(_validation_args(state, estimate))
        status = STATUS_VALIDATED if report["ok"] else STATUS_NEEDS_REVIEW
        span.set_attribute("status", status)
        span.set_attribute("warnings", len(report["warnings"]))
        issues = [
            f"[{issue['code']}] {issue['where'] or 'estimate'}: {issue['message']}"
            for issue in report["issues"]
        ]
        # Warnings (e.g. no_reference) are informative: they never change the status
        return {"status": status, "errors": issues}


async def flag_for_review(state: EstimationState, deps: GraphDependencies) -> dict[str, Any]:
    with logfire.span("node: flag_for_review", estimation_id=state.get("estimation_id")):
        return {
            "status": STATUS_NEEDS_REVIEW,
            "errors": ["flagged for human review: the estimate did not pass validation"],
        }


def route_after_validation(state: EstimationState) -> Literal["validated", "needs_review"]:
    """Conditional edge: anything but an explicit "validated" goes to review."""
    return "validated" if state.get("status") == STATUS_VALIDATED else "needs_review"
