"""Offline check of the loop mechanics: complex transcript + the kit's retrieval stub + a scripted model.

No network and no database: the "model" is a policy that reacts to the observations it receives,
like the real agent would (one search per component, then calculate, then answer). It proves the
wiring end to end (stub loaded by path -> backend -> tools -> loop -> trace), not the quality of
a real model's reasoning.
"""

import json
from typing import Any

import pytest

from app.generation.agentic.agent_loop import format_trace, run_agent
from app.generation.agentic.agent_tools import build_tool_registry
from scripts.run_agent_s12 import STUB_PATH, load_stub_search, make_stub_backend
from tests.agent_helpers import (
    FakeAsyncOpenAI,
    function_call,
    make_response,
    outputs_of,
    reasoning_item,
)

pytestmark = pytest.mark.anyio

COMPLEX_TRANSCRIPT = STUB_PATH.parent / "sample_transcript_complex.txt"
COMPONENT_QUERIES = {
    "backend": "business backend: orders, route assignment, shipment tracking and tariffs API",
    "sap": "SAP ERP integration middleware with IDocs, invoicing and retries",
    "mobile": "mobile app for couriers, Android iOS, offline mode with signature and photo",
    "analytics": "analytics dashboard with KPIs, filters and alerts for management",
}


def observations(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {item["call_id"]: json.loads(item["output"]) for item in outputs_of(kwargs)}


def search_turn(_: dict[str, Any]) -> Any:
    calls = [
        function_call("search_budgets", {"query": query, "filters": None}, name)
        for name, query in COMPONENT_QUERIES.items()
    ]
    return make_response(
        "resp_1", [reasoning_item("Four unrelated components: search each."), *calls]
    )


def calculate_turn(kwargs: dict[str, Any]) -> Any:
    seen = observations(kwargs)
    components = [
        {
            "name": name,
            "tasks": [
                {
                    "name": f"{name} core",
                    "references": [
                        {"budget_id": i["budget_id"], "estimated_hours": i["estimated_hours"]}
                        for i in seen[name]["items"]
                    ],
                }
            ],
        }
        for name in COMPONENT_QUERIES
    ]
    return make_response(
        "resp_2", [function_call("calculate_estimate", {"components": components}, "calc")]
    )


def answer_turn(kwargs: dict[str, Any]) -> Any:
    calc = observations(kwargs)["calc"]
    estimate = {
        "components": [
            {
                "name": c["name"],
                "hours": c["estimated_hours"],
                "reference_budget_ids": c["reference_budget_ids"],
            }
            for c in calc["components"]
        ],
        "total_hours": calc["total_hours"],
        "notes": "stub-backed run",
    }
    return make_response("resp_3", [], output_text=json.dumps(estimate))


async def test_the_complex_transcript_triggers_one_search_per_component_through_the_stub() -> None:
    assert STUB_PATH.is_file() and COMPLEX_TRANSCRIPT.is_file()
    transcript = COMPLEX_TRANSCRIPT.read_text(encoding="utf-8")
    client = FakeAsyncOpenAI([search_turn, calculate_turn, answer_turn])
    registry = build_tool_registry(make_stub_backend(load_stub_search()))

    result = await run_agent(
        transcript,
        client=client,
        registry=registry,
        model="gpt-5-mini",
        effort="minimal",
        max_iterations=10,
    )

    assert client.calls[0]["input"] == [{"role": "user", "content": transcript}]
    searches = [a for step in result.trace for a in step.actions if a.tool == "search_budgets"]
    assert len(searches) > 1 and len(searches) == len(COMPONENT_QUERIES)
    assert all(a.error is None and a.observation["count"] > 0 for a in searches)
    assert result.status == "done" and result.iterations == 3
    assert result.estimate is not None and result.estimate.total_hours > 0
    assert "STEP 1" in format_trace(result) and "search_budgets(" in format_trace(result)
    # The kit's stub items carry budget_id, so the new reference format works end to end
    calc = next(a for step in result.trace for a in step.actions if a.tool == "calculate_estimate")
    assert all(c["reference_budget_ids"] for c in calc.observation["components"])
    assert result.estimate is not None
    assert all(c.reference_budget_ids for c in result.estimate.components)
    assert result.coverage is not None and result.coverage.ratio == 1.0
    assert result.warnings == []
