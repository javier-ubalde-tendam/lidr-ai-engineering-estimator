"""Agent tools: strict flat schemas, the deterministic calculators and the search wrapper."""

import json
from typing import Any

import pytest

from app.generation.agentic.agent_loop import FINAL_ANSWER_FORMAT
from app.generation.agentic.agent_schemas import SearchBudgetsArgs, SearchBudgetsFilters
from app.generation.agentic.agent_tools import (
    CONTINGENCY_FACTOR,
    PREVIEW_MAX_CHARS,
    TOOL_SCHEMAS,
    build_tool_registry,
    calculate_estimate,
    execute_tool,
    validate_estimate,
)

pytestmark = pytest.mark.anyio


# --- Schemas the model reads -------------------------------------------------------------------


def _object_nodes(node: Any) -> list[dict[str, Any]]:
    """Every dict with ``properties`` (an object schema), at any depth."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "properties" in node:
            found.append(node)
        for value in node.values():
            found += _object_nodes(value)
    elif isinstance(node, list):
        for value in node:
            found += _object_nodes(value)
    return found


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda schema: schema["name"])
def test_tool_schemas_are_flat_and_strict(schema: dict[str, Any]) -> None:
    assert set(schema) == {"type", "name", "description", "parameters", "strict"}
    assert "function" not in schema  # Chat Completions nests under "function"; Responses is flat
    assert schema["type"] == "function" and schema["strict"] is True
    assert schema["description"].strip()

    objects = _object_nodes(schema["parameters"])
    assert objects, "parameters must describe an object"
    for node in objects:
        assert node["additionalProperties"] is False
        assert sorted(node["required"]) == sorted(node["properties"])


def test_the_final_answer_format_is_a_strict_json_schema() -> None:
    assert FINAL_ANSWER_FORMAT["type"] == "json_schema" and FINAL_ANSWER_FORMAT["strict"] is True
    objects = _object_nodes(FINAL_ANSWER_FORMAT["schema"])
    assert len(objects) >= 2  # the answer and its components, resolved through $defs
    for node in objects:
        assert node["additionalProperties"] is False
        assert sorted(node["required"]) == sorted(node["properties"])


def test_the_agent_has_the_two_required_tools_and_the_optional_validator() -> None:
    assert [schema["name"] for schema in TOOL_SCHEMAS] == [
        "search_budgets",
        "calculate_estimate",
        "validate_estimate",
    ]


def test_the_search_schema_exposes_only_filters_that_exist_in_the_args_model() -> None:
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "search_budgets")
    exposed = set(schema["parameters"]["properties"]["filters"]["properties"])

    assert exposed == set(SearchBudgetsFilters.model_fields)
    assert set(schema["parameters"]["properties"]) == set(SearchBudgetsArgs.model_fields)


# --- calculate_estimate ---------------------------------------------------------------------------


def task(name: str, *refs: float) -> dict[str, Any]:
    """A task whose references are ``refs`` hours, with ids ``<name>-<n>``."""
    return {
        "name": name,
        "references": [
            {"budget_id": f"{name}-{n}", "estimated_hours": hours} for n, hours in enumerate(refs)
        ],
    }


async def test_a_task_costs_the_median_of_its_references_plus_the_fixed_contingency() -> None:
    result = await calculate_estimate(
        {"components": [{"name": "Auth", "tasks": [task("Login", 100, 120, 1000)]}]}
    )

    # The 1000 h outlier does not move the median (120); the mean would give ~407
    login = result["components"][0]["tasks"][0]
    assert login["central_hours"] == 120.0
    assert login["estimated_hours"] == round(120 * (1 + CONTINGENCY_FACTOR), 1)
    assert login["unbudgeted"] is False


async def test_a_component_is_the_sum_of_the_medians_of_its_tasks() -> None:
    result = await calculate_estimate(
        {
            "components": [
                {
                    "name": "Backend",
                    "tasks": [task("Orders API", 10, 20, 30), task("Tracking", 40, 60)],
                },
                {"name": "Mobile", "tasks": [task("Offline sync", 100)]},
            ]
        }
    )

    backend, mobile = result["components"]
    # medians: 20 and 50 -> 70 h, plus 15% contingency on each task
    assert backend["estimated_hours"] == round(20 * 1.15 + 50 * 1.15, 1) == 80.5
    assert mobile["estimated_hours"] == 115.0
    assert result["total_hours"] == 195.5
    assert "2 components and 3 tasks" in result["summary"]


async def test_a_task_without_references_costs_zero_and_is_flagged() -> None:
    result = await calculate_estimate(
        {
            "components": [
                {"name": "Backend", "tasks": [task("Known", 200), task("Unknown")]},
            ]
        }
    )

    backend = result["components"][0]
    assert backend["tasks"][1]["estimated_hours"] == 0.0 and backend["tasks"][1]["unbudgeted"]
    assert backend["unbudgeted_tasks"] == ["Unknown"]
    assert result["total_hours"] == 230.0  # only the known task counts
    assert "Backend / Unknown" in result["summary"]


async def test_the_totals_are_rounded_sums_and_deterministic() -> None:
    args = {
        "components": [
            {"name": "A", "tasks": [task("a1", 33.33), task("a2", 1.11)]},
            {"name": "B", "tasks": [task("b1", 10.01, 20.03)]},
        ]
    }

    first, second = await calculate_estimate(args), await calculate_estimate(args)

    assert first == second
    for component in first["components"]:
        assert component["estimated_hours"] == round(
            sum(t["estimated_hours"] for t in component["tasks"]), 1
        )
    assert first["total_hours"] == round(sum(c["estimated_hours"] for c in first["components"]), 1)


async def test_budget_ids_come_back_per_task_and_per_component_without_duplicates() -> None:
    shared = {"budget_id": "B-SHARED", "estimated_hours": 50.0}
    result = await calculate_estimate(
        {
            "components": [
                {
                    "name": "Backend",
                    "tasks": [
                        {
                            "name": "A",
                            "references": [
                                {"budget_id": "B-2", "estimated_hours": 10.0},
                                shared,
                                {"budget_id": "B-2", "estimated_hours": 12.0},  # repeated id
                            ],
                        },
                        {"name": "B", "references": [shared, {"budget_id": "B-1", "estimated_hours": 5.0}]},
                        {"name": "C", "references": []},
                    ],
                }
            ]
        }
    )

    backend = result["components"][0]
    task_a, task_b, task_c = backend["tasks"]
    assert task_a["reference_budget_ids"] == ["B-2", "B-SHARED"]  # first-seen order, no duplicates
    assert task_b["reference_budget_ids"] == ["B-SHARED", "B-1"]
    assert task_c["reference_budget_ids"] == []
    assert backend["reference_budget_ids"] == ["B-2", "B-SHARED", "B-1"]
    # The median still runs over the hours of ALL the references (10, 50, 12 -> 12)
    assert task_a["central_hours"] == 12.0 and task_a["reference_count"] == 3


async def test_the_old_reference_amounts_format_is_rejected() -> None:
    registry = build_tool_registry(failing_backend)
    old = {"components": [{"name": "A", "tasks": [{"name": "t", "reference_amounts": [10]}]}]}

    result = await execute_tool(registry, "calculate_estimate", json.dumps(old))

    assert "error" in result and "references" in result["error"]


async def test_a_reference_needs_a_budget_id_and_non_negative_hours() -> None:
    registry = build_tool_registry(failing_backend)
    bad_hours = {"budget_id": "B-1", "estimated_hours": -5}
    no_id = {"budget_id": "", "estimated_hours": 5}

    for reference in (bad_hours, no_id):
        args = {"components": [{"name": "A", "tasks": [{"name": "t", "references": [reference]}]}]}
        result = await execute_tool(registry, "calculate_estimate", json.dumps(args))
        assert "error" in result


def test_the_reference_schema_is_an_object_with_id_and_hours_in_both_tools() -> None:
    for name in ("calculate_estimate", "validate_estimate"):
        schema = next(s for s in TOOL_SCHEMAS if s["name"] == name)
        text = json.dumps(schema)
        assert "reference_amounts" not in text
        refs = [n for n in _object_nodes(schema["parameters"]) if "budget_id" in n["properties"]]
        assert len(refs) == 1
        assert sorted(refs[0]["required"]) == ["budget_id", "estimated_hours"]
        assert refs[0]["properties"]["budget_id"]["type"] == "string"
        assert refs[0]["properties"]["estimated_hours"]["type"] == "number"


def test_the_descriptions_do_not_contradict_the_warning_semantics() -> None:
    validate = next(s for s in TOOL_SCHEMAS if s["name"] == "validate_estimate")["description"]
    calculate = next(s for s in TOOL_SCHEMAS if s["name"] == "calculate_estimate")["description"]

    assert "`warnings` are informative" in validate and "acceptable" in validate
    assert "stale_calculation" in validate
    assert "task with no reference" in validate
    assert "pure function" in calculate and "after your last search" in calculate


def test_coverage_of_counts_tasks_with_and_without_references() -> None:
    from app.generation.agentic.agent_tools import coverage_of

    calculation = {
        "components": [
            {"name": "A", "tasks": [{"name": "a1", "unbudgeted": False}, {"name": "a2", "unbudgeted": True}]},
            {"name": "B", "tasks": [{"name": "b1", "unbudgeted": True}]},
        ]
    }

    coverage, unbudgeted = coverage_of(calculation)

    assert (coverage.budgeted_tasks, coverage.total_tasks) == (1, 3)
    assert unbudgeted == ["A / a2", "B / b1"]


async def test_a_component_must_have_tasks() -> None:
    registry = build_tool_registry(failing_backend)

    result = await execute_tool(
        registry, "calculate_estimate", json.dumps({"components": [{"name": "A", "tasks": []}]})
    )

    assert "tasks" in result["error"]


# --- validate_estimate ---------------------------------------------------------------------------


def draft(name: str, hours: float, refs: list[float]) -> dict[str, Any]:
    return {
        "name": name,
        "estimated_hours": hours,
        "references": [
            {"budget_id": f"{name}-{n}", "estimated_hours": ref} for n, ref in enumerate(refs)
        ],
    }


def component(*tasks: dict[str, Any]) -> dict[str, Any]:
    return {"name": "Backend", "tasks": list(tasks)}


def codes(result: dict[str, Any]) -> set[str]:
    """Codes of the blocking issues."""
    return {issue["code"] for issue in result["issues"]}


def warning_codes(result: dict[str, Any]) -> set[str]:
    return {warning["code"] for warning in result["warnings"]}


async def test_a_consistent_draft_is_ok() -> None:
    result = await validate_estimate(
        {
            "components": [component(draft("A", 230, [200]), draft("B", 115, [100]))],
            "total_hours": 345,
        }
    )

    assert result["ok"] is True and result["issues"] == [] and result["warnings"] == []
    assert result["summary"] == "Estimate looks consistent."


async def test_findings_point_at_the_component_and_task() -> None:
    result = await validate_estimate(
        {"components": [component(draft("Tracking", 0, []))], "total_hours": 0}
    )

    assert result["warnings"][0]["where"] == "Backend / Tracking"


async def test_no_reference_is_a_warning_that_does_not_break_ok() -> None:
    result = await validate_estimate(
        {
            "components": [
                component(draft("Known", 115, [100]), draft("A", 0, []), draft("B", 0, []))
            ],
            "total_hours": 115,
        }
    )

    assert result["ok"] is True and result["issues"] == []
    assert warning_codes(result) == {"no_reference"} and len(result["warnings"]) == 2
    assert result["summary"] == "0 issues, 2 warnings: no_reference×2"


async def test_the_summary_counts_issues_and_warnings_with_their_codes() -> None:
    result = await validate_estimate(
        {
            "components": [component(draft("A", 900, [100, 120]), draft("B", 0, []))],
            "total_hours": 900,
        }
    )

    assert result["ok"] is False
    assert result["summary"] == "1 issue, 1 warning: out_of_range×1, no_reference×1"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            {"components": [component(draft("A", 0, []))], "total_hours": 0},
            {"non_positive_total"},  # no_reference is only a warning
        ),
        (
            {"components": [component(draft("A", 900, [100, 120]))], "total_hours": 900},
            {"out_of_range"},
        ),
        (
            {"components": [component(draft("A", 10, [100, 120]))], "total_hours": 10},
            {"out_of_range"},
        ),
        (
            {"components": [component(draft("A", 0, [100]))], "total_hours": 100},
            {"non_positive_hours", "total_mismatch"},
        ),
        (
            {"components": [component(draft("A", 115, [100]))], "total_hours": 500},
            {"total_mismatch"},
        ),
        (
            {
                "components": [component(draft("A", 115, [100]), draft("B", 60, [50]))],
                "total_hours": 115,  # forgot the second task
            },
            {"total_mismatch"},
        ),
        (
            {"components": [component(draft("A", 115, [100]))], "total_hours": -5},
            {"non_positive_total", "total_mismatch"},
        ),
        (
            {"components": [component(draft("A", 90_000, [80_000]))], "total_hours": 90_000},
            {"disproportionate_total"},
        ),
    ],
)
async def test_each_kind_of_issue_is_reported(args: dict[str, Any], expected: set[str]) -> None:
    result = await validate_estimate(args)

    assert result["ok"] is False
    assert codes(result) == expected


# --- execute_tool: errors are observations ----------------------------------------------------------


async def failing_backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
    raise RuntimeError("database is down")


async def test_unknown_tool_invalid_json_and_invalid_args_do_not_raise() -> None:
    registry = build_tool_registry(failing_backend)

    unknown = await execute_tool(registry, "delete_everything", "{}")
    bad_json = await execute_tool(registry, "calculate_estimate", "{not json")
    bad_args = await execute_tool(registry, "search_budgets", json.dumps({"query": "short"}))
    extra_key = await execute_tool(
        registry, "search_budgets", json.dumps({"query": "a long enough query", "limit": 3})
    )

    assert "Unknown tool" in unknown["error"] and "search_budgets" in unknown["error"]
    assert "not valid JSON" in bad_json["error"]
    assert "query" in bad_args["error"] and "at least 10" in bad_args["error"]
    assert "limit" in extra_key["error"]


async def test_a_tool_exception_becomes_an_error_observation() -> None:
    registry = build_tool_registry(failing_backend)

    result = await execute_tool(
        registry, "search_budgets", json.dumps({"query": "payment gateway integration"})
    )

    assert "database is down" in result["error"] and "search_budgets" in result["error"]


# --- search_budgets with a fake backend -----------------------------------------------------------


class RecordingBackend:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self.received: list[SearchBudgetsArgs] = []

    async def __call__(self, args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        self.received.append(args)
        return self.items


def item(item_id: int, hours: float | None, distance: float) -> dict[str, Any]:
    return {
        "id": item_id,
        "budget_id": f"B-{item_id}",
        "sector": "finance",
        "project_year": 2023,
        "content": "x" * 400,
        "estimated_hours": hours,
        "distance": distance,
    }


async def test_search_returns_a_compact_summary_never_a_chunk_dump() -> None:
    backend = RecordingBackend([item(1, 100, 0.2), item(2, 300, 0.3)])
    registry = build_tool_registry(backend)

    result = await execute_tool(
        registry, "search_budgets", json.dumps({"query": "payment gateway integration"})
    )

    assert result["count"] == 2 and result["low_confidence"] is False
    assert set(result["items"][0]) == {
        "id",
        "budget_id",
        "sector",
        "project_year",
        "content_preview",
        "estimated_hours",
        "distance",
    }
    assert len(result["items"][0]["content_preview"]) <= PREVIEW_MAX_CHARS
    assert "100-300" in result["summary"] and "median 200" in result["summary"]
    assert "hint" not in result


async def test_filters_and_query_reach_the_backend() -> None:
    backend = RecordingBackend([item(1, 100, 0.2)])
    registry = build_tool_registry(backend)
    arguments = {
        "query": "  payment gateway integration  ",
        "filters": {"sectors": ["finance"], "project_year_min": 2022, "project_year_max": None},
    }

    await execute_tool(registry, "search_budgets", json.dumps(arguments))

    received = backend.received[0]
    assert received.query == "payment gateway integration"
    assert received.filters is not None
    assert received.filters.sectors == ["finance"] and received.filters.project_year_min == 2022
    assert received.filters.project_year_max is None


async def test_no_results_is_a_low_confidence_signal_with_a_way_out() -> None:
    registry = build_tool_registry(RecordingBackend([]))

    result = await execute_tool(
        registry, "search_budgets", json.dumps({"query": "something very exotic"})
    )

    assert result["count"] == 0 and result["low_confidence"] is True
    assert "No historical items" in result["summary"]
    # It must say how to retry AND when to stop, otherwise the agent searches forever
    assert "generic wording" in result["hint"] and "unbudgeted" in result["hint"]
    assert result["filters_applied"] == {} and "without them" not in result["hint"]


async def test_an_empty_result_reports_the_filters_that_were_applied() -> None:
    registry = build_tool_registry(RecordingBackend([]))
    arguments = {
        "query": "something very exotic",
        "filters": {"sectors": ["industrial"], "project_year_min": None, "project_year_max": 2024},
    }

    result = await execute_tool(registry, "search_budgets", json.dumps(arguments))

    assert result["filters_applied"] == {"sectors": ["industrial"], "project_year_max": 2024}
    assert "retry once without them" in result["hint"]


async def test_weak_matches_are_flagged_and_items_without_hours_are_dropped() -> None:
    registry = build_tool_registry(RecordingBackend([item(1, 100, 0.55), item(2, None, 0.1)]))

    result = await execute_tool(
        registry, "search_budgets", json.dumps({"query": "something vaguely related"})
    )

    assert result["count"] == 1  # the item without hours is useless as a reference amount
    assert result["low_confidence"] is True
    assert "at most twice" in result["hint"] and "unbudgeted" in result["hint"]
