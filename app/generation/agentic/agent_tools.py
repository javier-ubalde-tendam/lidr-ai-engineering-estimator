"""Tools of the estimation agent (Session 12): schemas the model reads + their implementations.

Backends are **injected**: ``search_budgets`` receives an async callable instead of importing the
RAG layer (``generation/agentic`` must not know ``generation/rag``). The conductor
(``app/domain/estimation_service.py``) builds the real backend; tests and the ``--stub`` script
pass fakes.

Every tool returns a plain ``dict`` that is serialized as the ``function_call_output``. Failures
(bad arguments, unknown tool, exception) are **observations** too (``{"error": "..."}``): the model
reads them, fixes its call and the loop goes on.
"""

import json
import statistics
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from pydantic import ValidationError

from app.generation.agentic.agent_schemas import (
    TOTAL_TOLERANCE_HOURS,
    CalculateEstimateArgs,
    Coverage,
    SearchBudgetsArgs,
    ValidateEstimateArgs,
)

log = structlog.get_logger()

# Flat contingency added to every component's central estimate. Transparent: no hidden multipliers.
CONTINGENCY_FACTOR = 0.15

# search_budgets output: keep observations small, the model only needs the gist of each item
PREVIEW_MAX_CHARS = 160
# Beyond this cosine distance the best match is only loosely related to the query
WEAK_MATCH_DISTANCE = 0.45

# validate_estimate: an estimate is "reasonable" inside [min(refs) * 0.5, max(refs) * 1.5]
RANGE_LOWER_FACTOR = 0.5
RANGE_UPPER_FACTOR = 1.5
# ~10 engineer-years: a total above this is suspicious for a single estimate
MAX_REASONABLE_TOTAL_HOURS = 20_000.0

ERROR_MESSAGE_MAX_CHARS = 300

SearchBackend = Callable[[SearchBudgetsArgs], Awaitable[list[dict[str, Any]]]]
ToolFunction = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ToolRegistry = dict[str, ToolFunction]


# --- Schemas the model reads (Responses API: flat function tools, strict mode) -----------------

SEARCH_BUDGETS_DESCRIPTION = (
    "Search the historical database for past engineering TASKS similar to ONE task of the "
    "project and return the engineer-hours each one actually took. The items are functional "
    "tasks (a feature or a piece of a module, roughly 16 to 72 hours each): not whole systems, "
    "and not small chores or product-specific details. Break a component down to that level "
    "before searching. Make one call per task, each with its own query; do not mix unrelated "
    "tasks in a single search. Independent searches, including the rephrasings of a round, can "
    "be issued together in the same turn. "
    "Returns `items` (id, budget_id, sector, project_year, content_preview, estimated_hours = "
    "engineer-hours, distance = cosine distance: lower is more similar), `count`, a `summary`, "
    "the `filters_applied` and `low_confidence`. When `low_confidence` is true or `count` is 0 "
    "the evidence is weak: rephrase with more generic wording (what the task does, without "
    "product or protocol names) or drop the filters. If two rephrasings are still weak, the "
    "history has no equivalent for that task: report it as unbudgeted instead of searching "
    "further. For each relevant item, its `budget_id` and `estimated_hours` form one reference "
    "of that task in `calculate_estimate`; only budget_ids returned here may be cited later."
)

CALCULATE_ESTIMATE_DESCRIPTION = (
    "Deterministic cost model (no LLM) that turns historical hours into an estimate. Pass the "
    "project components, each with its `tasks`; every task has a `name` and its `references`: "
    "one `{budget_id, estimated_hours}` object per relevant item that `search_budgets` returned "
    "for that task (an empty list when nothing relevant was found). A task costs the median of "
    "the `estimated_hours` of its references plus a fixed 15% contingency; a component is the "
    "SUM of its tasks; the project total is the sum of the components. Returns per task and per "
    "component its `estimated_hours` and the `reference_budget_ids` it rests on (the ids to cite "
    "in the final answer), plus `total_hours`, the `unbudgeted_tasks` and a `summary`. A task "
    "without references costs 0 h (hours are never guessed): search again for it, or report it "
    "as unbudgeted. It is a pure function: it does not know what you searched. Run it after "
    "your last search; if you search again afterwards, run it again. Its output is the basis "
    "of the final answer."
)

VALIDATE_ESTIMATE_DESCRIPTION = (
    "Sanity-check a draft estimate before answering. Pass every component with its tasks (each "
    "with `name`, `estimated_hours` as returned by `calculate_estimate`, and its `references` as "
    "`{budget_id, estimated_hours}` objects), plus the `total_hours` you intend to report. "
    "Returns `ok`, `issues`, `warnings` and a `summary` with the counts and codes. `issues` are "
    "real defects (hours far outside the range of the task's references, non-positive hours, a "
    "total that does not match the sum of the tasks, a non-positive or disproportionate total): "
    "fix their cause and validate again. `warnings` are informative: a task with no reference "
    "(already declared as unbudgeted) is acceptable, no need to search again; report it in "
    "`notes`. A `stale_calculation` warning means you searched after your last calculation: "
    "recalculate."
)


def _function_tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    # Responses API shape: flat, NOT nested under a "function" key (that is Chat Completions)
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": parameters,
        "strict": True,
    }


_REFERENCES = {
    "type": "array",
    "description": "The relevant historical items found for this task, one object per item.",
    "items": {
        "type": "object",
        "properties": {
            "budget_id": {
                "type": "string",
                "description": "budget_id of the item, exactly as `search_budgets` returned it.",
            },
            "estimated_hours": {
                "type": "number",
                "description": "Engineer-hours of that item, as `search_budgets` returned them.",
            },
        },
        "required": ["budget_id", "estimated_hours"],
        "additionalProperties": False,
    },
}

SEARCH_BUDGETS_SCHEMA = _function_tool(
    "search_budgets",
    SEARCH_BUDGETS_DESCRIPTION,
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Descriptive text of the single task to look for (at least 10 characters): "
                    "what it does and its main technology, in generic functional wording, e.g. "
                    "'payment gateway integration with webhooks and refunds'. Product- or "
                    "protocol-specific details (codes, vendor names) match the history worse."
                ),
            },
            "filters": {
                "type": ["object", "null"],
                "description": "Optional metadata filters; null to search the whole corpus.",
                "properties": {
                    "sectors": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": (
                            "Keep only budgets of these client sectors (finance, ecommerce, "
                            "healthcare, industrial, logistics, education, media, government); "
                            "null for any sector."
                        ),
                    },
                    "project_year_min": {
                        "type": ["integer", "null"],
                        "description": "Oldest project year to include (e.g. 2022); null for no limit.",
                    },
                    "project_year_max": {
                        "type": ["integer", "null"],
                        "description": "Newest project year to include; null for no limit.",
                    },
                },
                "required": ["sectors", "project_year_min", "project_year_max"],
                "additionalProperties": False,
            },
        },
        "required": ["query", "filters"],
        "additionalProperties": False,
    },
)

CALCULATE_ESTIMATE_SCHEMA = _function_tool(
    "calculate_estimate",
    CALCULATE_ESTIMATE_DESCRIPTION,
    {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "description": "One entry per project component.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Component name."},
                        "tasks": {
                            "type": "array",
                            "description": "The engineering tasks that make up the component.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Task name."},
                                    "references": _REFERENCES,
                                },
                                "required": ["name", "references"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["name", "tasks"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["components"],
        "additionalProperties": False,
    },
)

VALIDATE_ESTIMATE_SCHEMA = _function_tool(
    "validate_estimate",
    VALIDATE_ESTIMATE_DESCRIPTION,
    {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "description": "The draft estimate, one entry per component.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Component name."},
                        "tasks": {
                            "type": "array",
                            "description": "The tasks of the component with their draft hours.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Task name."},
                                    "estimated_hours": {
                                        "type": "number",
                                        "description": "Hours estimated for the task.",
                                    },
                                    "references": _REFERENCES,
                                },
                                "required": ["name", "estimated_hours", "references"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["name", "tasks"],
                    "additionalProperties": False,
                },
            },
            "total_hours": {
                "type": "number",
                "description": "Total hours you intend to report for the whole project.",
            },
        },
        "required": ["components", "total_hours"],
        "additionalProperties": False,
    },
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    SEARCH_BUDGETS_SCHEMA,
    CALCULATE_ESTIMATE_SCHEMA,
    VALIDATE_ESTIMATE_SCHEMA,
]


# --- Implementations ------------------------------------------------------------------------


def _preview(item: dict[str, Any]) -> str:
    text = item.get("content_preview") or item.get("content") or ""
    return " ".join(str(text).split())[:PREVIEW_MAX_CHARS]


def _compact_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "budget_id": item.get("budget_id"),
        "sector": item.get("sector"),
        "project_year": item.get("project_year"),
        "content_preview": _preview(item),
        "estimated_hours": float(item["estimated_hours"]),
        "distance": round(float(item.get("distance", 1.0)), 3),
    }


def build_search_result(
    query: str, raw_items: list[dict[str, Any]], filters_applied: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compact, model-friendly view of the backend items (never a chunk dump)."""
    filters_applied = filters_applied or {}
    # An item without hours cannot be a reference amount: it is of no use to calculate_estimate
    usable = [item for item in raw_items if item.get("estimated_hours") is not None]
    items = [_compact_item(item) for item in usable]
    if not items:
        # Say WHY it may be empty and when to stop: otherwise "try other terms" never ends
        filter_note = (
            f" Filters applied: {json.dumps(filters_applied)}; retry once without them."
            if filters_applied
            else ""
        )
        return {
            "items": [],
            "count": 0,
            "summary": f"No historical items found for '{query}'.",
            "low_confidence": True,
            "filters_applied": filters_applied,
            "hint": (
                "Nothing in the history is close enough to this wording."
                f"{filter_note} Retry with more generic wording (what the task does, no product "
                "or protocol names). If that is also empty, the history has no equivalent: "
                "report the task as unbudgeted (0 h, noted in `notes`) and move on."
            ),
        }

    hours = [item["estimated_hours"] for item in items]
    best_distance = min(item["distance"] for item in items)
    low_confidence = best_distance > WEAK_MATCH_DISTANCE
    result: dict[str, Any] = {
        "items": items,
        "count": len(items),
        "summary": (
            f"{len(items)} historical items; hours {min(hours):g}-{max(hours):g} "
            f"(median {statistics.median(hours):g}); closest distance {best_distance}."
        ),
        "low_confidence": low_confidence,
        "filters_applied": filters_applied,
    }
    if low_confidence:
        result["hint"] = (
            f"Weak matches (closest distance {best_distance} > {WEAK_MATCH_DISTANCE}): use only "
            "the items that are clearly the same kind of work. Rephrase at most twice with more "
            "generic wording; if it stays weak, report the task as unbudgeted."
        )
    return result


def make_search_budgets(backend: SearchBackend) -> ToolFunction:
    async def search_budgets(raw_args: dict[str, Any]) -> dict[str, Any]:
        args = SearchBudgetsArgs.model_validate(raw_args)
        filters = args.filters.model_dump(exclude_none=True) if args.filters else {}
        return build_search_result(args.query, await backend(args), filters)

    return search_budgets


def _unique(ids: list[str]) -> list[str]:
    """Drop duplicates keeping the first-seen order (stable, deterministic output)."""
    return list(dict.fromkeys(ids))


async def calculate_estimate(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Cost each component as the SUM of its tasks, each task from its historical hours (no LLM).

    The history stores individual tasks, so a component is only comparable to it at task level:
    every task costs the **median** of its reference hours (not the mean: one outlier task, a
    gold-plated or a rushed one, drags the mean of a handful of samples far away while the
    median ignores it), plus a fixed, visible ``CONTINGENCY_FACTOR``. The component is the sum
    of its tasks and the project total the sum of the components. A task without references
    costs 0 h and is listed in ``unbudgeted_tasks``: inventing a number would defeat the point
    of grounding the estimate on history.

    References carry their ``budget_id``: the result returns the ids per task and per component
    (no duplicates, first-seen order) so the final answer can cite them without reconstructing
    them from memory. The function is pure: it cannot know whether the model searched afterwards.
    """
    args = CalculateEstimateArgs.model_validate(raw_args)
    components: list[dict[str, Any]] = []
    unbudgeted: list[str] = []
    for component in args.components:
        tasks: list[dict[str, Any]] = []
        for task in component.tasks:
            hours = [reference.estimated_hours for reference in task.references]
            central = statistics.median(hours) if hours else 0.0
            tasks.append(
                {
                    "name": task.name,
                    "reference_count": len(hours),
                    "reference_budget_ids": _unique(
                        [reference.budget_id for reference in task.references]
                    ),
                    "central_hours": round(central, 1),
                    "estimated_hours": round(central * (1 + CONTINGENCY_FACTOR), 1),
                    "unbudgeted": not hours,
                }
            )
            if not hours:
                unbudgeted.append(f"{component.name} / {task.name}")
        components.append(
            {
                "name": component.name,
                # Summing the rounded task values keeps every level consistent with what is shown
                "estimated_hours": round(sum(task["estimated_hours"] for task in tasks), 1),
                "reference_budget_ids": _unique(
                    [budget_id for task in tasks for budget_id in task["reference_budget_ids"]]
                ),
                "unbudgeted_tasks": [task["name"] for task in tasks if task["unbudgeted"]],
                "tasks": tasks,
            }
        )

    total = round(sum(component["estimated_hours"] for component in components), 1)
    task_count = sum(len(component["tasks"]) for component in components)
    summary = (
        f"total={total}h across {len(components)} components and {task_count} tasks "
        f"(sum of task medians + {CONTINGENCY_FACTOR:.0%} contingency)"
    )
    if unbudgeted:
        summary += f"; unbudgeted (no references): {', '.join(unbudgeted)}"
    return {
        "components": components,
        "total_hours": total,
        "contingency_factor": CONTINGENCY_FACTOR,
        "unbudgeted_tasks": unbudgeted,
        "summary": summary,
    }


def coverage_of(calculation: dict[str, Any]) -> tuple[Coverage, list[str]]:
    """Coverage and unbudgeted task names of a ``calculate_estimate`` result (system-computed)."""
    tasks = [
        (component["name"], task)
        for component in calculation.get("components", [])
        for task in component.get("tasks", [])
    ]
    unbudgeted = [f"{name} / {task['name']}" for name, task in tasks if task.get("unbudgeted")]
    total = len(tasks)
    budgeted = total - len(unbudgeted)
    ratio = round(budgeted / total, 3) if total else 0.0
    return Coverage(budgeted_tasks=budgeted, total_tasks=total, ratio=ratio), unbudgeted


def _finding(where: str | None, code: str, message: str) -> dict[str, Any]:
    return {"where": where, "code": code, "message": message}


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def summarize_findings(issues: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> str:
    """``"0 issues, 10 warnings: no_reference×10"``: counts first, then every code with its count."""
    if not issues and not warnings:
        return "Estimate looks consistent."
    counts: dict[str, int] = {}
    for finding in [*issues, *warnings]:
        counts[finding["code"]] = counts.get(finding["code"], 0) + 1
    codes = ", ".join(f"{code}×{count}" for code, count in counts.items())
    return f"{_plural(len(issues), 'issue')}, {_plural(len(warnings), 'warning')}: {codes}"


async def validate_estimate(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Deterministic sanity checks over a draft estimate; never raises on bad numbers.

    Two severities: ``issues`` are defects the model must fix (``ok`` is false); ``warnings``
    are informative and do not break ``ok``. A task without references is a warning: after the
    searches are over it is an already-declared gap (it costs 0 h), not something to fix.
    """
    args = ValidateEstimateArgs.model_validate(raw_args)
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    tasks_sum = 0.0

    for component in args.components:
        for task in component.tasks:
            where = f"{component.name} / {task.name}"
            refs = [reference.estimated_hours for reference in task.references]
            hours = task.estimated_hours
            tasks_sum += hours
            if not refs:
                warnings.append(
                    _finding(where, "no_reference", "No historical reference: unbudgeted task.")
                )
                continue
            if hours <= 0:
                issues.append(_finding(where, "non_positive_hours", "Hours must be positive."))
                continue
            low, high = min(refs) * RANGE_LOWER_FACTOR, max(refs) * RANGE_UPPER_FACTOR
            if not low <= hours <= high:
                issues.append(
                    _finding(
                        where,
                        "out_of_range",
                        f"{hours:g} h is outside {low:g}-{high:g} h, the range its references "
                        f"({min(refs):g}-{max(refs):g} h) support.",
                    )
                )

    if args.total_hours <= 0:
        issues.append(_finding(None, "non_positive_total", "Total hours must be positive."))
    elif args.total_hours > MAX_REASONABLE_TOTAL_HOURS:
        issues.append(
            _finding(
                None,
                "disproportionate_total",
                f"{args.total_hours:g} h exceeds {MAX_REASONABLE_TOTAL_HOURS:g} h: check units.",
            )
        )
    if abs(args.total_hours - tasks_sum) > TOTAL_TOLERANCE_HOURS:
        issues.append(
            _finding(
                None,
                "total_mismatch",
                f"Total {args.total_hours:g} h differs from the sum of the tasks "
                f"({round(tasks_sum, 1):g} h).",
            )
        )

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "summary": summarize_findings(issues, warnings),
    }


def build_tool_registry(search_backend: SearchBackend) -> ToolRegistry:
    """name -> async function, with the search backend injected."""
    return {
        "search_budgets": make_search_budgets(search_backend),
        "calculate_estimate": calculate_estimate,
        "validate_estimate": validate_estimate,
    }


def _validation_message(exc: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "; ".join(problems)[:ERROR_MESSAGE_MAX_CHARS]


async def execute_tool(
    registry: ToolRegistry, name: str, raw_args: str | dict[str, Any]
) -> dict[str, Any]:
    """Run one tool call. Never raises: every failure becomes an ``{"error": ...}`` observation."""
    tool = registry.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(sorted(registry))}."}

    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError as exc:
        return {"error": f"Arguments of '{name}' are not valid JSON ({exc.msg}). Resend them."}
    if not isinstance(args, dict):
        return {"error": f"Arguments of '{name}' must be a JSON object."}

    try:
        return await tool(args)
    except ValidationError as exc:
        return {
            "error": f"Invalid arguments for '{name}': {_validation_message(exc)}. "
            "Fix them and call the tool again."
        }
    except Exception as exc:  # noqa: BLE001 - a tool failure must reach the model, not abort the loop
        log.warning("agent_tool_failed", tool=name, error_type=type(exc).__name__)
        message = f"{type(exc).__name__}: {exc}"[:ERROR_MESSAGE_MAX_CHARS]
        return {"error": f"Tool '{name}' failed ({message}). Adjust the call or try another approach."}
