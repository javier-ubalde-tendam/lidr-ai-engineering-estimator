"""One Logfire span per node, captured in memory (no token, no export)."""

import pytest
from logfire.testing import CaptureLogfire

from app.domain.graph import build_graph
from tests.domain.graph.helpers import (
    TRANSCRIPT,
    ScriptedLLM,
    ScriptedSearch,
    estimate_of,
    make_deps,
)

pytestmark = pytest.mark.anyio

COMPONENTS = [("Payments", "backend")]
HOURS = {"Payments": [100.0]}


async def span_names(capfire: CaptureLogfire, estimate_hours: float) -> list[str]:
    estimate = estimate_of(("Payments", estimate_hours, ["B-Pay-0"]))
    graph = build_graph(None, make_deps(ScriptedLLM(estimate, COMPONENTS), ScriptedSearch(HOURS)))
    await graph.ainvoke({"transcript": TRANSCRIPT})
    return [span["name"] for span in capfire.exporter.exported_spans_as_dict()]


async def test_each_node_opens_a_span(capfire: CaptureLogfire) -> None:
    names = await span_names(capfire, 100.0)

    assert [n for n in names if n.startswith("node: ")] == [
        "node: extract_requirements",
        "node: classify_components",
        "node: search_budgets",
        "node: generate_estimate",
        "node: validate_and_consolidate",
    ]


async def test_review_branch_opens_the_flag_span(capfire: CaptureLogfire) -> None:
    names = await span_names(capfire, 900.0)  # far above 1.5 x the only reference

    assert "node: flag_for_review" in names
