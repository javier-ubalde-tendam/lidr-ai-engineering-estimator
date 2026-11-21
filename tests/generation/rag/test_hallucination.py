"""Semantic hallucination gate: numeric anchor, pure combiner, batched judge and aggregation.

The LLM is a fake injected as ``llm``; nothing here touches the network.
"""

from typing import Any

import pytest

from app.foundation.llm.wrapper import LLMCallMeta
from app.generation.rag.quality.hallucination import (
    JudgePanel,
    gate_estimate,
    gate_line,
    judge_estimate,
    numeric_anchor,
)
from app.generation.rag.schemas import (
    Estimate,
    LineVerdict,
    RetrievedChunk,
    SourceReference,
    TaskItem,
    WorkModule,
)

pytestmark = pytest.mark.anyio

META = LLMCallMeta(latency_ms=1, tokens_in=1, tokens_out=1, cost_usd=0.0, model="m", provider="p")


class FakeJudge:
    """Stands in for ``LLMWrapper``: returns a canned panel (or raises) and records the calls."""

    def __init__(self, result: JudgePanel | Exception) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[JudgePanel, LLMCallMeta]:
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result, META


def make_chunk(chunk_id: int, content: str = "x", hours: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id, content=content, chunk_type="budget_component", distance=0.2, estimated_hours=hours
    )


def make_task(
    name: str, days: int | None, *, grounded: bool = True, cited: tuple[str, ...] = ("101",)
) -> TaskItem:
    sources = [
        SourceReference(chunk_id=chunk_id, document_id="BUD-1", evidence="120 h")
        for chunk_id in cited
        if grounded
    ]
    return TaskItem(name=name, engineer_days=days, grounded=grounded, sources=sources)


def make_estimate(*tasks: TaskItem) -> Estimate:
    return Estimate(
        confidence="high", reasoning="r", modules=[WorkModule(name="Auth", tasks=list(tasks))]
    )


def test_anchor_prefers_metadata_hours_then_parses_the_text() -> None:
    from app.generation.rag.quality.hallucination import _chunk_hours

    assert _chunk_hours(make_chunk(1, "no figure", hours=120)) == 120.0
    assert _chunk_hours(make_chunk(1, "AUTH :: OAuth - 160 h and later 40 hours")) == 40.0
    assert _chunk_hours(make_chunk(1, "no hours at all")) is None


def test_numeric_anchor_sums_cited_hours_in_engineer_days() -> None:
    chunks = {"101": make_chunk(101, hours=120), "102": make_chunk(102, hours=40)}

    anchor = numeric_anchor(make_task("t", 10, cited=("101", "102")), chunks)

    assert anchor == (120 + 40) / 8


def test_numeric_anchor_is_none_without_cited_hours() -> None:
    chunks = {"101": make_chunk(101, "no figure")}

    assert numeric_anchor(make_task("t", 10), chunks) is None
    assert numeric_anchor(make_task("t", 10, cited=("999",)), chunks) is None  # not retrieved


def test_gate_line_grounded_within_tolerance_and_for_a_fraction_of_the_source() -> None:
    within = gate_line(make_task("a", 20), "Auth", 15.0, None, tolerance=0.5)  # +33 %
    fraction = gate_line(make_task("b", 2), "Auth", 15.0, None, tolerance=0.5)  # far LESS is fine

    assert within.status == "grounded" and fraction.status == "grounded"
    assert fraction.numeric_deviation == pytest.approx(13 / 15)


def test_gate_line_degraded_on_numeric_overshoot() -> None:
    gate = gate_line(make_task("inflated", 90), "Auth", 15.0, None, tolerance=0.5)

    assert gate.status == "degraded"
    assert gate.numeric_deviation is not None and gate.numeric_deviation > 0.5
    assert "90d" in gate.reason


def test_gate_line_without_anchor_relies_on_the_judge_only() -> None:
    entailed = LineVerdict(module="Auth", component="a", entailed=True, reason="ok")
    rejected = LineVerdict(module="Auth", component="a", entailed=False, reason="unrelated work")

    assert gate_line(make_task("a", 90), "Auth", None, None, tolerance=0.5).status == "grounded"
    assert gate_line(make_task("a", 90), "Auth", None, entailed, tolerance=0.5).status == "grounded"
    degraded = gate_line(make_task("a", 90), "Auth", None, rejected, tolerance=0.5)
    assert degraded.status == "degraded" and "unrelated work" in degraded.reason


def test_gate_line_insufficient_when_ungrounded() -> None:
    task = make_task("nodata", None, grounded=False)

    assert gate_line(task, "Auth", None, None, tolerance=0.5).status == "insufficient"


async def test_judge_makes_one_batched_call_with_the_verbatim_evidence() -> None:
    verdict = LineVerdict(module="Auth", component="ok", entailed=True, reason="same work")
    judge = FakeJudge(JudgePanel(verdicts=[verdict]))
    estimate = make_estimate(make_task("ok", 10), make_task("other", 5), make_task("nodata", None, grounded=False))

    verdicts = await judge_estimate(estimate, llm=judge, model="judge-model", max_tokens=123)  # type: ignore[arg-type]

    assert verdicts == {("Auth", "ok"): verdict}
    assert len(judge.calls) == 1
    call = judge.calls[0]
    assert call["model"] == "judge-model" and call["max_tokens"] == 123
    assert call["response_model"] is JudgePanel
    assert '"120 h"' in call["user_message"] and "component: other" in call["user_message"]
    assert "nodata" not in call["user_message"]  # ungrounded lines are not audited


async def test_judge_is_not_called_when_there_are_no_grounded_lines() -> None:
    judge = FakeJudge(JudgePanel())

    verdicts = await judge_estimate(
        make_estimate(make_task("nodata", None, grounded=False)), llm=judge, model="m"  # type: ignore[arg-type]
    )

    assert verdicts == {} and judge.calls == []


async def test_a_failing_judge_degrades_to_the_numeric_anchor() -> None:
    judge = FakeJudge(RuntimeError("provider down"))
    chunks = [make_chunk(101, hours=120)]
    estimate = make_estimate(make_task("ok", 10), make_task("inflated", 90))

    report = await gate_estimate(
        estimate, chunks, llm=judge, tolerance=0.5, judge_model="m"  # type: ignore[arg-type]
    )

    assert len(judge.calls) == 1
    assert (report.grounded_lines, report.degraded_lines) == (1, 1)


async def test_gate_estimate_aggregates_anchor_and_judge() -> None:
    rejected = LineVerdict(module="Auth", component="judged", entailed=False, reason="wrong scope")
    judge = FakeJudge(JudgePanel(verdicts=[rejected]))
    chunks = [make_chunk(101, hours=120)]
    estimate = make_estimate(
        make_task("ok", 10),
        make_task("inflated", 90),
        make_task("judged", 10),
        make_task("nodata", None, grounded=False),
    )

    report = await gate_estimate(
        estimate, chunks, llm=judge, tolerance=0.5, judge_model="m"  # type: ignore[arg-type]
    )

    assert report.total_lines == 4
    assert (report.grounded_lines, report.degraded_lines, report.insufficient_lines) == (1, 2, 1)
    assert report.has_degraded is True
    assert {line.component: line.status for line in report.lines} == {
        "ok": "grounded",
        "inflated": "degraded",
        "judged": "degraded",
        "nodata": "insufficient",
    }


async def test_gate_estimate_without_judge_never_calls_the_llm() -> None:
    judge = FakeJudge(JudgePanel())

    report = await gate_estimate(
        make_estimate(make_task("ok", 10)),
        [make_chunk(101, hours=120)],
        llm=judge,  # type: ignore[arg-type]
        tolerance=0.5,
        judge_model="m",
        use_judge=False,
    )

    assert judge.calls == [] and report.grounded_lines == 1
