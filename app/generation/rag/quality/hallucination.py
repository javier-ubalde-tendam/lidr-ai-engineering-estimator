"""Session 11 - semantic hallucination gate over a grounded estimate.

``verify_citations`` proves REFERENTIAL integrity: every cited ``chunk_id`` was retrieved. That is
necessary but not sufficient: a line can cite a real ``<source>`` and still invent the number. This
module adds the SEMANTIC layer on top:

* ``numeric_anchor``: a DETERMINISTIC anchor, the historical hours the cited chunks carry (metadata
  ``estimated_hours`` or a figure parsed from the text) converted to engineer-days. No LLM, no cost.
* ``judge_estimate``: a STRICT judge (one batched LLM call): does the cited evidence actually
  support the claimed number and scope?
* ``gate_line``: pure combiner of anchor and verdict into ``grounded`` / ``degraded`` /
  ``insufficient``.
* ``gate_estimate``: anchors every line, runs the judge once (unless ``use_judge=False``) and
  aggregates a ``HallucinationReport``.

The rule is deliberately asymmetric: a line may estimate LESS than its source (it is one slice of
a historical component), but a line that claims substantially MORE than the evidence it cites, or
that the judge cannot tie to that evidence, is a hallucination wearing a real citation.
"""

import asyncio
import re

import structlog
from pydantic import BaseModel, Field

from app.foundation.llm.wrapper import LLMWrapper
from app.generation.rag.schemas import (
    Estimate,
    HallucinationReport,
    LineGate,
    LineVerdict,
    RetrievedChunk,
    TaskItem,
)

log = structlog.get_logger()

# The corpus records hours, the estimate speaks in engineer-days
_HOURS_PER_DAY = 8

# "<number> h / hrs / hours": the hours a chunk carries when they are not in its metadata
_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:h|hrs|hours)\b", re.IGNORECASE)


class JudgePanel(BaseModel):
    """Structured output of the batched judge call (one verdict per audited line)."""

    verdicts: list[LineVerdict] = Field(default_factory=list)


def _chunk_hours(chunk: RetrievedChunk) -> float | None:
    """Historical hours a chunk records: metadata first, then the last figure in its text."""
    if chunk.estimated_hours is not None:
        return float(chunk.estimated_hours)
    matches = _HOURS_RE.findall(chunk.content or "")
    return float(matches[-1]) if matches else None


def numeric_anchor(task: TaskItem, chunks_by_id: dict[str, RetrievedChunk]) -> float | None:
    """Deterministic anchor for a line, in engineer-days.

    Sums the hours of the chunks the line cites (its ceiling: a line should not derive more effort
    than the sources it names support). ``None`` when no cited chunk carries hours.
    """
    hours = [
        h
        for ref in task.sources
        if (chunk := chunks_by_id.get(ref.chunk_id)) is not None
        and (h := _chunk_hours(chunk)) is not None
    ]
    if not hours:
        return None
    return sum(hours) / _HOURS_PER_DAY


def gate_line(
    task: TaskItem,
    module_name: str,
    anchor_days: float | None,
    verdict: LineVerdict | None,
    *,
    tolerance: float,
) -> LineGate:
    """Combine the numeric anchor and the judge verdict for one line (pure, no I/O).

    ``verdict`` is ``None`` when the judge did not run (numeric-only mode) or failed.
    """
    if not task.grounded:
        return LineGate(module=module_name, component=task.name, status="insufficient")

    deviation: float | None = None
    numeric_fail = False
    if anchor_days and anchor_days > 0 and task.engineer_days is not None:
        overage = (task.engineer_days - anchor_days) / anchor_days
        deviation = abs(task.engineer_days - anchor_days) / anchor_days
        # Only an OVER-estimate beyond the tolerance is a hallucination signal
        numeric_fail = overage > tolerance

    judge_fail = verdict is not None and not verdict.entailed
    if not (numeric_fail or judge_fail):
        return LineGate(
            module=module_name, component=task.name, status="grounded", numeric_deviation=deviation
        )

    reasons: list[str] = []
    if numeric_fail:
        reasons.append(
            f"claims {task.engineer_days}d but the cited evidence supports ~{anchor_days:.0f}d"
        )
    if verdict is not None and judge_fail:
        reasons.append(verdict.reason)
    return LineGate(
        module=module_name,
        component=task.name,
        status="degraded",
        numeric_deviation=deviation,
        reason="; ".join(reason for reason in reasons if reason),
    )


def _judge_system_prompt() -> str:
    return (
        "You are a strict estimation auditor. For each estimate line you are given the claimed "
        "effort in engineer-days and the VERBATIM evidence it cites from a historical budget. "
        "Decide, for each line, whether that evidence genuinely supports the claimed effort and "
        "scope.\n"
        "- entailed=true only when the cited evidence names the same work and its hours are "
        "consistent with the claim (a line may be a fraction of a larger component; that is "
        "fine).\n"
        "- entailed=false when the number is not supported by the evidence, e.g. the line claims "
        "far MORE than the source records, or the evidence is about different work. Default to "
        "false when in doubt.\n"
        "Echo back each line's module and component exactly so verdicts can be matched."
    )


def _judge_user_message(grounded: list[tuple[str, TaskItem]]) -> str:
    blocks = []
    for module_name, task in grounded:
        evidence = " | ".join(f'"{ref.evidence}"' for ref in task.sources)
        blocks.append(
            f"<line>\n"
            f"module: {module_name}\n"
            f"component: {task.name}\n"
            f"claimed_engineer_days: {task.engineer_days}\n"
            f"cited_evidence: {evidence}\n"
            f"</line>"
        )
    return (
        "Audit each of the following estimate lines. Return one verdict per line.\n\n"
        + "\n".join(blocks)
    )


async def judge_estimate(
    estimate: Estimate,
    *,
    llm: LLMWrapper,
    model: str,
    max_tokens: int | None = None,
) -> dict[tuple[str, str], LineVerdict]:
    """Run the strict judge over every grounded line in ONE batched LLM call.

    Returns ``{(module, component): LineVerdict}``; empty when there are no grounded lines. A judge
    failure degrades to an empty map (the gate then relies on the numeric anchor alone), so an
    unavailable judge never sinks an otherwise valid estimate.
    """
    grounded = [
        (module.name, task)
        for module in estimate.modules
        for task in module.tasks
        if task.grounded and task.sources
    ]
    if not grounded:
        return {}

    try:
        panel, _meta = await asyncio.to_thread(
            llm.complete_structured,
            system_prompt=_judge_system_prompt(),
            user_message=_judge_user_message(grounded),
            response_model=JudgePanel,
            model=model,
            max_tokens=max_tokens,
            purpose="hallucination_judge",
        )
    except Exception as exc:  # noqa: BLE001 - a judge failure must not sink the gate
        log.warning("hallucination_judge_failed", error=str(exc)[:200])
        return {}

    return {(v.module, v.component): v for v in panel.verdicts}


async def gate_estimate(
    estimate: Estimate,
    chunks: list[RetrievedChunk],
    *,
    llm: LLMWrapper,
    tolerance: float,
    judge_model: str,
    use_judge: bool = True,
    max_tokens: int | None = None,
) -> HallucinationReport:
    """Grade every line of ``estimate`` grounded / degraded / insufficient.

    ``use_judge=False`` is the anchor-only path (no LLM, no cost).
    """
    chunks_by_id = {str(chunk.id): chunk for chunk in chunks}
    verdicts = (
        await judge_estimate(estimate, llm=llm, model=judge_model, max_tokens=max_tokens)
        if use_judge
        else {}
    )

    lines = [
        gate_line(
            task,
            module.name,
            numeric_anchor(task, chunks_by_id),
            verdicts.get((module.name, task.name)),
            tolerance=tolerance,
        )
        for module in estimate.modules
        for task in module.tasks
    ]
    return HallucinationReport(
        total_lines=len(lines),
        grounded_lines=sum(line.status == "grounded" for line in lines),
        degraded_lines=sum(line.status == "degraded" for line in lines),
        insufficient_lines=sum(line.status == "insufficient" for line in lines),
        lines=lines,
    )
