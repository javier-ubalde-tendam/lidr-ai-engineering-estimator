"""Session 11 - two-stage synthesis of the hours of a task from its historical neighbours.

A weighted consensus of the nearest historical tasks is right when they agree and misleading when
they do not: one analog says 40h and another 90h, the 65h midpoint reads as a confident point
estimate while hiding a 2x disagreement. Synthesis makes the disagreement first-class:

* Stage 1, DETERMINISTIC: the consensus and its dispersion (coefficient of variation) come from
  the caller; this module only decides whether the dispersion crosses the contradiction threshold.
* Stage 2, JUDGEMENT: only on contradiction, emit an ``HourRange`` (low..high) with a reason. An
  optional cheap LLM phrases the reason; a deterministic fallback (min/max of the neighbours and a
  templated reason) runs when the LLM is disabled or fails, so the range is always produced.

The point estimate is left untouched (the human still edits one number); the range travels
alongside it as the honest-uncertainty signal. The per-task hours pipeline that feeds it is
``generation/rag/task_hours.py`` (the only caller, deterministic path: no LLM wording).
"""

import asyncio

import structlog
from pydantic import BaseModel, Field

from app.foundation.llm.wrapper import LLMWrapper
from app.generation.rag.schemas import HourRange, TaskNeighbor

log = structlog.get_logger()


class _Reason(BaseModel):
    reason: str = Field(description="One-sentence explanation of the hour spread.")


def is_contradiction(dispersion: float | None, threshold: float) -> bool:
    """True when the neighbour spread crosses the contradiction threshold."""
    return dispersion is not None and dispersion > threshold


def _deterministic_range(neighbors: list[TaskNeighbor]) -> HourRange:
    """Fallback range: min/max of the neighbour hours, with the conflict named."""
    hours = sorted(neighbor.estimated_hours for neighbor in neighbors)
    low, high = hours[0], hours[-1]
    return HourRange(
        low=low,
        high=high,
        reason=(
            f"historical sources disagree ({low}h vs {high}h across {len(neighbors)} analogs); "
            "estimate depends on scope not yet pinned down"
        ),
    )


def _reason_system_prompt() -> str:
    return (
        "You are an estimation analyst. The historical analogs for a task disagree on effort. "
        "Given the low and high hour figures and the analog snippets, write ONE short sentence "
        "explaining the most likely reason for the spread (e.g. a scope or integration "
        "difference), so a human can resolve it. Do not change the numbers; only explain the "
        "disagreement."
    )


async def synthesize_range(
    neighbors: list[TaskNeighbor],
    dispersion: float | None,
    *,
    threshold: float,
    llm: LLMWrapper | None = None,
    model: str | None = None,
) -> HourRange | None:
    """Return an ``HourRange`` when the neighbours contradict, else ``None``.

    Deterministic by default: the range is the neighbour min/max with a templated reason. When
    both ``llm`` and ``model`` are given a cheap model phrases the reason; any failure degrades to
    the deterministic reason.
    """
    if not neighbors or not is_contradiction(dispersion, threshold):
        return None

    base = _deterministic_range(neighbors)
    if llm is None or not model:
        return base

    snippets = "; ".join(f"{n.estimated_hours}h (dist {n.distance:.2f})" for n in neighbors)
    user_message = (
        f"Task analogs range from {base.low}h to {base.high}h. Analogs: {snippets}.\n"
        "Explain the disagreement in one sentence."
    )
    try:
        reason, _meta = await asyncio.to_thread(
            llm.complete_structured,
            system_prompt=_reason_system_prompt(),
            user_message=user_message,
            response_model=_Reason,
            model=model,
            purpose="synthesis_reason",
        )
    except Exception as exc:  # noqa: BLE001 - the deterministic range is the floor
        log.warning("synthesis_reason_failed", error=str(exc)[:200])
        return base
    return HourRange(low=base.low, high=base.high, reason=reason.reason)
