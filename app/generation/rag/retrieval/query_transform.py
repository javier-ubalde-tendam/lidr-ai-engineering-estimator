"""Query expansion and decomposition (session 10).

Two DISTINCT techniques, not interchangeable; the distinguishing question is *one thing said many
ways, or many things said at once?*

* **Expansion (multi-query)**: N reformulations of the SAME intent. Fights the formulation
  lottery (the query's vocabulary deciding what is retrieved). Fused by consensus (RRF).
* **Decomposition**: independent sub-queries, one per TOPIC. Fights the averaged embedding of a
  multi-topic query ("near everything, near nothing"). Fused by coverage (round-robin).

Which one (if any) applies is a cheap heuristic on the query's length/structure; the decision is
logged. The LLM call uses structured output on a small model with "short-leash" instructions: keep
exact domain terms, do not invent requirements, fewer is better, at most 4 sub-queries (the cap
lives in the schema AND the prompt). On any failure it degrades to the original query.
"""

import asyncio
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from app.foundation.llm.wrapper import LLMWrapper

log = structlog.get_logger(__name__)

# Technique labels (they also select the fusion strategy downstream)
DIRECT = "direct"
EXPAND = "expand"
DECOMPOSE = "decompose"

MAX_SUBQUERIES = 4
TRANSFORM_MAX_TOKENS = 1500

# Heuristic thresholds, deliberately simple and tuned in the live session
_SHORT_WORDS = 6  # below this: pass through untouched
_LONG_WORDS = 25  # at/above this: treat as multi-topic
_MULTITOPIC_CONNECTORS = 2  # this many "and"/comma/semicolon joins mean multi-topic


class SubQuery(BaseModel):
    """One generated query plus the topic it covers (the topic helps logging and coverage)."""

    topic: str = Field(description="Short label of the angle/topic this query covers.")
    query: str = Field(description="A self-contained search query in English.")


class QueryTransformation(BaseModel):
    """Structured output: 1-4 sub-queries."""

    subqueries: list[SubQuery] = Field(min_length=1, max_length=MAX_SUBQUERIES)


@dataclass(frozen=True)
class QueryPlan:
    """The chosen technique and the sub-queries to run."""

    technique: str  # DIRECT | EXPAND | DECOMPOSE
    subqueries: list[SubQuery]


_EXPAND_SYSTEM_PROMPT = (
    "Rewrite the user's search query into several alternative formulations of the SAME intent, "
    "to beat vocabulary mismatch. Keep every exact domain term (product names, technologies, "
    "acronyms like Stripe, SAP, OAuth) verbatim. Do NOT add requirements the query does not "
    "state. Produce at most 4 variants; fewer is better. Each variant must stand on its own. "
    "All output in English."
)

_DECOMPOSE_SYSTEM_PROMPT = (
    "Break the user's query into independent sub-queries, ONE per distinct topic, so each can be "
    "searched on its own (the query bundles several topics whose average embedding is near "
    "nothing). Keep every exact domain term (Stripe, SAP, OAuth, etc.) verbatim. Do NOT invent "
    "requirements. At most 4 sub-queries; if there is really one topic, return a single clean "
    "sub-query. All output in English."
)


def choose_technique(query_text: str) -> str:
    """Pick a technique from the query's shape (length + topic connectors)."""
    words = len(query_text.split())
    if words < _SHORT_WORDS:
        return DIRECT

    lowered = query_text.lower()
    connectors = (
        lowered.count(" and ")
        + query_text.count(",")
        + query_text.count(";")
        + lowered.count(" plus ")
        + lowered.count(" also ")
    )
    sentences = sum(query_text.count(mark) for mark in ".!?")
    if words >= _LONG_WORDS or connectors >= _MULTITOPIC_CONNECTORS or sentences >= 2:
        return DECOMPOSE
    return EXPAND


def _direct_plan(query_text: str) -> QueryPlan:
    return QueryPlan(DIRECT, [SubQuery(topic="original", query=query_text)])


async def transform_query(
    query_text: str,
    *,
    llm: LLMWrapper,
    model: str,
    enabled: bool = True,
    max_subqueries: int = MAX_SUBQUERIES,
) -> QueryPlan:
    """Return the sub-queries to run for ``query_text`` and the technique used.

    On any failure (or when disabled) it degrades to a single direct query, so retrieval never
    depends on the transformer succeeding.
    """
    technique = choose_technique(query_text) if enabled else DIRECT
    if technique == DIRECT:
        log.info("query_transform_decided", technique=DIRECT, subqueries=1)
        return _direct_plan(query_text)

    system_prompt = _EXPAND_SYSTEM_PROMPT if technique == EXPAND else _DECOMPOSE_SYSTEM_PROMPT
    try:
        transformation, _meta = await asyncio.to_thread(
            llm.complete_structured,
            system_prompt=system_prompt,
            user_message=query_text,
            response_model=QueryTransformation,
            model=model,
            max_tokens=TRANSFORM_MAX_TOKENS,
            purpose="retrieval_query_transform",
        )
        subqueries = transformation.subqueries[: min(max_subqueries, MAX_SUBQUERIES)]
        if not subqueries:
            raise ValueError("transformer returned no sub-queries")
    except Exception as exc:  # noqa: BLE001 - degrade to a single direct query
        log.warning(
            "query_transform_failed",
            technique=technique,
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return _direct_plan(query_text)

    log.info(
        "query_transform_decided",
        technique=technique,
        subqueries=len(subqueries),
        topics=[sq.topic for sq in subqueries],
    )
    return QueryPlan(technique, subqueries)
