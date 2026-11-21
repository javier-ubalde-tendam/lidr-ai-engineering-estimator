"""Multi-index routing in a cascade of increasing cost (session 10).

The corpus is heterogeneous, so searching every collection on every query lets the dominant type
flood the top-k. The router decides WHICH collections to search, resolving in order; the cheapest
signal that suffices wins:

0. **Explicit**: the caller named the collections in the request. The best router is no router.
1. **Rules**: deterministic vocabulary patterns (``collections.match_rules``). A query saying
   "budget"/"cost"/"hours" obviously wants budgets; no LLM needed.
2. **Classifier**: an LLM with structured output emits 1-3 targets plus a one-sentence reason.
   Ambiguity is modelled as MULTIPLE targets, never as a confidence threshold.
3. **Fallback**: search every collection. Graceful degradation, never worse than a single index.

Every decision is logged with its level and reason. The LLM wrapper and the model name are
injected by the caller: generation code never reaches for the composition root.
"""

import asyncio
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from app.foundation.llm.wrapper import LLMWrapper
from app.generation.rag.retrieval.collections import (
    ALL_COLLECTIONS,
    Collection,
    match_rules,
)

log = structlog.get_logger(__name__)

CLASSIFIER_MAX_TOKENS = 1000


class RouteClassification(BaseModel):
    """Structured output of the LLM router (closed target set, 1-3 destinations)."""

    targets: list[Collection] = Field(
        min_length=1,
        max_length=3,
        description="Which collections to search. One per genuinely relevant source type; "
        "model ambiguity as several targets, not a maybe.",
    )
    reason: str = Field(description="One sentence justifying the routing choice.")


@dataclass(frozen=True)
class RoutingDecision:
    """Where the query is routed and how that was decided."""

    targets: list[Collection]
    level: str  # "explicit" | "rules" | "classifier" | "fallback"
    reason: str


_CLASSIFIER_SYSTEM_PROMPT = (
    "You route a search query to the document collections most likely to answer it. The "
    "collections are:\n"
    "- budget: historical project budgets/quotes with components, hours and costs.\n"
    "- transcript: client meeting transcripts (what was discussed and decided).\n"
    "- technical_doc: internal technical documentation, specs and API references.\n"
    "Return between 1 and 3 targets. Pick MORE than one only when the query genuinely spans "
    "several source types; ambiguity is several targets, not a guess. Always add a one-sentence "
    "reason. Never invent collections."
)


async def route(
    query_text: str,
    *,
    llm: LLMWrapper,
    model: str,
    explicit: list[Collection] | None = None,
    rules_enabled: bool = True,
    classifier_enabled: bool = True,
    max_targets: int = 3,
) -> RoutingDecision:
    """Resolve the target collections through the cascade described in the module docstring."""
    if explicit:
        targets = list(dict.fromkeys(explicit))[:max_targets]
        log.info("routing_decided", level="explicit", targets=[t.value for t in targets])
        return RoutingDecision(targets, "explicit", "Caller specified collections.")

    if rules_enabled:
        hits = match_rules(query_text)
        if hits:
            targets = hits[:max_targets]
            reason = f"Vocabulary matched collections: {', '.join(t.value for t in targets)}."
            log.info("routing_decided", level="rules", targets=[t.value for t in targets])
            return RoutingDecision(targets, "rules", reason)

    if classifier_enabled:
        try:
            classification = await _classify(query_text, llm=llm, model=model)
        except Exception as exc:  # noqa: BLE001 - a classifier failure degrades to the fallback
            log.warning(
                "routing_classifier_failed", error_type=type(exc).__name__, error=str(exc)[:200]
            )
        else:
            targets = list(dict.fromkeys(classification.targets))[:max_targets]
            log.info(
                "routing_decided",
                level="classifier",
                targets=[t.value for t in targets],
                reason=classification.reason,
            )
            return RoutingDecision(targets, "classifier", classification.reason)

    targets = list(ALL_COLLECTIONS)
    log.info("routing_decided", level="fallback", targets=[t.value for t in targets])
    return RoutingDecision(
        targets, "fallback", "No rule or classifier signal; searching all collections."
    )


async def _classify(query_text: str, *, llm: LLMWrapper, model: str) -> RouteClassification:
    # The wrapper is synchronous (blocking HTTP): keep it off the event loop
    classification, _meta = await asyncio.to_thread(
        llm.complete_structured,
        system_prompt=_CLASSIFIER_SYSTEM_PROMPT,
        user_message=query_text,
        response_model=RouteClassification,
        model=model,
        max_tokens=CLASSIFIER_MAX_TOKENS,
        purpose="retrieval_routing",
    )
    return classification
