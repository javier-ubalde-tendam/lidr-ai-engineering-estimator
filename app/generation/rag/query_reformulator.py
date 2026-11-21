"""Query understanding: turn a raw meeting transcript into a structured brief.

A transcript is not a query: embedding ~600 tokens of digression gives a vector that averages
several intents plus conversational noise, with compressed distances and irrelevant hits. This
stage distills the transcript into an ``EstimationQuery`` and then composes a short technical
English search string aligned with the corpus.

Both the structured extraction and the degraded fallback go through the injected ``LLMWrapper``
(retries, cost tracking and structured-output re-prompting included).
"""

import asyncio

import structlog
from pydantic import BaseModel, Field

from app.foundation.llm.wrapper import LLMWrapper
from app.generation.rag.errors import ReformulationError
from app.generation.rag.schemas import EstimationQuery

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are a software-delivery analyst. Extract a structured project brief from "
    "a raw, messy client meeting transcript. Capture ONLY what the client wants to "
    "build and the constraints that bound it; ignore small talk, anecdotes and "
    "digressions. Normalise everything to concise technical English regardless of "
    "the transcript language. Leave a field empty/unknown when the transcript gives "
    "no evidence for it - never invent technologies, sectors or regulations."
)

_FALLBACK_SYSTEM_PROMPT = (
    "Rewrite the following client meeting transcript as a single short technical "
    "search query in English describing the software to build. One sentence, no "
    "preamble."
)


class _RewrittenQuery(BaseModel):
    """Free-text fallback output, wrapped in a schema because the wrapper is structured-only."""

    query: str = Field(description="One-sentence technical search query in English.")


async def reformulate_query(
    transcript: str, *, llm: LLMWrapper, model: str, max_tokens: int
) -> EstimationQuery:
    """Distill a transcript into a structured ``EstimationQuery``.

    ``max_tokens`` must be generous: the reformulation model is also a reasoning model, whose
    reasoning tokens count against the ceiling and would truncate the JSON otherwise.

    On the primary path every field the model could ground is populated; on the degraded
    fallback only ``function`` is set. Raises ``ReformulationError`` if both paths fail.
    """
    try:
        query, _meta = await asyncio.to_thread(
            llm.complete_structured,
            system_prompt=_SYSTEM_PROMPT,
            user_message=transcript,
            response_model=EstimationQuery,
            model=model,
            max_tokens=max_tokens,
            purpose="rag_reformulation",
        )
        return query
    except Exception as exc:  # noqa: BLE001 - degrade to a simpler rewrite
        log.warning(
            "reformulation_fallback",
            reason="structured_extraction_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )

    try:
        rewritten, _meta = await asyncio.to_thread(
            llm.complete_structured,
            system_prompt=_FALLBACK_SYSTEM_PROMPT,
            user_message=transcript,
            response_model=_RewrittenQuery,
            model=model,
            max_tokens=max_tokens,
            purpose="rag_reformulation_fallback",
        )
    except Exception as exc:
        raise ReformulationError("Query reformulation failed.") from exc

    function = rewritten.query.strip()
    if not function:
        raise ReformulationError("Fallback rewrite returned an empty query.")
    return EstimationQuery(function=function)


def compose_search_text(query: EstimationQuery) -> str:
    """Compose the short technical string to embed from an ``EstimationQuery``.

    Joins the grounded fields into one corpus-aligned phrase, e.g. ``"B2B payments marketplace
    with Stripe Connect, KYC for healthcare in Germany BaFin-compliant"``. Empty fields are dropped.
    """
    parts: list[str] = [query.function.strip()] if query.function.strip() else []

    if query.technologies:
        parts.append("with " + ", ".join(query.technologies))
    if query.sector:
        parts.append(f"for {query.sector}")
    if query.country:
        parts.append(f"in {query.country}")
    if query.regulations:
        parts.append(", ".join(query.regulations) + "-compliant")

    return " ".join(parts).strip()
