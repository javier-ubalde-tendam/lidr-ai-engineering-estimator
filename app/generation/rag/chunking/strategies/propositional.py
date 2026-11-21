"""Propositional chunker.

An LLM decomposes each component into self-contained atomic propositions; each
proposition becomes its own chunk. Precise and great for fact-level retrieval,
but expensive: one LLM call per component. The cost comes from the call metadata
of the injected ``LLMWrapper`` and is reported as extra ingestion cost.
"""

import time

import structlog
from pydantic import BaseModel, Field

from app.foundation.llm.wrapper import LLMWrapper
from app.generation.rag.chunking.base import Chunker, count_tokens, emit_chunking_done
from app.generation.rag.chunking.structural import component_metadata, render_component_text
from app.generation.rag.schemas import Budget, Chunk

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "Decompose the following content into clear, simple, self-contained "
    "propositions. Each proposition must:\n"
    "1. Express a single atomic fact.\n"
    "2. Be understandable on its own, without the surrounding text.\n"
    "3. Resolve pronouns and references to the explicit entity they refer to "
    "(e.g. replace 'it' with the component name).\n"
    "Return the list of propositions."
)


class _Propositions(BaseModel):
    propositions: list[str] = Field(default_factory=list)


class PropositionalChunker(Chunker):
    strategy_name = "propositional"

    def __init__(self, llm: LLMWrapper, model: str) -> None:
        self._llm = llm
        self._model = model

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        calls = 0
        cost = 0.0
        for budget in budgets:
            for component in budget.components:
                text = render_component_text(budget, component)
                props, call_cost = self._decompose(text)
                calls += 1
                cost += call_cost
                if not props:
                    # Fallback: keep the component as a single chunk so a failed
                    # decomposition never drops content silently.
                    props = [text]
                for j, prop in enumerate(props):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{budget.budget_id}::{component.component_id}::prop{j}",
                            text=prop,
                            metadata={**component_metadata(budget, component), "proposition": j},
                            token_count=count_tokens(prop),
                        )
                    )
        self.last_extra_api_calls = calls
        self.last_extra_cost_usd = cost
        emit_chunking_done(
            strategy=self.strategy_name,
            chunks=chunks,
            n_input_documents=len(budgets),
            extra_api_calls=calls,
            extra_cost_usd=cost,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return chunks

    def _decompose(self, text: str) -> tuple[list[str], float]:
        """Return (propositions, cost_usd). Logs and degrades on any LLM error."""
        try:
            result, meta = self._llm.complete_structured(
                system_prompt=_SYSTEM_PROMPT,
                user_message=text,
                response_model=_Propositions,
                model=self._model,
                purpose="chunking_propositional",
            )
        except Exception as exc:  # noqa: BLE001 - one failed component must not abort the corpus
            log.warning(
                "propositional_call_failed", error_type=type(exc).__name__, error=str(exc)[:200]
            )
            return [], 0.0
        return [p for p in result.propositions if p.strip()], meta.cost_usd
