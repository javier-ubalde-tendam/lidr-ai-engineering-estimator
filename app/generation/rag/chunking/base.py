"""Common interface for chunking strategies.

Every strategy implements :class:`Chunker` so they are interchangeable in the
comparison framework. The shared helpers keep token counting and the
``chunking_done`` log event identical across strategies.
"""

from abc import ABC, abstractmethod
from functools import lru_cache

import structlog
import tiktoken

from app.generation.rag.schemas import Budget, Chunk

log = structlog.get_logger()


@lru_cache
def get_encoding() -> tiktoken.Encoding:
    # All strategies count tokens against the embedding model's tokenizer (cl100k_base is what
    # text-embedding-3-small uses) so the stats are comparable. Loaded lazily: importing this
    # module must not touch the BPE vocabulary file.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Token count of ``text`` using the embedding model's encoding."""
    return len(get_encoding().encode(text))


def emit_chunking_done(
    *,
    strategy: str,
    chunks: list[Chunk],
    n_input_documents: int,
    extra_api_calls: int = 0,
    extra_cost_usd: float = 0.0,
    latency_ms: float = 0.0,
) -> None:
    """Emit the standard ``chunking_done`` event (identical shape per strategy)."""
    log.info(
        "chunking_done",
        strategy=strategy,
        n_chunks=len(chunks),
        n_input_documents=n_input_documents,
        extra_api_calls=extra_api_calls,
        extra_cost_usd=round(extra_cost_usd, 6),
        latency_ms=round(latency_ms, 1),
    )


class Chunker(ABC):
    """A chunking strategy: budgets in, chunks out.

    Subclasses set ``last_extra_api_calls`` / ``last_extra_cost_usd`` after each
    ``chunk()`` call so the comparator can report ingestion cost. Strategies that
    do not call an external API leave them at zero.
    """

    last_extra_api_calls: int = 0
    last_extra_cost_usd: float = 0.0

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Stable identifier used in logs and in the comparison framework."""

    @abstractmethod
    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        """Produce the chunks for a list of budgets."""
