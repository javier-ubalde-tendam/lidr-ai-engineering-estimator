"""Sentence-window chunker.

Indexes individual sentences (small, precise units, high recall) but carries a
+-N-sentence window in ``metadata.window_text``. The bare sentence is what gets
embedded; at retrieval time the window is what you would feed the generator, so
the LLM still sees surrounding context. The contrast to embed is that the
indexed unit and the returned unit are deliberately *different*.

The default sentence splitter is NLTK's ``punkt``, whose data is downloaded on first use
(network needed once). Tests and offline runs inject their own ``split_sentences``.
"""

import time
from collections.abc import Callable

from app.generation.rag.chunking.base import Chunker, count_tokens, emit_chunking_done
from app.generation.rag.chunking.structural import component_metadata
from app.generation.rag.schemas import Budget, Chunk

# Sentences of context kept on each side of the indexed sentence.
WINDOW_SIZE = 2


def nltk_sentence_splitter(text: str) -> list[str]:
    """Split ``text`` into sentences with NLTK, fetching the punkt data lazily."""
    import nltk  # lazy: importing the module must not require NLTK or its data

    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            break
        except LookupError:
            continue
    else:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
    return list(nltk.sent_tokenize(text))


class SentenceWindowChunker(Chunker):
    strategy_name = "sentence_window"

    def __init__(
        self, split_sentences: Callable[[str], list[str]] = nltk_sentence_splitter
    ) -> None:
        self._split_sentences = split_sentences

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                sentences = self._split_sentences(component.description)
                for i, sentence in enumerate(sentences):
                    lo = max(0, i - WINDOW_SIZE)
                    hi = min(len(sentences), i + WINDOW_SIZE + 1)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{budget.budget_id}::{component.component_id}::s{i}",
                            text=sentence,
                            metadata={
                                **component_metadata(budget, component),
                                "sentence_index": i,
                                "window_text": " ".join(sentences[lo:hi]),
                            },
                            token_count=count_tokens(sentence),
                        )
                    )
        self.last_extra_api_calls = 0
        self.last_extra_cost_usd = 0.0
        emit_chunking_done(
            strategy=self.strategy_name,
            chunks=chunks,
            n_input_documents=len(budgets),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return chunks
