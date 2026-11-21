"""Similarity helpers and the chunking comparator (fake chunkers + fake embedder, no network)."""

import pytest

from app.generation.rag.analysis.comparison import (
    OBESE_MIN_TOKENS,
    ORPHAN_MAX_TOKENS,
    ChunkingComparator,
)
from app.generation.rag.analysis.similarity import cosine_similarity, percentile
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.schemas import Budget, Chunk, EmbeddedChunk
from tests.embedding_helpers import make_budget


def test_cosine_similarity_basics() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_percentile_interpolates_and_handles_edges() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([7], 95) == 7.0
    assert percentile([10, 20, 30, 40], 50) == pytest.approx(25.0)
    assert percentile([1, 2, 3, 4, 5], 100) == 5.0


class ScriptedChunker(Chunker):
    """Returns chunks with fixed token counts and counts how many times it ran."""

    def __init__(self, name: str, token_counts: list[int], cost: float = 0.0) -> None:
        self._name = name
        self._token_counts = token_counts
        self.last_extra_cost_usd = cost
        self.runs = 0

    @property
    def strategy_name(self) -> str:
        return self._name

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        self.runs += 1
        return [
            Chunk(chunk_id=f"{self._name}-{i}", text=f"{self._name} text {i}", metadata={}, token_count=n)
            for i, n in enumerate(self._token_counts)
        ]


class FakeEmbedder:
    """The query is [1, 0]; a chunk whose text ends in ``i`` is rotated by 0.1*i radians away."""

    def __init__(self) -> None:
        self.embedded_batches = 0

    def embed_one(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        import math

        self.embedded_batches += 1
        return [
            EmbeddedChunk(
                **c.model_dump(),
                embedding=[math.cos(0.1 * int(c.text[-1])), math.sin(0.1 * int(c.text[-1]))],
            )
            for c in chunks
        ]


def test_stats_count_orphans_and_obese_chunks_and_report_cost() -> None:
    sizes = [ORPHAN_MAX_TOKENS - 1, 100, 200, OBESE_MIN_TOKENS + 1]
    comparator = ChunkingComparator([ScriptedChunker("a", sizes, cost=0.0123)], FakeEmbedder())  # type: ignore[arg-type]

    stats = comparator.compute_stats([make_budget()])["a"]

    assert stats.n_chunks == 4
    assert (stats.n_orphan_chunks, stats.n_obese_chunks) == (1, 1)
    assert stats.token_distribution.min == ORPHAN_MAX_TOKENS - 1
    assert stats.token_distribution.max == OBESE_MIN_TOKENS + 1
    assert stats.token_distribution.p50 == 150.0
    assert stats.ingestion_cost_usd == 0.0123


def test_stats_of_a_strategy_without_chunks_are_zero() -> None:
    comparator = ChunkingComparator([ScriptedChunker("empty", [])], FakeEmbedder())  # type: ignore[arg-type]

    stats = comparator.compute_stats([make_budget()])["empty"]

    assert (stats.n_chunks, stats.token_distribution.max, stats.token_distribution.p95) == (0, 0, 0.0)


def test_queries_rank_chunks_by_cosine_and_cut_at_top_k() -> None:
    comparator = ChunkingComparator([ScriptedChunker("a", [50, 50, 50, 50])], FakeEmbedder())  # type: ignore[arg-type]

    results = comparator.run_queries([make_budget()], ["q1"], top_k=2)

    top = results["a"][0].top_k
    assert [t.chunk_id for t in top] == ["a-0", "a-1"]
    assert top[0].cosine == 1.0
    assert top[0].cosine > top[1].cosine


def test_without_queries_nothing_is_embedded() -> None:
    embedder = FakeEmbedder()
    comparator = ChunkingComparator([ScriptedChunker("a", [50])], embedder)  # type: ignore[arg-type]

    assert comparator.run_queries([make_budget()], []) == {}
    assert embedder.embedded_batches == 0


def test_each_strategy_is_chunked_once_for_stats_and_queries() -> None:
    chunker = ScriptedChunker("a", [50, 60])
    comparator = ChunkingComparator([chunker], FakeEmbedder())  # type: ignore[arg-type]
    budgets = [make_budget()]

    comparator.compute_stats(budgets)
    comparator.run_queries(budgets, ["q"])

    # The paid strategies (propositional, contextual) must not be charged twice
    assert chunker.runs == 1
