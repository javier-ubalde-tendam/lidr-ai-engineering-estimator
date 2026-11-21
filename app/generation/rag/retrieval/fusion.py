"""Rank fusion: RRF (consensus) and round-robin (coverage).

RRF fuses by POSITION, not by score. The two branches produce scores that are not comparable
(cosine distance, lower = better, vs. ``ts_rank_cd``, higher = better and unbounded), and
normalising them is fragile and corpus-dependent; the order each branch produces is robust.
"""

from collections.abc import Callable, Hashable
from itertools import zip_longest
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[int]], *, k: int = DEFAULT_RRF_K
) -> list[tuple[int, float]]:
    """Fuse id rankings (each best first) into ``[(id, score)]`` sorted by score descending.

    score(id) = sum over rankings of ``1 / (k + rank)``, with 0-based rank. ``k`` smooths the
    curve: a large ``k`` flattens it (a chunk must rank well in BOTH branches to win); a small
    ``k`` lets a single #1 dominate. Duplicates inside one ranking count once (first position).
    Ties are broken by ascending id so the output is deterministic.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, chunk_id in enumerate(ranking):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            # dict.get(key, default) ~ Map.getOrDefault in Java
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _identity(item: Any) -> Hashable:
    return item  # type: ignore[no-any-return]


def round_robin_merge(
    ranked_lists: list[list[T]], *, key: Callable[[T], Hashable] = _identity
) -> list[T]:
    """Interleave several ranked lists position by position, dropping duplicates.

    Use it for query DECOMPOSITION (each sub-query owns a different topic) and to merge
    collections. RRF rewards consensus, which is wrong there: a topic covered by a single list
    must still be represented. Round-robin gives every list's #1 a slot before any list's #2, so
    coverage wins over agreement. The first occurrence of an item (by ``key``) is kept.

    ``key`` is the dedup identity. Chunk ids are only unique WITHIN a collection (each table has
    its own sequence), so across collections pass ``lambda c: (c.collection, c.id)``.
    """
    seen: set[Hashable] = set()
    merged: list[T] = []
    for column in zip_longest(*ranked_lists):
        for item in column:
            # zip_longest pads the shorter lists with None
            if item is None:
                continue
            identity = key(item)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(item)
    return merged
