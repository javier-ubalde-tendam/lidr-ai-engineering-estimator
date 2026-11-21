"""Reciprocal Rank Fusion (RRF): merge independent rankings into a single one.

RRF fuses by POSITION, not by score. The two branches produce scores that are not comparable
(cosine distance, lower = better, vs. ``ts_rank_cd``, higher = better and unbounded), and
normalising them is fragile and corpus-dependent; the order each branch produces is robust.
"""

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
