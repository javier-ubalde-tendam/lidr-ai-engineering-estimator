"""Pure metrics for the session 10 retrieval evaluation (no I/O)."""

import math
from collections.abc import Sequence


def precision_at_k(
    retrieved_budget_ids: Sequence[str | None], relevant_budget_ids: set[str], k: int = 5
) -> float:
    """Share of the first ``k`` chunks whose parent budget is relevant. Always divides by ``k``.

    Missing positions (fewer than k results) count as misses. Limitation: several chunks of the
    same budget can fill the top-k, so this measures chunk relevance, not budget diversity.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    hits = sum(1 for budget_id in retrieved_budget_ids[:k] if budget_id in relevant_budget_ids)
    return hits / k


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (pct in (0, 100]): always one of the measured values."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 < pct <= 100:
        raise ValueError(f"pct must be in (0, 100], got {pct}")
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[rank - 1]
