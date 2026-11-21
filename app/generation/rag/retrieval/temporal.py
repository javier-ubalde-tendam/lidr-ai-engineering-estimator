"""Temporal decay: down-weight stale finalists (session 10).

An embedding encodes what a chunk SAYS, not when it was written or whether it is still true: the
2019 AngularJS budget can be semantically perfect and operationally dangerous. Temporal decay is
the SOFT counterpart of a hard date filter: instead of excluding old chunks it multiplies their
relevance by an exponential decay, so recency breaks ties without silencing history.

``weight = 0.5 ** (age_days / half_life_days)``: a chunk exactly one half-life old keeps half its
score. It is applied LAST, on the finalists and AFTER reranking: the cross-encoder decides
relevance, decay only adjusts the final order. Applied earlier it would distort the candidate set
the reranker sees.

Precondition: the base score must be NON-NEGATIVE. Decay is a 0..1 multiplier, so a negative base
would move UP toward zero and invert the order. Cross-encoder logits can be negative, which is why
``sigmoid`` exists here: the advanced pipeline maps logits through it before decaying.

Pure and synchronous (no I/O); the reference date is injected so tests are deterministic.
"""

import math
from datetime import date

from app.generation.rag.schemas import RetrievedChunk


def sigmoid(value: float) -> float:
    """Map a cross-encoder logit to a non-negative (0, 1) score, preserving the order."""
    # Two branches keep exp() from overflowing for large negative logits
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def decay_weight(age_days: float, half_life_days: float) -> float:
    """Exponential decay multiplier in ``(0, 1]``.

    A non-positive ``half_life_days`` disables decay (weight 1.0); a chunk dated in the future or
    with no age yields 1.0 (no penalty for being fresh).
    """
    if half_life_days <= 0 or age_days <= 0:
        return 1.0
    return float(0.5 ** (age_days / half_life_days))


def apply_temporal_decay(
    scored_chunks: list[tuple[RetrievedChunk, float]],
    *,
    half_life_days: float,
    reference_date: date,
) -> list[tuple[RetrievedChunk, float]]:
    """Re-weight ``(chunk, base_score)`` finalists by recency and re-sort best first.

    Chunks without a ``document_date`` keep weight 1.0. Returns ``(chunk, decayed_score)`` pairs.
    """
    weighted: list[tuple[RetrievedChunk, float]] = []
    for chunk, base in scored_chunks:
        if chunk.document_date is None:
            weight = 1.0
        else:
            weight = decay_weight((reference_date - chunk.document_date).days, half_life_days)
        weighted.append((chunk, base * weight))

    # sort() is stable: equal scores keep their incoming (relevance) order
    weighted.sort(key=lambda pair: pair[1], reverse=True)
    return weighted
