"""Temporal decay (soft recency re-weighting) and the sigmoid that makes it safe over logits."""

from datetime import date

import pytest

from app.generation.rag.retrieval.temporal import apply_temporal_decay, decay_weight, sigmoid
from app.generation.rag.schemas import RetrievedChunk


def chunk(chunk_id: int, document_date: date | None) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=f"chunk-{chunk_id}",
        chunk_type="budget_component",
        distance=0.2,
        document_date=document_date,
    )


def test_decay_weight_halves_after_one_half_life() -> None:
    assert decay_weight(900, 900) == 0.5
    assert decay_weight(1800, 900) == 0.25


def test_decay_weight_guards() -> None:
    assert decay_weight(0, 900) == 1.0  # fresh
    assert decay_weight(-10, 900) == 1.0  # future date
    assert decay_weight(900, 0) == 1.0  # decay disabled (non-positive half-life)


def test_recent_chunk_overtakes_a_slightly_more_relevant_stale_one() -> None:
    old_but_relevant = chunk(1, date(2018, 1, 1))
    recent = chunk(2, date(2023, 12, 1))

    out = apply_temporal_decay(
        [(old_but_relevant, 1.0), (recent, 0.9)], half_life_days=900, reference_date=date(2024, 1, 1)
    )

    assert [c.id for c, _ in out] == [2, 1]


def test_chunks_without_a_date_keep_full_weight() -> None:
    out = apply_temporal_decay(
        [(chunk(1, date(2016, 1, 1)), 0.8), (chunk(2, None), 0.8)],
        half_life_days=900,
        reference_date=date(2024, 1, 1),
    )

    assert out[0][0].id == 2
    assert out[0][1] == pytest.approx(0.8)


def test_without_aging_the_relevance_order_is_untouched() -> None:
    same_day = date(2024, 1, 1)
    out = apply_temporal_decay(
        [(chunk(1, same_day), 0.9), (chunk(2, same_day), 0.5)],
        half_life_days=900,
        reference_date=same_day,
    )

    assert [c.id for c, _ in out] == [1, 2]


def test_empty_input() -> None:
    assert apply_temporal_decay([], half_life_days=900, reference_date=date(2024, 1, 1)) == []


def test_sigmoid_is_non_negative_monotonic_and_overflow_safe() -> None:
    logits = [-800.0, -5.0, -0.1, 0.0, 0.1, 5.0, 800.0]
    scores = [sigmoid(logit) for logit in logits]

    assert scores == sorted(scores)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert sigmoid(0.0) == 0.5


def test_decay_over_a_negative_base_rewards_stale_chunks_but_over_sigmoid_it_penalises_them() -> None:
    # Why the pipeline applies sigmoid first: a multiplier in (0, 1) moves a negative base UP
    today = date(2024, 1, 1)
    stale, fresh = chunk(1, date(2016, 1, 1)), chunk(2, date(2024, 1, 1))

    raw = dict_by_id(apply_temporal_decay([(stale, -1.0), (fresh, -2.0)], half_life_days=900, reference_date=today))
    squashed = dict_by_id(
        apply_temporal_decay(
            [(stale, sigmoid(-1.0)), (fresh, sigmoid(-2.0))], half_life_days=900, reference_date=today
        )
    )

    assert raw[1] > -1.0  # the stale chunk got BETTER after "decay": the bug the sigmoid avoids
    assert squashed[1] < sigmoid(-1.0)  # penalised, as decay intends
    assert squashed[2] == pytest.approx(sigmoid(-2.0))  # a same-day chunk is untouched


def dict_by_id(scored: list[tuple[RetrievedChunk, float]]) -> dict[int, float]:
    return {c.id: score for c, score in scored}
