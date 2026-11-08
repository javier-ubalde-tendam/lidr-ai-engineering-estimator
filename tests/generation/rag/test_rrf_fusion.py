import pytest

from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion


def test_hand_computed_example() -> None:
    # k=60, 0-based ranks. vector: [1, 2, 3]; lexical: [3, 1]
    #   1 -> 1/60 + 1/61 ; 2 -> 1/61 ; 3 -> 1/62 + 1/60
    fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60))

    assert fused[1] == pytest.approx(1 / 60 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 61)
    assert fused[3] == pytest.approx(1 / 62 + 1 / 60)
    assert [chunk_id for chunk_id, _ in reciprocal_rank_fusion([[1, 2, 3], [3, 1]])] == [1, 3, 2]


def test_single_ranking_keeps_its_order() -> None:
    assert [chunk_id for chunk_id, _ in reciprocal_rank_fusion([[5, 9, 2]])] == [5, 9, 2]


def test_empty_rankings() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    # An empty branch adds nothing: the other one decides
    assert [chunk_id for chunk_id, _ in reciprocal_rank_fusion([[4, 7], []])] == [4, 7]


def test_duplicates_within_a_ranking_count_once_at_first_position() -> None:
    fused = dict(reciprocal_rank_fusion([[1, 2, 1]], k=10))

    assert fused[1] == pytest.approx(1 / 10)
    assert fused[2] == pytest.approx(1 / 11)


def test_ties_break_by_ascending_id() -> None:
    # 7 and 3 are each first in one branch: identical scores
    assert [chunk_id for chunk_id, _ in reciprocal_rank_fusion([[7], [3]])] == [3, 7]


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_raises(k: int) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1]], k=k)


def test_k_controls_whether_a_single_top_hit_dominates() -> None:
    # 1 is #1 in the vector branch only; 2 is #3 in BOTH branches (consensus)
    rankings = [[1, 9, 2], [8, 7, 2]]

    small = dict(reciprocal_rank_fusion(rankings, k=1))
    large = dict(reciprocal_rank_fusion(rankings, k=60))

    # k=1: 1/1 = 1.0 beats 2 * 1/3 -> the single #1 dominates
    assert small[1] > small[2]
    # k=60: 1/60 < 2 * 1/62 -> ranking well in both branches wins
    assert large[2] > large[1]
