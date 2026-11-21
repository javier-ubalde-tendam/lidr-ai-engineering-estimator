"""Round-robin merge (the decomposition fusion): coverage over consensus. My RRF has its own tests."""

from app.generation.rag.retrieval.fusion import round_robin_merge


def test_interleaves_lists_position_by_position() -> None:
    assert round_robin_merge([[1, 2, 3], [4, 5, 6]]) == [1, 4, 2, 5, 3, 6]


def test_dedupes_keeping_the_first_occurrence() -> None:
    assert round_robin_merge([[1, 2], [1, 3]]) == [1, 2, 3]


def test_uneven_lists_drain_the_longer_one() -> None:
    assert round_robin_merge([[1], [2, 3, 4]]) == [1, 2, 3, 4]


def test_coverage_beats_consensus() -> None:
    # The niche list's only item gets a slot before the popular list's second one
    assert round_robin_merge([[10, 11, 12, 13], [99]])[:2] == [10, 99]


def test_composite_key_keeps_equal_ids_of_different_collections() -> None:
    budgets = [("budget", 1), ("budget", 2)]
    transcripts = [("transcript", 1)]

    merged = round_robin_merge([budgets, transcripts], key=lambda pair: pair)

    assert merged == [("budget", 1), ("transcript", 1), ("budget", 2)]


def test_without_a_composite_key_colliding_ids_are_dropped() -> None:
    # Documents WHY the pipeline must pass (collection, id): the plain id loses a chunk
    assert round_robin_merge([[1, 2], [1]]) == [1, 2]


def test_empty_input() -> None:
    assert round_robin_merge([]) == []
    assert round_robin_merge([[], []]) == []
