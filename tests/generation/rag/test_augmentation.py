"""Context augmentation passes: extractive compression and edge-loading reorder."""

from app.generation.rag.quality.augmentation import (
    augment_chunks,
    compress_chunk,
    extract_key_points,
    reorder_edge_loaded,
)
from app.generation.rag.schemas import RetrievedChunk


def make_chunk(chunk_id: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id, content=content, sector="ecommerce", chunk_type="budget_component", distance=0.1
    )


def test_extract_key_points_keeps_figures_and_ids_and_drops_filler() -> None:
    chunk = make_chunk(1, "Intro prose with no data\nAUTH-001 :: OAuth backend - 120 h\nmore filler")

    key_points = extract_key_points(chunk)

    assert key_points == "AUTH-001 :: OAuth backend - 120 h"


def test_extract_key_points_never_empties_a_chunk() -> None:
    assert extract_key_points(make_chunk(1, "\n  just some prose \nsecond line")) == "just some prose"
    assert extract_key_points(make_chunk(1, "")) == ""


def test_compress_chunk_returns_a_copy_and_keeps_the_id() -> None:
    chunk = make_chunk(7, "filler\nPAY :: Stripe - 80 h")

    compressed = compress_chunk(chunk)

    assert compressed.id == 7 and compressed.content == "PAY :: Stripe - 80 h"
    assert chunk.content.startswith("filler")  # the input is not mutated


def test_reorder_edge_loaded_puts_the_strongest_at_both_ends() -> None:
    chunks = [make_chunk(i, f"c{i}") for i in range(5)]  # best first: 0..4

    assert [c.id for c in reorder_edge_loaded(chunks)] == [0, 2, 4, 3, 1]


def test_reorder_edge_loaded_handles_short_lists() -> None:
    assert reorder_edge_loaded([]) == []
    assert [c.id for c in reorder_edge_loaded([make_chunk(1, "a")])] == [1]
    assert [c.id for c in reorder_edge_loaded([make_chunk(1, "a"), make_chunk(2, "b")])] == [1, 2]


def test_augment_chunks_applies_each_pass_independently() -> None:
    chunks = [make_chunk(i, f"ITEM-{i} :: work - {i * 10} h\nfiller") for i in range(4)]

    both = augment_chunks(chunks)
    only_compress = augment_chunks(chunks, reorder=False)
    only_reorder = augment_chunks(chunks, compress=False)
    neither = augment_chunks(chunks, compress=False, reorder=False)

    assert sorted(c.id for c in both) == [0, 1, 2, 3]
    assert all("filler" not in c.content for c in both)
    assert [c.id for c in both] == [0, 2, 3, 1]
    assert [c.id for c in only_compress] == [0, 1, 2, 3]
    assert all("filler" not in c.content for c in only_compress)
    assert [c.id for c in only_reorder] == [0, 2, 3, 1]
    assert all("filler" in c.content for c in only_reorder)
    assert neither == chunks and neither is not chunks
