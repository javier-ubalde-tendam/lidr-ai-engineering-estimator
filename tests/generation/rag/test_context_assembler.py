"""Context assembly: <source> XML blocks and the whole-chunk token budget."""

from app.generation.rag.context_assembler import build_context_block, truncate_to_token_budget
from app.generation.rag.schemas import RetrievedChunk


class CharEncoder:
    """1 character = 1 token: deterministic budgets without downloading tiktoken data."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text)


def chunk(chunk_id: int, content: str = "Payments module: 120 hours", **kwargs: object) -> RetrievedChunk:
    defaults: dict[str, object] = {
        "sector": "finance",
        "project_year": 2023,
        "chunk_type": "budget_component",
        "distance": 0.1234567,
        "budget_id": "BUD-2023-001",
    }
    return RetrievedChunk(id=chunk_id, content=content, **{**defaults, **kwargs})  # type: ignore[arg-type]


def test_each_chunk_is_wrapped_in_a_source_element_with_its_citable_ids() -> None:
    block = build_context_block([chunk(7)])

    assert block == (
        '<source id="7" document_id="BUD-2023-001" sector="finance" project_year="2023" '
        'chunk_type="budget_component" distance="0.1235">\n'
        "Payments module: 120 hours\n"
        "</source>"
    )


def test_chunks_without_a_budget_id_get_an_unknown_document_id() -> None:
    assert 'document_id="unknown"' in build_context_block([chunk(1, budget_id=None)])


def test_order_is_preserved_and_empty_input_is_an_empty_block() -> None:
    block = build_context_block([chunk(2), chunk(1)])

    assert block.index('id="2"') < block.index('id="1"')
    assert build_context_block([]) == ""


def test_truncate_keeps_whole_leading_chunks_within_the_budget() -> None:
    chunks = [chunk(1), chunk(2), chunk(3)]
    one_chunk_cost = len(build_context_block([chunks[0]]))

    kept = truncate_to_token_budget(chunks, one_chunk_cost * 2 + 5, CharEncoder())

    assert [c.id for c in kept] == [1, 2]


def test_truncate_counts_the_xml_wrapper_not_just_the_content() -> None:
    chunks = [chunk(1)]
    content_only = len(chunks[0].content)

    assert truncate_to_token_budget(chunks, content_only, CharEncoder()) == []


def test_truncate_stops_at_the_first_overflow_even_if_a_later_chunk_would_fit() -> None:
    small, big = chunk(1, "x"), chunk(2, "y" * 500)
    small_cost = len(build_context_block([small]))

    kept = truncate_to_token_budget([small, big, small], small_cost + 10, CharEncoder())

    assert [c.id for c in kept] == [1]


def test_zero_budget_keeps_nothing() -> None:
    assert truncate_to_token_budget([chunk(1)], 0, CharEncoder()) == []
