"""Citation verification, coherence guard and the line-level grounding contract of TaskItem."""

import pytest
from pydantic import ValidationError

from app.generation.rag.schemas import (
    Estimate,
    SourceCitation,
    SourceReference,
    TaskItem,
    WorkModule,
)
from app.generation.rag.validation import check_coherence, verify_citations


def ref(chunk_id: str) -> SourceReference:
    return SourceReference(chunk_id=chunk_id, document_id="BUD-1", evidence="Payments: 120 hours")


def grounded(name: str, *chunk_ids: str, days: int = 5) -> TaskItem:
    return TaskItem(
        name=name, engineer_days=days, grounded=True, sources=[ref(c) for c in chunk_ids]
    )


def estimate(*tasks: TaskItem, sources: list[SourceCitation] | None = None) -> Estimate:
    return Estimate(
        total_engineer_days=sum(t.engineer_days or 0 for t in tasks),
        modules=[WorkModule(name="Payments", tasks=list(tasks))],
        sources=sources or [],
        confidence="medium",
        reasoning="derived from sources",
    )


def test_all_citations_resolve_to_retrieved_chunks() -> None:
    report = verify_citations(estimate(grounded("Checkout", "1", "2")), {"1", "2"})

    assert (report.total_lines, report.grounded_lines, report.dangling_lines) == (1, 1, 0)
    assert report.verified_citations == 2
    assert not report.has_dangling and report.lines[0].status == "grounded"


def test_a_line_citing_an_unretrieved_chunk_is_dangling() -> None:
    report = verify_citations(estimate(grounded("Checkout", "1", "99")), {"1"})

    assert report.has_dangling and report.dangling_citations == ["99"]
    assert (report.grounded_lines, report.dangling_lines, report.verified_citations) == (0, 1, 1)
    assert report.lines[0].dangling_chunk_ids == ["99"]


def test_a_fabricated_estimate_global_citation_is_dangling_too() -> None:
    global_source = SourceCitation(source_id=42, relevance="primary", used_for="payments")

    report = verify_citations(estimate(grounded("Checkout", "1"), sources=[global_source]), {"1"})

    assert report.dangling_citations == ["42"]


def test_ungrounded_lines_are_reported_as_insufficient() -> None:
    report = verify_citations(estimate(TaskItem(name="Novel scope")), {"1"})

    assert (report.insufficient_lines, report.grounded_lines) == (1, 0)
    assert not report.has_dangling


def test_with_nothing_retrieved_every_cited_id_is_dangling() -> None:
    report = verify_citations(estimate(grounded("A", "3"), grounded("B", "1", "3")), set())

    assert report.dangling_citations == ["1", "3"]  # sorted and de-duplicated


def test_insufficient_estimate_with_no_numbers_is_coherent() -> None:
    insufficient = Estimate(
        confidence="insufficient", reasoning="r", insufficient_context_explanation="no data"
    )

    assert check_coherence(insufficient)


@pytest.mark.parametrize(
    "update",
    [
        {"total_engineer_days": 10},
        {"duration_weeks": 3},
        {"modules": [WorkModule(name="X")]},
        {"insufficient_context_explanation": None},
    ],
    ids=["days", "weeks", "modules", "no-explanation"],
)
def test_insufficient_estimate_violating_its_shape_is_incoherent(update: dict[str, object]) -> None:
    base = Estimate(
        confidence="insufficient", reasoning="r", insufficient_context_explanation="no data"
    )

    assert not check_coherence(base.model_copy(update=update))


def test_other_confidence_levels_are_always_coherent() -> None:
    assert check_coherence(estimate(grounded("A", "1")))


def test_grounded_line_requires_a_source() -> None:
    with pytest.raises(ValidationError):
        TaskItem(name="A", engineer_days=3, grounded=True)


def test_ungrounded_line_cannot_carry_hours_or_sources() -> None:
    with pytest.raises(ValidationError):
        TaskItem(name="A", engineer_days=3, grounded=False)
    with pytest.raises(ValidationError):
        TaskItem(name="A", grounded=False, sources=[ref("1")])
