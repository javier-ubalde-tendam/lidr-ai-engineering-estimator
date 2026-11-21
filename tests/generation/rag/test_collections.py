"""Collection registry: table mapping, rule routing, dates, uniform chunk mapping, hard filters.

Pure introspection, no DB. Pins the per-collection knowledge the router and the advanced pipeline
rely on.
"""

from datetime import date
from types import SimpleNamespace

from app.generation.rag.retrieval.collections import (
    ALL_COLLECTIONS,
    Collection,
    HardFilters,
    match_rules,
    spec_for,
)
from app.generation.rag.store.models import (
    BudgetChunkRow,
    TechnicalDocChunkRow,
    TranscriptChunkRow,
)


def test_registry_maps_each_collection_to_its_table() -> None:
    assert spec_for(Collection.BUDGET).model is BudgetChunkRow
    assert spec_for(Collection.TRANSCRIPT).model is TranscriptChunkRow
    assert spec_for(Collection.TECHNICAL_DOC).model is TechnicalDocChunkRow
    assert set(ALL_COLLECTIONS) == set(Collection)


def test_rules_route_by_vocabulary() -> None:
    assert match_rules("how much did the budget cost in hours") == [Collection.BUDGET]
    assert match_rules("what did the client say in the kickoff meeting") == [Collection.TRANSCRIPT]
    assert match_rules("API reference for the architecture spec") == [Collection.TECHNICAL_DOC]
    assert match_rules("the weather today") == []  # no signal: escalate to the classifier


def test_a_query_spanning_collections_matches_all_of_them_in_registry_order() -> None:
    hits = match_rules("what was discussed in the meeting about the budget and the api reference")

    assert hits == [Collection.BUDGET, Collection.TRANSCRIPT, Collection.TECHNICAL_DOC]


def test_date_extraction_per_collection() -> None:
    assert spec_for(Collection.BUDGET).date_of({"year": 2021}) == date(2021, 1, 1)
    assert spec_for(Collection.TRANSCRIPT).date_of({"meeting_date": "2023-09-10"}) == date(
        2023, 9, 10
    )
    # Technical docs carry no usable date; missing or malformed values degrade to None
    assert spec_for(Collection.TECHNICAL_DOC).date_of({"version": "v1.4"}) is None
    assert spec_for(Collection.BUDGET).date_of({}) is None
    assert spec_for(Collection.TRANSCRIPT).date_of({"meeting_date": "not-a-date"}) is None


def test_to_chunk_carries_provenance_source_id_and_date() -> None:
    row = SimpleNamespace(
        id=7,
        content="meeting about Stripe",
        chunk_type="meeting_segment",
        metadata_={"transcript_id": "TR-2024-002", "meeting_date": "2024-03-20"},
    )

    chunk = spec_for(Collection.TRANSCRIPT).to_chunk(row, distance=0.3)

    assert chunk.collection == "transcript"
    assert chunk.source_id == "TR-2024-002"
    assert chunk.document_date == date(2024, 3, 20)
    # A transcript has no budget fields: the budget-centric ones fall back to their defaults
    assert (chunk.budget_id, chunk.sector, chunk.project_year) == (None, "unknown", 0)


def test_budget_to_chunk_keeps_the_existing_mapping() -> None:
    row = SimpleNamespace(
        id=1,
        content="c",
        chunk_type="budget_component",
        metadata_={"budget_id": "BUD-1", "client_sector": "finance", "year": 2023},
    )

    chunk = spec_for(Collection.BUDGET).to_chunk(row, distance=0.1)

    assert (chunk.budget_id, chunk.sector, chunk.project_year) == ("BUD-1", "finance", 2023)
    assert (chunk.collection, chunk.source_id) == ("budget", "BUD-1")
    assert chunk.document_date == date(2023, 1, 1)


def test_hard_filters_emit_only_axes_the_collection_understands() -> None:
    filters = HardFilters(
        technologies=("node",), sectors=("ecommerce",), min_date=date(2023, 1, 1), version="v3.2"
    )

    # Budgets understand sector + technology + min_date (3 clauses), not version
    assert len(spec_for(Collection.BUDGET).hard_filter_clauses(filters)) == 3
    assert len(spec_for(Collection.TRANSCRIPT).hard_filter_clauses(filters)) == 1  # min_date
    assert len(spec_for(Collection.TECHNICAL_DOC).hard_filter_clauses(filters)) == 1  # version
    # No filters: no clauses, the search runs unconstrained
    assert spec_for(Collection.BUDGET).hard_filter_clauses(HardFilters()) == []
