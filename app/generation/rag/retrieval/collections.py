"""Collection registry of the multi-index retriever (session 10).

The corpus is heterogeneous (historical budgets, meeting transcripts, internal technical docs) and
lives in three tables (see ``store/models.py``). This module is the single place that knows, per
collection:

* which ORM model backs it,
* how to read a date from its (divergent) metadata schema, for temporal decay,
* how to map one of its DB rows onto the uniform ``RetrievedChunk`` contract, so every later stage
  sees "list of chunks" whatever the origin,
* the deterministic vocabulary the rule-based router matches.

Keeping that knowledge in one registry is what lets the router, the store and the pipelines stay
collection-agnostic.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from sqlalchemy import ColumnElement, Date, Integer, cast

from app.generation.rag.schemas import RetrievedChunk
from app.generation.rag.store.models import (
    BudgetChunkRow,
    TechnicalDocChunkRow,
    TranscriptChunkRow,
)


class Collection(StrEnum):
    """Closed set of collections: the router's target enum and the registry key.

    A ``StrEnum`` is the closest thing to a Java ``enum`` with a string value: Pydantic, JSON and
    the LLM's structured output all see the plain string.
    """

    BUDGET = "budget"
    TRANSCRIPT = "transcript"
    TECHNICAL_DOC = "technical_doc"


@dataclass(frozen=True)
class HardFilters:
    """Metadata constraints applied BEFORE the vector ranking.

    Each collection consumes the subset it understands and ignores the rest. ``None``/empty means
    "do not constrain on this axis".
    """

    technologies: tuple[str, ...] = ()
    sectors: tuple[str, ...] = ()
    min_date: date | None = None  # budgets: year >= min_date.year; transcripts: meeting_date >=
    version: str | None = None  # technical docs only


@dataclass(frozen=True)
class CollectionSpec:
    """Everything the pipeline needs to treat a collection uniformly."""

    collection: Collection
    model: Any  # the mapped class backing the table (BudgetChunkRow, ...)
    chunk_type: str  # stamped on every chunk of the collection at ingest
    source_id_key: str  # JSONB key with the traceable id of the parent document
    date_key: str | None  # JSONB key with the chunk's date (temporal decay)
    date_kind: str | None  # "year" (int) | "iso" (YYYY-MM-DD string) | None
    rule_patterns: tuple[str, ...] = field(default_factory=tuple)

    def date_of(self, metadata: dict[str, Any]) -> date | None:
        """Read a ``date`` from this collection's metadata; ``None`` when absent or malformed."""
        if not self.date_key or self.date_kind is None:
            return None
        raw = (metadata or {}).get(self.date_key)
        if raw in (None, ""):
            return None
        try:
            if self.date_kind == "year":
                return date(int(raw), 1, 1)
            if self.date_kind == "iso":
                return date.fromisoformat(str(raw))
        except (ValueError, TypeError):
            return None
        return None

    def to_chunk(self, row: Any, *, distance: float) -> RetrievedChunk:
        """Map a DB row of this collection onto the uniform retrieval contract."""
        metadata = row.metadata_ or {}
        budget_id = metadata.get("budget_id")
        source_id = metadata.get(self.source_id_key)
        return RetrievedChunk(
            id=row.id,
            content=row.content,
            sector=str(metadata.get("client_sector", "unknown")),
            project_year=int(metadata.get("year", 0) or 0),
            chunk_type=row.chunk_type,
            distance=distance,
            budget_id=str(budget_id) if budget_id is not None else None,
            collection=self.collection.value,
            source_id=str(source_id) if source_id is not None else None,
            document_date=self.date_of(metadata),
            estimated_hours=metadata.get("estimated_hours"),
        )

    def hard_filter_clauses(self, filters: HardFilters) -> list[ColumnElement[bool]]:
        """SQL predicates embedding ``filters`` into the search query.

        Only the axes this collection understands are emitted; empty means no constraint, so the
        search behaves exactly as if no filter had been passed.
        """
        metadata = self.model.metadata_
        clauses: list[ColumnElement[bool]] = []

        if self.collection is Collection.BUDGET:
            if filters.sectors:
                clauses.append(metadata["client_sector"].astext.in_(list(filters.sectors)))
            if filters.technologies:
                clauses.append(metadata["main_technology"].astext.in_(list(filters.technologies)))
            if filters.min_date is not None:
                clauses.append(cast(metadata["year"].astext, Integer) >= filters.min_date.year)
        elif self.collection is Collection.TRANSCRIPT:
            if filters.min_date is not None:
                clauses.append(cast(metadata["meeting_date"].astext, Date) >= filters.min_date)
        elif self.collection is Collection.TECHNICAL_DOC:
            if filters.version is not None:
                clauses.append(metadata["version"].astext == filters.version)

        return clauses


# Registry order is the canonical "search everything" order (router fallback)
COLLECTIONS: dict[Collection, CollectionSpec] = {
    Collection.BUDGET: CollectionSpec(
        collection=Collection.BUDGET,
        model=BudgetChunkRow,
        chunk_type="budget_component",
        source_id_key="budget_id",
        date_key="year",
        date_kind="year",
        rule_patterns=(
            r"\bbudget(s)?\b",
            r"\bquote(s|d)?\b",
            r"\bestimat(e|ed|ion|es)\b",
            r"\bcost(s|ed)?\b",
            r"\bhours?\b",
            r"\bhow much\b",
            r"\bprice(s|d)?\b",
        ),
    ),
    Collection.TRANSCRIPT: CollectionSpec(
        collection=Collection.TRANSCRIPT,
        model=TranscriptChunkRow,
        chunk_type="meeting_segment",
        source_id_key="transcript_id",
        date_key="meeting_date",
        date_kind="iso",
        rule_patterns=(
            r"\bmeeting(s)?\b",
            r"\btranscript(s)?\b",
            r"\bcall(s)?\b",
            r"\bdiscuss(ed|ion)?\b",
            r"\bclient (said|asked|wants?|mentioned)\b",
            r"\bkick-?off\b",
            r"\bstand-?up\b",
        ),
    ),
    Collection.TECHNICAL_DOC: CollectionSpec(
        collection=Collection.TECHNICAL_DOC,
        model=TechnicalDocChunkRow,
        chunk_type="doc_section",
        source_id_key="doc_id",
        date_key=None,
        date_kind=None,
        rule_patterns=(
            r"\bdocument(ation|s)?\b",
            r"\bspec(ification)?s?\b",
            r"\bapi reference\b",
            r"\barchitecture\b",
            r"\bhow (do|to) (we|i) (implement|configure|integrate)\b",
            r"\brunbook\b",
            r"\bversion\b",
        ),
    ),
}

ALL_COLLECTIONS: tuple[Collection, ...] = tuple(COLLECTIONS)


def spec_for(collection: Collection) -> CollectionSpec:
    return COLLECTIONS[collection]


def match_rules(query_text: str) -> list[Collection]:
    """Deterministic vocabulary routing (cascade level 1).

    Returns every collection whose vocabulary appears in the query, in registry order. Empty
    means no rule fired, and the cascade escalates to the classifier. The query is lower-cased,
    so the patterns are written in lower case.
    """
    lowered = query_text.lower()
    return [
        collection
        for collection, spec in COLLECTIONS.items()
        if any(re.search(pattern, lowered) for pattern in spec.rule_patterns)
    ]
