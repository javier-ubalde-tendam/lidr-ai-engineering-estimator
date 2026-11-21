"""Pseudonym mapping repository.

``lookup_or_create`` is the operation the pseudonymizer calls per detected entity. It must be
idempotent under contention: two parallel calls with the same ``(entity_type, original_hash)`` MUST
converge to the same pseudonym, otherwise the corpus would get two different pseudonyms for the
same person and the consistency guarantee breaks.

The UNIQUE constraint enforces it: insert first, fall back to a SELECT when the insert conflicts.
That is cheaper than SELECT-then-INSERT and survives concurrent writers without explicit locking.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.foundation.persistence.models import PseudonymMappingRow


@dataclass(frozen=True)
class Mapping:
    """In-memory view of a row. Repositories never leak SQLAlchemy types."""

    entity_type: str
    original_hash: str
    pseudonym: str
    created_at: datetime


class MappingsRepository:
    """Thin wrapper around the ``pseudonym_mappings`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def lookup(self, entity_type: str, original_hash: str) -> Mapping | None:
        result = await self._session.execute(
            select(PseudonymMappingRow).where(
                PseudonymMappingRow.entity_type == entity_type,
                PseudonymMappingRow.original_hash == original_hash,
            )
        )
        row = result.scalar_one_or_none()
        return _to_mapping(row) if row is not None else None

    async def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> Mapping:
        """Return the existing mapping or insert a new one, deterministically.

        ``new_pseudonym_factory`` is only invoked when no prior mapping exists: generating a
        Faker value is cheap but pointless on a cache hit.
        """
        existing = await self.lookup(entity_type, original_hash)
        if existing is not None:
            return existing

        statement = (
            pg_insert(PseudonymMappingRow)
            .values(
                entity_type=entity_type,
                original_hash=original_hash,
                pseudonym=new_pseudonym_factory(),
            )
            .on_conflict_do_nothing(index_elements=["entity_type", "original_hash"])
            .returning(PseudonymMappingRow)
        )
        inserted = (await self._session.execute(statement)).scalar_one_or_none()
        await self._session.commit()
        if inserted is not None:
            return _to_mapping(inserted)

        # Another writer won the race: read the winning row
        winner = await self.lookup(entity_type, original_hash)
        if winner is None:
            raise RuntimeError("pseudonym mapping vanished after a unique-constraint conflict")
        return winner

    async def forget(self, entity_type: str, original_hash: str) -> bool:
        """Delete a mapping (right to be forgotten). Returns True if a row existed."""
        result = await self._session.execute(
            delete(PseudonymMappingRow).where(
                PseudonymMappingRow.entity_type == entity_type,
                PseudonymMappingRow.original_hash == original_hash,
            )
        )
        await self._session.commit()
        return isinstance(result, CursorResult) and result.rowcount > 0


def _to_mapping(row: PseudonymMappingRow) -> Mapping:
    return Mapping(
        entity_type=row.entity_type,
        original_hash=row.original_hash,
        pseudonym=row.pseudonym,
        created_at=row.created_at,
    )
