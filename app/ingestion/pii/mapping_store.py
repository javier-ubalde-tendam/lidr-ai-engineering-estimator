"""The mapping store: persistent backing for consistent pseudonymization.

Two implementations of the same async contract:

* :class:`PostgresMappingStore`: the production path. Wraps the :class:`MappingsRepository`. The
  HMAC of the original value (not the value itself) hits the database; the value never does.
  That asymmetry is what makes "right to be forgotten" auditable: deleting the row by hash kills
  the reversibility without ever having stored the plaintext.

* :class:`InMemoryMappingStore`: for tests and one-off scripts. Same contract, no DB.

Both stores are *idempotent*: looking up the same ``(entity_type, hash)`` twice returns the same
pseudonym, regardless of the order of calls. Concurrent callers converge thanks to the unique
constraint upstream.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession

from app.foundation.persistence.repositories.mappings import MappingsRepository


@runtime_checkable
class MappingStore(Protocol):
    async def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> str: ...

    async def forget(self, entity_type: str, original_hash: str) -> bool: ...


class PostgresMappingStore:
    """Persistent mapping store backed by Postgres."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = MappingsRepository(session)

    async def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> str:
        mapping = await self._repo.lookup_or_create(
            entity_type=entity_type,
            original_hash=original_hash,
            new_pseudonym_factory=new_pseudonym_factory,
        )
        return mapping.pseudonym

    async def forget(self, entity_type: str, original_hash: str) -> bool:
        return await self._repo.forget(entity_type, original_hash)


class InMemoryMappingStore:
    """Simple dict-backed mapping store for tests and scripts."""

    def __init__(self) -> None:
        self._mappings: dict[tuple[str, str], str] = {}

    async def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> str:
        key = (entity_type, original_hash)
        if key not in self._mappings:
            self._mappings[key] = new_pseudonym_factory()
        return self._mappings[key]

    async def forget(self, entity_type: str, original_hash: str) -> bool:
        return self._mappings.pop((entity_type, original_hash), None) is not None
