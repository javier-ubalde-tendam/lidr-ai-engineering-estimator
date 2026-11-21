"""Incremental corpus expansion: add NEW budget documents to the corpus and index them.

A thin batch loop over the single-document ingestion (chunk -> embed -> persist, with its
duplicate guard), so no ingest logic is re-implemented. It:

* skips documents already present (same ``source_path``) instead of failing the whole batch,
* reports per-document progress through ``on_progress`` so the job the router runs can show a
  live counter.

The existing corpus is NOT re-embedded (that would be an embedding-model change): new documents
use the SAME model and land in ``budget_chunks``, where the HNSW index picks them up on insert.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.errors import DocumentAlreadyIngestedError
from app.generation.rag.ingest_service import IngestionService
from app.generation.rag.schemas import Budget, IngestRequest, IngestResponse

log = structlog.get_logger()


class DocumentIngester(Protocol):
    """Ingests one document; raises ``DocumentAlreadyIngestedError`` on a duplicate."""

    async def ingest(self, request: IngestRequest) -> IngestResponse: ...


class SessionIngester:
    """Ingests each document in its own short-lived session (so one failure cannot poison the
    rest of a long batch, and no session outlives the background task that uses it)."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        chunker: JSONStructuralChunker,
        embedder: OpenAIEmbedder,
    ) -> None:
        self._session_factory = session_factory
        self._chunker = chunker
        self._embedder = embedder

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        async with self._session_factory() as session:
            service = IngestionService(session, self._chunker, self._embedder)
            return await service.ingest(request)


@dataclass(frozen=True)
class CorpusExpansionResult:
    """Outcome of a batch corpus expansion."""

    documents_indexed: int
    documents_skipped: int
    chunks_created: int


class CorpusIndexService:
    """Batch-adds new documents to the corpus through a :class:`DocumentIngester`."""

    def __init__(self, ingester: DocumentIngester) -> None:
        self._ingester = ingester

    async def expand(
        self,
        documents: list[Budget],
        *,
        document_type: str = "historical_budget",
        chunk_type: str = "budget_component",
        source_prefix: str = "corpus-expansion",
        on_progress: Callable[[int], None] | None = None,
    ) -> CorpusExpansionResult:
        """Add every document; a duplicate (same ``source_path``) is skipped, not an error.

        ``on_progress`` receives the running processed count (indexed + skipped) after each
        document.
        """
        indexed = skipped = chunks = 0
        for document in documents:
            source_path = f"{source_prefix}::{document.budget_id}"
            try:
                response = await self._ingester.ingest(
                    IngestRequest(
                        source_path=source_path,
                        document_type=document_type,
                        content=document,
                        chunk_type=chunk_type,
                    )
                )
                indexed += 1
                chunks += response.chunks_created
            except DocumentAlreadyIngestedError:
                skipped += 1
                log.info("corpus_expansion_skip_duplicate", source_path=source_path)
            if on_progress is not None:
                on_progress(indexed + skipped)

        log.info(
            "corpus_expansion_done",
            documents_indexed=indexed,
            documents_skipped=skipped,
            chunks_created=chunks,
        )
        return CorpusExpansionResult(
            documents_indexed=indexed, documents_skipped=skipped, chunks_created=chunks
        )
