import asyncio
import time

import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import EMBEDDING_DIMENSION
from app.db.models import SOURCE_PATH_INDEX, Chunk
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.errors import DocumentAlreadyIngestedError, ensure_valid_vectors
from app.embedding_pipeline.repository import DocumentRepository
from app.embedding_pipeline.schemas import (
    BudgetComponent,
    EmbeddedChunk,
    IngestRequest,
    IngestResponse,
)

logger = structlog.get_logger(__name__)

CHUNK_TYPE_BUDGET_COMPONENT = "budget_component"
UNIQUE_VIOLATION_SQLSTATE = "23505"


def is_source_path_violation(exc: IntegrityError) -> bool:
    # Solo la unicidad de source_path cuenta como duplicado; cualquier otro IntegrityError es un bug
    cause = getattr(exc.orig, "__cause__", None) or exc.orig
    return (
        getattr(cause, "sqlstate", None) == UNIQUE_VIOLATION_SQLSTATE
        and getattr(cause, "constraint_name", None) == SOURCE_PATH_INDEX
    )


class IngestionService:
    def __init__(
        self, session: AsyncSession, chunker: JSONStructuralChunker, embedder: OpenAIEmbedder
    ) -> None:
        self._session = session
        self._repository = DocumentRepository(session)
        self._chunker = chunker
        self._embedder = embedder

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        started = time.perf_counter()
        try:
            document_id, chunks_created = await self._ingest_in_transaction(request)
        except IntegrityError as exc:
            if not is_source_path_violation(exc):
                raise
            # Carrera entre dos peticiones iguales: la transacción perdedora ya hizo rollback
            async with self._session.begin():
                existing_id = await self._repository.find_document_id(request.source_path)
            if existing_id is None:
                raise
            raise DocumentAlreadyIngestedError(existing_id) from exc

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        logger.info(
            "document_ingested",
            document_id=document_id,
            chunks_created=chunks_created,
            ingestion_time_ms=elapsed_ms,
        )
        return IngestResponse(
            document_id=document_id,
            chunks_created=chunks_created,
            embedding_dimension=EMBEDDING_DIMENSION,
            ingestion_time_ms=elapsed_ms,
        )

    async def _ingest_in_transaction(self, request: IngestRequest) -> tuple[int, int]:
        # `begin()` hace commit al salir bien y rollback ante cualquier excepción: o entra todo o nada
        async with self._session.begin():
            existing_id = await self._repository.find_document_id(request.source_path)
            if existing_id is not None:
                # Antes de pagar embeddings
                raise DocumentAlreadyIngestedError(existing_id)

            budget = request.content
            document = await self._repository.add_document(
                source_path=request.source_path,
                document_type=request.document_type,
                metadata={
                    **budget.model_dump(exclude={"components"}),
                    "component_count": len(budget.components),
                },
            )

            chunks = self._chunker.chunk([budget])
            # El embedder es síncrono (time.sleep en el backoff): se aparta del event loop.
            # Al hilo solo viajan los textos, nunca la AsyncSession.
            embedded = await asyncio.to_thread(self._embedder.embed_many, chunks)
            ensure_valid_vectors([item.embedding for item in embedded], expected_count=len(chunks))

            components = {component.component_id: component for component in budget.components}
            await self._repository.add_chunks(
                [
                    build_chunk_row(document.id, item, components[str(item.metadata["component_id"])])
                    for item in embedded
                ]
            )
            return document.id, len(embedded)


def build_chunk_row(document_id: int, item: EmbeddedChunk, component: BudgetComponent) -> Chunk:
    return Chunk(
        document_id=document_id,
        chunk_type=CHUNK_TYPE_BUDGET_COMPONENT,
        content=item.text,
        embedding=item.embedding,
        metadata_={
            **item.metadata,
            # chunk_id lógico (budget_id::component_id) != id BIGINT de la tabla
            "chunk_id": item.chunk_id,
            "token_count": item.token_count,
            "component_name": component.name,
            "tech_stack": component.tech_stack,
            "dependencies": component.dependencies,
        },
    )
