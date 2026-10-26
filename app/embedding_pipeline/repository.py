from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document


class DocumentRepository:
    """Acceso a datos de documents/chunks. Nunca hace commit: la transacción es del servicio."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_document_id(self, source_path: str) -> int | None:
        result = await self._session.execute(
            select(Document.id).where(Document.source_path == source_path)
        )
        return result.scalar_one_or_none()

    async def add_document(
        self, source_path: str, document_type: str, metadata: dict[str, object]
    ) -> Document:
        document = Document(source_path=source_path, document_type=document_type, metadata_=metadata)
        self._session.add(document)
        # flush (no commit): obtiene el id y detecta ya aquí una violación de unicidad
        await self._session.flush()
        return document

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        self._session.add_all(chunks)
        await self._session.flush()
