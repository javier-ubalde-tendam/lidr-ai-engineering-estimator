import asyncio
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.errors import ensure_valid_vectors
from app.embedding_pipeline.schemas import SearchResponse, SearchResult


class Retriever:
    """Búsqueda semántica exacta: el ranking por distancia coseno lo hace PostgreSQL."""

    def __init__(self, session: AsyncSession, embedder: OpenAIEmbedder) -> None:
        self._session = session
        self._embedder = embedder

    async def search(self, query: str, k: int) -> SearchResponse:
        started = time.perf_counter()
        # Mismo embedder (y por tanto mismo modelo/dimensión) que usa la ingesta
        query_vector = await asyncio.to_thread(self._embedder.embed_one, query)
        ensure_valid_vectors([query_vector], expected_count=1)

        distance = Chunk.embedding.cosine_distance(query_vector)
        statement = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.chunk_type,
                Chunk.content,
                Chunk.metadata_,
                distance.label("distance"),
            )
            # No es un filtro de negocio: sin vector la distancia sería NULL
            .where(Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(k)
        )
        async with self._session.begin():
            rows = (await self._session.execute(statement)).all()

        results = [
            SearchResult(
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                content=row.content,
                distance=row.distance,
                metadata=row.metadata_,
            )
            for row in rows
        ]
        return SearchResponse(
            query=query,
            k=k,
            search_time_ms=round((time.perf_counter() - started) * 1000),
            results=results,
        )
