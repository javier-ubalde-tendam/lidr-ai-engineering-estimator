from typing import Any, Literal

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import ColumnElement, Integer, Row, Text, cast, func, select, text
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import EMBEDDING_DIMENSION
from app.generation.rag.retrieval.collections import Collection, spec_for
from app.generation.rag.schemas import EmbeddedChunk
from app.generation.rag.store.models import TEXT_SEARCH_CONFIG, Chunk, Document

LexicalMatchMode = Literal["or", "and"]

# pgvector's HNSW scan only keeps ``hnsw.ef_search`` candidates (default 40, max 1000), so a
# ``LIMIT`` above it silently returns fewer rows. Never go below the default: it is also the
# recall/latency trade-off pgvector ships with.
HNSW_DEFAULT_EF_SEARCH = 40
HNSW_MAX_EF_SEARCH = 1000


def hnsw_ef_search_for(limit: int) -> int:
    """``hnsw.ef_search`` able to return ``limit`` rows, clamped to pgvector's valid range."""
    return min(max(limit, HNSW_DEFAULT_EF_SEARCH), HNSW_MAX_EF_SEARCH)


def halfvec_cosine_distance(model: Any, query_vector: list[float]) -> ColumnElement[float]:
    """Cosine distance written EXACTLY like the HNSW index expression (migration 0004).

    The indexes are built on ``embedding::halfvec(1536)`` with ``halfvec_cosine_ops``; Postgres
    only uses an index when the query expression and operator match it, so a plain
    ``embedding <=> :q`` would silently fall back to a sequential scan.
    """
    expression: ColumnElement[float] = cast(model.embedding, HALFVEC(EMBEDDING_DIMENSION)).cosine_distance(
        query_vector
    )
    return expression


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

    async def persist_document_with_chunks(
        self,
        *,
        source_path: str,
        document_type: str,
        doc_metadata: dict[str, object],
        embedded_chunks: list[EmbeddedChunk],
        chunk_type: str | None = None,
        collection: Collection = Collection.BUDGET,
    ) -> int:
        """Insert a document and its embedded chunks into ``collection``'s table; returns its id.

        ``chunk_type`` defaults to the collection's own (e.g. ``budget_component``). Same
        transaction contract as the rest of the repository (flush, never commit), so the caller
        decides when anything lands.
        """
        document = await self.add_document(source_path, document_type, doc_metadata)
        spec = spec_for(collection)
        self._session.add_all(
            spec.model(
                document_id=document.id,
                chunk_type=chunk_type or spec.chunk_type,
                content=item.text,
                embedding=item.embedding,
                metadata_=item.metadata,
            )
            for item in embedded_chunks
        )
        await self._session.flush()
        return document.id


class ChunkSearchStore:
    """Read-only queries over the chunk tables for the RAG retrieval. Writes stay in DocumentRepository.

    Stateless: the session comes in each call and the caller owns the transaction. Every SQL
    predicate lives here (filters included), so retrieval code above stays free of SQL. Each
    query takes a ``collection`` (default: budgets) that selects the table it runs against.
    """

    @staticmethod
    def _structural_filters(
        *,
        sectors: list[str] | None,
        project_year_min: int | None,
        project_year_max: int | None,
        chunk_types: list[str] | None,
        collection: Collection = Collection.BUDGET,
        extra_filters: list[ColumnElement[bool]] | None = None,
    ) -> list[ColumnElement[bool]]:
        """Metadata predicates shared by both branches. ``None``/empty = no filter on that axis.

        Sector and year live in the JSONB ``metadata`` (``client_sector``, ``year``); the chunk
        type has its own column. ``extra_filters`` are ready-made predicates (the collection
        registry's hard filters) ANDed with the structural ones.
        """
        model = spec_for(collection).model
        filters: list[ColumnElement[bool]] = []
        if sectors:
            filters.append(model.metadata_["client_sector"].astext.in_(sectors))
        if project_year_min is not None or project_year_max is not None:
            year = cast(model.metadata_["year"].astext, Integer)
            if project_year_min is not None:
                filters.append(year >= project_year_min)
            if project_year_max is not None:
                filters.append(year <= project_year_max)
        if chunk_types:
            filters.append(model.chunk_type.in_(chunk_types))
        filters.extend(extra_filters or [])
        return filters

    async def corpus_stats(self, session: AsyncSession) -> list[tuple[str, int, int, bool]]:
        """Per-collection ``(name, documents, chunks, hnsw_indexed)``.

        Documents are the distinct parents owning chunks in the collection. The HNSW flag comes
        from ``pg_indexes`` (index name convention of migration 0004).
        """
        stats: list[tuple[str, int, int, bool]] = []
        for collection in Collection:
            model = spec_for(collection).model
            chunks = (await session.execute(select(func.count()).select_from(model))).scalar_one()
            documents = (
                await session.execute(select(func.count(func.distinct(model.document_id))))
            ).scalar_one()
            indexed: bool = (
                await session.execute(
                    text("SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = :name)"),
                    {"name": f"ix_{model.__tablename__}_embedding_hnsw"},
                )
            ).scalar_one()
            stats.append((collection.value, int(documents), int(chunks), bool(indexed)))
        return stats

    async def search_filtered(
        self,
        session: AsyncSession,
        *,
        query_vector: list[float],
        top_k: int,
        distance_threshold: float,
        collection: Collection = Collection.BUDGET,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
        extra_filters: list[ColumnElement[bool]] | None = None,
    ) -> tuple[list[Row[Any]], int]:
        """k-NN by cosine distance, dropping chunks farther than ``distance_threshold``.

        The optional metadata filters narrow the candidate space BEFORE the ranking.
        Returns ``(rows ascending by distance, candidates_evaluated)``; the latter counts the
        chunks with an embedding that match the filters, i.e. the space the k-NN looked at
        before threshold/limit.
        """
        model = spec_for(collection).model
        filters = self._structural_filters(
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
            collection=collection,
            extra_filters=extra_filters,
        )
        distance = halfvec_cosine_distance(model, query_vector)
        # set_config(..., is_local=true) == SET LOCAL (scoped to the current transaction) but,
        # unlike SET, it accepts a bind parameter. The setting is harmless for a seq scan.
        await session.execute(
            select(func.set_config("hnsw.ef_search", str(hnsw_ef_search_for(top_k)), True))
        )
        candidates_evaluated = await session.scalar(
            select(func.count())
            .select_from(model)
            .where(model.embedding.is_not(None))
            .where(*filters)
        )
        statement = (
            select(
                model.id,
                model.chunk_type,
                model.content,
                model.metadata_,
                distance.label("distance"),
            )
            .where(model.embedding.is_not(None))
            .where(*filters)
            .where(distance <= distance_threshold)
            .order_by(distance)
            .limit(top_k)
        )
        rows = list((await session.execute(statement)).all())
        return rows, int(candidates_evaluated or 0)

    async def search_lexical(
        self,
        session: AsyncSession,
        *,
        query_text: str,
        top_k: int,
        match_mode: LexicalMatchMode = "or",
        collection: Collection = Collection.BUDGET,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
        extra_filters: list[ColumnElement[bool]] | None = None,
    ) -> list[Row[Any]]:
        """Full-text branch: ``@@`` filters on ``content_tsv``, ``ts_rank_cd`` ranks (desc).

        ``plainto_tsquery`` joins every lexeme with AND. Real queries are long project
        descriptions, so requiring all lexemes in one chunk almost always returns 0 rows and the
        hybrid search silently degrades to vector-only. In "or" mode the ``&`` are rewritten to
        ``|``; ``ts_rank_cd`` still favours chunks covering more terms. A query made only of
        stop-words yields an empty tsquery, which matches nothing (``[]``). The metadata filters
        mirror ``search_filtered`` so both branches see the same candidate space.
        """
        model = spec_for(collection).model
        filters = self._structural_filters(
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
            collection=collection,
            extra_filters=extra_filters,
        )
        tsquery: Any = func.plainto_tsquery(TEXT_SEARCH_CONFIG, query_text)
        if match_mode == "or":
            tsquery = cast(func.replace(cast(tsquery, Text), " & ", " | "), TSQUERY)
        rank = func.ts_rank_cd(model.content_tsv, tsquery)
        statement = (
            select(
                model.id,
                model.chunk_type,
                model.content,
                model.metadata_,
                rank.label("rank"),
            )
            .where(model.content_tsv.op("@@")(tsquery))
            .where(*filters)
            # Ties broken by id: deterministic ordering for RRF
            .order_by(rank.desc(), model.id)
            .limit(top_k)
        )
        return list((await session.execute(statement)).all())
