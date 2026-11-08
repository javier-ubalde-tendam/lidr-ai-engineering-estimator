from typing import Any, Literal

from sqlalchemy import Row, Text, cast, func, select
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TEXT_SEARCH_CONFIG, Chunk

LexicalMatchMode = Literal["or", "and"]


class ChunkSearchStore:
    """Read-only queries over ``chunks`` for the RAG retrieval. Writes stay in DocumentRepository.

    Stateless: the session comes in each call and the caller owns the transaction.
    """

    async def search_filtered(
        self,
        session: AsyncSession,
        *,
        query_vector: list[float],
        top_k: int,
        distance_threshold: float,
    ) -> tuple[list[Row[Any]], int]:
        """k-NN by cosine distance, dropping chunks farther than ``distance_threshold``.

        Returns ``(rows ascending by distance, candidates_evaluated)``; the latter counts the
        chunks with an embedding, i.e. the space the k-NN looked at before threshold/limit.
        """
        distance = Chunk.embedding.cosine_distance(query_vector)
        candidates_evaluated = await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
        )
        statement = (
            select(
                Chunk.id,
                Chunk.chunk_type,
                Chunk.content,
                Chunk.metadata_,
                distance.label("distance"),
            )
            .where(Chunk.embedding.is_not(None))
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
    ) -> list[Row[Any]]:
        """Full-text branch: ``@@`` filters on ``content_tsv``, ``ts_rank_cd`` ranks (desc).

        ``plainto_tsquery`` joins every lexeme with AND. Real queries are long project
        descriptions, so requiring all lexemes in one chunk almost always returns 0 rows and the
        hybrid search silently degrades to vector-only. In "or" mode the ``&`` are rewritten to
        ``|``; ``ts_rank_cd`` still favours chunks covering more terms. A query made only of
        stop-words yields an empty tsquery, which matches nothing (``[]``).
        """
        tsquery: Any = func.plainto_tsquery(TEXT_SEARCH_CONFIG, query_text)
        if match_mode == "or":
            tsquery = cast(func.replace(cast(tsquery, Text), " & ", " | "), TSQUERY)
        rank = func.ts_rank_cd(Chunk.content_tsv, tsquery)
        statement = (
            select(
                Chunk.id,
                Chunk.chunk_type,
                Chunk.content,
                Chunk.metadata_,
                rank.label("rank"),
            )
            .where(Chunk.content_tsv.op("@@")(tsquery))
            # Ties broken by id: deterministic ordering for RRF
            .order_by(rank.desc(), Chunk.id)
            .limit(top_k)
        )
        return list((await session.execute(statement)).all())
