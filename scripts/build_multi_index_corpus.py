#!/usr/bin/env python3
"""Ingest the session 10 transcript + technical-doc collections (multi-index).

The budgets corpus is ingested over HTTP by ``query_examples.py``; this script seeds the two NEW
collections (``transcript_chunks``, ``technical_doc_chunks``) through ``DocumentRepository``,
reusing the app's embedder and async session factory. One chunk per transcript segment / per doc
section, each with its collection's own metadata schema (transcripts: speakers + meeting_date;
docs: version) so routing, hard filters and temporal decay return sensible results.

Idempotent: each document is keyed by ``source_path`` and skipped if already ingested, so
re-running never duplicates data. Needs OPENAI_API_KEY (it embeds) and the database up with the
migrations applied (``alembic upgrade head``). Wipe with::

    DELETE FROM documents WHERE document_type IN ('meeting_transcript', 'technical_doc');

Usage::

    uv run python scripts/build_multi_index_corpus.py
"""

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dependencies import (
    get_async_session_factory,
    get_embedder,
    get_token_encoder,
)
from app.generation.rag.retrieval.collections import Collection
from app.generation.rag.schemas import Chunk
from app.generation.rag.store.repository import DocumentRepository

TRANSCRIPTS_PATH = ROOT / "data" / "transcripts_sample.json"
DOCS_PATH = ROOT / "data" / "technical_docs_sample.json"

ChunkBuilder = Callable[[Any, dict[str, Any]], list[Chunk]]


def build_chunk(
    encoder: Any, *, chunk_id: str, header: str, body: str, metadata: dict[str, Any]
) -> Chunk:
    """One embeddable chunk: a small contextual header + the body text."""
    text = f"{header}\n{body}".strip()
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=metadata,
        token_count=len(encoder.encode(text)),
    )


def transcript_chunks(encoder: Any, transcript: dict[str, Any]) -> list[Chunk]:
    transcript_id = transcript["transcript_id"]
    header = f"Meeting: {transcript['project']} ({transcript['meeting_date']})"
    return [
        build_chunk(
            encoder,
            chunk_id=f"{transcript_id}::{segment['segment_id']}",
            header=f"{header} - topic: {segment['topic']}",
            body=segment["text"],
            metadata={
                "transcript_id": transcript_id,
                "segment_id": segment["segment_id"],
                "project": transcript["project"],
                "meeting_date": transcript["meeting_date"],
                "speakers": segment.get("speakers", []),
                "topic": segment["topic"],
            },
        )
        for segment in transcript["segments"]
    ]


def doc_chunks(encoder: Any, doc: dict[str, Any]) -> list[Chunk]:
    doc_id = doc["doc_id"]
    header = f"Doc: {doc['title']} (version {doc['version']})"
    return [
        build_chunk(
            encoder,
            chunk_id=f"{doc_id}::{section['section_id']}",
            header=f"{header} - {section['heading']}",
            body=section["text"],
            metadata={
                "doc_id": doc_id,
                "section_id": section["section_id"],
                "heading": section["heading"],
                "version": doc["version"],
                "component": doc.get("component", "unknown"),
            },
        )
        for section in doc["sections"]
    ]


async def ingest_documents(
    *,
    label: str,
    items: list[dict[str, Any]],
    source_file: str,
    document_type: str,
    collection: Collection,
    build_chunks: ChunkBuilder,
) -> None:
    """Embed and persist each document of one collection, skipping duplicates."""
    embedder = get_embedder()
    encoder = get_token_encoder()
    session_factory = get_async_session_factory()

    created = skipped = 0
    for item in items:
        document_key = item.get("transcript_id") or item.get("doc_id")
        source_path = f"{source_file}::{document_key}"

        async with session_factory() as session, session.begin():
            if await DocumentRepository(session).find_document_id(source_path) is not None:
                skipped += 1
                continue

        # The embedder is synchronous (blocking HTTP): keep it off the event loop
        embedded = await asyncio.to_thread(embedder.embed_many, build_chunks(encoder, item))

        # One transaction per document: the document row and its chunks land together or not at all
        async with session_factory() as session, session.begin():
            await DocumentRepository(session).persist_document_with_chunks(
                source_path=source_path,
                document_type=document_type,
                doc_metadata={"source_id": document_key},
                embedded_chunks=embedded,
                collection=collection,
            )
        created += 1

    print(f"{label}: {len(items)} documents - {created} ingested, {skipped} already present.")


async def main() -> int:
    transcripts = json.loads(TRANSCRIPTS_PATH.read_text(encoding="utf-8"))
    docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))

    await ingest_documents(
        label="Transcripts",
        items=transcripts,
        source_file="data/transcripts_sample.json",
        document_type="meeting_transcript",
        collection=Collection.TRANSCRIPT,
        build_chunks=transcript_chunks,
    )
    await ingest_documents(
        label="Technical docs",
        items=docs,
        source_file="data/technical_docs_sample.json",
        document_type="technical_doc",
        collection=Collection.TECHNICAL_DOC,
        build_chunks=doc_chunks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
