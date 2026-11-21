"""Incremental corpus expansion + index status, under ``/embeddings/index``.

* ``POST /runs``      add a batch of new documents to the corpus. Registers a job, runs the
  expansion as a background task and returns 202 with the ``job_id`` straight away.
* ``GET  /jobs/{id}`` poll the job's progress (documents processed / status).
* ``GET  /stats``     per-collection corpus size and whether it is HNSW-indexed (corpus growth).

Same posture as ``POST /embeddings/ingest`` (no API key). Jobs live in memory
(``generation/rag/index_jobs.py``): they are per-process and lost on restart, which is enough for
a client that polls while the batch runs.
"""

import uuid

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_chunk_store, get_corpus_index_service, get_index_job_store
from app.foundation.persistence.database import get_db_session
from app.generation.rag.index_jobs import IndexJobStore
from app.generation.rag.index_service import CorpusIndexService
from app.generation.rag.schemas import (
    Budget,
    CollectionStats,
    CorpusStats,
    IndexJobView,
    IndexRunRequest,
    IndexRunResponse,
)
from app.generation.rag.store.repository import ChunkSearchStore

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings/index", tags=["corpus-index"])


async def _run_expansion(
    *,
    job_id: uuid.UUID,
    documents: list[Budget],
    document_type: str,
    chunk_type: str,
    service: CorpusIndexService,
    jobs: IndexJobStore,
) -> None:
    """Background task body: expand the corpus, updating the job as it goes.

    Nothing may escape: the job row is the only place a failure of a background task is visible.
    """
    jobs.mark_running(job_id)
    try:
        result = await service.expand(
            documents,
            document_type=document_type,
            chunk_type=chunk_type,
            on_progress=lambda processed: jobs.set_progress(job_id, processed),
        )
        jobs.mark_completed(
            job_id, documents_processed=result.documents_indexed + result.documents_skipped
        )
    except Exception as exc:  # noqa: BLE001 - record the failure instead of losing it
        message = str(exc)
        log.error(
            "corpus_expansion_failed",
            job_id=str(job_id),
            error_type=type(exc).__name__,
            error=message[:400],
        )
        jobs.mark_failed(job_id, error_message=f"{type(exc).__name__}: {message}")


@router.post("/runs", response_model=IndexRunResponse, status_code=202)
async def create_index_run(
    request: IndexRunRequest,
    background: BackgroundTasks,
    service: CorpusIndexService = Depends(get_corpus_index_service),
    jobs: IndexJobStore = Depends(get_index_job_store),
) -> IndexRunResponse:
    """Add a batch of new documents to the corpus (asynchronous): 202 + ``job_id``."""
    job = jobs.create(documents_total=len(request.documents))
    background.add_task(
        _run_expansion,
        job_id=job.job_id,
        documents=request.documents,
        document_type=request.document_type,
        chunk_type=request.chunk_type,
        service=service,
        jobs=jobs,
    )
    log.info("corpus_index_run_created", job_id=str(job.job_id), documents_total=job.documents_total)
    return IndexRunResponse(
        job_id=job.job_id, documents_total=job.documents_total, status=job.status
    )


@router.get("/jobs/{job_id}", response_model=IndexJobView)
async def get_index_job(
    job_id: uuid.UUID, jobs: IndexJobStore = Depends(get_index_job_store)
) -> IndexJobView:
    """Poll a corpus-expansion job's progress."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return IndexJobView(
        job_id=job.job_id,
        status=job.status,
        documents_processed=job.documents_processed,
        error_message=job.error_message,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/stats", response_model=CorpusStats)
async def get_corpus_stats(
    session: AsyncSession = Depends(get_db_session),
    store: ChunkSearchStore = Depends(get_chunk_store),
) -> CorpusStats:
    """Per-collection corpus size and HNSW index state."""
    try:
        rows = await store.corpus_stats(session)
    except (SQLAlchemyError, OSError) as exc:
        log.error("corpus_stats_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        raise HTTPException(status_code=502, detail="Database error") from exc
    collections = [
        CollectionStats(collection=name, documents=docs, chunks=chunks, hnsw_indexed=indexed)
        for name, docs, chunks, indexed in rows
    ]
    return CorpusStats(collections=collections, total_chunks=sum(c.chunks for c in collections))
