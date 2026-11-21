"""In-memory tracking of the corpus-expansion jobs.

The teacher keeps these in an ``ingestion_jobs`` table (part of the optional offline-ingestion
extra, 8a, which this project does not have). A job only needs to live as long as the client is
polling it, so a bounded per-process dict is enough: state is lost on restart and is not shared
between workers. Everything runs on the event loop (the background task and the endpoints), so no
locking is needed.
"""

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.generation.rag.schemas import IndexJobStatus

# Oldest jobs are evicted beyond this: the store must not grow without bound
MAX_TRACKED_JOBS = 200


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class IndexJob:
    job_id: uuid.UUID
    documents_total: int
    status: IndexJobStatus = "pending"
    documents_processed: int = 0
    error_message: str | None = None
    started_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None


class IndexJobStore:
    """Bounded in-memory registry of the corpus-expansion jobs."""

    def __init__(self, max_jobs: int = MAX_TRACKED_JOBS) -> None:
        self._max_jobs = max_jobs
        self._jobs: OrderedDict[uuid.UUID, IndexJob] = OrderedDict()

    def create(self, documents_total: int) -> IndexJob:
        job = IndexJob(job_id=uuid.uuid4(), documents_total=documents_total)
        self._jobs[job.job_id] = job
        while len(self._jobs) > self._max_jobs:
            self._jobs.popitem(last=False)
        return job

    def get(self, job_id: uuid.UUID) -> IndexJob | None:
        return self._jobs.get(job_id)

    def mark_running(self, job_id: uuid.UUID) -> None:
        if job := self._jobs.get(job_id):
            job.status = "running"

    def set_progress(self, job_id: uuid.UUID, processed: int) -> None:
        if job := self._jobs.get(job_id):
            job.documents_processed = processed

    def mark_completed(self, job_id: uuid.UUID, *, documents_processed: int) -> None:
        if job := self._jobs.get(job_id):
            job.status = "completed"
            job.documents_processed = documents_processed
            job.finished_at = _now()

    def mark_failed(self, job_id: uuid.UUID, *, error_message: str) -> None:
        if job := self._jobs.get(job_id):
            job.status = "failed"
            job.error_message = error_message[:2048]
            job.finished_at = _now()
