"""Ingestion-job repository.

A job moves through ``pending -> running -> completed | failed``. The router creates the row, the
background task transitions it, the GET endpoint reads it. Callers never see ORM types: they get
plain frozen dataclasses. Every write commits on its own, so progress is visible to a poller
while the job is still running.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.foundation.persistence.models import ERROR_MESSAGE_MAX_LENGTH, IngestionJobRow

JobStatusName = Literal["pending", "running", "completed", "failed"]


@dataclass(frozen=True)
class Job:
    job_id: uuid.UUID
    source_name: str
    status: JobStatusName
    documents_count: int
    error_message: str | None
    started_at: datetime
    finished_at: datetime | None


class JobsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, source_name: str) -> Job:
        row = IngestionJobRow(source_name=source_name, status="pending")
        self._session.add(row)
        await self._session.commit()
        # started_at is a server default: reload it so the returned view is complete
        await self._session.refresh(row)
        return _to_job(row)

    async def get(self, job_id: uuid.UUID) -> Job | None:
        row = await self._session.get(IngestionJobRow, job_id)
        return _to_job(row) if row is not None else None

    async def mark_running(self, job_id: uuid.UUID) -> None:
        await self._update(job_id, status="running")

    async def mark_completed(self, job_id: uuid.UUID, *, documents_count: int) -> None:
        await self._update(
            job_id,
            status="completed",
            documents_count=documents_count,
            finished_at=datetime.now(UTC),
        )

    async def mark_failed(self, job_id: uuid.UUID, *, error_message: str) -> None:
        await self._update(
            job_id,
            status="failed",
            error_message=error_message[:ERROR_MESSAGE_MAX_LENGTH],
            finished_at=datetime.now(UTC),
        )

    async def list_recent(self, *, limit: int = 50) -> list[Job]:
        """Most recent jobs first (debug/preflight helper, not exposed over HTTP)."""
        result = await self._session.execute(
            select(IngestionJobRow).order_by(IngestionJobRow.started_at.desc()).limit(limit)
        )
        return [_to_job(row) for row in result.scalars()]

    async def _update(self, job_id: uuid.UUID, **fields: Any) -> None:
        row = await self._session.get(IngestionJobRow, job_id)
        if row is None:
            return
        for key, value in fields.items():
            setattr(row, key, value)
        await self._session.commit()


def _to_job(row: IngestionJobRow) -> Job:
    return Job(
        job_id=row.job_id,
        source_name=row.source_name,
        status=cast(JobStatusName, row.status),
        documents_count=row.documents_count,
        error_message=row.error_message,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )
