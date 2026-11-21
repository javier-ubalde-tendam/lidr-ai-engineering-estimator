"""POST /api/v1/ingestion/runs   +   GET /api/v1/ingestion/jobs/{job_id}.

The router is intentionally thin: it validates the source against the catalog, records the job row,
dispatches the orchestrator as a BackgroundTask and returns 202 immediately. All ingestion logic
lives in ``app.ingestion.orchestrator``; the router never imports parsers.

Same posture as the teacher's project and as ``POST /embeddings/ingest``: no API key. The run only
reads the local seed folder (no paid call), but anybody who can reach the API can create job rows.
"""

import uuid

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import DEV_PSEUDONYM_HASH_SALT, get_settings
from app.dependencies import (
    get_async_session_factory,
    get_catalog,
    get_filesystem_loader,
    get_parser_registry,
)
from app.domain.schemas.ingestion import (
    IngestionJobView,
    IngestionRunRequest,
    IngestionRunResponse,
)
from app.foundation.persistence.database import get_db_session
from app.foundation.persistence.repositories.jobs import JobsRepository
from app.ingestion.catalog.models import CatalogDecision, DataCatalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.orchestrator import ingest_source
from app.ingestion.parsers.registry import ParserRegistry

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1/ingestion", tags=["ingestion"])


def warm_up_catalog() -> None:
    """Load the catalog at startup so a malformed one shows up in the logs right away.

    Fail-soft like the teacher's app: the rest of the API must keep working, and the first
    ingestion request will surface the same error again.
    """
    settings = get_settings()
    try:
        catalog = get_catalog()
    except Exception as exc:  # noqa: BLE001 - fail-soft: a broken catalog must not stop the API
        log.error("catalog_load_failed", path=str(settings.CATALOG_PATH), error=str(exc)[:400])
    else:
        log.info(
            "catalog_loaded",
            version=catalog.version,
            sources_total=len(catalog.sources),
            sources_included=len(catalog.included_sources()),
        )
    if settings.PSEUDONYM_HASH_SALT == DEV_PSEUDONYM_HASH_SALT:
        log.warning(
            "pseudonym_salt_is_dev_placeholder",
            hint="set PSEUDONYM_HASH_SALT to a long random secret before pseudonymizing real data",
        )


async def _run_in_background(
    *,
    job_id: uuid.UUID,
    source_name: str,
    catalog: DataCatalog,
    loader: FileSystemLoader,
    registry: ParserRegistry,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """BackgroundTask body. It owns its session: the request-scoped one is closed by then."""
    try:
        async with session_factory() as session:
            await ingest_source(
                catalog=catalog,
                source_name=source_name,
                loader=loader,
                registry=registry,
                jobs_repo=JobsRepository(session),
                job_id=job_id,
            )
    except Exception as exc:  # noqa: BLE001 - nothing may escape a background task
        # The orchestrator already wrote the failure on the job row; nothing may escape here
        log.error(
            "ingestion_background_failed",
            job_id=str(job_id),
            source_name=source_name,
            error=str(exc)[:400],
        )


@router.post("/runs", response_model=IngestionRunResponse, status_code=202)
async def create_ingestion_run(
    request: IngestionRunRequest,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_async_session_factory),
    catalog: DataCatalog = Depends(get_catalog),
    loader: FileSystemLoader = Depends(get_filesystem_loader),
    registry: ParserRegistry = Depends(get_parser_registry),
) -> IngestionRunResponse:
    source = catalog.find(request.source_name)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail={"reason": "unknown_source", "source_name": request.source_name},
        )
    if source.decision is not CatalogDecision.INCLUDE:
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "source_not_included",
                "source_name": request.source_name,
                "decision": source.decision.value,
                "decision_reason": source.decision_reason,
            },
        )

    job = await JobsRepository(session).create(source_name=request.source_name)
    background.add_task(
        _run_in_background,
        job_id=job.job_id,
        source_name=request.source_name,
        catalog=catalog,
        loader=loader,
        registry=registry,
        session_factory=session_factory,
    )
    return IngestionRunResponse(job_id=job.job_id, source_name=job.source_name, status=job.status)


@router.get("/jobs/{job_id}", response_model=IngestionJobView)
async def get_ingestion_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> IngestionJobView:
    job = await JobsRepository(session).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return IngestionJobView(
        job_id=job.job_id,
        source_name=job.source_name,
        status=job.status,
        documents_count=job.documents_count,
        error_message=job.error_message,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )
