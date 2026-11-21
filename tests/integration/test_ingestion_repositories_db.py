"""Ingestion jobs and pseudonym mappings against real Postgres (migration 0005)."""

import asyncio
import textwrap
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.dependencies import (
    get_async_session_factory,
    get_catalog,
    get_filesystem_loader,
    get_parser_registry,
)
from app.foundation.persistence.repositories.jobs import JobsRepository
from app.foundation.persistence.repositories.mappings import MappingsRepository
from app.ingestion.catalog.loader import load_catalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.registry import default_registry
from app.main import app
from tests.integration.conftest import run

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def clean_ingestion_tables(session_factory: async_sessionmaker[AsyncSession]) -> None:
    async def truncate() -> None:
        async with session_factory() as session, session.begin():
            await session.execute(text("TRUNCATE ingestion_jobs, pseudonym_mappings RESTART IDENTITY"))

    run(truncate())


def test_job_lifecycle_is_persisted_step_by_step(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async def scenario() -> list[Any]:
        async with session_factory() as session:
            repo = JobsRepository(session)
            created = await repo.create("presupuestos_json")
            assert (created.status, created.documents_count) == ("pending", 0)
            assert created.started_at is not None  # server default, reloaded after the INSERT

            await repo.mark_running(created.job_id)
            running = await repo.get(created.job_id)
            await repo.mark_completed(created.job_id, documents_count=6)
            done = await repo.get(created.job_id)
            return [running, done]

    running, done = run(scenario())
    assert running.status == "running" and running.finished_at is None
    assert done.status == "completed" and done.documents_count == 6
    assert done.finished_at is not None


def test_failed_job_truncates_the_error_message(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async def scenario() -> Any:
        async with session_factory() as session:
            repo = JobsRepository(session)
            job = await repo.create("presupuestos_json")
            await repo.mark_failed(job.job_id, error_message="x" * 5000)
            return await repo.get(job.job_id)

    failed = run(scenario())
    assert failed.status == "failed"
    assert len(failed.error_message) == 2048


def test_unknown_job_is_none_and_updates_on_it_are_ignored(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async def scenario() -> Any:
        async with session_factory() as session:
            repo = JobsRepository(session)
            await repo.mark_running(uuid.uuid4())
            return await repo.get(uuid.uuid4())

    assert run(scenario()) is None


def test_list_recent_returns_newest_first(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async def scenario() -> list[str]:
        async with session_factory() as session:
            repo = JobsRepository(session)
            for name in ("first", "second", "third"):
                await repo.create(name)
            return [job.source_name for job in await repo.list_recent(limit=2)]

    assert run(scenario()) == ["third", "second"]


def test_lookup_or_create_returns_the_same_pseudonym_and_calls_the_factory_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    calls: list[str] = []

    def factory() -> str:
        calls.append("called")
        return "Ana Pseudónimo"

    async def scenario() -> list[str]:
        async with session_factory() as session:
            repo = MappingsRepository(session)
            first = await repo.lookup_or_create("PERSON", "hash-1", factory)
            second = await repo.lookup_or_create("PERSON", "hash-1", factory)
            other_type = await repo.lookup_or_create("LOCATION", "hash-1", lambda: "Teruel")
            return [first.pseudonym, second.pseudonym, other_type.pseudonym]

    assert run(scenario()) == ["Ana Pseudónimo", "Ana Pseudónimo", "Teruel"]
    assert calls == ["called"]  # the cache hit never generated a second pseudonym


def test_concurrent_writers_converge_on_one_pseudonym(
    session_factory: async_sessionmaker[AsyncSession], sql: Callable[..., list[Any]]
) -> None:
    async def writer(index: int) -> str:
        # Each writer has its own session, as two parallel ingestion runs would
        async with session_factory() as session:
            mapping = await MappingsRepository(session).lookup_or_create(
                "PERSON", "same-hash", lambda: f"pseudonym-{index}"
            )
            return mapping.pseudonym

    async def scenario() -> list[str]:
        return list(await asyncio.gather(*(writer(i) for i in range(8))))

    pseudonyms = run(scenario())
    assert len(set(pseudonyms)) == 1
    assert sql("SELECT count(*) FROM pseudonym_mappings")[0][0] == 1


def test_forget_deletes_the_mapping_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async def scenario() -> list[Any]:
        async with session_factory() as session:
            repo = MappingsRepository(session)
            await repo.lookup_or_create("PERSON", "hash-9", lambda: "x")
            return [
                await repo.forget("PERSON", "hash-9"),
                await repo.lookup("PERSON", "hash-9"),
                await repo.forget("PERSON", "hash-9"),
            ]

    assert run(scenario()) == [True, None, False]


def test_ingestion_run_end_to_end_through_the_api(
    api: TestClient, session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    (tmp_path / "catalog.yaml").write_text(
        textwrap.dedent(
            """
            version: "1.0.0"
            sources:
              - name: tiny_json
                location: budgets
                format: json
                quality: {completeness: 4, consistency: 4, actuality: 5, reliability: 4}
                sensitivity: {has_pii: false, pii_flags: []}
                decision: include
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "budgets").mkdir()
    (tmp_path / "budgets" / "BUDGET-2024-0001.json").write_text(
        '{"budget_id":"BUDGET-2024-0001","currency":"EUR","total_amount":1,'
        '"signed_at":"2024-01-01","phases":[]}',
        encoding="utf-8",
    )
    catalog = load_catalog(tmp_path / "catalog.yaml")
    app.dependency_overrides[get_catalog] = lambda: catalog
    app.dependency_overrides[get_filesystem_loader] = lambda: FileSystemLoader(tmp_path)
    app.dependency_overrides[get_parser_registry] = lambda: default_registry()
    app.dependency_overrides[get_async_session_factory] = lambda: session_factory

    response = api.post("/api/v1/ingestion/runs", json={"source_name": "tiny_json"})
    assert response.status_code == 202
    job = api.get(f"/api/v1/ingestion/jobs/{response.json()['job_id']}").json()
    assert (job["status"], job["documents_count"]) == ("completed", 1)
    assert job["finished_at"] is not None
