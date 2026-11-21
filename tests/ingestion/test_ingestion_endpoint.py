"""HTTP-level tests of the ingestion endpoints.

Postgres is bypassed: ``JobsRepository`` is replaced by an in-memory fake with the same async
contract (pending -> running -> completed | failed) and the sessions are stubs, so the router and
the orchestrator wiring are exercised in isolation. The real repository has its own integration
tests (tests/integration/test_ingestion_repositories_db.py).
"""

import textwrap
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from app.dependencies import (
    get_async_session_factory,
    get_catalog,
    get_filesystem_loader,
    get_parser_registry,
)
from app.foundation.persistence.database import get_db_session
from app.ingestion.catalog.loader import load_catalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.registry import default_registry
from app.main import app


class _FakeJob:
    def __init__(self, source_name: str) -> None:
        self.job_id = uuid.uuid4()
        self.source_name = source_name
        self.status = "pending"
        self.documents_count = 0
        self.error_message: str | None = None
        self.started_at = datetime.now(UTC)
        self.finished_at: datetime | None = None


class _FakeJobsRepo:
    """Class-level store: the BackgroundTask and the GET endpoint build separate instances."""

    store: ClassVar[dict[uuid.UUID, _FakeJob]] = {}

    def __init__(self, session: Any = None) -> None:
        pass

    async def create(self, *, source_name: str) -> _FakeJob:
        job = _FakeJob(source_name)
        self.store[job.job_id] = job
        return job

    async def get(self, job_id: uuid.UUID) -> _FakeJob | None:
        return self.store.get(job_id)

    async def mark_running(self, job_id: uuid.UUID) -> None:
        self.store[job_id].status = "running"

    async def mark_completed(self, job_id: uuid.UUID, *, documents_count: int) -> None:
        job = self.store[job_id]
        job.status = "completed"
        job.documents_count = documents_count
        job.finished_at = datetime.now(UTC)

    async def mark_failed(self, job_id: uuid.UUID, *, error_message: str) -> None:
        job = self.store[job_id]
        job.status = "failed"
        job.error_message = error_message
        job.finished_at = datetime.now(UTC)


class _FakeSessionFactory:
    """Stands in for async_sessionmaker: ``async with factory() as session``."""

    def __call__(self) -> "_FakeSessionFactory":
        return self

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc_info: object) -> None:
        return None


_CATALOG = """
version: "1.0.0"
sources:
  - name: tiny_json
    location: budgets
    format: json
    quality: {completeness: 4, consistency: 4, actuality: 5, reliability: 4}
    sensitivity: {has_pii: false, pii_flags: []}
    decision: include
  - name: missing_folder
    location: does_not_exist
    format: json
    quality: {completeness: 4, consistency: 4, actuality: 5, reliability: 4}
    sensitivity: {has_pii: false, pii_flags: []}
    decision: include
  - name: excluded_xlsx
    location: rate_card.xlsx
    format: xlsx
    quality: {completeness: 4, consistency: 5, actuality: 1, reliability: 4}
    sensitivity: {has_pii: false, pii_flags: []}
    decision: exclude
    decision_reason: "actuality=1: outdated"
"""


@pytest.fixture
def ingestion_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    (tmp_path / "catalog.yaml").write_text(textwrap.dedent(_CATALOG), encoding="utf-8")
    budgets = tmp_path / "budgets"
    budgets.mkdir()
    (budgets / "BUDGET-2024-0001.json").write_text(
        '{"budget_id":"BUDGET-2024-0001","currency":"EUR",'
        '"total_amount":100,"signed_at":"2024-01-01","phases":[]}',
        encoding="utf-8",
    )
    catalog = load_catalog(tmp_path / "catalog.yaml")
    loader = FileSystemLoader(data_root=tmp_path)
    registry = default_registry()

    async def no_session() -> Any:
        yield None

    app.dependency_overrides[get_catalog] = lambda: catalog
    app.dependency_overrides[get_filesystem_loader] = lambda: loader
    app.dependency_overrides[get_parser_registry] = lambda: registry
    app.dependency_overrides[get_db_session] = no_session
    app.dependency_overrides[get_async_session_factory] = lambda: _FakeSessionFactory()
    monkeypatch.setattr("app.api.ingestion.JobsRepository", _FakeJobsRepo)
    _FakeJobsRepo.store.clear()

    yield TestClient(app)
    app.dependency_overrides.clear()


def test_post_run_returns_202_and_creates_job(ingestion_client: TestClient) -> None:
    response = ingestion_client.post("/api/v1/ingestion/runs", json={"source_name": "tiny_json"})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["source_name"] == "tiny_json"
    uuid.UUID(body["job_id"])


def test_get_job_reflects_completed_state(ingestion_client: TestClient) -> None:
    # TestClient runs the BackgroundTask before returning the POST response
    job_id = ingestion_client.post(
        "/api/v1/ingestion/runs", json={"source_name": "tiny_json"}
    ).json()["job_id"]
    body = ingestion_client.get(f"/api/v1/ingestion/jobs/{job_id}").json()
    assert body["status"] == "completed"
    assert body["documents_count"] == 1
    assert body["finished_at"] is not None


def test_unresolvable_location_marks_the_job_failed(ingestion_client: TestClient) -> None:
    response = ingestion_client.post("/api/v1/ingestion/runs", json={"source_name": "missing_folder"})
    assert response.status_code == 202  # the failure is asynchronous: it shows up on the job
    body = ingestion_client.get(f"/api/v1/ingestion/jobs/{response.json()['job_id']}").json()
    assert body["status"] == "failed"
    assert "does not resolve" in body["error_message"]


def test_unknown_source_returns_404(ingestion_client: TestClient) -> None:
    response = ingestion_client.post("/api/v1/ingestion/runs", json={"source_name": "nope"})
    assert response.status_code == 404
    assert response.json()["detail"]["reason"] == "unknown_source"


def test_excluded_source_returns_400(ingestion_client: TestClient) -> None:
    response = ingestion_client.post("/api/v1/ingestion/runs", json={"source_name": "excluded_xlsx"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["reason"] == "source_not_included"
    assert detail["decision"] == "exclude"
    assert not _FakeJobsRepo.store  # a rejected request never creates a job


def test_extra_fields_in_the_request_are_rejected(ingestion_client: TestClient) -> None:
    response = ingestion_client.post(
        "/api/v1/ingestion/runs", json={"source_name": "tiny_json", "force": True}
    )
    assert response.status_code == 422


def test_get_unknown_job_returns_404(ingestion_client: TestClient) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    assert ingestion_client.get(f"/api/v1/ingestion/jobs/{missing}").status_code == 404
