"""POST /embeddings/index/runs, GET /embeddings/index/jobs/{id} and GET /embeddings/index/stats.

The corpus service runs against a fake ingester (no DB, no embeddings) and the stats endpoint
against a fake store; the router, the background task and the in-memory job store are the real ones.
"""

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_chunk_store, get_corpus_index_service, get_index_job_store
from app.foundation.persistence.database import get_db_session
from app.generation.rag.index_jobs import IndexJobStore
from app.generation.rag.index_service import CorpusIndexService
from app.generation.rag.schemas import IndexJobView
from app.main import app
from tests.index_helpers import FakeIngester, budget_dict


class FakeStore:
    async def corpus_stats(self, session: Any) -> list[tuple[str, int, int, bool]]:
        return [("budget", 10, 64, True), ("transcript", 2, 11, True), ("technical_doc", 1, 8, False)]


@pytest.fixture
def ingester() -> FakeIngester:
    return FakeIngester(duplicates={"DUP"}, fail_on="BOOM")


@pytest.fixture
def jobs() -> IndexJobStore:
    return IndexJobStore()


@pytest.fixture
def client(ingester: FakeIngester, jobs: IndexJobStore) -> Iterator[TestClient]:
    async def no_session() -> Any:
        yield None

    app.dependency_overrides[get_db_session] = no_session
    app.dependency_overrides[get_corpus_index_service] = lambda: CorpusIndexService(ingester)
    app.dependency_overrides[get_index_job_store] = lambda: jobs
    app.dependency_overrides[get_chunk_store] = lambda: FakeStore()
    yield TestClient(app)
    app.dependency_overrides.clear()


def run(client: TestClient, *documents: dict[str, Any], **extra: Any) -> Any:
    return client.post("/embeddings/index/runs", json={"documents": list(documents), **extra})


def job_of(client: TestClient, response: Any) -> dict[str, Any]:
    return client.get(f"/embeddings/index/jobs/{response.json()['job_id']}").json()


def test_a_run_returns_202_with_the_job_id_and_the_total(client: TestClient) -> None:
    response = run(client, budget_dict("NEW-1"), budget_dict("NEW-2"))

    assert response.status_code == 202
    body = response.json()
    assert body["documents_total"] == 2 and body["status"] == "pending"
    uuid.UUID(body["job_id"])


def test_the_job_completes_with_its_progress(client: TestClient) -> None:
    # TestClient runs the background task before returning the response
    response = run(client, budget_dict("NEW-1", 3))

    job = job_of(client, response)
    assert job["status"] == "completed" and job["documents_processed"] == 1
    assert job["error_message"] is None and job["finished_at"] is not None
    IndexJobView.model_validate(job)


def test_the_document_and_chunk_types_reach_the_ingester(
    client: TestClient, ingester: FakeIngester
) -> None:
    run(client, budget_dict("NEW-1"), document_type="historical_task_breakdown", chunk_type="historical_task")

    request = ingester.requests[0]
    assert request.source_path == "corpus-expansion::NEW-1"
    assert request.document_type == "historical_task_breakdown"
    assert request.chunk_type == "historical_task"


def test_a_duplicate_is_skipped_and_counted_as_processed(client: TestClient) -> None:
    response = run(client, budget_dict("NEW-1"), budget_dict("DUP"))

    job = job_of(client, response)
    assert job["status"] == "completed" and job["documents_processed"] == 2


def test_a_failure_marks_the_job_failed_with_the_error(client: TestClient) -> None:
    response = run(client, budget_dict("NEW-1"), budget_dict("BOOM"))

    job = job_of(client, response)
    assert job["status"] == "failed"
    assert "embedding provider down" in job["error_message"]
    assert job["documents_processed"] == 1  # progress made before the failure is kept


def test_stats_report_the_corpus_size_and_the_index_state(client: TestClient) -> None:
    response = client.get("/embeddings/index/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks"] == 64 + 11 + 8
    by_name = {c["collection"]: c for c in body["collections"]}
    assert by_name["budget"] == {
        "collection": "budget",
        "documents": 10,
        "chunks": 64,
        "hnsw_indexed": True,
    }
    assert by_name["technical_doc"]["hnsw_indexed"] is False


@pytest.mark.parametrize(
    "body",
    [
        {"documents": []},
        {"documents": [budget_dict("A")], "chunk_type": ""},
        {"documents": [budget_dict(f"B-{i}") for i in range(201)]},
    ],
)
def test_an_invalid_batch_is_422(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/embeddings/index/runs", json=body).status_code == 422


def test_an_unknown_job_is_404(client: TestClient) -> None:
    assert client.get(f"/embeddings/index/jobs/{uuid.uuid4()}").status_code == 404
    assert client.get("/embeddings/index/jobs/not-a-uuid").status_code == 422


def test_the_job_store_evicts_the_oldest_jobs() -> None:
    store = IndexJobStore(max_jobs=2)

    first, second, third = (store.create(documents_total=1) for _ in range(3))

    assert store.get(first.job_id) is None
    assert store.get(second.job_id) is second and store.get(third.job_id) is third
