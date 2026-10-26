from collections.abc import Iterator

import openai
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.dependencies import get_ingestion_service, get_retriever
from app.embedding_pipeline.errors import DocumentAlreadyIngestedError, InvalidEmbeddingError
from app.embedding_pipeline.schemas import (
    IngestRequest,
    IngestResponse,
    SearchResponse,
    SearchResult,
)
from app.main import app
from tests.embedding_helpers import make_budget

SOURCE_PATH = "data/budgets_sample.json::BUD-1"


class FakeIngestionService:
    def __init__(self, outcome: IngestResponse | Exception) -> None:
        self._outcome = outcome
        self.requests: list[IngestRequest] = []

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        self.requests.append(request)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class FakeRetriever:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, k: int) -> SearchResponse:
        self.calls.append((query, k))
        if self._error:
            raise self._error
        return SearchResponse(
            query=query,
            k=k,
            search_time_ms=7,
            results=[
                SearchResult(
                    chunk_id=156,
                    document_id=12,
                    chunk_type="budget_component",
                    content="Backend service",
                    distance=0.231,
                    metadata={"scope": "backend", "technologies": ["python", "fastapi"]},
                )
            ],
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


def ingest_payload() -> dict:
    return {
        "source_path": SOURCE_PATH,
        "document_type": "historical_budget",
        "content": make_budget().model_dump(),
    }


OK_RESPONSE = IngestResponse(
    document_id=42, chunks_created=2, embedding_dimension=1536, ingestion_time_ms=1240
)


# --- POST /embeddings/ingest ---


def test_ingest_200_has_exact_contract(client: TestClient) -> None:
    service = FakeIngestionService(OK_RESPONSE)
    app.dependency_overrides[get_ingestion_service] = lambda: service

    response = client.post("/embeddings/ingest", json=ingest_payload())

    assert response.status_code == 200
    assert response.json() == {
        "document_id": 42,
        "chunks_created": 2,
        "embedding_dimension": 1536,
        "ingestion_time_ms": 1240,
    }
    assert service.requests[0].content.budget_id == "BUD-1"


def test_ingest_409_has_detail_and_document_id_at_top_level(client: TestClient) -> None:
    service = FakeIngestionService(DocumentAlreadyIngestedError(42))
    app.dependency_overrides[get_ingestion_service] = lambda: service

    response = client.post("/embeddings/ingest", json=ingest_payload())

    assert response.status_code == 409
    assert response.json() == {"detail": "Document already ingested", "document_id": 42}


def test_ingest_strips_source_path(client: TestClient) -> None:
    service = FakeIngestionService(OK_RESPONSE)
    app.dependency_overrides[get_ingestion_service] = lambda: service
    payload = ingest_payload() | {"source_path": f"  {SOURCE_PATH}  "}

    assert client.post("/embeddings/ingest", json=payload).status_code == 200
    assert service.requests[0].source_path == SOURCE_PATH


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_path": ""},
        {"source_path": "   "},
        {"source_path": "x" * 1001},
        {"document_type": ""},
        {"document_type": "x" * 51},
        {"content": None},
    ],
)
def test_ingest_invalid_fields_return_422(client: TestClient, mutation: dict) -> None:
    service = FakeIngestionService(OK_RESPONSE)
    app.dependency_overrides[get_ingestion_service] = lambda: service

    response = client.post("/embeddings/ingest", json=ingest_payload() | mutation)

    assert response.status_code == 422
    assert service.requests == []


def test_ingest_old_contract_is_rejected(client: TestClient) -> None:
    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService(OK_RESPONSE)

    response = client.post("/embeddings/ingest", json={"budgets": [make_budget().model_dump()]})

    assert response.status_code == 422


def test_ingest_invalid_budget_sector_returns_422(client: TestClient) -> None:
    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService(OK_RESPONSE)
    payload = ingest_payload()
    payload["content"]["client_metadata"]["sector"] = "aerospace"

    assert client.post("/embeddings/ingest", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (openai.OpenAIError("secret detail sk-should-not-leak"), "Embedding service error"),
        (InvalidEmbeddingError("sk-should-not-leak"), "Embedding service error"),
        (SQLAlchemyError("password=estimator sk-should-not-leak"), "Database error"),
        (ConnectionRefusedError("postgres:5432 sk-should-not-leak"), "Database error"),
    ],
)
def test_ingest_external_errors_are_generic(client: TestClient, error: Exception, detail: str) -> None:
    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService(error)

    response = client.post("/embeddings/ingest", json=ingest_payload())

    assert response.status_code == 500
    assert response.json() == {"detail": detail}


def test_openapi_documents_409_and_search_without_api_prefix(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    responses = schema["paths"]["/embeddings/ingest"]["post"]["responses"]
    assert "409" in responses
    assert "/search" in schema["paths"]
    assert "/api/v1/search" not in schema["paths"]


# --- POST /search ---


def test_search_returns_contract_and_serializes_metadata(client: TestClient) -> None:
    retriever = FakeRetriever()
    app.dependency_overrides[get_retriever] = lambda: retriever

    response = client.post("/search", json={"query": "REST API", "k": 3})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"query", "k", "search_time_ms", "results"}
    assert body["results"][0] == {
        "chunk_id": 156,
        "document_id": 12,
        "chunk_type": "budget_component",
        "content": "Backend service",
        "distance": 0.231,
        "metadata": {"scope": "backend", "technologies": ["python", "fastapi"]},
    }
    assert retriever.calls == [("REST API", 3)]


def test_search_default_k_is_5_and_query_is_trimmed(client: TestClient) -> None:
    retriever = FakeRetriever()
    app.dependency_overrides[get_retriever] = lambda: retriever

    response = client.post("/search", json={"query": "  hello  "})

    assert response.status_code == 200
    assert retriever.calls == [("hello", 5)]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   \n\t"},
        {"k": 5},
        {"query": "x", "k": 0},
        {"query": "x", "k": 101},
        {"query": "x", "k": -1},
        {"query": "x", "k": 2.5},
        {"query": "x", "k": "many"},
    ],
)
def test_search_invalid_requests_return_422(client: TestClient, payload: dict) -> None:
    retriever = FakeRetriever()
    app.dependency_overrides[get_retriever] = lambda: retriever

    assert client.post("/search", json=payload).status_code == 422
    assert retriever.calls == []


@pytest.mark.parametrize("k", [1, 100])
def test_search_k_bounds_are_accepted(client: TestClient, k: int) -> None:
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()

    assert client.post("/search", json={"query": "x", "k": k}).status_code == 200


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (openai.OpenAIError("secret detail sk-should-not-leak"), "Embedding service error"),
        (SQLAlchemyError("password=estimator"), "Database error"),
    ],
)
def test_search_external_errors_are_generic(client: TestClient, error: Exception, detail: str) -> None:
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever(error)

    response = client.post("/search", json={"query": "x"})

    assert response.status_code == 500
    assert response.json() == {"detail": detail}
