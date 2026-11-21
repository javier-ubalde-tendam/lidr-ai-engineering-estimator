"""Rate limiting (429 + Retry-After, per-key buckets) and the X-Request-ID middleware."""

import openai
import pytest
import structlog
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.dependencies import get_embedder
from app.main import app

SEARCH_BODY = {"query_text": "ecommerce storefront with card checkout"}
RETRIEVAL_LIMIT = 120  # @limiter.limit("120/minute") on POST /v1/retrieval/search


@pytest.fixture
def client(retrieval_stubs: None) -> TestClient:
    return TestClient(app)


def test_retrieval_returns_429_with_retry_after_beyond_the_limit(
    client: TestClient, retrieval_api_key: str
) -> None:
    headers = {"X-API-Key": retrieval_api_key}

    statuses = [
        client.post("/v1/retrieval/search", json=SEARCH_BODY, headers=headers).status_code
        for _ in range(RETRIEVAL_LIMIT)
    ]
    blocked = client.post("/v1/retrieval/search", json=SEARCH_BODY, headers=headers)

    assert statuses == [200] * RETRIEVAL_LIMIT
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "60"
    assert blocked.json()["retry_after_seconds"] == 60
    assert blocked.headers["X-Request-ID"]  # the middleware also wraps the 429


def test_the_limit_is_per_api_key(client: TestClient, retrieval_api_key: str) -> None:
    exhausted = {"X-API-Key": retrieval_api_key}
    for _ in range(RETRIEVAL_LIMIT + 1):
        client.post("/v1/retrieval/search", json=SEARCH_BODY, headers=exhausted)

    other_key = client.post(
        "/v1/retrieval/search", json=SEARCH_BODY, headers={"X-API-Key": "another-fake-key"}
    )

    # A different key has its own bucket: it is rate-limit-free but fails auth (401), not 429
    assert other_key.status_code == 401


def test_request_id_is_generated_when_absent() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert len(response.headers["X-Request-ID"]) == 36  # uuid4


def test_request_id_is_echoed_when_valid() -> None:
    response = TestClient(app).get("/health", headers={"X-Request-ID": "trace-abc-123"})

    assert response.headers["X-Request-ID"] == "trace-abc-123"


def test_request_id_is_replaced_when_oversized() -> None:
    response = TestClient(app).get("/health", headers={"X-Request-ID": "x" * 129})

    assert response.headers["X-Request-ID"] != "x" * 129
    assert len(response.headers["X-Request-ID"]) == 36


def test_request_id_is_bound_to_the_logs_of_the_request(
    client: TestClient, retrieval_api_key: str
) -> None:
    class FailingEmbedder:
        def embed_one(self, text: str) -> list[float]:
            raise openai.OpenAIError("boom")

    app.dependency_overrides[get_embedder] = lambda: FailingEmbedder()
    processors = [structlog.contextvars.merge_contextvars]  # same as configure_logging()

    with capture_logs(processors=processors) as logs:
        response = client.post(
            "/v1/retrieval/search",
            json=SEARCH_BODY,
            headers={"X-API-Key": retrieval_api_key, "X-Request-ID": "bound-id"},
        )

    assert response.status_code == 502
    failure = next(log for log in logs if log["event"] == "retrieval_embedding_failed")
    assert failure["request_id"] == "bound-id"
    assert "request_id" not in structlog.contextvars.get_contextvars()  # unbound afterwards