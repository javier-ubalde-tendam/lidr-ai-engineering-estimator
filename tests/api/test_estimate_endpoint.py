"""POST /v1/estimate/from-transcript: auth, validation, rate limit, soft-fail and errors."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_transcript_estimator
from app.generation.rag.errors import RetrievalError
from app.main import app
from tests.estimator_helpers import (
    TRANSCRIPT,
    ScriptedLLM,
    good_estimate,
    make_estimator,
    patch_retrieve,
    result_with,
)

URL = "/v1/estimate/from-transcript"
ESTIMATE_LIMIT = 10  # @limiter.limit("10/minute")


@pytest.fixture
def estimate_api_key() -> str:
    return "test-estimate-key"  # same fake value as tests/conftest.py


@pytest.fixture
def llm() -> ScriptedLLM:
    # Enough scripted answers for every call the tests below make
    return ScriptedLLM(*[good_estimate("1") for _ in range(ESTIMATE_LIMIT + 2)])


@pytest.fixture
def client(
    llm: ScriptedLLM, monkeypatch: pytest.MonkeyPatch, estimate_api_key: str
) -> Iterator[TestClient]:
    patch_retrieve(monkeypatch, result_with(1))
    estimator = make_estimator(llm)
    app.dependency_overrides[get_transcript_estimator] = lambda: estimator
    yield TestClient(app, headers={"X-API-Key": estimate_api_key})
    app.dependency_overrides.clear()


def post(client: TestClient, **body: Any) -> Any:
    return client.post(URL, json={"transcript": TRANSCRIPT, **body})


def test_returns_the_grounded_estimate(client: TestClient) -> None:
    response = post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == "high" and body["total_engineer_days"] == 10
    assert body["modules"][0]["tasks"][0]["sources"][0]["chunk_id"] == "1"
    assert response.headers["X-Request-ID"]


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_missing_or_wrong_key_is_401(client: TestClient, headers: dict[str, str]) -> None:
    response = TestClient(app).post(URL, json={"transcript": TRANSCRIPT}, headers=headers)

    assert response.status_code == 401


def test_the_retrieval_key_does_not_open_the_estimate_endpoint(
    client: TestClient, retrieval_api_key: str
) -> None:
    response = TestClient(app).post(
        URL, json={"transcript": TRANSCRIPT}, headers={"X-API-Key": retrieval_api_key}
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    "body",
    [
        {"transcript": "too short"},
        {"transcript": "x" * 50_001},
        {"transcript": TRANSCRIPT, "idempotency_key": ""},
        {"transcript": TRANSCRIPT, "idempotency_key": "k" * 129},
        {"transcript": TRANSCRIPT, "unexpected": 1, "idempotency_key": 5},
    ],
)
def test_invalid_payload_is_422(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post(URL, json=body).status_code == 422


def test_rate_limit_is_ten_per_minute_with_retry_after(client: TestClient) -> None:
    statuses = [post(client).status_code for _ in range(ESTIMATE_LIMIT)]
    blocked = post(client)

    assert statuses == [200] * ESTIMATE_LIMIT
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "60"


def test_nothing_retrieved_is_a_200_with_insufficient_confidence(
    llm: ScriptedLLM, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    patch_retrieve(monkeypatch, result_with())

    response = post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == "insufficient"
    assert body["total_engineer_days"] is None and body["modules"] == []
    assert body["insufficient_context_explanation"]
    assert llm.generation_calls == []


def test_pipeline_failure_is_a_502_with_a_generic_message(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    patch_retrieve(monkeypatch, RetrievalError("password=hunter2 leaked in DSN"))

    response = post(client)

    assert response.status_code == 502
    assert response.json() == {"detail": "Failed to produce an estimate."}
