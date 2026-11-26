"""POST /v1/estimate/graph: auth, validation and status mapping with a fake compiled graph."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app

URL = "/v1/estimate/graph"
BODY = {"transcript": "The client wants a customer portal with payments. " * 4}
HEADERS = {"X-API-Key": "test-estimate-key"}  # same fake value as tests/conftest.py

DEFAULT_RESULT: dict[str, Any] = {
    "status": "validated",
    "estimate": {"components": [], "total_hours": 10.0, "notes": "n"},
    "requirements": ["pay"],
    "components": [{"name": "Payments", "category": "backend"}],
    "budget_matches": [
        {"component": "Payments", "reference_budget_id": None, "amount": 10.0, "distance": 0.1}
    ],
    "errors": [],
}


class FakeGraph:
    def __init__(
        self, result: dict[str, Any] | None = None, error: Exception | None = None
    ) -> None:
        self.result = DEFAULT_RESULT if result is None else result
        self.error = error
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, config))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client() -> Iterator[TestClient]:
    previous = getattr(app.state, "graph", None)
    yield TestClient(app)
    app.state.graph = previous


def test_returns_the_estimate_and_uses_estimation_id_as_thread_id(client: TestClient) -> None:
    graph = FakeGraph()
    app.state.graph = graph

    response = client.post(URL, json={**BODY, "estimation_id": "est-1"}, headers=HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "validated"
    assert body["estimation_id"] == "est-1"
    assert body["components"] == [{"name": "Payments", "category": "backend"}]
    assert graph.calls[0][1] == {"configurable": {"thread_id": "est-1"}}


def test_generates_an_estimation_id_and_defaults_status_to_needs_review(
    client: TestClient,
) -> None:
    app.state.graph = FakeGraph(result={})

    body = client.post(URL, json=BODY, headers=HEADERS).json()

    assert body["status"] == "needs_review"
    assert len(body["estimation_id"]) == 36  # uuid4


def test_requires_the_api_key(client: TestClient) -> None:
    app.state.graph = FakeGraph()
    assert client.post(URL, json=BODY).status_code == 401


def test_503_when_the_graph_is_not_available(client: TestClient) -> None:
    app.state.graph = None
    assert client.post(URL, json=BODY, headers=HEADERS).status_code == 503


def test_422_when_the_transcript_is_too_short(client: TestClient) -> None:
    app.state.graph = FakeGraph()
    assert client.post(URL, json={"transcript": "too short"}, headers=HEADERS).status_code == 422


def test_502_when_the_graph_raises(client: TestClient) -> None:
    app.state.graph = FakeGraph(error=RuntimeError("boom"))
    assert client.post(URL, json=BODY, headers=HEADERS).status_code == 502
