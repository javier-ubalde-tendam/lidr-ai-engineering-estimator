"""POST /v1/retrieval/advanced-search: auth, validation, stage precedence, response and errors.

The pipeline itself is covered in tests/generation/rag/test_advanced_pipeline.py; here it is
replaced by a spy so only the HTTP layer is exercised.
"""

from collections.abc import Iterator
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routers import retrieval_advanced
from app.dependencies import (
    get_async_session_factory,
    get_chunk_store,
    get_embedder,
    get_llm_wrapper,
    get_reranker,
    get_runtime_config,
    get_runtime_retrieval_config,
)
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.advanced_pipeline import AdvancedRetrievalOutcome
from app.generation.rag.retrieval.collections import Collection
from app.generation.rag.retrieval.query_transform import SubQuery
from app.generation.rag.retrieval.router import RoutingDecision
from app.generation.rag.schemas import RetrievedChunk
from app.main import app
from tests.runtime_helpers import FakeRuntimeModelConfig, FakeRuntimeRetrievalConfig

URL = "/v1/retrieval/advanced-search"
BODY = {"query_text": "what did the client say about the Stripe checkout"}
LIMIT = 120  # @limiter.limit("120/minute")


class PipelineSpy:
    """Replaces ``advanced_retrieve``: records the call and returns a canned outcome."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    async def __call__(self, **kwargs: Any) -> AdvancedRetrievalOutcome:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return AdvancedRetrievalOutcome(
            chunks=[
                RetrievedChunk(
                    id=3,
                    content="Stripe Connect kickoff",
                    chunk_type="meeting_segment",
                    distance=0.2,
                    collection="transcript",
                    source_id="TR-2024-002",
                    relevance_score=0.9,
                    document_date=date(2024, 3, 20),
                )
            ],
            routing=RoutingDecision([Collection.TRANSCRIPT], "rules", "Vocabulary matched."),
            technique="expand",
            subqueries=[SubQuery(topic="original", query=BODY["query_text"])],
            cardinality={"transcript": 12},
            low_confidence=False,
        )


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[PipelineSpy]:
    pipeline = PipelineSpy()
    monkeypatch.setattr(retrieval_advanced, "advanced_retrieve", pipeline)
    app.dependency_overrides[get_embedder] = lambda: object()
    app.dependency_overrides[get_chunk_store] = lambda: object()
    app.dependency_overrides[get_reranker] = lambda: object()
    app.dependency_overrides[get_llm_wrapper] = lambda: object()
    app.dependency_overrides[get_async_session_factory] = lambda: object()
    app.dependency_overrides[get_runtime_config] = lambda: FakeRuntimeModelConfig()
    app.dependency_overrides[get_runtime_retrieval_config] = lambda: FakeRuntimeRetrievalConfig()
    yield pipeline
    app.dependency_overrides.clear()


@pytest.fixture
def client(spy: PipelineSpy, retrieval_api_key: str) -> TestClient:
    return TestClient(app, headers={"X-API-Key": retrieval_api_key})


def test_requires_the_retrieval_key(spy: PipelineSpy) -> None:
    response = TestClient(app).post(URL, json=BODY)

    assert response.status_code == 401
    assert spy.calls == []  # rejected before any retrieval work


def test_another_services_key_is_rejected(spy: PipelineSpy) -> None:
    response = TestClient(app).post(URL, json=BODY, headers={"X-API-Key": "test-estimate-key"})

    assert response.status_code == 401


def test_returns_chunks_and_how_they_were_retrieved(client: TestClient) -> None:
    response = client.post(URL, json=BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["low_confidence"] is False
    assert body["routing"] == {
        "level": "rules",
        "targets": ["transcript"],
        "reason": "Vocabulary matched.",
    }
    assert body["technique"] == "expand"
    assert body["subqueries"] == [{"topic": "original", "query": BODY["query_text"]}]
    assert body["cardinality"] == {"transcript": 12}
    (chunk,) = body["chunks"]
    assert (chunk["collection"], chunk["source_id"], chunk["relevance_score"]) == (
        "transcript",
        "TR-2024-002",
        0.9,
    )
    assert chunk["document_date"] == "2024-03-20"
    assert response.headers["X-Request-ID"]


def test_request_fields_reach_the_pipeline(client: TestClient, spy: PipelineSpy) -> None:
    client.post(
        URL,
        json={
            **BODY,
            "collections": ["budget", "technical_doc"],
            "top_k": 3,
            "sectors": ["finance"],
            "technologies": ["node"],
            "min_date": "2023-01-01",
            "doc_version": "v2",
        },
    )

    (call,) = spy.calls
    assert call["explicit_collections"] == [Collection.BUDGET, Collection.TECHNICAL_DOC]
    assert call["stages"].top_k == 3
    hard_filters = call["hard_filters"]
    assert hard_filters.sectors == ("finance",) and hard_filters.technologies == ("node",)
    assert hard_filters.min_date == date(2023, 1, 1) and hard_filters.version == "v2"
    assert call["query_text"] == BODY["query_text"]


def test_stage_toggles_follow_request_then_runtime_then_environment(
    client: TestClient, spy: PipelineSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RETRIEVAL_ROUTING_ENABLED", "true")
    monkeypatch.setenv("TEMPORAL_DECAY_ENABLED", "false")
    app.dependency_overrides[get_runtime_retrieval_config] = lambda: FakeRuntimeRetrievalConfig(
        search_mode="hybrid", rerank=True, query_transform=False
    )

    client.post(URL, json={**BODY, "rerank": False, "routing_enabled": False})
    stages = spy.calls[-1]["stages"]

    assert stages.rerank is False  # request beats the runtime override
    assert stages.routing_enabled is False  # request beats the environment
    assert stages.search_mode == "hybrid"  # runtime override (no request field)
    assert stages.query_transform_enabled is False  # runtime override
    assert stages.temporal_decay_enabled is False  # environment


@pytest.mark.parametrize(
    "payload",
    [
        {"query_text": "short"},
        {**BODY, "top_k": 0},
        {**BODY, "top_k": 31},
        {**BODY, "collections": ["emails"]},
        {**BODY, "search_mode": "fuzzy"},
        {**BODY, "sectors": ["not-a-sector"]},
        {**BODY, "unknown_field": 1},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, spy: PipelineSpy, payload: dict) -> None:
    assert client.post(URL, json=payload).status_code == 422
    assert spy.calls == []


def test_retrieval_failure_is_a_generic_502(client: TestClient, spy: PipelineSpy) -> None:
    spy.error = RetrievalError("postgres password=hunter2 refused")

    response = client.post(URL, json=BODY)

    assert response.status_code == 502
    assert response.json() == {"detail": "Retrieval service error"}  # no internal detail leaks


def test_is_rate_limited_at_120_per_minute(client: TestClient) -> None:
    statuses = [client.post(URL, json=BODY).status_code for _ in range(LIMIT + 1)]

    assert statuses[:LIMIT] == [200] * LIMIT
    assert statuses[LIMIT] == 429
