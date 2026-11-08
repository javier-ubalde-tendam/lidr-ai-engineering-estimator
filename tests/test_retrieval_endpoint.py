"""POST /v1/retrieval/search with dependency_overrides (no DB, no OpenAI, no torch)."""

from collections.abc import Iterator
from typing import Any

import openai
import pytest
from fastapi.testclient import TestClient

from app.config import EMBEDDING_DIMENSION
from app.db.session import get_db_session
from app.dependencies import get_chunk_store, get_embedder, get_reranker
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import RetrievalResult, RetrievedChunk
from app.main import app
from app.routers import retrieval as retrieval_router

QUERY = "E-commerce checkout with cart and payments"


class FakeEmbedder:
    def __init__(self, error: Exception | None = None, dimension: int = EMBEDDING_DIMENSION):
        self._error = error
        self._dimension = dimension
        self.queries: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self.queries.append(text)
        if self._error:
            raise self._error
        return [0.1] * self._dimension


class RetrieveSpy:
    """Replaces pipeline.retrieve inside the router: records kwargs, returns a canned result."""

    def __init__(self, outcome: RetrievalResult | Exception) -> None:
        self._outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, session: Any, store: Any, **kwargs: Any) -> RetrievalResult:
        self.calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome.model_copy(
            update={"search_mode": kwargs["search_mode"], "rerank": kwargs["rerank"]}
        )


ONE_CHUNK = RetrievalResult(
    chunks=[
        RetrievedChunk(
            id=7,
            content="Cart and checkout service",
            sector="ecommerce",
            project_year=2024,
            chunk_type="budget_component",
            distance=0.21,
            budget_id="BUD-2024-005",
        )
    ],
    low_confidence=False,
    candidates_evaluated=60,
)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def client(embedder: FakeEmbedder) -> Iterator[TestClient]:
    async def no_session() -> Any:
        yield None

    app.dependency_overrides[get_db_session] = no_session
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_chunk_store] = lambda: object()
    app.dependency_overrides[get_reranker] = lambda: CrossEncoderReranker("fake")
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> RetrieveSpy:
    spy = RetrieveSpy(ONE_CHUNK)
    monkeypatch.setattr(retrieval_router, "retrieve", spy)
    return spy


def post(client: TestClient, **body: Any) -> Any:
    return client.post("/v1/retrieval/search", json={"query_text": QUERY, **body})


def test_200_returns_chunks_and_embeds_the_query(
    client: TestClient, spy: RetrieveSpy, embedder: FakeEmbedder
) -> None:
    response = post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["chunks"][0]["budget_id"] == "BUD-2024-005"
    assert body["low_confidence"] is False
    assert embedder.queries == [QUERY]
    assert spy.calls[0]["query_text"] == QUERY
    assert len(spy.calls[0]["query_embedding"]) == EMBEDDING_DIMENSION


@pytest.mark.parametrize(
    ("search_mode", "rerank"),
    [("vector", False), ("hybrid", False), ("vector", True), ("hybrid", True)],
    ids=["A", "B", "C", "D"],
)
def test_the_four_configurations_from_the_request_body(
    client: TestClient, spy: RetrieveSpy, search_mode: str, rerank: bool
) -> None:
    body = post(client, search_mode=search_mode, rerank=rerank).json()

    assert (spy.calls[0]["search_mode"], spy.calls[0]["rerank"]) == (search_mode, rerank)
    assert (body["search_mode"], body["rerank"]) == (search_mode, rerank)


@pytest.mark.parametrize(
    ("env_mode", "env_rerank", "expected"),
    [
        ("vector", "false", ("vector", False)),
        ("hybrid", "false", ("hybrid", False)),
        ("vector", "true", ("vector", True)),
        ("hybrid", "true", ("hybrid", True)),
    ],
    ids=["A", "B", "C", "D"],
)
def test_the_four_configurations_from_environment_variables(
    client: TestClient,
    spy: RetrieveSpy,
    monkeypatch: pytest.MonkeyPatch,
    env_mode: str,
    env_rerank: str,
    expected: tuple[str, bool],
) -> None:
    monkeypatch.setenv("RETRIEVAL_SEARCH_MODE", env_mode)
    monkeypatch.setenv("RERANKER_ENABLED", env_rerank)

    post(client)

    assert (spy.calls[0]["search_mode"], spy.calls[0]["rerank"]) == expected


def test_request_fields_beat_environment(
    client: TestClient, spy: RetrieveSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RETRIEVAL_SEARCH_MODE", "hybrid")
    monkeypatch.setenv("RERANKER_ENABLED", "true")

    post(client, search_mode="vector", rerank=False)

    assert (spy.calls[0]["search_mode"], spy.calls[0]["rerank"]) == ("vector", False)


def test_settings_knobs_are_forwarded(
    client: TestClient, spy: RetrieveSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RETRIEVAL_RECALL_TOP_K", "40")
    monkeypatch.setenv("RERANK_TOP_N", "3")
    monkeypatch.setenv("RRF_K", "10")
    monkeypatch.setenv("LEXICAL_MATCH_MODE", "and")

    post(client, top_k=4, distance_threshold=1.5)

    call = spy.calls[0]
    assert (call["recall_k"], call["rerank_top_n"], call["rrf_k"]) == (40, 3, 10)
    assert call["lexical_match_mode"] == "and"
    assert (call["top_k"], call["distance_threshold"]) == (4, 1.5)


def test_soft_fail_is_200_with_low_confidence(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = RetrievalResult(chunks=[], low_confidence=True, candidates_evaluated=60)
    monkeypatch.setattr(retrieval_router, "retrieve", RetrieveSpy(empty))

    response = post(client)

    assert response.status_code == 200
    assert response.json()["low_confidence"] is True and response.json()["chunks"] == []


def test_retrieval_error_is_502_with_generic_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_detail = RetrievalError("password=hunter2 leaked in DSN")
    monkeypatch.setattr(retrieval_router, "retrieve", RetrieveSpy(secret_detail))

    response = post(client)

    assert response.status_code == 502
    assert response.json() == {"detail": "Retrieval service error"}


@pytest.mark.parametrize(
    "embedder_override",
    [
        FakeEmbedder(error=openai.APIConnectionError(request=None)),  # type: ignore[arg-type]
        FakeEmbedder(dimension=3),
    ],
    ids=["openai-error", "invalid-vector"],
)
def test_embedding_errors_are_502(
    client: TestClient, spy: RetrieveSpy, embedder_override: FakeEmbedder
) -> None:
    app.dependency_overrides[get_embedder] = lambda: embedder_override

    response = post(client)

    assert response.status_code == 502
    assert response.json() == {"detail": "Embedding service error"}
    assert spy.calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"query_text": "short"},
        {"query_text": "x" * 2001},
        {"top_k": 0},
        {"top_k": 31},
        {"distance_threshold": -0.1},
        {"distance_threshold": 2.1},
        {"search_mode": "keyword"},
        {"sectors": ["finance"]},  # metadata filters are not exposed
    ],
)
def test_invalid_requests_are_422(
    client: TestClient, spy: RetrieveSpy, mutation: dict[str, Any]
) -> None:
    response = client.post("/v1/retrieval/search", json={"query_text": QUERY, **mutation})

    assert response.status_code == 422
    assert spy.calls == []


def test_existing_search_endpoint_is_still_registered() -> None:
    paths = set(app.openapi()["paths"])

    assert {"/search", "/embeddings/ingest", "/v1/retrieval/search"} <= paths
