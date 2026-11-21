"""POST /embeddings/compare and the chunker registry that feeds it (everything faked)."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import dependencies
from app.config import Settings
from app.dependencies import ALL_STRATEGIES, get_chunker_builder, get_embedder
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.chunking.strategies import FixedSizeChunker, HierarchicalChunker
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.schemas import Chunk, EmbeddedChunk
from app.main import app
from tests.embedding_helpers import make_budget

URL = "/embeddings/compare"


class FakeEmbedder:
    def embed_one(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [EmbeddedChunk(**c.model_dump(), embedding=[1.0, 0.0]) for c in chunks]


class RaisingChunker(Chunker):
    strategy_name = "exploding"

    def chunk(self, budgets: list[Any]) -> list[Chunk]:
        raise RuntimeError("secret detail that must not reach the client")


def builder_for(registry: dict[str, Chunker]) -> Any:
    def build(names: list[str]) -> list[Chunker]:
        return [registry[name] for name in names]  # KeyError on unknown names, like the real one

    return build


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_chunker_builder] = lambda: builder_for(
        {"structural": JSONStructuralChunker(), "fixed_size": FixedSizeChunker()}
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def body(**extra: Any) -> dict[str, Any]:
    return {"budgets": [make_budget(n_components=2).model_dump()], **extra}


def test_compare_returns_stats_and_top_k_per_strategy(client: TestClient) -> None:
    response = client.post(
        URL, json=body(strategies=["structural", "fixed_size"], queries=["q"], top_k=1)
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data["stats_per_strategy"]) == {"structural", "fixed_size"}
    assert data["stats_per_strategy"]["structural"]["n_chunks"] == 2
    assert len(data["queries_per_strategy"]["fixed_size"][0]["top_k"]) == 1


def test_compare_defaults_to_every_registered_strategy(client: TestClient) -> None:
    seen: list[list[str]] = []

    def build(names: list[str]) -> list[Chunker]:
        seen.append(names)
        return []

    app.dependency_overrides[get_chunker_builder] = lambda: build

    response = client.post(URL, json=body())

    assert response.status_code == 200
    assert seen == [ALL_STRATEGIES]


def test_compare_without_queries_skips_the_top_k_section(client: TestClient) -> None:
    data = client.post(URL, json=body(strategies=["structural"])).json()

    assert data["queries_per_strategy"] == {}


def test_unknown_strategy_is_a_400(client: TestClient) -> None:
    response = client.post(URL, json=body(strategies=["nope"]))

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown strategy: nope"


def test_missing_provider_key_is_reported_by_name(client: TestClient) -> None:
    def build(names: list[str]) -> list[Chunker]:
        raise RuntimeError("ContextualRetrievalChunker requires ANTHROPIC_API_KEY.")

    app.dependency_overrides[get_chunker_builder] = lambda: build

    response = client.post(URL, json=body(strategies=["contextual_retrieval"]))

    assert response.status_code == 500
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_chunker_failure_is_a_generic_500(client: TestClient) -> None:
    app.dependency_overrides[get_chunker_builder] = lambda: builder_for({"x": RaisingChunker()})

    response = client.post(URL, json=body(strategies=["x"]))

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to run chunking comparison"}


@pytest.mark.parametrize(
    "payload",
    [
        {"budgets": []},
        {"budgets": [make_budget().model_dump()], "top_k": 0},
        {"budgets": [make_budget().model_dump()], "queries": ["q"] * 21},
    ],
)
def test_invalid_requests_are_422(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post(URL, json=payload).status_code == 422


def test_ingest_route_is_still_registered() -> None:
    paths = app.openapi()["paths"]

    assert "post" in paths["/embeddings/ingest"]
    assert "post" in paths["/embeddings/compare"]


# --- registry -------------------------------------------------------------------------------


class FakeModels:
    def effective(self, key: str) -> str:
        return f"{key}-resolved"


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_registry_lists_the_structural_and_the_seven_comparison_strategies() -> None:
    assert ALL_STRATEGIES == [
        "structural",
        "fixed_size",
        "recursive",
        "sentence_window",
        "semantic",
        "propositional",
        "contextual_retrieval",
        "hierarchical",
    ]


def test_build_chunkers_builds_keyless_strategies_in_order() -> None:
    chunkers = dependencies.build_chunkers(["hierarchical", "fixed_size", "structural"])

    assert [c.strategy_name for c in chunkers] == ["hierarchical", "fixed_size", "structural"]
    assert isinstance(chunkers[0], HierarchicalChunker)


def test_build_chunkers_rejects_unknown_names() -> None:
    with pytest.raises(KeyError):
        dependencies.build_chunkers(["nope"])


def test_contextual_retrieval_requires_the_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dependencies, "get_settings", lambda: settings(OPENAI_API_KEY="sk-x", ANTHROPIC_API_KEY=None)
    )
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        dependencies.build_chunkers(["contextual_retrieval"])


def test_llm_strategies_require_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: settings(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="ak-x", OPENAI_API_KEY=None),
    )

    for name in ("propositional", "semantic"):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            dependencies.build_chunkers([name])


def test_llm_strategies_use_the_runtime_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: settings(OPENAI_API_KEY="sk-x", ANTHROPIC_API_KEY="ak-x"),
    )
    monkeypatch.setattr(dependencies, "get_runtime_config", lambda: FakeModels())

    propositional, contextual = dependencies.build_chunkers(["propositional", "contextual_retrieval"])

    assert propositional._model == "PROPOSITIONAL_CHUNKER_MODEL-resolved"  # type: ignore[attr-defined]
    assert contextual._model == "CONTEXTUAL_CHUNKER_MODEL-resolved"  # type: ignore[attr-defined]
