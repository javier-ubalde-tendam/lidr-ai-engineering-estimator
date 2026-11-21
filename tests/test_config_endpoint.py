"""HTTP tests for GET/PUT /api/v1/config/{models,retrieval} (runtime settings)."""

from collections.abc import Iterator

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.dependencies import get_runtime_config, get_runtime_retrieval_config
from app.foundation.llm.runtime_config import (
    MODEL_KEYS,
    RETRIEVAL_KEYS,
    RuntimeConfigUnavailable,
    RuntimeModelConfig,
    RuntimeRetrievalConfig,
)
from app.main import app


def make_settings(**overrides: object) -> Settings:
    defaults = {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "sk-ant-test"}
    return Settings(_env_file=None, **{**defaults, **overrides})


@pytest.fixture
def fake_store() -> Iterator[RuntimeModelConfig]:
    settings = make_settings()
    store = RuntimeModelConfig(fakeredis.FakeRedis(decode_responses=True), settings)
    app.dependency_overrides[get_runtime_config] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    yield store
    app.dependency_overrides.clear()


@pytest.fixture
def retrieval_store() -> Iterator[RuntimeRetrievalConfig]:
    settings = make_settings()
    store = RuntimeRetrievalConfig(fakeredis.FakeRedis(decode_responses=True), settings)
    app.dependency_overrides[get_runtime_retrieval_config] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    yield store
    app.dependency_overrides.clear()


@pytest.fixture
def client(fake_store: RuntimeModelConfig) -> TestClient:
    return TestClient(app)


def test_get_models_returns_full_snapshot(client: TestClient) -> None:
    response = client.get("/api/v1/config/models")

    assert response.status_code == 200
    body = response.json()
    assert body["models"]["GENERATION_MODEL"] == {
        "effective": "gpt-5",
        "default": "gpt-5",
        "overridden": False,
    }
    assert set(body["models"]) == set(MODEL_KEYS)
    assert "gpt-4o" in body["available_models"]
    assert "claude-sonnet-4-5" in body["available_models"]
    assert body["embedding_model"] == "text-embedding-3-small"


def test_available_models_are_filtered_by_configured_keys(fake_store: RuntimeModelConfig) -> None:
    settings = make_settings(ANTHROPIC_API_KEY=None)
    app.dependency_overrides[get_settings] = lambda: settings

    body = TestClient(app).get("/api/v1/config/models").json()

    assert all(not model.startswith("claude") for model in body["available_models"])
    assert "gpt-4o-mini" in body["available_models"]


def test_put_models_overrides_and_returns_fresh_snapshot(
    client: TestClient, fake_store: RuntimeModelConfig
) -> None:
    response = client.put("/api/v1/config/models", json={"models": {"GENERATION_MODEL": "gpt-4o"}})

    assert response.status_code == 200
    assert response.json()["models"]["GENERATION_MODEL"] == {
        "effective": "gpt-4o",
        "default": "gpt-5",
        "overridden": True,
    }
    assert fake_store.effective("GENERATION_MODEL") == "gpt-4o"


def test_put_models_null_resets_override(
    client: TestClient, fake_store: RuntimeModelConfig
) -> None:
    fake_store.set("GENERATION_MODEL", "gpt-4o")

    response = client.put("/api/v1/config/models", json={"models": {"GENERATION_MODEL": None}})

    assert response.status_code == 200
    assert response.json()["models"]["GENERATION_MODEL"]["overridden"] is False
    assert fake_store.get("GENERATION_MODEL") is None


def test_put_models_unknown_key_is_422(client: TestClient) -> None:
    response = client.put("/api/v1/config/models", json={"models": {"EMBEDDING_MODEL": "gpt-4o"}})

    assert response.status_code == 422
    assert "Unknown model key" in response.json()["detail"]


def test_put_models_outside_catalog_is_422(client: TestClient) -> None:
    response = client.put(
        "/api/v1/config/models", json={"models": {"GENERATION_MODEL": "gpt-99-ultra"}}
    )

    assert response.status_code == 422
    assert "not in the catalog" in response.json()["detail"]


def test_put_models_with_missing_provider_key_is_400(fake_store: RuntimeModelConfig) -> None:
    settings = make_settings(ANTHROPIC_API_KEY=None)
    app.dependency_overrides[get_settings] = lambda: settings

    response = TestClient(app).put(
        "/api/v1/config/models", json={"models": {"GENERATION_MODEL": "claude-sonnet-4-5"}}
    )

    assert response.status_code == 400
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_put_models_is_all_or_nothing(client: TestClient, fake_store: RuntimeModelConfig) -> None:
    response = client.put(
        "/api/v1/config/models",
        json={"models": {"GENERATION_MODEL": "gpt-4o", "BOGUS_KEY": "gpt-4o"}},
    )

    assert response.status_code == 422
    assert fake_store.is_overridden("GENERATION_MODEL") is False


def test_get_retrieval_returns_every_toggle(retrieval_store: RuntimeRetrievalConfig) -> None:
    response = TestClient(app).get("/api/v1/config/retrieval")

    assert response.status_code == 200
    body = response.json()
    assert set(body["retrieval"]) == set(RETRIEVAL_KEYS)
    assert body["retrieval"]["RETRIEVAL_SEARCH_MODE"]["effective"] == "vector"
    assert body["reranker_model"]


def test_put_retrieval_applies_only_the_sent_fields(
    retrieval_store: RuntimeRetrievalConfig,
) -> None:
    client = TestClient(app)

    response = client.put(
        "/api/v1/config/retrieval", json={"search_mode": "hybrid", "routing_enabled": False}
    )

    assert response.status_code == 200
    retrieval = response.json()["retrieval"]
    assert retrieval["RETRIEVAL_SEARCH_MODE"]["effective"] == "hybrid"
    assert retrieval["RETRIEVAL_ROUTING_ENABLED"] == {
        "effective": False,
        "default": True,
        "overridden": True,
    }
    assert retrieval["RERANKER_ENABLED"]["overridden"] is False

    reset = client.put("/api/v1/config/retrieval", json={"search_mode": None}).json()
    assert reset["retrieval"]["RETRIEVAL_SEARCH_MODE"]["overridden"] is False
    assert reset["retrieval"]["RETRIEVAL_ROUTING_ENABLED"]["overridden"] is True


def test_put_retrieval_invalid_search_mode_is_422(
    retrieval_store: RuntimeRetrievalConfig,
) -> None:
    response = TestClient(app).put("/api/v1/config/retrieval", json={"search_mode": "keyword"})

    assert response.status_code == 422


def test_put_retrieval_maps_store_outage_to_503(retrieval_store: RuntimeRetrievalConfig) -> None:
    def fail(_: bool | None) -> None:
        raise RuntimeConfigUnavailable("down")

    retrieval_store.set_rerank = fail  # type: ignore[method-assign]

    response = TestClient(app).put("/api/v1/config/retrieval", json={"rerank": True})

    assert response.status_code == 503


def test_put_retrieval_sets_validates_and_resets_the_task_hours_knobs(
    retrieval_store: RuntimeRetrievalConfig,
) -> None:
    client = TestClient(app)

    response = client.put(
        "/api/v1/config/retrieval",
        json={"task_hours_top_k": 8, "task_hours_distance_threshold": 0.3},
    )

    assert response.status_code == 200
    retrieval = response.json()["retrieval"]
    assert retrieval["TASK_HOURS_TOP_K"] == {"effective": 8, "default": 5, "overridden": True}
    assert retrieval["TASK_HOURS_DISTANCE_THRESHOLD"]["effective"] == 0.3
    assert retrieval_store.effective_task_hours_top_k() == 8

    reset = client.put(
        "/api/v1/config/retrieval",
        json={"task_hours_top_k": None, "task_hours_distance_threshold": None},
    ).json()
    assert reset["retrieval"]["TASK_HOURS_TOP_K"]["overridden"] is False
    assert reset["retrieval"]["TASK_HOURS_DISTANCE_THRESHOLD"]["overridden"] is False


@pytest.mark.parametrize(
    "body",
    [
        {"task_hours_top_k": 0},
        {"task_hours_top_k": 31},
        {"task_hours_distance_threshold": -0.1},
        {"task_hours_distance_threshold": 2.1},
    ],
)
def test_put_retrieval_rejects_out_of_range_task_hours_knobs(
    retrieval_store: RuntimeRetrievalConfig, body: dict[str, float]
) -> None:
    assert TestClient(app).put("/api/v1/config/retrieval", json=body).status_code == 422
