"""Unit tests for the Redis-backed runtime model and retrieval-toggle stores (fakeredis)."""

from unittest.mock import MagicMock

import fakeredis
import pytest
import redis as redis_lib

from app.config import Settings
from app.foundation.llm.runtime_config import (
    BOOL_TOGGLE_KEYS,
    HASH_KEY,
    MODEL_KEYS,
    RETRIEVAL_HASH_KEY,
    RETRIEVAL_KEYS,
    ROUTING_KEY,
    SEARCH_MODE_KEY,
    TASK_HOURS_DISTANCE_THRESHOLD_KEY,
    TASK_HOURS_TOP_K_KEY,
    RuntimeConfigUnavailable,
    RuntimeModelConfig,
    RuntimeRetrievalConfig,
)


def make_settings(**overrides: object) -> Settings:
    return Settings(OPENAI_API_KEY="sk-test", _env_file=None, **overrides)


def broken_redis() -> MagicMock:
    broken = MagicMock()
    for method in ("hget", "hgetall", "hset", "hdel", "delete"):
        getattr(broken, method).side_effect = redis_lib.RedisError("down")
    return broken


@pytest.fixture
def store() -> RuntimeModelConfig:
    return RuntimeModelConfig(fakeredis.FakeRedis(decode_responses=True), make_settings())


@pytest.fixture
def retrieval_store() -> RuntimeRetrievalConfig:
    return RuntimeRetrievalConfig(fakeredis.FakeRedis(decode_responses=True), make_settings())


def test_model_keys_are_the_honoured_settings_fields() -> None:
    assert set(MODEL_KEYS) == {
        "REFORMULATION_MODEL",
        "GENERATION_MODEL",
        "ROUTER_MODEL",
        "QUERY_TRANSFORM_MODEL",
        "HALLUCINATION_JUDGE_MODEL",
        "AUGMENTATION_MODEL",
        "PROPOSITIONAL_CHUNKER_MODEL",
        "CONTEXTUAL_CHUNKER_MODEL",
    }
    settings = make_settings()
    assert all(isinstance(getattr(settings, key), str) for key in MODEL_KEYS)


def test_effective_returns_settings_default_when_no_override(store: RuntimeModelConfig) -> None:
    assert store.effective("GENERATION_MODEL") == "gpt-5"
    assert store.is_overridden("GENERATION_MODEL") is False


def test_set_and_effective_round_trip(store: RuntimeModelConfig) -> None:
    store.set("GENERATION_MODEL", "gpt-4o")

    assert store.get("GENERATION_MODEL") == "gpt-4o"
    assert store.effective("GENERATION_MODEL") == "gpt-4o"
    assert store.is_overridden("GENERATION_MODEL") is True
    assert store.is_overridden("ROUTER_MODEL") is False


def test_set_none_resets_to_default(store: RuntimeModelConfig) -> None:
    store.set("GENERATION_MODEL", "gpt-4o")
    store.set("GENERATION_MODEL", None)

    assert store.get("GENERATION_MODEL") is None
    assert store.effective("GENERATION_MODEL") == "gpt-5"


def test_unknown_or_env_only_key_raises(store: RuntimeModelConfig) -> None:
    for key in ("EMBEDDING_MODEL", "CRITIC_MODEL", "NOT_A_KEY"):
        with pytest.raises(ValueError, match="Unknown model key"):
            store.effective(key)
    with pytest.raises(ValueError, match="Unknown model key"):
        store.set("NOT_A_KEY", "gpt-4o")


def test_snapshot_covers_every_key(store: RuntimeModelConfig) -> None:
    store.set("ROUTER_MODEL", "gpt-4o")
    snapshot = store.snapshot()

    assert set(snapshot) == set(MODEL_KEYS)
    assert snapshot["ROUTER_MODEL"] == {
        "effective": "gpt-4o",
        "default": "gpt-4o-mini",
        "overridden": True,
    }
    assert snapshot["GENERATION_MODEL"]["overridden"] is False


def test_reset_all_clears_every_override(store: RuntimeModelConfig) -> None:
    store.set("GENERATION_MODEL", "gpt-4o")
    store.set("ROUTER_MODEL", "gpt-4o")

    store.reset_all()

    assert not any(entry["overridden"] for entry in store.snapshot().values())


def test_model_overrides_live_in_their_own_hash(store: RuntimeModelConfig) -> None:
    store.set("GENERATION_MODEL", "gpt-4o")

    assert store._redis.hgetall(HASH_KEY) == {"GENERATION_MODEL": "gpt-4o"}


def test_model_reads_degrade_to_defaults_when_redis_is_down() -> None:
    store = RuntimeModelConfig(broken_redis(), make_settings())

    assert store.get("GENERATION_MODEL") is None
    assert store.effective("GENERATION_MODEL") == "gpt-5"
    assert store.snapshot()["GENERATION_MODEL"]["overridden"] is False


def test_model_writes_raise_when_redis_is_down() -> None:
    store = RuntimeModelConfig(broken_redis(), make_settings())

    with pytest.raises(RuntimeConfigUnavailable):
        store.set("GENERATION_MODEL", "gpt-4o")
    with pytest.raises(RuntimeConfigUnavailable):
        store.reset_all()


def test_retrieval_defaults_come_from_settings(retrieval_store: RuntimeRetrievalConfig) -> None:
    assert retrieval_store.effective_search_mode() == "vector"
    assert retrieval_store.effective_rerank() is False
    assert retrieval_store.effective_routing() is True
    assert retrieval_store.effective_temporal_decay() is False
    assert retrieval_store.effective_hallucination_gate() is True


def test_retrieval_overrides_win_over_defaults(retrieval_store: RuntimeRetrievalConfig) -> None:
    retrieval_store.set_search_mode("hybrid")
    retrieval_store.set_rerank(True)
    retrieval_store.set_bool(ROUTING_KEY, False)

    assert retrieval_store.effective_search_mode() == "hybrid"
    assert retrieval_store.effective_rerank() is True
    assert retrieval_store.effective_routing() is False

    retrieval_store.set_bool(ROUTING_KEY, None)
    assert retrieval_store.effective_routing() is True


def test_retrieval_rejects_invalid_values(retrieval_store: RuntimeRetrievalConfig) -> None:
    with pytest.raises(ValueError, match="Invalid search mode"):
        retrieval_store.set_search_mode("keyword")
    with pytest.raises(ValueError, match="Unknown retrieval toggle"):
        retrieval_store.set_bool(SEARCH_MODE_KEY, True)


def test_retrieval_snapshot_covers_every_toggle(retrieval_store: RuntimeRetrievalConfig) -> None:
    retrieval_store.set_search_mode("hybrid")
    snapshot = retrieval_store.snapshot()

    assert set(snapshot) == set(RETRIEVAL_KEYS)
    assert set(BOOL_TOGGLE_KEYS) < set(RETRIEVAL_KEYS)
    assert snapshot[SEARCH_MODE_KEY] == {
        "effective": "hybrid",
        "default": "vector",
        "overridden": True,
    }
    assert snapshot[ROUTING_KEY]["overridden"] is False
    assert retrieval_store._redis.hgetall(RETRIEVAL_HASH_KEY) == {SEARCH_MODE_KEY: "hybrid"}


def test_retrieval_reads_degrade_and_writes_raise_when_redis_is_down() -> None:
    store = RuntimeRetrievalConfig(broken_redis(), make_settings())

    assert store.effective_search_mode() == "vector"
    assert store.effective_synthesis() is True
    assert store.snapshot()[ROUTING_KEY]["overridden"] is False
    with pytest.raises(RuntimeConfigUnavailable):
        store.set_rerank(True)


def test_task_hours_knobs_default_to_settings_and_can_be_overridden(
    retrieval_store: RuntimeRetrievalConfig,
) -> None:
    assert retrieval_store.effective_task_hours_top_k() == 5
    assert retrieval_store.effective_task_hours_distance_threshold() == 0.45

    retrieval_store.set_task_hours_top_k(9)
    retrieval_store.set_task_hours_distance_threshold(0.3)

    assert retrieval_store.effective_task_hours_top_k() == 9
    assert retrieval_store.effective_task_hours_distance_threshold() == 0.3
    snapshot = retrieval_store.snapshot()
    assert snapshot[TASK_HOURS_TOP_K_KEY] == {"effective": 9, "default": 5, "overridden": True}
    assert snapshot[TASK_HOURS_DISTANCE_THRESHOLD_KEY]["overridden"] is True

    retrieval_store.set_task_hours_top_k(None)
    retrieval_store.set_task_hours_distance_threshold(None)
    assert retrieval_store.effective_task_hours_top_k() == 5
    assert retrieval_store.snapshot()[TASK_HOURS_TOP_K_KEY]["overridden"] is False


def test_task_hours_knobs_reject_out_of_range_values(
    retrieval_store: RuntimeRetrievalConfig,
) -> None:
    with pytest.raises(ValueError, match="TASK_HOURS_TOP_K"):
        retrieval_store.set_task_hours_top_k(0)
    with pytest.raises(ValueError, match="TASK_HOURS_DISTANCE_THRESHOLD"):
        retrieval_store.set_task_hours_distance_threshold(2.5)
    with pytest.raises(ValueError, match="Unknown retrieval toggle"):
        retrieval_store.set_bool(TASK_HOURS_TOP_K_KEY, True)


def test_a_corrupt_task_hours_override_falls_back_to_the_default(
    retrieval_store: RuntimeRetrievalConfig,
) -> None:
    retrieval_store._redis.hset(RETRIEVAL_HASH_KEY, TASK_HOURS_TOP_K_KEY, "many")
    retrieval_store._redis.hset(RETRIEVAL_HASH_KEY, TASK_HOURS_DISTANCE_THRESHOLD_KEY, "far")

    assert retrieval_store.effective_task_hours_top_k() == 5
    assert retrieval_store.effective_task_hours_distance_threshold() == 0.45
