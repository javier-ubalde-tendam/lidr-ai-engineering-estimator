"""Idempotent replay of the estimate endpoint and the IdempotencyStore backends."""

import time
from collections.abc import Iterator
from typing import Any

import fakeredis
import pytest
import redis
from fastapi.testclient import TestClient

from app.dependencies import get_transcript_estimator
from app.generation.rag.idempotency import IdempotencyStore
from app.generation.rag.schemas import Estimate
from app.main import app
from tests.estimator_helpers import (
    TRANSCRIPT,
    RetrieveSpy,
    ScriptedLLM,
    good_estimate,
    make_estimator,
    patch_retrieve,
    result_with,
)

URL = "/v1/estimate/from-transcript"


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM(*[good_estimate("1") for _ in range(4)])


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> RetrieveSpy:
    return patch_retrieve(monkeypatch, result_with(1))


@pytest.fixture
def client(llm: ScriptedLLM, spy: RetrieveSpy) -> Iterator[TestClient]:
    estimator = make_estimator(llm, idempotency=IdempotencyStore())
    app.dependency_overrides[get_transcript_estimator] = lambda: estimator
    yield TestClient(app, headers={"X-API-Key": "test-estimate-key"})
    app.dependency_overrides.clear()


def post(client: TestClient, **body: Any) -> Any:
    return client.post(URL, json={"transcript": TRANSCRIPT, **body})


def test_repeated_key_serves_the_cached_estimate_without_running_the_pipeline(
    client: TestClient, llm: ScriptedLLM, spy: RetrieveSpy
) -> None:
    first = post(client, idempotency_key="order-1")
    calls_after_first = len(llm.calls)
    second = post(client, idempotency_key="order-1")

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()
    assert len(llm.calls) == calls_after_first  # no LLM cost on the replay
    assert len(spy.calls) == 1


def test_distinct_keys_and_missing_keys_run_the_pipeline_each_time(
    client: TestClient, llm: ScriptedLLM
) -> None:
    post(client, idempotency_key="a")
    post(client, idempotency_key="b")
    post(client)  # no key: nothing is cached
    post(client)

    assert len(llm.generation_calls) == 4


def test_soft_fail_estimates_are_cached_too(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    empty_spy = patch_retrieve(monkeypatch, result_with())

    first = post(client, idempotency_key="empty")
    second = post(client, idempotency_key="empty")

    assert first.json() == second.json() and first.json()["confidence"] == "insufficient"
    assert len(empty_spy.calls) == 1


def sample_estimate() -> Estimate:
    return good_estimate("1")


def test_memory_backend_roundtrip_and_delete() -> None:
    store = IdempotencyStore()

    assert store.get("k") is None
    store.set("k", sample_estimate())
    assert store.get("k") == sample_estimate()
    store.delete("k")
    assert store.get("k") is None


def test_memory_backend_entries_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    store = IdempotencyStore(ttl=60)
    store.set("k", sample_estimate())

    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now + 61)

    assert store.get("k") is None


def test_redis_backend_roundtrip_with_ttl() -> None:
    client = fakeredis.FakeRedis(decode_responses=True)
    store = IdempotencyStore(redis_client=client, ttl=120)

    store.set("k", sample_estimate())

    assert store.get("k") == sample_estimate()
    assert 0 < client.ttl("idempotency:estimate:k") <= 120
    store.delete("k")
    assert store.get("k") is None


def test_corrupt_entries_are_a_miss_and_are_removed() -> None:
    client = fakeredis.FakeRedis(decode_responses=True)
    client.set("idempotency:estimate:k", "{not json")
    store = IdempotencyStore(redis_client=client)

    assert store.get("k") is None
    assert client.get("idempotency:estimate:k") is None


def test_redis_outage_never_breaks_a_request() -> None:
    class DownRedis:
        def get(self, *_: object, **__: object) -> None:
            raise redis.ConnectionError("down")

        set = delete = get

    store = IdempotencyStore(redis_client=DownRedis())

    assert store.get("k") is None  # a read error is a miss
    store.set("k", sample_estimate())  # a write error only loses replay protection
    store.delete("k")
