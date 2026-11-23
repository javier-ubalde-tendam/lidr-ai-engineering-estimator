"""POST /v1/estimate/agent: auth, validation, status mapping and rate limit (no real agent)."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_agent_estimation_service, get_async_openai_client
from app.domain.estimation_service import AgentUnavailableError
from app.generation.agentic.agent_loop import AgentLoopError
from app.generation.agentic.agent_schemas import (
    AgentAction,
    AgentEstimate,
    AgentResult,
    AgentStep,
    AgentWarning,
    Coverage,
    TokenUsage,
)
from app.main import app
from tests.agent_helpers import valid_estimate

URL = "/v1/estimate/agent"
BODY = {"transcript": "The client wants a logistics platform with several parts. " * 4}
API_KEY = "test-estimate-key"  # same fake value as tests/conftest.py


def done_result() -> AgentResult:
    return AgentResult(
        status="done",
        estimate=AgentEstimate.model_validate(valid_estimate()),
        trace=[
            AgentStep(
                step=1,
                reasoning="r",
                actions=[
                    AgentAction(
                        tool="search_budgets",
                        call_id="c1",
                        arguments={"query": "q" * 12},
                        observation={"count": 1},
                        duration_ms=3,
                    )
                ],
            )
        ],
        iterations=1,
        usage=TokenUsage(input_tokens=1, output_tokens=1, reasoning_tokens=0, total_tokens=2),
    )


class FakeAgentService:
    def __init__(self, result: AgentResult | None = None, error: Exception | None = None) -> None:
        self.result = result or done_result()
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def estimate_with_agent(self, transcript: str, **overrides: Any) -> AgentResult:
        self.calls.append((transcript, overrides))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def service() -> FakeAgentService:
    return FakeAgentService()


@pytest.fixture
def client(service: FakeAgentService) -> Iterator[TestClient]:
    app.dependency_overrides[get_agent_estimation_service] = lambda: service
    app.dependency_overrides[get_async_openai_client] = lambda: object()
    yield TestClient(app, headers={"X-API-Key": API_KEY})
    app.dependency_overrides.clear()


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_missing_or_wrong_key_is_401(client: TestClient, headers: dict[str, str]) -> None:
    assert TestClient(app).post(URL, json=BODY, headers=headers).status_code == 401


def test_a_finished_run_returns_the_estimate_the_trace_and_the_usage(
    client: TestClient, service: FakeAgentService
) -> None:
    response = client.post(URL, json=BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done" and body["estimate"]["total_hours"] == 115.0
    assert body["trace"][0]["actions"][0]["tool"] == "search_budgets"
    assert body["usage"]["total_tokens"] == 2
    assert service.calls == [
        (BODY["transcript"], {"model": None, "reasoning_effort": None, "max_iterations": None})
    ]


def test_the_new_fields_are_added_and_the_existing_ones_do_not_change(
    client: TestClient, service: FakeAgentService
) -> None:
    service.result = done_result().model_copy(
        update={
            "coverage": Coverage(budgeted_tasks=11, total_tasks=21, ratio=0.524),
            "unbudgeted_tasks": ["Backend / Tariffs"],
            "total_is_lower_bound": True,
            "warnings": [AgentWarning(code="stale_calculation", message="m")],
        }
    )

    body = client.post(URL, json=BODY).json()

    assert body["coverage"] == {"budgeted_tasks": 11, "total_tasks": 21, "ratio": 0.524}
    assert body["unbudgeted_tasks"] == ["Backend / Tariffs"]
    assert body["total_is_lower_bound"] is True
    assert body["warnings"] == [{"code": "stale_calculation", "message": "m"}]
    # The pre-existing contract is untouched (only additions)
    assert {"status", "estimate", "trace", "iterations", "usage", "error"} <= set(body)
    assert set(body["estimate"]) == {"components", "total_hours", "notes"}
    assert set(body["trace"][0]) >= {"step", "reasoning", "actions", "usage", "duration_ms"}


def test_a_run_without_calculation_has_empty_coverage_fields(client: TestClient) -> None:
    body = client.post(URL, json=BODY).json()

    assert body["coverage"] is None and body["unbudgeted_tasks"] == []
    assert body["total_is_lower_bound"] is False and body["warnings"] == []


def test_the_overrides_are_forwarded_to_the_conductor(
    client: TestClient, service: FakeAgentService
) -> None:
    client.post(
        URL,
        json={**BODY, "model": "gpt-5-mini", "reasoning_effort": "minimal", "max_iterations": 4},
    )

    assert service.calls[0][1] == {
        "model": "gpt-5-mini",
        "reasoning_effort": "minimal",
        "max_iterations": 4,
    }


def test_max_steps_exceeded_is_a_200_with_that_status_and_the_partial_trace(
    client: TestClient, service: FakeAgentService
) -> None:
    service.result = AgentResult(
        status="max_steps_exceeded", trace=done_result().trace, iterations=1, error="Stopped"
    )

    response = client.post(URL, json=BODY)

    assert response.status_code == 200
    assert response.json()["status"] == "max_steps_exceeded"
    assert response.json()["estimate"] is None and len(response.json()["trace"]) == 1


@pytest.mark.parametrize(
    "extra",
    [
        {"model": "gpt-4o-mini"},  # in AVAILABLE_MODELS but not a reasoning model of the agent
        {"model": "claude-sonnet-4-5"},
        {"model": "not-a-model"},
        {"reasoning_effort": "extreme"},
        {"max_iterations": 0},
        {"max_iterations": 100},
    ],
)
def test_invalid_overrides_are_422_and_never_reach_the_agent(
    client: TestClient, service: FakeAgentService, extra: dict[str, Any]
) -> None:
    assert client.post(URL, json={**BODY, **extra}).status_code == 422
    assert service.calls == []


def test_a_short_transcript_is_422(client: TestClient) -> None:
    assert client.post(URL, json={"transcript": "too short"}).status_code == 422


def test_without_an_openai_client_the_answer_is_503(service: FakeAgentService) -> None:
    app.dependency_overrides[get_agent_estimation_service] = lambda: service
    app.dependency_overrides[get_async_openai_client] = lambda: None
    try:
        response = TestClient(app).post(URL, json=BODY, headers={"X-API-Key": API_KEY})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503 and "OPENAI_API_KEY" in response.json()["detail"]
    assert service.calls == []


def test_the_503_does_not_bypass_authentication() -> None:
    app.dependency_overrides[get_async_openai_client] = lambda: None
    try:
        response = TestClient(app).post(URL, json=BODY)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401


def test_a_conductor_reporting_the_agent_unavailable_is_503(
    client: TestClient, service: FakeAgentService
) -> None:
    service.error = AgentUnavailableError("no retriever")

    assert client.post(URL, json=BODY).status_code == 503


def test_an_openai_failure_is_a_generic_502(client: TestClient, service: FakeAgentService) -> None:
    service.error = AgentLoopError("secret-key leaked in message", [])

    response = client.post(URL, json=BODY)

    assert response.status_code == 502
    assert "secret-key" not in response.text


def test_rate_limit_is_five_per_minute(client: TestClient) -> None:
    statuses = [client.post(URL, json=BODY).status_code for _ in range(6)]

    assert statuses[:5] == [200] * 5 and statuses[5] == 429
