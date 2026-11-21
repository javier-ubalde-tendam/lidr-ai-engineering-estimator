"""POST /v1/estimate/tasks/hours: auth, validation, rate limit, pass-through and errors."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_task_hours_estimator
from app.generation.rag.errors import RetrievalError
from app.generation.rag.schemas import (
    TaskHoursEstimate,
    TaskHoursModuleInput,
    TaskHoursResult,
    TaskNeighbor,
)
from app.main import app

URL = "/v1/estimate/tasks/hours"
BODY = {
    "modules": [
        {"name": "Auth", "tasks": [{"name": "Login", "description": "OAuth"}, {"name": "RBAC"}]}
    ]
}


class FakeTaskHoursEstimator:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.received: list[TaskHoursModuleInput] = []

    async def estimate_all(self, modules: list[TaskHoursModuleInput]) -> TaskHoursResult:
        self.received = modules
        if self.error:
            raise self.error
        return TaskHoursResult(
            tasks=[
                TaskHoursEstimate(
                    module=module.name,
                    task=task.name,
                    estimated_hours=24,
                    reliability=0.8,
                    has_match=True,
                    neighbors=[TaskNeighbor(source_id=1, estimated_hours=24, distance=0.1)],
                )
                for module in modules
                for task in module.tasks
            ]
        )


@pytest.fixture
def estimator() -> FakeTaskHoursEstimator:
    return FakeTaskHoursEstimator()


@pytest.fixture
def client(estimator: FakeTaskHoursEstimator, estimate_api_key: str) -> Iterator[TestClient]:
    app.dependency_overrides[get_task_hours_estimator] = lambda: estimator
    yield TestClient(app, headers={"X-API-Key": estimate_api_key})
    app.dependency_overrides.clear()


@pytest.fixture
def estimate_api_key() -> str:
    return "test-estimate-key"  # same fake value as tests/conftest.py


def test_returns_the_hours_in_submission_order(
    client: TestClient, estimator: FakeTaskHoursEstimator
) -> None:
    response = client.post(URL, json=BODY)

    assert response.status_code == 200
    tasks = response.json()["tasks"]
    assert [(t["module"], t["task"]) for t in tasks] == [("Auth", "Login"), ("Auth", "RBAC")]
    assert tasks[0]["estimated_hours"] == 24 and tasks[0]["neighbors"][0]["source_id"] == 1
    assert [t.name for t in estimator.received[0].tasks] == ["Login", "RBAC"]


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_missing_or_wrong_key_is_401(client: TestClient, headers: dict[str, str]) -> None:
    assert TestClient(app).post(URL, json=BODY, headers=headers).status_code == 401


def test_the_retrieval_key_does_not_open_it(client: TestClient, retrieval_api_key: str) -> None:
    response = TestClient(app).post(URL, json=BODY, headers={"X-API-Key": retrieval_api_key})

    assert response.status_code == 401


@pytest.mark.parametrize(
    "body",
    [{}, {"modules": []}, {"modules": [{"name": "", "tasks": []}]}, {"modules": [{"name": "A", "tasks": [{"name": ""}]}]}],
)
def test_invalid_payload_is_422(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post(URL, json=body).status_code == 422


def test_a_pipeline_failure_is_a_generic_502(
    client: TestClient, estimator: FakeTaskHoursEstimator
) -> None:
    estimator.error = RetrievalError("password=hunter2")

    response = client.post(URL, json=BODY)

    assert response.status_code == 502
    assert response.json() == {"detail": "Failed to estimate task hours."}


def test_rate_limit_is_thirty_per_minute(client: TestClient) -> None:
    statuses = [client.post(URL, json=BODY).status_code for _ in range(31)]

    assert statuses[:30] == [200] * 30 and statuses[30] == 429
