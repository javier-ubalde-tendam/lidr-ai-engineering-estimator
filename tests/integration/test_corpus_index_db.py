"""Corpus expansion and per-task hours against real Postgres (embeddings are the fake client)."""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.dependencies import get_corpus_index_service, get_index_job_store
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.index_jobs import IndexJobStore
from app.generation.rag.index_service import CorpusIndexService, SessionIngester
from app.generation.rag.schemas import TaskHoursModuleInput
from app.generation.rag.store.repository import ChunkSearchStore
from app.generation.rag.task_hours import TaskHoursEstimator
from app.main import app
from tests.embedding_helpers import FakeOpenAIClient, make_embedder
from tests.index_helpers import budget_dict
from tests.integration.conftest import run
from tests.runtime_helpers import FakeRuntimeRetrievalConfig

pytestmark = pytest.mark.integration

RUN_URL = "/embeddings/index/runs"


@pytest.fixture
def indexing_api(
    api: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
    fake_openai: FakeOpenAIClient,
    chunker: JSONStructuralChunker,
) -> Iterator[TestClient]:
    ingester = SessionIngester(session_factory, chunker, make_embedder(fake_openai))
    app.dependency_overrides[get_corpus_index_service] = lambda: CorpusIndexService(ingester)
    jobs = IndexJobStore()
    app.dependency_overrides[get_index_job_store] = lambda: jobs
    yield api


def task_run(api: TestClient, *budget_ids: str) -> dict[str, Any]:
    response = api.post(
        RUN_URL,
        json={
            "documents": [budget_dict(budget_id, 2) for budget_id in budget_ids],
            "document_type": "historical_task_breakdown",
            "chunk_type": "historical_task",
        },
    )
    assert response.status_code == 202
    return api.get(f"/embeddings/index/jobs/{response.json()['job_id']}").json()  # type: ignore[no-any-return]


def test_expansion_persists_task_chunks_and_skips_what_is_already_there(
    indexing_api: TestClient, sql: Callable[..., list[Any]]
) -> None:
    first = task_run(indexing_api, "TASK-1", "TASK-2")
    again = task_run(indexing_api, "TASK-1", "TASK-3")

    assert first["status"] == "completed" and first["documents_processed"] == 2
    assert again["status"] == "completed" and again["documents_processed"] == 2
    rows = sql(
        "SELECT d.source_path, d.document_type, c.chunk_type, c.metadata->>'module' "
        "FROM budget_chunks c JOIN documents d ON d.id = c.document_id ORDER BY d.source_path, c.id"
    )
    assert len(rows) == 6  # TASK-1 was NOT embedded twice: 3 documents x 2 tasks
    assert {row[0] for row in rows} == {f"corpus-expansion::TASK-{i}" for i in (1, 2, 3)}
    assert {(row[1], row[2], row[3]) for row in rows} == {
        ("historical_task_breakdown", "historical_task", "Core")
    }


def test_the_stats_follow_the_corpus_growth(indexing_api: TestClient) -> None:
    before = indexing_api.get("/embeddings/index/stats").json()
    task_run(indexing_api, "TASK-1", "TASK-2")
    after = indexing_api.get("/embeddings/index/stats").json()

    assert [c["collection"] for c in after["collections"]] == [
        "budget",
        "transcript",
        "technical_doc",
    ]
    assert before["total_chunks"] == 0 and after["total_chunks"] == 4
    budget = after["collections"][0]
    assert (budget["documents"], budget["chunks"], budget["hnsw_indexed"]) == (2, 4, True)
    assert all(c["chunks"] == 0 and c["hnsw_indexed"] for c in after["collections"][1:])


def test_task_hours_come_from_the_historical_tasks_only(
    indexing_api: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
    fake_openai: FakeOpenAIClient,
    sql: Callable[..., list[Any]],
) -> None:
    task_run(indexing_api, "TASK-1")
    # A budget-type chunk that would be the nearest neighbour must be filtered out
    indexing_api.post(
        "/embeddings/ingest",
        json={
            "source_path": "data/budgets_sample.json::BUD-9",
            "document_type": "historical_budget",
            "content": budget_dict("BUD-9", 1) | {"components": [_component(hours=900)]},
        },
    )
    estimator = TaskHoursEstimator(
        embedder=make_embedder(fake_openai),
        session_factory=session_factory,
        store=ChunkSearchStore(),
        retrieval_config=FakeRuntimeRetrievalConfig(  # type: ignore[arg-type]
            search_mode="vector", rerank=False, synthesis=False, task_hours_distance_threshold=0.5
        ),
        settings=get_settings(),
    )
    modules = [TaskHoursModuleInput(name="Core", tasks=[{"name": "Component 1"}])]  # type: ignore[list-item]

    result = run(estimator.estimate_all(modules))

    (task,) = result.tasks
    assert task.has_match is True and task.estimated_hours == 100
    assert len(task.neighbors) == 2  # the two historical tasks, never the 900h budget component


def _component(hours: int) -> dict[str, Any]:
    return {
        "component_id": "X-1",
        "name": "Budget component",
        "description": "not a task",
        "tech_stack": ["Python"],
        "estimated_hours": hours,
        "complexity": "low",
        "dependencies": [],
    }
