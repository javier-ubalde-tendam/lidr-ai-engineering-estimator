"""Per-task hours: the pure consensus math and ``TaskHoursEstimator`` with fake collaborators."""

from types import SimpleNamespace
from typing import Any

import pytest

from app.config import get_settings
from app.generation.rag import task_hours
from app.generation.rag.errors import RetrievalError
from app.generation.rag.schemas import RetrievalResult, RetrievedChunk, TaskHoursModuleInput
from app.generation.rag.task_hours import TaskHoursEstimator, compose_task_search_text, consensus
from tests.estimator_helpers import FakeEmbedder, FakeSessionFactory, RetrieveSpy
from tests.runtime_helpers import FakeRuntimeRetrievalConfig

pytestmark = pytest.mark.anyio


def task_chunk(chunk_id: int, hours: int | None, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=f"historical task ~{hours}h",
        chunk_type="historical_task",
        distance=distance,
        budget_id="TASK-2024-0001",
        estimated_hours=hours,
    )


def result_of(*chunks: RetrievedChunk) -> RetrievalResult:
    return RetrievalResult(
        chunks=list(chunks), low_confidence=not chunks, candidates_evaluated=10
    )


def make_estimator(
    spy: RetrieveSpy | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    *,
    config: FakeRuntimeRetrievalConfig | None = None,
    embedder: Any = None,
) -> TaskHoursEstimator:
    if spy is not None and monkeypatch is not None:
        monkeypatch.setattr(task_hours, "retrieve", spy)
    return TaskHoursEstimator(
        embedder=embedder or FakeEmbedder(),  # type: ignore[arg-type]
        session_factory=FakeSessionFactory(),  # type: ignore[arg-type]
        store=SimpleNamespace(),  # type: ignore[arg-type]
        retrieval_config=config or FakeRuntimeRetrievalConfig(),  # type: ignore[arg-type]
        settings=get_settings(),
    )


# --- pure functions ----------------------------------------------------------------------------


def test_the_search_text_mirrors_the_corpus_chunk_shape() -> None:
    assert compose_task_search_text("Payments", "Gateway", "PSP integration") == (
        "Module: Payments\nTask: Gateway\nPSP integration"
    )
    assert compose_task_search_text("", "Gateway", None) == "Task: Gateway"


def test_a_single_neighbour_gives_its_hours_and_no_dispersion() -> None:
    hours, reliability, dispersion = consensus([(40, 0.05)])

    assert hours == 40 and dispersion == 0.0
    assert reliability == pytest.approx(0.95, abs=0.01)


def test_the_closest_neighbour_weighs_more() -> None:
    hours, _reliability, _dispersion = consensus([(30, 0.4), (50, 0.02)])

    assert hours > 45


def test_disagreeing_neighbours_lower_the_reliability() -> None:
    tight = consensus([(40, 0.1), (42, 0.1)])[1]
    spread = consensus([(10, 0.1), (90, 0.1)])[1]

    assert spread < tight


def test_an_exact_match_does_not_blow_up_the_weights() -> None:
    hours, reliability, _dispersion = consensus([(40, 0.0), (80, 0.3)])

    assert 40 <= hours < 45 and 0.0 <= reliability <= 1.0


# --- estimate_one ------------------------------------------------------------------------------


async def test_a_match_returns_the_consensus_with_its_neighbours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = RetrieveSpy(result_of(task_chunk(1, 40, 0.1), task_chunk(2, 48, 0.2)))
    estimator = make_estimator(spy, monkeypatch)

    estimate = await estimator.estimate_one(
        "Payments", "Gateway", "PSP", top_k=5, distance_threshold=0.45
    )

    assert estimate.has_match is True
    assert estimate.estimated_hours is not None and 40 <= estimate.estimated_hours <= 48
    assert estimate.reliability is not None and 0.0 <= estimate.reliability <= 1.0
    assert [n.source_id for n in estimate.neighbors] == [1, 2]
    assert estimate.neighbors[0].budget_id == "TASK-2024-0001"
    assert estimate.hours_range is None


async def test_the_search_is_filtered_to_the_historical_task_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = RetrieveSpy(result_of(task_chunk(1, 40, 0.1)))
    embedder = FakeEmbedder()
    estimator = make_estimator(spy, monkeypatch, embedder=embedder)

    await estimator.estimate_one(
        "Payments",
        "Gateway",
        None,
        top_k=3,
        distance_threshold=0.3,
        search_mode="hybrid",
        rerank=True,
    )

    call = spy.calls[0]
    assert call["chunk_types"] == ["historical_task"]
    assert call["top_k"] == 3 and call["rerank_top_n"] == 3 and call["distance_threshold"] == 0.3
    assert call["search_mode"] == "hybrid" and call["rerank"] is True
    assert call["recall_k"] >= 3
    assert embedder.texts == ["Module: Payments\nTask: Gateway"]


async def test_no_neighbour_under_the_threshold_is_flagged_without_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = make_estimator(RetrieveSpy(result_of()), monkeypatch)

    estimate = await estimator.estimate_one(
        "Quantum", "Teleporter", None, top_k=5, distance_threshold=0.45
    )

    assert estimate.has_match is False
    assert estimate.estimated_hours is None and estimate.reliability is None
    assert estimate.neighbors == [] and estimate.hours_range is None


async def test_chunks_without_hours_are_not_neighbours(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = RetrieveSpy(result_of(task_chunk(1, None, 0.1), task_chunk(2, 30, 0.2)))
    estimator = make_estimator(spy, monkeypatch)

    estimate = await estimator.estimate_one("M", "T", None, top_k=5, distance_threshold=0.45)

    assert [n.source_id for n in estimate.neighbors] == [2]


async def test_contradicting_neighbours_surface_a_range_when_synthesis_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = RetrieveSpy(result_of(task_chunk(1, 40, 0.1), task_chunk(2, 90, 0.1)))
    estimator = make_estimator(spy, monkeypatch)

    with_synthesis = await estimator.estimate_one(
        "Notifications", "Delivery", None, top_k=5, distance_threshold=0.45, synthesis=True
    )
    without = await estimator.estimate_one(
        "Notifications", "Delivery", None, top_k=5, distance_threshold=0.45, synthesis=False
    )

    assert with_synthesis.hours_range is not None
    assert (with_synthesis.hours_range.low, with_synthesis.hours_range.high) == (40, 90)
    assert with_synthesis.estimated_hours == 65  # the point estimate still carries the consensus
    assert without.hours_range is None


async def test_agreeing_neighbours_get_no_range_even_with_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = RetrieveSpy(result_of(task_chunk(1, 40, 0.1), task_chunk(2, 42, 0.1)))
    estimator = make_estimator(spy, monkeypatch)

    estimate = await estimator.estimate_one(
        "M", "T", None, top_k=5, distance_threshold=0.45, synthesis=True
    )

    assert estimate.hours_range is None


async def test_an_embedding_failure_is_a_retrieval_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEmbedder:
        def embed_one(self, text: str) -> list[float]:
            raise RuntimeError("quota exceeded")

    estimator = make_estimator(RetrieveSpy(result_of()), monkeypatch, embedder=BrokenEmbedder())

    with pytest.raises(RetrievalError):
        await estimator.estimate_one("M", "T", None, top_k=5, distance_threshold=0.45)


# --- estimate_all ------------------------------------------------------------------------------


async def test_estimate_all_keeps_the_submission_order_and_uses_the_runtime_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = RetrieveSpy(result_of(task_chunk(1, 24, 0.15)))
    config = FakeRuntimeRetrievalConfig(
        search_mode="hybrid",
        rerank=True,
        synthesis=False,
        task_hours_top_k=7,
        task_hours_distance_threshold=0.3,
    )
    estimator = make_estimator(spy, monkeypatch, config=config)
    modules = [
        TaskHoursModuleInput(name="Auth", tasks=[{"name": "Login"}, {"name": "RBAC"}]),  # type: ignore[list-item]
        TaskHoursModuleInput(name="Payments", tasks=[{"name": "Gateway"}]),  # type: ignore[list-item]
    ]

    result = await estimator.estimate_all(modules)

    assert [t.task for t in result.tasks] == ["Login", "RBAC", "Gateway"]
    assert [t.module for t in result.tasks] == ["Auth", "Auth", "Payments"]
    assert all(t.estimated_hours == 24 for t in result.tasks)
    assert len(spy.calls) == 3
    assert all(
        c["top_k"] == 7 and c["distance_threshold"] == 0.3 and c["search_mode"] == "hybrid"
        for c in spy.calls
    )


async def test_estimate_all_flags_only_the_unmatched_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PerTaskSpy(RetrieveSpy):
        async def __call__(self, session: Any, store: Any, **kwargs: Any) -> RetrievalResult:
            self.calls.append(kwargs)
            return result_of() if "Teleporter" in kwargs["query_text"] else result_of(
                task_chunk(1, 10, 0.1)
            )

    estimator = make_estimator(PerTaskSpy(result_of()), monkeypatch)
    modules = [
        TaskHoursModuleInput(name="M", tasks=[{"name": "Login"}, {"name": "Teleporter"}]),  # type: ignore[list-item]
    ]

    result = await estimator.estimate_all(modules)

    assert [t.has_match for t in result.tasks] == [True, False]
