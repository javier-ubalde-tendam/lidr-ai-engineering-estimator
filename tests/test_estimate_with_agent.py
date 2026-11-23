"""The conductor side of the agent: the real ``search_budgets`` backend and ``estimate_with_agent``."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.domain.estimation_service import (
    AgentUnavailableError,
    EstimationService,
    build_budget_search_backend,
)
from app.generation.agentic.agent_schemas import SearchBudgetsArgs
from app.generation.agentic.agent_tools import build_tool_registry, execute_tool
from app.generation.rag.schemas import RetrievalRequest, RetrievalResult, RetrievedChunk
from tests.agent_helpers import (
    FakeAsyncOpenAI,
    calc_args,
    final_response,
    function_call,
    make_response,
    reference,
    valid_estimate,
)

pytestmark = pytest.mark.anyio

TASK_CONTENT = (
    "[Project: Payments platform]\n[Client sector: finance | Year: 2023 | Main tech: Python]\n\n"
    "Module: Payments\nComponent: Gateway\nDescription: PSP integration\nEstimated hours: 80"
)


def chunk(chunk_id: int = 7, hours: int | None = 80) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=TASK_CONTENT,
        sector="finance",
        project_year=2023,
        chunk_type="historical_task",
        distance=0.21,
        budget_id="TASK-2023-0007",
        estimated_hours=hours,
    )


class FakeRetriever:
    """Stands in for ``TranscriptEstimator``: records the ``RetrievalRequest`` it receives."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.requests: list[RetrievalRequest] = []
        self._chunks = [chunk()] if chunks is None else chunks

    async def retrieve_chunks(self, request: RetrievalRequest) -> RetrievalResult:
        self.requests.append(request)
        return RetrievalResult(
            chunks=self._chunks, low_confidence=not self._chunks, candidates_evaluated=5
        )


async def test_the_backend_translates_the_tool_args_into_a_real_retrieval_request() -> None:
    settings = get_settings()
    retriever = FakeRetriever()
    backend = build_budget_search_backend(retriever, settings)
    args = SearchBudgetsArgs.model_validate(
        {
            "query": "payment gateway integration",
            "filters": {"sectors": ["finance"], "project_year_min": 2022, "project_year_max": 2024},
        }
    )

    items = await backend(args)

    request = retriever.requests[0]
    assert request.query_text == "payment gateway integration"
    assert request.chunk_types == ["historical_task"]
    assert request.top_k == settings.AGENT_SEARCH_TOP_K
    assert request.distance_threshold == settings.AGENT_SEARCH_DISTANCE_THRESHOLD
    assert request.sectors == ["finance"]
    assert (request.project_year_min, request.project_year_max) == (2022, 2024)
    assert items == [
        {
            "id": 7,
            "budget_id": "TASK-2023-0007",
            "sector": "finance",
            "project_year": 2023,
            # The parent-budget header is dropped: the task detail is what helps the model
            "content_preview": TASK_CONTENT.split("\n\n", 1)[1],
            "estimated_hours": 80,
            "distance": 0.21,
        }
    ]


async def test_no_filters_means_no_metadata_filtering_but_always_historical_tasks() -> None:
    retriever = FakeRetriever()
    backend = build_budget_search_backend(retriever, get_settings())

    await backend(SearchBudgetsArgs.model_validate({"query": "payment gateway integration"}))

    request = retriever.requests[0]
    assert request.sectors is None and request.project_year_min is None
    assert request.chunk_types == ["historical_task"]


async def test_an_unknown_sector_or_bad_year_range_reaches_the_model_as_an_error() -> None:
    registry = build_tool_registry(build_budget_search_backend(FakeRetriever(), get_settings()))

    bad_sector = await execute_tool(
        registry,
        "search_budgets",
        json.dumps({"query": "payment gateway integration", "filters": {"sectors": ["space"]}}),
    )
    bad_years = await execute_tool(
        registry,
        "search_budgets",
        json.dumps(
            {
                "query": "payment gateway integration",
                "filters": {"project_year_min": 2024, "project_year_max": 2022},
            }
        ),
    )

    assert "sectors" in bad_sector["error"] and "finance" in bad_sector["error"]
    assert "project_year_min" in bad_years["error"]


def test_the_request_contract_still_rejects_short_queries() -> None:
    # The agent schema mirrors this limit: if retrieval ever relaxes it, revisit MIN_QUERY_LENGTH
    with pytest.raises(ValidationError):
        RetrievalRequest(query_text="too short")


def make_service(**collaborators: Any) -> EstimationService:
    return EstimationService(
        openai_client=object(),
        exact_cache=object(),  # type: ignore[arg-type]
        semantic_cache=object(),  # type: ignore[arg-type]
        **collaborators,
    )


async def test_estimate_with_agent_runs_the_loop_with_the_real_backend_and_settings_defaults() -> None:
    retriever = FakeRetriever()
    client = FakeAsyncOpenAI(
        [
            make_response(
                "resp_1",
                [function_call("search_budgets", {"query": "payment gateway integration"}, "c1")],
            ),
            make_response(
                "resp_2",
                [
                    function_call(
                        "calculate_estimate",
                        calc_args(
                            {
                                "name": "Gateway",
                                "references": [reference("TASK-2023-0007", 80.0)],
                            }
                        ),
                        "c2",
                    )
                ],
            ),
            final_response("resp_3", valid_estimate(92.0, ids=("TASK-2023-0007",))),
        ]
    )
    service = make_service(async_openai_client=client, budget_retriever=retriever)

    result = await service.estimate_with_agent("a transcript")

    settings = get_settings()
    assert result.status == "done"
    assert len(retriever.requests) == 1 and retriever.requests[0].chunk_types == ["historical_task"]
    assert client.calls[0]["model"] == settings.AGENT_MODEL
    assert client.calls[0]["reasoning"]["effort"] == settings.AGENT_REASONING_EFFORT


async def test_estimate_with_agent_applies_the_overrides_and_the_iteration_cap() -> None:
    client = FakeAsyncOpenAI(
        [
            make_response(f"resp_{n}", [function_call("calculate_estimate", {"components": []}, "c")])
            for n in range(1, 4)
        ]
    )
    service = make_service(async_openai_client=client, budget_retriever=FakeRetriever())

    result = await service.estimate_with_agent(
        "a transcript", model="gpt-5-mini", reasoning_effort="low", max_iterations=2
    )

    assert result.status == "max_steps_exceeded" and len(client.calls) == 2
    assert client.calls[0]["model"] == "gpt-5-mini" and client.calls[0]["reasoning"]["effort"] == "low"


@pytest.mark.parametrize(
    "collaborators",
    [{"budget_retriever": FakeRetriever()}, {"async_openai_client": FakeAsyncOpenAI([])}],
    ids=["no-openai-client", "no-retriever"],
)
async def test_estimate_with_agent_needs_both_collaborators(collaborators: dict[str, Any]) -> None:
    with pytest.raises(AgentUnavailableError):
        await make_service(**collaborators).estimate_with_agent("a transcript")
