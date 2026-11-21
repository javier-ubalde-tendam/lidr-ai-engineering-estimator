"""POST /v1/estimate/stages/*: auth boundary, stateless stage contracts and grounding signals.

The LLM, embedder, DB session and retrieval are fakes (``tests/estimator_helpers.py``); the context
assembler, citation check and coherence check are the REAL ones.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_token_encoder, get_transcript_estimator
from app.generation.rag.errors import GenerationError, RetrievalError
from app.generation.rag.quality.hallucination import JudgePanel
from app.generation.rag.schemas import (
    Estimate,
    EstimationQuery,
    LineVerdict,
    SourceReference,
    TaskItem,
    WorkModule,
)
from app.main import app
from tests.estimator_helpers import (
    TRANSCRIPT,
    CharEncoder,
    FakeEmbedder,
    ScriptedLLM,
    good_estimate,
    make_estimator,
    patch_retrieve,
    result_with,
    retrieved_chunk,
)

BASE = "/v1/estimate/stages"
QUERY = EstimationQuery(function="payments marketplace", sector="finance").model_dump()


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM(good_estimate("1"))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def client(
    llm: ScriptedLLM,
    embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    estimate_api_key: str,
) -> Iterator[TestClient]:
    patch_retrieve(monkeypatch, result_with(1, 2))
    estimator = make_estimator(llm, embedder=embedder)
    app.dependency_overrides[get_transcript_estimator] = lambda: estimator
    app.dependency_overrides[get_token_encoder] = lambda: CharEncoder()
    yield TestClient(app, headers={"X-API-Key": estimate_api_key})
    app.dependency_overrides.clear()


@pytest.fixture
def estimate_api_key() -> str:
    return "test-estimate-key"  # same fake value as tests/conftest.py


STAGE_BODIES: list[tuple[str, dict[str, Any]]] = [
    ("reformulate", {"transcript": TRANSCRIPT}),
    ("retrieve", {"query_text": "payments marketplace with kyc"}),
    ("assemble", {"chunks": []}),
    ("structure", {"query": QUERY}),
    ("generate", {"context_block": "<source/>", "query": QUERY}),
    ("verify", {"estimate": good_estimate("1").model_dump()}),
]


@pytest.mark.parametrize(("stage", "body"), STAGE_BODIES)
def test_every_stage_requires_the_estimate_key(
    client: TestClient, retrieval_api_key: str, stage: str, body: dict[str, Any]
) -> None:
    anonymous = TestClient(app)

    assert anonymous.post(f"{BASE}/{stage}", json=body).status_code == 401
    # The retrieval key must NOT open the estimation wizard
    wrong = anonymous.post(f"{BASE}/{stage}", json=body, headers={"X-API-Key": retrieval_api_key})
    assert wrong.status_code == 401


def test_reformulate_returns_the_brief_and_the_search_text(client: TestClient) -> None:
    response = client.post(f"{BASE}/reformulate", json={"transcript": TRANSCRIPT})

    assert response.status_code == 200
    body = response.json()
    assert body["query"]["sector"] == "finance"
    assert "payments marketplace" in body["search_text"]


def test_reformulate_rejects_a_short_transcript(client: TestClient) -> None:
    assert client.post(f"{BASE}/reformulate", json={"transcript": "short"}).status_code == 422


def test_retrieve_embeds_the_text_and_applies_the_request_knobs(
    client: TestClient, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = patch_retrieve(monkeypatch, result_with(1, 2))

    response = client.post(
        f"{BASE}/retrieve",
        json={
            "query_text": "payments marketplace with kyc",
            "top_k": 3,
            "distance_threshold": 0.4,
            "sectors": ["finance"],
            "chunk_types": ["historical_task"],
            "search_mode": "hybrid",
            "rerank": False,
        },
    )

    assert response.status_code == 200
    assert [c["id"] for c in response.json()["chunks"]] == [1, 2]
    assert embedder.texts == ["payments marketplace with kyc"]
    call = spy.calls[0]
    assert call["top_k"] == 3 and call["distance_threshold"] == 0.4
    assert call["sectors"] == ["finance"] and call["chunk_types"] == ["historical_task"]
    assert call["search_mode"] == "hybrid" and call["rerank"] is False


def test_retrieve_passes_the_soft_fail_through(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_retrieve(monkeypatch, result_with())

    response = client.post(f"{BASE}/retrieve", json={"query_text": "dog grooming blockchain"})

    assert response.status_code == 200
    assert response.json()["chunks"] == [] and response.json()["low_confidence"] is True


def test_retrieve_failure_is_a_generic_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_retrieve(monkeypatch, RetrievalError("password=hunter2"))

    response = client.post(f"{BASE}/retrieve", json={"query_text": "payments marketplace"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Retrieval failed."}


def test_assemble_wraps_the_chunks_in_source_elements(client: TestClient) -> None:
    chunks = [retrieved_chunk(1).model_dump(), retrieved_chunk(2).model_dump()]

    response = client.post(f"{BASE}/assemble", json={"chunks": chunks})

    assert response.status_code == 200
    body = response.json()
    assert '<source id="1"' in body["context_block"]
    assert body["dropped_count"] == 0 and len(body["kept_chunks"]) == 2
    assert body["token_count"] == len(body["context_block"]) and body["augmented"] is False


def test_assemble_drops_whole_chunks_over_the_budget(client: TestClient) -> None:
    chunks = [retrieved_chunk(i, content="word " * 300).model_dump() for i in range(1, 6)]

    response = client.post(f"{BASE}/assemble", json={"chunks": chunks, "max_context_tokens": 2000})

    body = response.json()
    assert response.status_code == 200
    assert body["dropped_count"] > 0 and len(body["kept_chunks"]) + body["dropped_count"] == 5


def test_assemble_augmentation_compresses_and_keeps_every_id(client: TestClient) -> None:
    chunks = [
        retrieved_chunk(i, content=f"filler prose line\nITEM-{i} :: work - {i * 10} h").model_dump()
        for i in range(1, 4)
    ]

    response = client.post(f"{BASE}/assemble", json={"chunks": chunks, "augment": True})

    body = response.json()
    assert body["augmented"] is True and "filler prose" not in body["context_block"]
    assert {c["id"] for c in body["kept_chunks"]} == {1, 2, 3}


def test_structure_is_a_free_decomposition_without_sources(
    llm: ScriptedLLM, client: TestClient
) -> None:
    llm._estimates = [
        Estimate(
            confidence="high",
            reasoning="decomposed from the brief",
            modules=[WorkModule(name="Auth", tasks=[TaskItem(name="OAuth login")])],
        )
    ]

    response = client.post(f"{BASE}/structure", json={"query": QUERY})

    assert response.status_code == 200
    body = response.json()
    assert body["estimate"]["modules"][0]["tasks"][0]["engineer_days"] is None
    assert body["fabricated_source_ids"] == [] and body["coherent"] is True
    assert body["citation_report"] is None
    call = llm.generation_calls[0]
    assert "STRUCTURE-ONLY" in call["system_prompt"] and "<sources>" not in call["user_message"]
    assert call["purpose"] == "rag_structure"


def test_structure_failure_is_a_502(llm: ScriptedLLM, client: TestClient) -> None:
    llm.complete_structured = _raising  # type: ignore[method-assign]

    response = client.post(f"{BASE}/structure", json={"query": QUERY})

    assert response.status_code == 502
    assert response.json() == {"detail": "Structure generation failed."}


def _raising(**_: Any) -> Any:
    raise GenerationError("boom")


def _generate_body(**extra: Any) -> dict[str, Any]:
    return {
        "context_block": '<source id="1">...</source>',
        "query": QUERY,
        "kept_chunks": [retrieved_chunk(1).model_dump()],
        **extra,
    }


def test_generate_reports_the_grounding_signals_without_retrying(
    llm: ScriptedLLM, client: TestClient
) -> None:
    # The line cites chunk 999, which was never in kept_chunks: reported, NOT auto-corrected
    llm._estimates = [good_estimate("999")]

    response = client.post(f"{BASE}/generate", json=_generate_body())

    assert response.status_code == 200
    body = response.json()
    assert body["fabricated_source_ids"] == ["999"]
    assert body["citation_report"]["dangling_lines"] == 1
    assert body["coherent"] is True
    assert len(llm.generation_calls) == 1


def test_generate_flags_an_incoherent_insufficient_estimate(
    llm: ScriptedLLM, client: TestClient
) -> None:
    llm._estimates = [
        Estimate(
            confidence="insufficient",
            reasoning="r",
            total_engineer_days=40,
            insufficient_context_explanation="should not carry numbers",
        )
    ]

    response = client.post(f"{BASE}/generate", json=_generate_body())

    assert response.json()["coherent"] is False


def test_generate_include_hours_false_asks_for_the_structure_only_prompt(
    llm: ScriptedLLM, client: TestClient
) -> None:
    client.post(f"{BASE}/generate", json=_generate_body(include_hours=False))
    client.post(f"{BASE}/generate", json=_generate_body())

    structure_prompt, hours_prompt = (c["system_prompt"] for c in llm.generation_calls)
    assert "DO NOT estimate hours" in structure_prompt
    assert "DO NOT estimate hours" not in hours_prompt


def _inflated_estimate() -> Estimate:
    # Claims 90 engineer-days against a cited source that records 120h (15d)
    return Estimate(
        confidence="high",
        reasoning="r",
        total_engineer_days=90,
        modules=[
            WorkModule(
                name="Auth",
                tasks=[
                    TaskItem(
                        name="OAuth login",
                        engineer_days=90,
                        grounded=True,
                        sources=[
                            SourceReference(chunk_id="1", document_id="BUD-1", evidence="120 h")
                        ],
                    )
                ],
            )
        ],
    )


def test_verify_flags_a_degraded_line_with_the_anchor_only(
    llm: ScriptedLLM, client: TestClient
) -> None:
    body = {
        "estimate": _inflated_estimate().model_dump(),
        "kept_chunks": [retrieved_chunk(1, content="OAuth backend - 120 h").model_dump()],
        "use_judge": False,
    }

    response = client.post(f"{BASE}/verify", json=body)

    assert response.status_code == 200
    report = response.json()
    assert report["total_lines"] == 1 and report["degraded_lines"] == 1
    assert report["lines"][0]["status"] == "degraded"
    assert llm.judge_calls == []


def test_verify_runs_the_judge_by_default(llm: ScriptedLLM, client: TestClient) -> None:
    llm._judge = JudgePanel(
        verdicts=[
            LineVerdict(module="Auth", component="OAuth login", entailed=False, reason="not backed")
        ]
    )
    estimate = _inflated_estimate().model_copy(update={"total_engineer_days": 15})
    estimate.modules[0].tasks[0].engineer_days = 15
    body = {
        "estimate": estimate.model_dump(),
        "kept_chunks": [retrieved_chunk(1, content="OAuth backend - 120 h").model_dump()],
    }

    response = client.post(f"{BASE}/verify", json=body)

    assert response.json()["lines"][0]["status"] == "degraded"
    assert len(llm.judge_calls) == 1


def test_a_stage_is_rate_limited_with_retry_after(client: TestClient) -> None:
    body = {"query": QUERY}
    statuses = [client.post(f"{BASE}/structure", json=body).status_code for _ in range(16)]

    assert 429 in statuses and statuses[:15].count(429) == 0
