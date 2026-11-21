"""Fakes to drive ``TranscriptEstimator`` end to end without network, DB, Redis or torch."""

from types import SimpleNamespace
from typing import Any, Self

import pytest

from app.config import EMBEDDING_DIMENSION, Settings, get_settings
from app.foundation.llm.wrapper import LLMCallMeta
from app.generation.rag import estimator as estimator_module
from app.generation.rag.estimator import TranscriptEstimator
from app.generation.rag.idempotency import IdempotencyStore
from app.generation.rag.quality.hallucination import JudgePanel
from app.generation.rag.schemas import (
    Estimate,
    EstimationQuery,
    RetrievalResult,
    RetrievedChunk,
    SourceReference,
    TaskItem,
    WorkModule,
)
from tests.runtime_helpers import FakeRuntimeModelConfig, FakeRuntimeRetrievalConfig

META = LLMCallMeta(latency_ms=1, tokens_in=1, tokens_out=1, cost_usd=0.0, model="m", provider="p")
TRANSCRIPT = "The client wants a payments marketplace with Stripe Connect and KYC. " * 4


class CharEncoder:
    """1 character = 1 token: deterministic budgets without tiktoken's data download."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text)


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.1] * EMBEDDING_DIMENSION


class FakeSessionFactory:
    """``async with factory() as session``: the session is a placeholder (retrieve is patched)."""

    def __call__(self) -> Self:
        return self

    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_: object) -> None:
        return None


class ScriptedLLM:
    """Answers reformulation with ``brief``, the judge with ``judge`` and each generation with the
    next scripted Estimate. The default judge panel has no verdicts (numeric anchor only)."""

    def __init__(
        self,
        *estimates: Estimate,
        brief: EstimationQuery | None = None,
        judge: JudgePanel | Exception | None = None,
    ) -> None:
        self._estimates = list(estimates)
        self._brief = brief or EstimationQuery(function="payments marketplace", sector="finance")
        self._judge = judge or JudgePanel()
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[Any, LLMCallMeta]:
        self.calls.append(kwargs)
        if kwargs["response_model"] is EstimationQuery:
            return self._brief, META
        if kwargs["response_model"] is JudgePanel:
            if isinstance(self._judge, Exception):
                raise self._judge
            return self._judge, META
        return self._estimates.pop(0), META

    @property
    def generation_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["response_model"] is Estimate]

    @property
    def judge_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["response_model"] is JudgePanel]


def retrieved_chunk(chunk_id: int, content: str = "Payments module: 120 hours") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=content,
        sector="finance",
        project_year=2023,
        chunk_type="budget_component",
        distance=0.2,
        budget_id="BUD-2023-001",
    )


def result_with(*chunk_ids: int) -> RetrievalResult:
    return RetrievalResult(
        chunks=[retrieved_chunk(i) for i in chunk_ids],
        low_confidence=not chunk_ids,
        candidates_evaluated=60,
    )


def good_estimate(*cited_ids: str) -> Estimate:
    return Estimate(
        total_engineer_days=10,
        duration_weeks=4,
        modules=[
            WorkModule(
                name="Payments",
                tasks=[
                    TaskItem(
                        name="Checkout",
                        engineer_days=10,
                        grounded=True,
                        sources=[
                            SourceReference(
                                chunk_id=cid, document_id="BUD-2023-001", evidence="120 hours"
                            )
                            for cid in cited_ids
                        ],
                    )
                ],
            )
        ],
        confidence="high",
        reasoning="derived from the sources",
    )


class RetrieveSpy:
    """Replaces ``retrieve`` inside the estimator module: records kwargs, returns a canned result."""

    def __init__(self, result: RetrievalResult | Exception) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, session: Any, store: Any, **kwargs: Any) -> RetrievalResult:
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def patch_retrieve(
    monkeypatch: pytest.MonkeyPatch, result: RetrievalResult | Exception
) -> RetrieveSpy:
    spy = RetrieveSpy(result)
    monkeypatch.setattr(estimator_module, "retrieve", spy)
    return spy


def make_estimator(
    llm: ScriptedLLM,
    *,
    embedder: FakeEmbedder | None = None,
    models: FakeRuntimeModelConfig | None = None,
    retrieval_config: FakeRuntimeRetrievalConfig | None = None,
    idempotency: IdempotencyStore | None = None,
    settings: Settings | None = None,
) -> TranscriptEstimator:
    return TranscriptEstimator(
        llm=llm,  # type: ignore[arg-type]
        models=models or FakeRuntimeModelConfig(),  # type: ignore[arg-type]
        retrieval_config=retrieval_config or FakeRuntimeRetrievalConfig(),  # type: ignore[arg-type]
        embedder=embedder or FakeEmbedder(),  # type: ignore[arg-type]
        session_factory=FakeSessionFactory(),  # type: ignore[arg-type]
        store=SimpleNamespace(),  # type: ignore[arg-type]
        idempotency=idempotency or IdempotencyStore(),
        token_encoder=CharEncoder(),
        settings=settings or get_settings(),
    )
