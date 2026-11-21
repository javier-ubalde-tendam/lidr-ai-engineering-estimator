from collections.abc import Iterator
from typing import Any

import pytest

from app.api.routers import retrieval as retrieval_router
from app.config import EMBEDDING_DIMENSION
from app.dependencies import (
    get_chunk_store,
    get_embedder,
    get_reranker,
    get_runtime_retrieval_config,
)
from app.foundation.persistence.database import get_db_session
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import RetrievalResult
from app.main import app
from tests.runtime_helpers import FakeRuntimeRetrievalConfig


class _FakeEmbedder:
    def embed_one(self, text: str) -> list[float]:
        return [0.1] * EMBEDDING_DIMENSION


@pytest.fixture
def retrieval_stubs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub everything behind POST /v1/retrieval/search so only auth/limits are exercised."""

    async def no_session() -> Any:
        yield None

    async def fake_retrieve(session: Any, store: Any, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(chunks=[], low_confidence=True, candidates_evaluated=0)

    app.dependency_overrides[get_db_session] = no_session
    app.dependency_overrides[get_embedder] = lambda: _FakeEmbedder()
    app.dependency_overrides[get_chunk_store] = lambda: object()
    app.dependency_overrides[get_reranker] = lambda: CrossEncoderReranker("fake")
    app.dependency_overrides[get_runtime_retrieval_config] = lambda: FakeRuntimeRetrievalConfig()
    monkeypatch.setattr(retrieval_router, "retrieve", fake_retrieve)
    yield
    app.dependency_overrides.clear()
