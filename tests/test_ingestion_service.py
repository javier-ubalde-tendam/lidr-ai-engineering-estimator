from types import SimpleNamespace
from typing import Any

import openai
import pytest

from app.config import EMBEDDING_DIMENSION, SUPPORTED_EMBEDDING_MODEL
from app.db.models import Chunk, Document
from app.dependencies import get_embedder, get_ingestion_service, get_retriever
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.errors import (
    DocumentAlreadyIngestedError,
    InvalidEmbeddingError,
    ensure_valid_vectors,
)
from app.embedding_pipeline.ingestion import IngestionService, is_source_path_violation
from app.embedding_pipeline.retriever import Retriever
from app.embedding_pipeline.schemas import IngestRequest
from tests.embedding_helpers import FakeOpenAIClient, angle_vector, make_budget, make_embedder

# Plugin pytest de anyio (viene con FastAPI): ejecuta los tests `async def` sin pytest-asyncio
pytestmark = pytest.mark.anyio


class FakeTransaction:
    def __init__(self, session: "FakeSession") -> None:
        self._session = session

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self._session.commits += 1
        else:
            self._session.rollbacks += 1


class FakeSession:
    """Solo comprueba el orden de operaciones; la atomicidad SQL real está en tests/integration."""

    def __init__(self, existing_id: int | None = None) -> None:
        self.existing_id = existing_id
        self.added: list[Any] = []
        self.commits = 0
        self.rollbacks = 0

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self)

    async def execute(self, _statement: Any) -> Any:
        return SimpleNamespace(scalar_one_or_none=lambda: self.existing_id)

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    def add_all(self, objs: list[Any]) -> None:
        self.added.extend(objs)

    async def flush(self) -> None:
        for obj in self.added:
            if isinstance(obj, Document) and obj.id is None:
                obj.id = 1


def make_request(n_components: int = 3) -> IngestRequest:
    return IngestRequest(
        source_path="data/budgets_sample.json::BUD-1",
        document_type="historical_budget",
        content=make_budget(n_components=n_components),
    )


def make_service(session: FakeSession, client: FakeOpenAIClient) -> IngestionService:
    return IngestionService(
        session=session,  # type: ignore[arg-type]
        chunker=JSONStructuralChunker(),
        embedder=make_embedder(client),
    )


async def test_ingest_persists_document_and_chunks_in_one_transaction() -> None:
    session, client = FakeSession(), FakeOpenAIClient()

    response = await make_service(session, client).ingest(make_request(3))

    assert (response.document_id, response.chunks_created, response.embedding_dimension) == (1, 3, 1536)
    assert session.commits == 1 and session.rollbacks == 0
    document, *chunks = session.added
    assert document.metadata_["budget_id"] == "BUD-1"
    assert document.metadata_["component_count"] == 3
    assert "components" not in document.metadata_
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert {chunk.chunk_type for chunk in chunks} == {"budget_component"}
    assert chunks[0].metadata_["chunk_id"] == "BUD-1::C-0"
    assert chunks[0].metadata_["client_sector"] == "finance"
    assert chunks[0].metadata_["tech_stack"] == ["python", "redis"]
    assert chunks[0].document_id == 1


async def test_ingest_batches_embeddings_in_a_single_call_with_the_contract_model() -> None:
    session, client = FakeSession(), FakeOpenAIClient()

    await make_service(session, client).ingest(make_request(5))

    assert len(client.calls) == 1
    assert len(client.calls[0]["input"]) == 5  # type: ignore[arg-type]
    assert client.calls[0]["model"] == SUPPORTED_EMBEDDING_MODEL


async def test_duplicate_is_detected_before_paying_for_embeddings() -> None:
    session, client = FakeSession(existing_id=42), FakeOpenAIClient()

    with pytest.raises(DocumentAlreadyIngestedError) as exc_info:
        await make_service(session, client).ingest(make_request())

    assert exc_info.value.document_id == 42
    assert client.calls == []
    assert session.added == []
    assert session.rollbacks == 1


async def test_embedder_failure_rolls_back_the_transaction() -> None:
    session = FakeSession()
    client = FakeOpenAIClient()

    def failing_create(model: str, input: list[str]) -> Any:
        raise openai.OpenAIError("boom")

    client.embeddings = SimpleNamespace(create=failing_create)

    with pytest.raises(openai.OpenAIError):
        await make_service(session, client).ingest(make_request())

    assert session.commits == 0 and session.rollbacks == 1
    assert not any(isinstance(obj, Chunk) for obj in session.added)


async def test_wrong_dimension_is_rejected_before_persisting_chunks() -> None:
    session = FakeSession()
    client = FakeOpenAIClient(vector_for=lambda _text, _index: [0.1, 0.2])

    with pytest.raises(InvalidEmbeddingError):
        await make_service(session, client).ingest(make_request())

    assert session.commits == 0 and session.rollbacks == 1
    assert not any(isinstance(obj, Chunk) for obj in session.added)


# --- vectores y errores ---


def test_ensure_valid_vectors_checks_count_and_dimension() -> None:
    good = [angle_vector(0.1)]
    ensure_valid_vectors(good, expected_count=1)
    with pytest.raises(InvalidEmbeddingError):
        ensure_valid_vectors(good, expected_count=2)
    with pytest.raises(InvalidEmbeddingError):
        ensure_valid_vectors([[0.0] * (EMBEDDING_DIMENSION - 1)], expected_count=1)


def fake_integrity_error(sqlstate: str, constraint: str) -> Any:
    cause = SimpleNamespace(sqlstate=sqlstate, constraint_name=constraint)
    orig = SimpleNamespace(__cause__=cause)
    return SimpleNamespace(orig=orig)


def test_only_source_path_unique_violation_counts_as_duplicate() -> None:
    assert is_source_path_violation(fake_integrity_error("23505", "ix_documents_source_path"))
    assert not is_source_path_violation(fake_integrity_error("23505", "chunks_pkey"))
    assert not is_source_path_violation(fake_integrity_error("23503", "chunks_document_id_fkey"))


# --- Retriever: misma dimensión que la ingesta ---


async def test_search_rejects_query_vectors_with_wrong_dimension() -> None:
    client = FakeOpenAIClient(vector_for=lambda _text, _index: [0.1, 0.2])
    retriever = Retriever(session=None, embedder=make_embedder(client))  # type: ignore[arg-type]

    with pytest.raises(InvalidEmbeddingError):
        await retriever.search("hello", 5)
    assert client.calls[0]["model"] == SUPPORTED_EMBEDDING_MODEL


def test_ingestion_and_search_share_the_same_embedder() -> None:
    embedder = make_embedder(FakeOpenAIClient())
    service = get_ingestion_service(session=None, chunker=JSONStructuralChunker(), embedder=embedder)  # type: ignore[arg-type]
    retriever = get_retriever(session=None, embedder=embedder)  # type: ignore[arg-type]

    assert service._embedder is retriever._embedder


def test_default_embedder_uses_the_contract_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.dependencies.get_openai_client", lambda: FakeOpenAIClient())

    embedder = get_embedder.__wrapped__()

    assert embedder._model == SUPPORTED_EMBEDDING_MODEL
