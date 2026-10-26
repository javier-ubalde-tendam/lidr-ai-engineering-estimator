import asyncio
import time
from collections.abc import Callable
from typing import Any

import openai
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import SUPPORTED_EMBEDDING_MODEL
from app.db.models import Chunk, Document
from app.embedding_pipeline import ingestion
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.errors import DocumentAlreadyIngestedError
from app.embedding_pipeline.ingestion import IngestionService, is_source_path_violation
from app.embedding_pipeline.schemas import IngestRequest
from tests.embedding_helpers import FakeOpenAIClient, make_budget, make_embedder
from tests.integration.conftest import run

pytestmark = pytest.mark.integration

SOURCE_PATH = "data/budgets_sample.json::BUD-1"


def payload(n_components: int = 3, source_path: str = SOURCE_PATH) -> dict[str, Any]:
    return {
        "source_path": source_path,
        "document_type": "historical_budget",
        "content": make_budget(n_components=n_components).model_dump(),
    }


def counts(sql: Callable[..., list[Any]]) -> tuple[int, int]:
    row = sql("SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)")[0]
    return row[0], row[1]


def test_ingest_persists_document_and_chunks(
    api: TestClient, fake_openai: FakeOpenAIClient, sql: Callable[..., list[Any]]
) -> None:
    response = api.post("/embeddings/ingest", json=payload(3))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"document_id", "chunks_created", "embedding_dimension", "ingestion_time_ms"}
    assert (body["chunks_created"], body["embedding_dimension"]) == (3, 1536)
    assert counts(sql) == (1, 3)

    document = sql("SELECT id, source_path, document_type, metadata, ingested_at FROM documents")[0]
    assert document.id == body["document_id"]
    assert (document.source_path, document.document_type) == (SOURCE_PATH, "historical_budget")
    assert document.metadata["budget_id"] == "BUD-1"
    assert document.metadata["client_metadata"]["sector"] == "finance"
    assert document.metadata["component_count"] == 3
    assert "components" not in document.metadata
    assert document.ingested_at is not None

    chunks = sql(
        "SELECT chunk_type, content, vector_dims(embedding) AS dims, metadata FROM chunks ORDER BY id"
    )
    assert {chunk.chunk_type for chunk in chunks} == {"budget_component"}
    assert {chunk.dims for chunk in chunks} == {1536}
    assert chunks[0].content.startswith("[Project: Mobile banking API]")
    assert chunks[0].metadata["chunk_id"] == "BUD-1::C-0"
    assert chunks[0].metadata["component_id"] == "C-0"
    assert chunks[0].metadata["client_sector"] == "finance"
    assert chunks[0].metadata["tech_stack"] == ["python", "redis"]
    # Un único lote con los 3 textos y el modelo del contrato (no una llamada por chunk)
    assert len(fake_openai.calls) == 1
    assert len(fake_openai.calls[0]["input"]) == 3  # type: ignore[arg-type]
    assert fake_openai.calls[0]["model"] == SUPPORTED_EMBEDDING_MODEL


def test_duplicate_returns_409_without_new_rows_or_embedding_calls(
    api: TestClient, fake_openai: FakeOpenAIClient, sql: Callable[..., list[Any]]
) -> None:
    first = api.post("/embeddings/ingest", json=payload(2)).json()

    response = api.post("/embeddings/ingest", json=payload(2))

    assert response.status_code == 409
    assert response.json() == {"detail": "Document already ingested", "document_id": first["document_id"]}
    assert counts(sql) == (1, 2)
    assert len(fake_openai.calls) == 1


def test_embedder_failure_rolls_back_everything(
    api: TestClient, fake_openai: FakeOpenAIClient, sql: Callable[..., list[Any]]
) -> None:
    def failing_create(model: str, input: list[str]) -> Any:
        raise openai.OpenAIError("boom sk-secret")

    fake_openai.embeddings.create = failing_create  # type: ignore[method-assign]

    response = api.post("/embeddings/ingest", json=payload())

    assert response.status_code == 500
    assert response.json() == {"detail": "Embedding service error"}
    assert counts(sql) == (0, 0)


def test_wrong_vector_dimension_rolls_back_the_flushed_document(
    api: TestClient, fake_openai: FakeOpenAIClient, sql: Callable[..., list[Any]]
) -> None:
    fake_openai._vector_for = lambda _text, _index: [0.5] * 10

    response = api.post("/embeddings/ingest", json=payload())

    assert response.status_code == 500
    assert counts(sql) == (0, 0)


def test_sql_failure_after_document_flush_rolls_back_document_and_chunks(
    api: TestClient, sql: Callable[..., list[Any]]
) -> None:
    # PostgreSQL rechaza NUL en text: falla al insertar el 2.º chunk, ya con el document flusheado
    body = payload(3)
    body["content"]["components"][2]["description"] = "bad \x00 byte"

    response = api.post("/embeddings/ingest", json=body)

    assert response.status_code == 500
    assert response.json() == {"detail": "Database error"}
    assert counts(sql) == (0, 0)

    # La BD sigue operativa y el mismo source_path puede ingestarse bien después
    assert api.post("/embeddings/ingest", json=payload(3)).status_code == 200
    assert counts(sql) == (1, 3)


def test_concurrent_duplicates_are_resolved_by_the_unique_index(
    session_factory: async_sessionmaker[AsyncSession],
    chunker: JSONStructuralChunker,
    sql: Callable[..., list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeOpenAIClient()
    original_create = client.embeddings.create

    def slow_create(model: str, input: list[str]) -> Any:
        # Mantiene abierta la transacción ganadora mientras la otra intenta insertar
        time.sleep(0.5)
        return original_create(model=model, input=input)

    client.embeddings.create = slow_create  # type: ignore[method-assign]
    race_detections: list[bool] = []
    real_check = ingestion.is_source_path_violation
    monkeypatch.setattr(
        ingestion,
        "is_source_path_violation",
        lambda exc: race_detections.append(True) or real_check(exc),
    )
    request = IngestRequest.model_validate(payload(2))

    async def attempt() -> Any:
        async with session_factory() as session:
            service = IngestionService(session, chunker, make_embedder(client))
            return await service.ingest(request)

    async def both() -> list[Any]:
        return await asyncio.gather(attempt(), attempt(), return_exceptions=True)

    results = run(both())

    winners = [r for r in results if not isinstance(r, Exception)]
    losers = [r for r in results if isinstance(r, DocumentAlreadyIngestedError)]
    assert len(winners) == 1 and len(losers) == 1
    assert losers[0].document_id == winners[0].document_id
    assert race_detections, "la carrera debe resolverse por el índice único, no por la comprobación previa"
    assert counts(sql) == (1, 2)


def test_is_source_path_violation_only_matches_the_unique_index(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async def duplicate_document() -> IntegrityError:
        async with session_factory() as session:
            try:
                async with session.begin():
                    session.add_all(
                        [
                            Document(source_path="dup", document_type="t", metadata_={}),
                            Document(source_path="dup", document_type="t", metadata_={}),
                        ]
                    )
                    await session.flush()
            except IntegrityError as exc:
                return exc
        raise AssertionError("se esperaba IntegrityError")

    async def orphan_chunk() -> IntegrityError:
        async with session_factory() as session:
            try:
                async with session.begin():
                    session.add(
                        Chunk(document_id=999_999, chunk_type="t", content="c", metadata_={})
                    )
                    await session.flush()
            except IntegrityError as exc:
                return exc
        raise AssertionError("se esperaba IntegrityError")

    assert is_source_path_violation(run(duplicate_document())) is True
    assert is_source_path_violation(run(orphan_chunk())) is False


def test_deleting_a_document_cascades_to_its_chunks(
    api: TestClient, sql: Callable[..., list[Any]]
) -> None:
    api.post("/embeddings/ingest", json=payload(3))
    api.post("/embeddings/ingest", json=payload(2, source_path="other"))
    assert counts(sql) == (2, 5)

    sql_delete = "DELETE FROM documents WHERE source_path = :p RETURNING id"
    deleted = sql(sql_delete, p=SOURCE_PATH)

    assert len(deleted) == 1
    assert counts(sql) == (1, 2)
