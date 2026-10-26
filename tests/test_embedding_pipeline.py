import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import BATCH_SIZE, OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import Chunk
from tests.embedding_helpers import make_budget

ROOT = Path(__file__).resolve().parent.parent


def load_cosine_similarity():
    # scripts/ no es un paquete: se carga el fichero por ruta con importlib
    spec = importlib.util.spec_from_file_location("compare", ROOT / "scripts" / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cosine_similarity


def make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"B::{i}", text=f"text {i}", metadata={}, token_count=2) for i in range(n)
    ]


class FakeEmbeddings:
    """Imita client.embeddings: cada texto 'text N' devuelve el vector [N]."""

    def __init__(self, errors: list[Exception] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._errors = errors or []

    def create(self, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append(input)
        if self._errors:
            raise self._errors.pop(0)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(t.split()[-1])]) for t in input]
        )


def make_embedder(
    errors: list[Exception] | None = None,
) -> tuple[OpenAIEmbedder, FakeEmbeddings, list[float]]:
    fake = FakeEmbeddings(errors)
    sleeps: list[float] = []
    client = SimpleNamespace(embeddings=fake)
    return OpenAIEmbedder(client=client, model="m", sleep=sleeps.append), fake, sleeps  # type: ignore[arg-type]


def rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com"))
    return openai.RateLimitError("rate limited", response=response, body=None)


# --- Chunker ---


def test_chunker_one_chunk_per_component() -> None:
    chunks = JSONStructuralChunker().chunk([make_budget("A", 2), make_budget("B", 3)])
    assert len(chunks) == 5


def test_chunker_chunk_id_format() -> None:
    chunks = JSONStructuralChunker().chunk([make_budget("BUD-1", 1)])
    assert chunks[0].chunk_id == "BUD-1::C-0"


def test_chunker_text_has_parent_header_and_component() -> None:
    text = JSONStructuralChunker().chunk([make_budget()])[0].text
    assert text.startswith(
        "[Project: Mobile banking API]\n"
        "[Client sector: finance | Year: 2024 | Main tech: python]\n"
        "\n"
        "Component: Component 0\n"
    )
    assert "Tech stack: python, redis" in text
    assert text.endswith("Estimated hours: 10")


def test_chunker_metadata_keys() -> None:
    metadata = JSONStructuralChunker().chunk([make_budget()])[0].metadata
    assert set(metadata) == {
        "budget_id",
        "component_id",
        "client_sector",
        "main_technology",
        "year",
        "complexity",
        "estimated_hours",
    }
    assert metadata["client_sector"] == "finance"


def test_chunker_token_count_positive() -> None:
    assert all(c.token_count > 0 for c in JSONStructuralChunker().chunk([make_budget()]))


# --- Embedder ---


def test_embed_many_batches_and_keeps_order() -> None:
    embedder, fake, _ = make_embedder()
    result = embedder.embed_many(make_chunks(250))

    assert [len(call) for call in fake.calls] == [BATCH_SIZE, BATCH_SIZE, 50]
    assert [c.chunk_id for c in result] == [f"B::{i}" for i in range(250)]
    assert [c.embedding for c in result] == [[float(i)] for i in range(250)]


def test_embed_one_returns_vector() -> None:
    embedder, _, _ = make_embedder()
    assert embedder.embed_one("text 7") == [7.0]


def test_rate_limit_retries_then_succeeds() -> None:
    embedder, fake, sleeps = make_embedder(errors=[rate_limit_error(), rate_limit_error()])
    result = embedder.embed_many(make_chunks(1))

    assert len(result) == 1
    assert len(fake.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_rate_limit_exhausted_reraises() -> None:
    embedder, fake, sleeps = make_embedder(errors=[rate_limit_error() for _ in range(4)])

    with pytest.raises(openai.RateLimitError):
        embedder.embed_many(make_chunks(1))
    assert len(fake.calls) == 4  # 1 intento inicial + 3 reintentos
    assert sleeps == [1.0, 2.0, 4.0]


def test_other_errors_are_not_retried() -> None:
    request = httpx.Request("POST", "https://api.openai.com")
    embedder, fake, sleeps = make_embedder(errors=[openai.APIConnectionError(request=request)])

    with pytest.raises(openai.APIConnectionError):
        embedder.embed_many(make_chunks(1))
    assert len(fake.calls) == 1
    assert sleeps == []


def test_estimate_cost_usd() -> None:
    assert estimate_cost_usd(1_000_000) == pytest.approx(0.02)
    assert estimate_cost_usd(0) == 0.0


# --- Cosine ---


def test_cosine_identical_vectors_is_one() -> None:
    assert load_cosine_similarity()([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero() -> None:
    assert load_cosine_similarity()([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero() -> None:
    assert load_cosine_similarity()([0.0, 0.0], [1.0, 2.0]) == 0.0
