"""The Session 7 chunking strategies, with every external collaborator faked (no network, no key).

The sentence splitter, the LangChain embeddings, the LLM wrapper and the Anthropic client are
injected, so nothing here needs NLTK data, OpenAI or Anthropic.
"""

import re
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from app.foundation.llm.wrapper import LLMCallMeta
from app.generation.rag.chunking.base import Chunker, count_tokens
from app.generation.rag.chunking.strategies import (
    ContextualRetrievalChunker,
    FixedSizeChunker,
    HierarchicalChunker,
    PropositionalChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceWindowChunker,
    contextual_retrieval,
    fixed_size,
)
from app.generation.rag.chunking.structural import (
    JSONStructuralChunker,
    component_metadata,
    render_component_text,
    serialize_budget,
)
from app.generation.rag.schemas import Budget
from tests.embedding_helpers import make_budget


def budget_with_descriptions(*descriptions: str, module: str | None = None) -> Budget:
    budget = make_budget(n_components=len(descriptions))
    for component, description in zip(budget.components, descriptions, strict=True):
        component.description = description
        component.module = module
    return budget


def regex_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


# --- shared helpers / structural ------------------------------------------------------------


def test_component_metadata_omits_module_when_absent() -> None:
    budget = make_budget(n_components=1)

    assert "module" not in component_metadata(budget, budget.components[0])


def test_component_metadata_includes_module_when_present() -> None:
    budget = budget_with_descriptions("Does something", module="Payments")

    metadata = component_metadata(budget, budget.components[0])

    assert metadata["module"] == "Payments"
    assert "Module: Payments\n" in render_component_text(budget, budget.components[0])


def test_structural_chunker_is_a_chunker_and_keeps_its_output() -> None:
    budget = make_budget(n_components=1)

    chunker = JSONStructuralChunker()
    (chunk,) = chunker.chunk([budget])

    assert isinstance(chunker, Chunker)
    assert chunker.strategy_name == "structural"
    assert chunk.chunk_id == "BUD-1::C-0"
    assert chunk.text == render_component_text(budget, budget.components[0])
    assert chunk.metadata == component_metadata(budget, budget.components[0])
    assert chunk.token_count == count_tokens(chunk.text)


# --- fixed size -----------------------------------------------------------------------------


def test_fixed_size_short_component_is_one_window() -> None:
    chunks = FixedSizeChunker().chunk([make_budget(n_components=2)])

    assert [c.chunk_id for c in chunks] == ["BUD-1::C-0::p0", "BUD-1::C-1::p0"]
    assert all(c.metadata["part"] == 0 for c in chunks)


def test_fixed_size_long_component_slides_overlapping_windows() -> None:
    budget = budget_with_descriptions("alpha beta gamma delta " * 300)

    chunks = FixedSizeChunker().chunk([budget])

    assert len(chunks) > 1
    assert all(c.token_count <= fixed_size.CHUNK_SIZE_TOKENS for c in chunks)
    assert [c.metadata["part"] for c in chunks] == list(range(len(chunks)))
    # Consecutive windows share OVERLAP_TOKENS tokens, so the end of one reappears in the next
    tail = chunks[0].text[-40:]
    assert tail in chunks[1].text


# --- recursive ------------------------------------------------------------------------------


def test_recursive_splits_a_long_budget_within_the_token_budget() -> None:
    budget = budget_with_descriptions(*[f"Feature {i}. " + "word " * 200 for i in range(6)])

    chunks = RecursiveChunker().chunk([budget])

    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == [f"BUD-1::r{i}" for i in range(len(chunks))]
    assert all(c.token_count <= 600 for c in chunks)
    assert chunks[0].metadata["client_sector"] == "finance"


# --- sentence window ------------------------------------------------------------------------


def test_sentence_window_indexes_sentences_and_keeps_the_window() -> None:
    sentences = [f"Sentence number {i}." for i in range(6)]
    budget = budget_with_descriptions(" ".join(sentences))

    chunks = SentenceWindowChunker(split_sentences=regex_sentences).chunk([budget])

    assert [c.text for c in chunks] == sentences
    assert chunks[0].chunk_id == "BUD-1::C-0::s0"
    # Window = +-2 sentences, clipped at both ends of the description
    assert chunks[0].metadata["window_text"] == " ".join(sentences[0:3])
    assert chunks[3].metadata["window_text"] == " ".join(sentences[1:6])
    assert chunks[5].metadata["window_text"] == " ".join(sentences[3:6])
    assert chunks[2].metadata["sentence_index"] == 2


def test_default_sentence_splitter_is_nltk_loaded_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    # Building the chunker must not import NLTK nor fetch punkt data (offline-safe)
    monkeypatch.delitem(sys.modules, "nltk", raising=False)

    SentenceWindowChunker()

    assert "nltk" not in sys.modules


# --- hierarchical ---------------------------------------------------------------------------


def test_hierarchical_emits_one_parent_per_budget_and_children_pointing_to_it() -> None:
    budget = make_budget(n_components=2)

    chunks = HierarchicalChunker().chunk([budget])

    parent, *children = chunks
    assert parent.chunk_id == "BUD-1::parent"
    assert parent.text == serialize_budget(budget)
    assert parent.metadata["level"] == "parent"
    assert [c.chunk_id for c in children] == ["BUD-1::C-0", "BUD-1::C-1"]
    assert all(c.metadata["level"] == "child" for c in children)
    assert all(c.metadata["parent_chunk_id"] == "BUD-1::parent" for c in children)


# --- semantic -------------------------------------------------------------------------------


class FakeLangchainEmbeddings:
    """LangChain's ``Embeddings`` interface: two topics -> two orthogonal vectors."""

    def __init__(self) -> None:
        self.documents_calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents_calls.append(list(texts))
        return [[1.0, 0.0] if "auth" in t.lower() else [0.0, 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def test_semantic_cuts_on_topic_shift_and_reports_embedding_cost() -> None:
    embeddings = FakeLangchainEmbeddings()
    budget = budget_with_descriptions(
        "Auth login flow. Auth token refresh. Auth session expiry. "
        "Invoice export. Invoice tax report. Invoice payment reminders."
    )
    chunker = SemanticChunker(embeddings)

    chunks = chunker.chunk([budget])

    assert len(chunks) >= 2
    assert [c.chunk_id for c in chunks] == [f"BUD-1::sem{i}" for i in range(len(chunks))]
    # One batched embed_documents call per budget, and it is reported as ingestion cost
    assert len(embeddings.documents_calls) == 1
    assert chunker.last_extra_api_calls == 1
    assert chunker.last_extra_cost_usd > 0


# --- propositional --------------------------------------------------------------------------


class FakeLLM:
    """Stands in for ``LLMWrapper.complete_structured``."""

    def __init__(self, propositions: list[str] | None = None, error: Exception | None = None):
        self.propositions = propositions or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[Any, LLMCallMeta]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        result = kwargs["response_model"](propositions=self.propositions)
        meta = LLMCallMeta(
            latency_ms=1, tokens_in=10, tokens_out=5, cost_usd=0.001, model="m", provider="openai"
        )
        return result, meta


def test_propositional_one_chunk_per_proposition_with_cost() -> None:
    llm = FakeLLM(["The API uses Redis.", "The API is written in Python."])
    chunker = PropositionalChunker(llm=llm, model="cheap-model")  # type: ignore[arg-type]

    chunks = chunker.chunk([make_budget(n_components=2)])

    assert [c.text for c in chunks[:2]] == ["The API uses Redis.", "The API is written in Python."]
    assert [c.chunk_id for c in chunks[:2]] == ["BUD-1::C-0::prop0", "BUD-1::C-0::prop1"]
    assert len(chunks) == 4
    assert chunker.last_extra_api_calls == 2
    assert chunker.last_extra_cost_usd == pytest.approx(0.002)
    assert {call["model"] for call in llm.calls} == {"cheap-model"}


@pytest.mark.parametrize("llm", [FakeLLM([]), FakeLLM(error=RuntimeError("boom"))])
def test_propositional_falls_back_to_the_whole_component(llm: FakeLLM) -> None:
    budget = make_budget(n_components=1)
    chunker = PropositionalChunker(llm=llm, model="m")  # type: ignore[arg-type]

    (chunk,) = chunker.chunk([budget])

    assert chunk.text == render_component_text(budget, budget.components[0])
    assert chunker.last_extra_api_calls == 1


# --- contextual retrieval -------------------------------------------------------------------


class FakeAnthropic:
    def __init__(self, text: str = "Situating context.", error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._text = text
        self._error = error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        usage = SimpleNamespace(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return SimpleNamespace(content=[SimpleNamespace(text=f"  {self._text}  ")], usage=usage)


def test_contextual_retrieval_prepends_context_and_caches_the_parent_document() -> None:
    client = FakeAnthropic()
    budget = make_budget(n_components=2)
    chunker = ContextualRetrievalChunker(client=client, model="claude-test")  # type: ignore[arg-type]

    chunks = chunker.chunk([budget])

    assert [c.chunk_id for c in chunks] == ["BUD-1::C-0", "BUD-1::C-1"]
    base = render_component_text(budget, budget.components[0])
    assert chunks[0].text == f"Situating context.\n\n{base}"
    assert chunks[0].metadata["generated_context"] == "Situating context."
    call = client.calls[0]
    assert call["model"] == "claude-test"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert serialize_budget(budget) in call["system"][0]["text"]
    assert chunker.last_extra_api_calls == 2
    # 1M input tokens per call at the Sonnet input price, twice
    assert chunker.last_extra_cost_usd == pytest.approx(2 * contextual_retrieval.SONNET_INPUT_PER_MILLION_USD)


def test_contextual_retrieval_keeps_the_plain_chunk_when_the_call_fails() -> None:
    budget = make_budget(n_components=1)
    chunker = ContextualRetrievalChunker(client=FakeAnthropic(error=RuntimeError("down")), model="m")  # type: ignore[arg-type]

    (chunk,) = chunker.chunk([budget])

    assert chunk.text == render_component_text(budget, budget.components[0])
    assert chunk.metadata["generated_context"] == ""
    assert chunker.last_extra_cost_usd == 0.0
