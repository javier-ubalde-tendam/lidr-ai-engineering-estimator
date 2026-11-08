import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from app.generation.rag.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder: score = value looked up per document."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [self._scores[document] for _query, document in pairs]


@dataclass
class Candidate:
    id: int
    content: str


def make_reranker(scores: dict[str, float]) -> tuple[CrossEncoderReranker, FakeCrossEncoder]:
    reranker = CrossEncoderReranker("fake-model")
    fake = FakeCrossEncoder(scores)
    # Injecting the model skips the lazy load: no download, no torch
    reranker._model = fake
    return reranker, fake


def test_rerank_sorts_by_score_descending() -> None:
    reranker, _ = make_reranker({"a": 0.1, "b": 0.9, "c": 0.5})
    candidates = [Candidate(1, "a"), Candidate(2, "b"), Candidate(3, "c")]

    ranked = reranker.rerank("q", candidates, top_n=3)

    assert [c.id for c in ranked] == [2, 3, 1]


def test_rerank_truncates_to_top_n() -> None:
    reranker, _ = make_reranker({"a": 0.1, "b": 0.9, "c": 0.5})
    candidates = [Candidate(1, "a"), Candidate(2, "b"), Candidate(3, "c")]

    assert [c.id for c in reranker.rerank("q", candidates, top_n=2)] == [2, 3]


def test_rerank_is_stable_on_ties() -> None:
    reranker, _ = make_reranker({"x": 0.5, "y": 0.5, "z": 0.7})
    candidates = [Candidate(1, "x"), Candidate(2, "y"), Candidate(3, "z")]

    # Equal scores keep the recall order (1 before 2)
    assert [c.id for c in reranker.rerank("q", candidates, top_n=3)] == [3, 1, 2]


def test_rerank_empty_candidates_does_not_call_the_model() -> None:
    reranker, fake = make_reranker({})

    assert reranker.rerank("q", [], top_n=5) == []
    assert fake.calls == []


def test_score_uses_a_single_predict_call_in_input_order() -> None:
    reranker, fake = make_reranker({"a": 1.0, "b": 2.0})

    assert reranker.score("q", ["a", "b"]) == [1.0, 2.0]
    assert fake.calls == [[("q", "a"), ("q", "b")]]


def test_custom_text_of_extractor() -> None:
    reranker, _ = make_reranker({"low": 0.0, "high": 1.0})
    candidates = [{"text": "low"}, {"text": "high"}]

    ranked = reranker.rerank("q", candidates, top_n=1, text_of=lambda c: c["text"])

    assert ranked == [{"text": "high"}]


def test_from_settings_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "some/other-model")

    assert CrossEncoderReranker.from_settings().model_name == "some/other-model"


def test_constructing_does_not_load_the_model() -> None:
    reranker = CrossEncoderReranker("fake-model")

    assert reranker._model is None


def test_importing_the_module_does_not_import_torch() -> None:
    # Fresh interpreter: other tests in this process cannot pollute sys.modules
    code = (
        "import sys, app.generation.rag.retrieval.reranker; "
        "assert 'torch' not in sys.modules and 'sentence_transformers' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
