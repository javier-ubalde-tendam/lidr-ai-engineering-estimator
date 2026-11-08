"""Cross-encoder reranker (session 10).

A bi-encoder (our OpenAI embeddings) encodes query and document separately and compares two
vectors; a cross-encoder reads the (query, document) pair together and outputs one relevance
score. Much more accurate, much slower: that is why it only rescores the small pool recalled by
the cheap branches (recall-then-rerank). The orchestration (recall width, final cut) lives in
``pipeline.retrieve``; this module only loads the model and scores pairs.

Importing this module must stay cheap: ``torch``/``sentence_transformers`` are imported inside
the lazy loader, so app startup and unit tests never pay for them.
"""

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def _content_of(candidate: Any) -> str:
    # Default text extractor: ORM rows and RetrievedChunk both expose `.content`
    return str(candidate.content)


class CrossEncoderReranker:
    """Lazily loaded cross-encoder that rescores (query, document) pairs."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        # sentence_transformers.CrossEncoder once loaded; tests inject a fake here
        self._model: Any = None
        self._load_lock = threading.Lock()

    @classmethod
    def from_settings(cls) -> "CrossEncoderReranker":
        return cls(get_settings().RERANKER_MODEL)

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        """Force the model load now (verify script and evaluation warm-up)."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        # Double-checked locking: two concurrent first requests must not load the model twice
        with self._load_lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                started = time.perf_counter()
                self._model = CrossEncoder(self._model_name)
                logger.info(
                    "reranker_loaded",
                    model=self._model_name,
                    load_ms=round((time.perf_counter() - started) * 1000),
                )
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance of each document for the query (higher = better), in input order."""
        if not documents:
            return []
        model = self._ensure_loaded()
        # One predict() call with every pair: the library batches internally
        scores = model.predict([(query, document) for document in documents])
        return [float(score) for score in scores]

    def rerank(
        self,
        query: str,
        candidates: Sequence[T],
        *,
        top_n: int,
        text_of: Callable[[T], str] = _content_of,
    ) -> list[T]:
        """Sort candidates by cross-encoder score (descending) and keep the first ``top_n``.

        ``sorted`` is stable, so equal scores keep the recall order.
        """
        if not candidates:
            return []
        scores = self.score(query, [text_of(candidate) for candidate in candidates])
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [candidate for candidate, _ in ranked[:top_n]]
