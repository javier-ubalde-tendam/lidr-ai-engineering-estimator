"""Per-stage structured logging for the RAG pipeline.

``log_stage`` wraps each pipeline step so every stage emits a consistent ``stage.started`` /
``stage.completed`` (with ``duration_ms``) / ``stage.failed`` trio, all correlated by a shared
``request_id``. A single request's journey through reformulation -> retrieval -> augmentation
-> generation is then trivially greppable in the logs.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

log = structlog.get_logger()


@contextmanager
def log_stage(stage: str, request_id: str, **context: Any) -> Iterator[None]:
    """Log the lifecycle of one pipeline ``stage`` bound to ``request_id``.

    ``context`` adds extra structured fields to all events. Any exception raised in the body
    is logged as ``stage.failed`` (with ``duration_ms`` and the error type) and re-raised.
    """
    log.info("stage.started", stage=stage, request_id=request_id, **context)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        log.error(
            "stage.failed",
            stage=stage,
            request_id=request_id,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error_type=type(exc).__name__,
            error=str(exc)[:300],
            **context,
        )
        raise
    log.info(
        "stage.completed",
        stage=stage,
        request_id=request_id,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        **context,
    )
