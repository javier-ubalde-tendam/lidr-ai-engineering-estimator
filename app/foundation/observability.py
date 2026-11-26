"""Logfire setup (Session 13). Not to be confused with ``generation/rag/observability.py``.

``configure_logfire`` is a *safe no-op*: it never breaks startup, is idempotent, and only exports
when a token is configured. Settings reads ``.env`` but does not export it to ``os.environ``, which
is where Logfire would look for ``LOGFIRE_TOKEN``, so the token is passed explicitly.
"""

from typing import Any

import logfire
import structlog

from app.config import get_settings

log = structlog.get_logger(__name__)

_configured = False
_instrumented_apps: set[int] = set()


def configure_logfire(app: Any = None) -> bool:
    """Configure Logfire and the instrumentations once; return whether it is active."""
    global _configured
    try:
        if not _configured:
            settings = get_settings()
            logfire.configure(
                # Export only when a token exists; without one the spans stay local
                send_to_logfire="if-token-present",
                token=settings.LOGFIRE_TOKEN or None,
                service_name=settings.LOGFIRE_SERVICE_NAME,
                console=False,
            )
            # httpx = the OpenAI calls (via LiteLLM); asyncpg = the retrieval queries.
            # No request/response bodies are captured: transcripts may hold personal data
            logfire.instrument_httpx()
            logfire.instrument_asyncpg()
            _configured = True
        if app is not None and id(app) not in _instrumented_apps:
            logfire.instrument_fastapi(app)
            _instrumented_apps.add(id(app))
    except Exception as exc:  # noqa: BLE001 - observability must never prevent the service from starting
        log.warning(
            "logfire_configuration_failed", error_type=type(exc).__name__, error=str(exc)[:200]
        )
    return _configured
