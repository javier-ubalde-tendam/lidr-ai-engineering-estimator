"""Checkpointer DSN conversion and the safe no-op behaviour of configure_logfire."""

from types import SimpleNamespace
from typing import Any

import pytest

from app.foundation import observability
from app.foundation.persistence.checkpointer import saver_conninfo


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://u:p@db:5432/estimator",
        "postgresql+psycopg://u:p@db:5432/estimator",
        "postgresql://u:p@db:5432/estimator",
    ],
)
def test_saver_conninfo_returns_a_plain_libpq_dsn(url: str) -> None:
    assert saver_conninfo(url) == "postgresql://u:p@db:5432/estimator"


@pytest.fixture
def fresh_observability(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"configure": [], "fastapi": 0}
    monkeypatch.setattr(observability, "_configured", False)
    monkeypatch.setattr(observability, "_instrumented_apps", set())
    monkeypatch.setattr(
        observability.logfire, "configure", lambda **kw: calls["configure"].append(kw)
    )
    monkeypatch.setattr(observability.logfire, "instrument_httpx", lambda: None)
    monkeypatch.setattr(observability.logfire, "instrument_asyncpg", lambda: None)
    monkeypatch.setattr(
        observability.logfire,
        "instrument_fastapi",
        lambda app: calls.update(fastapi=calls["fastapi"] + 1),
    )
    monkeypatch.setattr(
        observability,
        "get_settings",
        lambda: SimpleNamespace(LOGFIRE_TOKEN="", LOGFIRE_SERVICE_NAME="svc"),
    )
    return calls


def test_configure_is_idempotent_and_only_exports_with_a_token(
    fresh_observability: dict[str, Any],
) -> None:
    app = object()
    assert observability.configure_logfire(app) is True
    assert observability.configure_logfire(app) is True

    assert len(fresh_observability["configure"]) == 1
    kwargs = fresh_observability["configure"][0]
    assert kwargs["token"] is None  # an empty token never reaches Logfire
    assert kwargs["send_to_logfire"] == "if-token-present"
    assert fresh_observability["fastapi"] == 1  # the same app is instrumented once


def test_configure_never_breaks_startup(
    fresh_observability: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_: Any) -> None:
        raise RuntimeError("logfire down")

    monkeypatch.setattr(observability.logfire, "configure", boom)

    assert observability.configure_logfire() is False
