"""The real Postgres checkpointer: open it and create its tables (needs Postgres; -m integration)."""

import asyncio
import sys

import pytest

from app.foundation.persistence.checkpointer import open_checkpointer

pytestmark = pytest.mark.integration


async def _open_and_setup(url: str) -> bool:
    async with open_checkpointer(url) as checkpointer:
        await checkpointer.setup()  # idempotent: a second call must not fail
        return await checkpointer.aget_tuple({"configurable": {"thread_id": "missing"}}) is None


def test_open_checkpointer_sets_up_its_tables(test_database_url: str) -> None:
    # psycopg's async mode needs the selector loop on Windows
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    assert asyncio.run(_open_and_setup(test_database_url), loop_factory=factory)
