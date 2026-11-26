"""LangGraph checkpointer on the project's own Postgres (Session 13).

``AsyncPostgresSaver`` speaks psycopg3 (libpq), not the SQLAlchemy/asyncpg URL the rest of the app
uses, so the URL is converted to a plain DSN. ``setup()`` creates the checkpoint tables in the
same database (the one with pgvector); it is idempotent, so it is safe to call on every startup.
"""

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.conninfo import make_conninfo

from app.config import DatabaseSettings

# Fail fast when Postgres is down: the service must still start without the graph
CONNECT_TIMEOUT_SECONDS = 5

_DRIVER_SUFFIX = re.compile(r"^(postgres(?:ql)?)\+\w+://")


def saver_conninfo(database_url: str | None = None) -> str:
    """``postgresql+asyncpg://...`` (SQLAlchemy URL) -> ``postgresql://...`` (libpq DSN)."""
    url = database_url or DatabaseSettings().DATABASE_URL
    return _DRIVER_SUFFIX.sub(r"\1://", url)


@asynccontextmanager
async def open_checkpointer(database_url: str | None = None) -> AsyncIterator[AsyncPostgresSaver]:
    """Open the Postgres checkpointer (one connection) and make sure its tables exist."""
    conninfo = make_conninfo(saver_conninfo(database_url), connect_timeout=CONNECT_TIMEOUT_SECONDS)
    async with AsyncPostgresSaver.from_conn_string(conninfo) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
