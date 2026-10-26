"""Tests de integración contra PostgreSQL + pgvector REAL, en una BD exclusiva de pruebas.

Se ejecutan con `uv run pytest -m integration` (ver README). Si Postgres no está disponible, fallan
con el error real de conexión: no hay skips que lo oculten.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from alembic import command
from app.config import DatabaseSettings
from app.db.session import get_db_session
from app.dependencies import get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.main import app
from tests.embedding_helpers import FakeOpenAIClient, make_embedder

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DATABASE_URL = "postgresql+asyncpg://estimator:estimator@localhost:5432/estimator_test"

T = TypeVar("T")


def run(coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)
    test_db = make_url(url).database
    dev_db = make_url(DatabaseSettings().DATABASE_URL).database
    # Protección: estos tests hacen TRUNCATE; nunca contra la BD de desarrollo
    if not test_db or not test_db.endswith("_test") or test_db == dev_db:
        raise pytest.UsageError(
            f"TEST_DATABASE_URL debe apuntar a una BD distinta de la de desarrollo ({dev_db!r}) "
            f"cuyo nombre termine en '_test'; recibido: {test_db!r}"
        )
    return url


async def _ensure_database_exists(url: str) -> None:
    parsed = make_url(url)
    admin = create_async_engine(
        parsed.set(database="postgres"), poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as connection:
        exists = await connection.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": parsed.database}
        )
        if not exists:
            # El nombre ya se validó (termina en _test); CREATE DATABASE no admite parámetros
            await connection.execute(text(f'CREATE DATABASE "{parsed.database}"'))
    await admin.dispose()


@pytest.fixture(scope="session")
def migrated_database(test_database_url: str) -> str:
    # Fixture síncrona: env.py de Alembic usa asyncio.run y no puede ejecutarse dentro de un loop
    run(_ensure_database_exists(test_database_url))
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", test_database_url.replace("%", "%%"))
    command.upgrade(config, "head")
    return test_database_url


@pytest.fixture
def session_factory(migrated_database: str) -> async_sessionmaker[AsyncSession]:
    # NullPool: cada sesión abre su conexión; así sirve en cualquier event loop (TestClient, asyncio.run)
    engine = create_async_engine(migrated_database, poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def sql(session_factory: async_sessionmaker[AsyncSession]) -> Callable[..., list[Any]]:
    def execute(statement: str, **params: Any) -> list[Any]:
        async def go() -> list[Any]:
            async with session_factory() as session, session.begin():
                return list((await session.execute(text(statement), params)).all())

        return run(go())

    return execute


@pytest.fixture(autouse=True)
def clean_tables(session_factory: async_sessionmaker[AsyncSession]) -> None:
    async def truncate() -> None:
        async with session_factory() as session, session.begin():
            await session.execute(text("TRUNCATE chunks, documents RESTART IDENTITY CASCADE"))

    run(truncate())


@pytest.fixture
def fake_openai() -> FakeOpenAIClient:
    return FakeOpenAIClient()


@pytest.fixture(scope="module")
def chunker() -> JSONStructuralChunker:
    return JSONStructuralChunker()


@pytest.fixture
def api(
    session_factory: async_sessionmaker[AsyncSession],
    fake_openai: FakeOpenAIClient,
    chunker: JSONStructuralChunker,
) -> Iterator[TestClient]:
    async def db_session() -> Any:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = db_session
    app.dependency_overrides[get_embedder] = lambda: make_embedder(fake_openai)
    app.dependency_overrides[get_chunker] = lambda: chunker
    yield TestClient(app)
    app.dependency_overrides.clear()
