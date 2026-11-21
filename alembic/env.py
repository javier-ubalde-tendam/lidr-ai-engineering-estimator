import asyncio
from logging.config import fileConfig

import pgvector.sqlalchemy
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from app.config import DatabaseSettings
from app.foundation.persistence.models import Base
from app.generation.rag.store import models as _rag_models  # noqa: F401  (registers the tables)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata

# Expression indexes with their own operator class (migration 0004) that the ORM does not declare;
# Alembic reflects them as "removed" on every `alembic check`, so they are excluded from autogenerate
HNSW_INDEX_SUFFIX = "_embedding_hnsw"


def include_object(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    return not (type_ == "index" and reflected and name is not None and name.endswith(HNSW_INDEX_SUFFIX))


def get_url() -> str:
    # Los tests de integración inyectan su URL con config.set_main_option; si no, entorno/.env
    return config.get_main_option("sqlalchemy.url") or DatabaseSettings().DATABASE_URL


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    # Permite reflejar/autogenerar columnas vector (sin esto `alembic check` no reconoce el tipo)
    connection.dialect.ischema_names["vector"] = pgvector.sqlalchemy.Vector  # type: ignore[attr-defined]
    context.configure(
        connection=connection, target_metadata=target_metadata, include_object=include_object
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(get_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
