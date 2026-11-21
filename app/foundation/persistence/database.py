from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    # Singleton perezoso: importar la app no abre conexiones; el pool se conecta en el primer uso
    return create_async_engine(get_settings().DATABASE_URL, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: tras el commit los objetos siguen legibles sin un refresh implícito
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    # Sin commit automático: cada servicio abre y cierra su propia transacción de forma explícita
    async with get_sessionmaker()() as session:
        yield session


async def dispose_engine() -> None:
    # Solo si se llegó a crear: no abrir un engine en el shutdown de una app que nunca usó la BD
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
        get_sessionmaker.cache_clear()
        get_engine.cache_clear()
