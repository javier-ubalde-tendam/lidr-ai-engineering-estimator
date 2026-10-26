from functools import lru_cache

from fastapi import Depends
from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import get_db_session
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.ingestion import IngestionService
from app.embedding_pipeline.retriever import Retriever
from app.sessions.store import SessionStore


@lru_cache
def get_openai_client() -> OpenAI:
    # lru_cache sin argumentos = singleton perezoso: se construye una sola vez, en el primer uso
    settings = get_settings()
    return OpenAI(api_key=settings.OPENAI_API_KEY)


@lru_cache
def get_session_store() -> SessionStore:
    # Mismo patrón que get_openai_client: singleton perezoso, sustituible en tests vía
    # app.dependency_overrides[get_session_store]
    settings = get_settings()
    return SessionStore(max_conversation_turns=settings.MAX_CONVERSATION_TURNS)


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    # Singleton perezoso: el chunker carga el tokenizer una sola vez
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    # Reutiliza el cliente OpenAI compartido; el embedder solo conoce la interfaz del cliente
    return OpenAIEmbedder(client=get_openai_client(), model=get_settings().EMBEDDING_MODEL)


# Servicios por petición: comparten la AsyncSession (una por request) y los singletons anteriores.
# Los tests los sustituyen enteros o solo get_db_session/get_embedder.
def get_ingestion_service(
    session: AsyncSession = Depends(get_db_session),
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> IngestionService:
    return IngestionService(session=session, chunker=chunker, embedder=embedder)


def get_retriever(
    session: AsyncSession = Depends(get_db_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> Retriever:
    return Retriever(session=session, embedder=embedder)
