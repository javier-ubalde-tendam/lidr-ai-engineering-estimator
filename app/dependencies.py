from functools import lru_cache

from openai import OpenAI

from app.config import get_settings
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
