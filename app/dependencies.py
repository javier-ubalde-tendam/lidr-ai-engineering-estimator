from functools import lru_cache

from openai import OpenAI

from app.config import get_settings


@lru_cache
def get_openai_client() -> OpenAI:
    # lru_cache sin argumentos = singleton perezoso: se construye una sola vez, en el primer uso
    settings = get_settings()
    return OpenAI(api_key=settings.OPENAI_API_KEY)
