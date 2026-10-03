import hashlib

import redis
import structlog

from app.config import get_settings

settings = get_settings()
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
logger = structlog.get_logger(__name__)

CACHE_TTL_SECONDS = 3600


def build_cache_key(system_prompt: str, user_prompt: str) -> str:
    # sha256 en vez de guardar el texto crudo como clave: tamaño fijo y evita problemas de encoding/longitud
    # Ambos prompts ya incluyen project_type, detail_level y output_format: cualquier cambio invalida la clave
    raw = f"{system_prompt}::{user_prompt}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"estimation:{digest}"


def get_cached_estimation(cache_key: str) -> str | None:
    try:
        return redis_client.get(cache_key)
    except redis.RedisError as exc:
        logger.warning("cache_unavailable", operation="get", error=str(exc))
        return None


def set_cached_estimation(cache_key: str, estimation: str) -> None:
    try:
        redis_client.set(cache_key, estimation, ex=CACHE_TTL_SECONDS)
    except redis.RedisError as exc:
        logger.warning("cache_unavailable", operation="set", error=str(exc))