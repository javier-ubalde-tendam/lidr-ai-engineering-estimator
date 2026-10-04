import array

import structlog
from openai import OpenAIError
from redis import RedisError
from redisvl.exceptions import RedisVLError
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema

from app.config import get_settings
from app.schemas.estimation import EstimationRequest, EstimationResult

# Errores esperables de un cache semántico "caído": Redis/RediSearch no disponible, o fallo de la API de embeddings
_SOFT_FAILURE_ERRORS = (RedisVLError, RedisError, OpenAIError)

settings = get_settings()
logger = structlog.get_logger(__name__)

# dims de text-embedding-3-small; si cambias EMBEDDING_MODEL por uno con otro tamaño de
# vector (p.ej. text-embedding-3-large = 3072), actualiza esta constante y recrea el índice
EMBEDDING_DIMS = 1536

_SCHEMA = IndexSchema.from_dict(
    {
        "index": {"name": "semantic_cache", "prefix": "semantic_cache", "storage_type": "hash"},
        "fields": [
            {"name": "bucket", "type": "tag"},
            {"name": "result_json", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": EMBEDDING_DIMS,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                },
            },
        ],
    }
)
index = SearchIndex(_SCHEMA, redis_url=settings.REDIS_URL)

try:
    # A diferencia de redis.Redis.from_url (conexión perezosa), esto habla con Redis ya mismo
    # (FT.INFO): si el servidor no tiene el módulo RediSearch (redis-stack) o está caído, no
    # debe tumbar la app entera al importar este módulo — el cache semántico simplemente no
    # funcionará (cada lookup/store fallará también en soft, más abajo)
    index.create()
except (RedisVLError, RedisError) as exc:
    logger.warning("semantic_cache_index_unavailable", error_type=type(exc).__name__, error=str(exc))


def _to_bytes(embedding: list[float]) -> bytes:
    # Redis guarda el vector como hash field: hace falta empaquetarlo como float32 crudo
    return array.array("f", embedding).tobytes()


def _index_missing(exc: Exception) -> bool:
    # Pasa tras un reinicio del contenedor Redis: los datos sobreviven (RDB) pero el índice no
    return "no such index" in str(exc).lower()


def _run_with_index_recovery(operation):
    """Ejecuta operation(); si falla porque el índice no existe, lo recrea una vez y reintenta."""
    try:
        return operation()
    except _SOFT_FAILURE_ERRORS as exc:
        if not _index_missing(exc):
            raise
        logger.warning("semantic_cache_index_missing_retrying", error=str(exc))
        try:
            index.create()
        except _SOFT_FAILURE_ERRORS as create_exc:
            raise exc from create_exc
        return operation()


def build_semantic_bucket(request: EstimationRequest, prompt_version: str) -> str:
    # output_format no cambia hoy el prompt generado en v2 (solo la presentación en Streamlit),
    # pero se incluye igualmente por si una futura versión del prompt sí varía con él
    return (
        f"{prompt_version}:{request.project_type.value}:"
        f"{request.detail_level.value}:{request.output_format.value}"
    )


def _embed(description: str, openai_client) -> list[float]:
    response = openai_client.embeddings.create(model=settings.EMBEDDING_MODEL, input=description)
    return response.data[0].embedding


def semantic_cache_lookup(
    request: EstimationRequest, prompt_version: str, openai_client
) -> EstimationResult | None:
    bucket = build_semantic_bucket(request, prompt_version)
    try:
        embedding = _embed(request.description, openai_client)
    except _SOFT_FAILURE_ERRORS as exc:
        # Fail soft: un cache semántico caído nunca debe impedir la estimación, solo degradarla a miss
        logger.warning("semantic_cache_lookup_failed", error_type=type(exc).__name__, error=str(exc))
        return None

    query = VectorQuery(
        vector=_to_bytes(embedding),
        vector_field_name="embedding",
        return_fields=["result_json"],
        filter_expression=Tag("bucket") == bucket,
        num_results=1,
    )
    try:
        results = _run_with_index_recovery(lambda: index.query(query))
    except _SOFT_FAILURE_ERRORS as exc:
        logger.warning("semantic_cache_lookup_failed", error_type=type(exc).__name__, error=str(exc))
        return None

    if not results:
        logger.info("semantic_cache_miss", bucket=bucket, reason="empty_bucket")
        return None

    # Redis define la distancia coseno como 1 - similaridad_coseno
    similarity = 1.0 - float(results[0]["vector_distance"])
    if similarity < settings.SEMANTIC_CACHE_THRESHOLD:
        logger.info("semantic_cache_miss", bucket=bucket, similarity=round(similarity, 4))
        return None

    if settings.SEMANTIC_CACHE_LOG_ONLY:
        # Modo de despliegue seguro: se mide el hit potencial pero nunca se sirve ni se salta el LLM
        logger.info("semantic_cache_hit_log_only", bucket=bucket, similarity=round(similarity, 4))
        return None

    logger.info("semantic_cache_hit", bucket=bucket, similarity=round(similarity, 4))
    return EstimationResult.model_validate_json(results[0]["result_json"])


def semantic_cache_store(
    request: EstimationRequest, result: EstimationResult, prompt_version: str, openai_client
) -> None:
    bucket = build_semantic_bucket(request, prompt_version)
    try:
        embedding = _embed(request.description, openai_client)
    except _SOFT_FAILURE_ERRORS as exc:
        logger.warning("semantic_cache_store_failed", error_type=type(exc).__name__, error=str(exc))
        return

    doc = {"bucket": bucket, "result_json": result.model_dump_json(), "embedding": _to_bytes(embedding)}
    try:
        _run_with_index_recovery(lambda: index.load([doc], ttl=settings.SEMANTIC_CACHE_TTL))
    except _SOFT_FAILURE_ERRORS as exc:
        logger.warning("semantic_cache_store_failed", error_type=type(exc).__name__, error=str(exc))
        return
    logger.info("semantic_cache_stored", bucket=bucket)
