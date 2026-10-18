import time
from collections.abc import Callable

import structlog
from openai import OpenAI, RateLimitError

from app.config import get_settings
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

logger = structlog.get_logger(__name__)

BATCH_SIZE = 100
# Esperas entre reintentos ante RateLimitError (backoff exponencial): 1s, 2s, 4s
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)

# Precio tomado del enunciado del ejercicio; cambia con el tiempo, verificar en la página
# oficial de precios de OpenAI.
PRICE_PER_MILLION_TOKENS_USD = 0.02


def estimate_cost_usd(total_tokens: int) -> float:
    return total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD


class OpenAIEmbedder:
    def __init__(
        self,
        client: OpenAI,
        model: str | None = None,
        # Función de espera inyectable: en Java sería una interfaz Sleeper; en Python basta
        # con pasar un callable (los tests pasan uno falso y no esperan de verdad).
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # El cliente llega por inyección: el embedder no sabe cómo se construye (desacoplado)
        self._client = client
        self._model = model or get_settings().EMBEDDING_MODEL
        self._sleep = sleep

    def embed_one(self, text: str) -> list[float]:
        return self._create([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embedded: list[EmbeddedChunk] = []
        for batch_index, start in enumerate(range(0, len(chunks), BATCH_SIZE)):
            batch = chunks[start : start + BATCH_SIZE]
            started = time.perf_counter()
            vectors = self._create([chunk.text for chunk in batch])
            logger.info(
                "embedding_batch_processed",
                batch_index=batch_index,
                chunks=len(batch),
                tokens=sum(chunk.token_count for chunk in batch),
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
            # La API devuelve los vectores en el orden del input, así que zip conserva el orden
            embedded.extend(
                EmbeddedChunk(**chunk.model_dump(), embedding=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            )
        return embedded

    def _create(self, texts: list[str]) -> list[list[float]]:
        # No se pasa `dimensions`: se usa la dimensión por defecto del modelo (1536)
        for attempt, delay in enumerate(RETRY_DELAYS_SECONDS, start=1):
            try:
                return self._call_api(texts)
            except RateLimitError as exc:
                logger.warning(
                    "embedding_rate_limited", attempt=attempt, retry_in_s=delay, error=str(exc)[:200]
                )
                self._sleep(delay)
        # Último intento tras agotar las esperas: si falla, el error se propaga al llamador
        return self._call_api(texts)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
