"""Idempotency cache for the transcript-estimate endpoint.

A client may safely retry ``POST /v1/estimate/from-transcript`` with the same
``idempotency_key`` and get the original result back without paying for the LLM pipeline again.
Two backends behind one contract (``get`` / ``set`` / ``delete``):

* Redis when it is reachable: survives restarts, shared across workers, native TTL.
* An in-process dict otherwise: thread-safe, manual expiry, so the feature also works in tests
  and single-process dev without Redis.

Redis failures at runtime never break a request: a read error is a cache miss and a write error
only loses the replay protection.
"""

import threading
import time
from typing import Any

import redis
import structlog

from app.config import Settings
from app.generation.rag.schemas import Estimate

log = structlog.get_logger()

_KEY_PREFIX = "idempotency:estimate:"
# A hung Redis must not stall the request: fail fast and fall back
_REDIS_TIMEOUT_SECONDS = 2


class IdempotencyStore:
    """Store ``Estimate`` results keyed by the client idempotency key."""

    def __init__(self, redis_client: Any | None = None, ttl: int = 86400) -> None:
        self._redis = redis_client
        self._ttl = ttl
        self._mem: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "IdempotencyStore":
        """Build the store, preferring Redis and degrading to the dict fallback."""
        redis_client = None
        if settings.REDIS_URL:
            try:
                redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
                    socket_timeout=_REDIS_TIMEOUT_SECONDS,
                )
                redis_client.ping()
            except Exception as exc:  # noqa: BLE001 - fall back to in-memory
                log.warning(
                    "idempotency_redis_unavailable",
                    reason="setup_failed",
                    error_type=type(exc).__name__,
                )
                redis_client = None
        return cls(redis_client=redis_client, ttl=settings.IDEMPOTENCY_TTL)

    def get(self, key: str) -> Estimate | None:
        """Return the cached estimate for ``key``, or None on a miss."""
        raw = self._get_raw(key)
        if raw is None:
            return None
        try:
            return Estimate.model_validate_json(raw)
        except ValueError:  # a corrupt entry is a miss
            self.delete(key)
            return None

    def set(self, key: str, estimate: Estimate) -> None:
        """Cache ``estimate`` under ``key`` with the configured TTL."""
        raw = estimate.model_dump_json()
        if self._redis is not None:
            try:
                self._redis.set(_KEY_PREFIX + key, raw, ex=self._ttl)
            except redis.RedisError as exc:
                log.warning("idempotency_redis_write_failed", error_type=type(exc).__name__)
            return
        with self._lock:
            self._mem[key] = (raw, time.monotonic() + self._ttl)

    def delete(self, key: str) -> None:
        """Remove ``key`` from the store (no-op if absent)."""
        if self._redis is not None:
            try:
                self._redis.delete(_KEY_PREFIX + key)
            except redis.RedisError as exc:
                log.warning("idempotency_redis_write_failed", error_type=type(exc).__name__)
            return
        with self._lock:
            self._mem.pop(key, None)

    def _get_raw(self, key: str) -> str | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(_KEY_PREFIX + key)
            except redis.RedisError as exc:
                log.warning("idempotency_redis_read_failed", error_type=type(exc).__name__)
                return None
            return raw  # type: ignore[no-any-return]
        with self._lock:
            entry = self._mem.get(key)
            if entry is None:
                return None
            raw, expires_at = entry
            if time.monotonic() >= expires_at:
                self._mem.pop(key, None)
                return None
            return raw
