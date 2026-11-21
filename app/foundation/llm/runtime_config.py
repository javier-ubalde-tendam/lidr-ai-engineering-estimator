"""Runtime-mutable model and retrieval-stage configuration, backed by Redis.

``Settings`` (.env) stays the immutable layer of *defaults*; this module holds runtime
*overrides* as deltas in Redis hashes, so models and stages can be switched mid-session
without editing .env or recreating containers. Overrides survive ``--reload`` restarts and
are shared across workers (Redis is the source of truth, read on every call).

Failure semantics:
- Reads degrade gracefully: if Redis is down, ``effective()`` falls back to the settings
  default (the service behaves exactly as configured by .env).
- Writes raise ``RuntimeConfigUnavailable``: a failed override must be visible to the
  caller (the API maps it to a 503).
"""

import redis
import structlog

from app.config import Settings

log = structlog.get_logger()

# Model knobs that can be overridden at runtime; each name is also a ``Settings`` field (the
# default). Only the knobs the RAG pipeline actually resolves through this store are listed:
# the aux-model call sites that predate it (primary/fallback, critic, metadata extractor,
# compression) stay .env-only. EMBEDDING_MODEL is deliberately absent: changing it would
# invalidate every stored vector.
MODEL_KEYS: tuple[str, ...] = (
    "REFORMULATION_MODEL",
    "GENERATION_MODEL",
    "ROUTER_MODEL",
    "QUERY_TRANSFORM_MODEL",
    "HALLUCINATION_JUDGE_MODEL",
    "AUGMENTATION_MODEL",
    "PROPOSITIONAL_CHUNKER_MODEL",
    "CONTEXTUAL_CHUNKER_MODEL",
)

HASH_KEY = "estimator:runtime_config"

# Short timeouts: a hung Redis must not stall every request that reads a runtime override
_REDIS_TIMEOUTS = {"socket_timeout": 2, "socket_connect_timeout": 2}


class RuntimeConfigUnavailable(Exception):
    """Raised when an override WRITE fails (Redis unreachable)."""


class RuntimeModelConfig:
    """Redis-hash-backed override store for the LLM model knobs."""

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> "RuntimeModelConfig":
        return cls(redis.from_url(url, decode_responses=True, **_REDIS_TIMEOUTS), settings)

    def get(self, key: str) -> str | None:
        """Return the override for ``key``, or ``None`` (unset / Redis down)."""
        self._validate_key(key)
        try:
            return self._redis.hget(HASH_KEY, key)
        except redis.RedisError as exc:
            log.warning("runtime_config_read_failed", key=key, error=str(exc)[:200])
            return None

    def set(self, key: str, value: str | None) -> None:
        """Set the override for ``key``; ``None`` clears it (back to default)."""
        self._validate_key(key)
        try:
            if value is None:
                self._redis.hdel(HASH_KEY, key)
            else:
                self._redis.hset(HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def effective(self, key: str) -> str:
        """The value the pipeline should use: override if set, else the default."""
        return self.get(key) or self.default(key)

    def default(self, key: str) -> str:
        self._validate_key(key)
        return getattr(self._settings, key)

    def is_overridden(self, key: str) -> bool:
        return self.get(key) is not None

    def snapshot(self) -> dict[str, dict[str, str | bool]]:
        """One entry per model key: ``{effective, default, overridden}``."""
        try:
            overrides = self._redis.hgetall(HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_config_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        return {
            key: {
                "effective": overrides.get(key) or self.default(key),
                "default": self.default(key),
                "overridden": key in overrides,
            }
            for key in MODEL_KEYS
        }

    def reset_all(self) -> None:
        """Drop every override (back to .env defaults)."""
        try:
            self._redis.delete(HASH_KEY)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    @staticmethod
    def _validate_key(key: str) -> None:
        if key not in MODEL_KEYS:
            raise ValueError(f"Unknown model key: {key}")


# --- Retrieval / generation-quality stage toggles ----------------------------
# Same override pattern as the model knobs, in a SEPARATE hash so resetting the model
# config does not clear the toggles and vice versa. The settings fields are the defaults.
RETRIEVAL_HASH_KEY = "estimator:runtime_retrieval"

SEARCH_MODE_KEY = "RETRIEVAL_SEARCH_MODE"
RERANK_KEY = "RERANKER_ENABLED"
ROUTING_KEY = "RETRIEVAL_ROUTING_ENABLED"
QUERY_TRANSFORM_KEY = "QUERY_TRANSFORM_ENABLED"
TEMPORAL_DECAY_KEY = "TEMPORAL_DECAY_ENABLED"
# Per-task hours knobs (numeric, not toggles): the distance threshold is the red-flag floor that
# gets calibrated against the corpus
TASK_HOURS_TOP_K_KEY = "TASK_HOURS_TOP_K"
TASK_HOURS_DISTANCE_THRESHOLD_KEY = "TASK_HOURS_DISTANCE_THRESHOLD"
HALLUCINATION_GATE_KEY = "HALLUCINATION_GATE_ENABLED"
AUGMENTATION_KEY = "AUGMENTATION_ENABLED"
SYNTHESIS_KEY = "SYNTHESIS_ENABLED"

# The boolean stage toggles (everything in the hash except the search mode)
BOOL_TOGGLE_KEYS: tuple[str, ...] = (
    RERANK_KEY,
    ROUTING_KEY,
    QUERY_TRANSFORM_KEY,
    TEMPORAL_DECAY_KEY,
    HALLUCINATION_GATE_KEY,
    AUGMENTATION_KEY,
    SYNTHESIS_KEY,
)
RETRIEVAL_KEYS: tuple[str, ...] = (
    SEARCH_MODE_KEY,
    *BOOL_TOGGLE_KEYS,
    TASK_HOURS_TOP_K_KEY,
    TASK_HOURS_DISTANCE_THRESHOLD_KEY,
)

_VALID_SEARCH_MODES = ("vector", "hybrid")


class RuntimeRetrievalConfig:
    """Redis-hash-backed override store for the retrieval / generation-quality toggles.

    Reads degrade to the settings default if Redis is down (the pipeline keeps serving
    exactly as .env configured it); writes raise ``RuntimeConfigUnavailable``.
    """

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> "RuntimeRetrievalConfig":
        return cls(redis.from_url(url, decode_responses=True, **_REDIS_TIMEOUTS), settings)

    def _get_raw(self, key: str) -> str | None:
        try:
            return self._redis.hget(RETRIEVAL_HASH_KEY, key)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key=key, error=str(exc)[:200])
            return None

    def _set_raw(self, key: str, value: str | None) -> None:
        try:
            if value is None:
                self._redis.hdel(RETRIEVAL_HASH_KEY, key)
            else:
                self._redis.hset(RETRIEVAL_HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def effective_search_mode(self) -> str:
        override = self._get_raw(SEARCH_MODE_KEY)
        if override in _VALID_SEARCH_MODES:
            return override
        return self._settings.RETRIEVAL_SEARCH_MODE

    def _effective_bool(self, key: str, default: bool) -> bool:
        override = self._get_raw(key)
        if override is None:
            return default
        return override.lower() == "true"

    def effective_rerank(self) -> bool:
        return self._effective_bool(RERANK_KEY, self._settings.RERANKER_ENABLED)

    def effective_routing(self) -> bool:
        return self._effective_bool(ROUTING_KEY, self._settings.RETRIEVAL_ROUTING_ENABLED)

    def effective_query_transform(self) -> bool:
        return self._effective_bool(QUERY_TRANSFORM_KEY, self._settings.QUERY_TRANSFORM_ENABLED)

    def effective_temporal_decay(self) -> bool:
        return self._effective_bool(TEMPORAL_DECAY_KEY, self._settings.TEMPORAL_DECAY_ENABLED)

    def effective_hallucination_gate(self) -> bool:
        return self._effective_bool(
            HALLUCINATION_GATE_KEY, self._settings.HALLUCINATION_GATE_ENABLED
        )

    def effective_augmentation(self) -> bool:
        return self._effective_bool(AUGMENTATION_KEY, self._settings.AUGMENTATION_ENABLED)

    def effective_synthesis(self) -> bool:
        return self._effective_bool(SYNTHESIS_KEY, self._settings.SYNTHESIS_ENABLED)

    def effective_task_hours_top_k(self) -> int:
        override = self._get_raw(TASK_HOURS_TOP_K_KEY)
        if override is None:
            return self._settings.TASK_HOURS_TOP_K
        try:
            return int(override)
        except ValueError:
            return self._settings.TASK_HOURS_TOP_K

    def effective_task_hours_distance_threshold(self) -> float:
        override = self._get_raw(TASK_HOURS_DISTANCE_THRESHOLD_KEY)
        if override is None:
            return self._settings.TASK_HOURS_DISTANCE_THRESHOLD
        try:
            return float(override)
        except ValueError:
            return self._settings.TASK_HOURS_DISTANCE_THRESHOLD

    def set_task_hours_top_k(self, value: int | None) -> None:
        if value is not None and value < 1:
            raise ValueError("TASK_HOURS_TOP_K must be >= 1")
        self._set_raw(TASK_HOURS_TOP_K_KEY, None if value is None else str(int(value)))

    def set_task_hours_distance_threshold(self, value: float | None) -> None:
        if value is not None and not (0.0 <= value <= 2.0):
            raise ValueError("TASK_HOURS_DISTANCE_THRESHOLD must be in [0, 2]")
        self._set_raw(
            TASK_HOURS_DISTANCE_THRESHOLD_KEY, None if value is None else str(float(value))
        )

    def set_search_mode(self, value: str | None) -> None:
        if value is not None and value not in _VALID_SEARCH_MODES:
            raise ValueError(f"Invalid search mode: {value}")
        self._set_raw(SEARCH_MODE_KEY, value)

    def set_rerank(self, value: bool | None) -> None:
        self._set_raw(RERANK_KEY, None if value is None else str(value).lower())

    def set_bool(self, key: str, value: bool | None) -> None:
        """Set one boolean stage toggle (routing, query transform, decay, gate, ...)."""
        if key not in BOOL_TOGGLE_KEYS:
            raise ValueError(f"Unknown retrieval toggle: {key}")
        self._set_raw(key, None if value is None else str(value).lower())

    def snapshot(self) -> dict[str, dict[str, object]]:
        """``{effective, default, overridden}`` per toggle (for the UI)."""
        try:
            overrides = self._redis.hgetall(RETRIEVAL_HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        effective_and_default = {
            SEARCH_MODE_KEY: (self.effective_search_mode(), self._settings.RETRIEVAL_SEARCH_MODE),
            RERANK_KEY: (self.effective_rerank(), self._settings.RERANKER_ENABLED),
            ROUTING_KEY: (self.effective_routing(), self._settings.RETRIEVAL_ROUTING_ENABLED),
            QUERY_TRANSFORM_KEY: (
                self.effective_query_transform(),
                self._settings.QUERY_TRANSFORM_ENABLED,
            ),
            TEMPORAL_DECAY_KEY: (
                self.effective_temporal_decay(),
                self._settings.TEMPORAL_DECAY_ENABLED,
            ),
            HALLUCINATION_GATE_KEY: (
                self.effective_hallucination_gate(),
                self._settings.HALLUCINATION_GATE_ENABLED,
            ),
            AUGMENTATION_KEY: (self.effective_augmentation(), self._settings.AUGMENTATION_ENABLED),
            SYNTHESIS_KEY: (self.effective_synthesis(), self._settings.SYNTHESIS_ENABLED),
            TASK_HOURS_TOP_K_KEY: (
                self.effective_task_hours_top_k(),
                self._settings.TASK_HOURS_TOP_K,
            ),
            TASK_HOURS_DISTANCE_THRESHOLD_KEY: (
                self.effective_task_hours_distance_threshold(),
                self._settings.TASK_HOURS_DISTANCE_THRESHOLD,
            ),
        }
        return {
            key: {"effective": effective, "default": default, "overridden": key in overrides}
            for key, (effective, default) in effective_and_default.items()
        }
