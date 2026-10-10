"""Wrapper observable para las llamadas estructuradas al LLM (Instructor sobre LiteLLM).

Cada llamada devuelve un `LLMCallMeta` (latencia, tokens, coste, modelo, proveedor) y emite
siempre `llm_call_completed` / `llm_call_failed` con esos mismos seis campos, de modo que el
coste y la latencia de cualquier parte del sistema se pueden agregar desde los logs.
"""

import time
from collections.abc import Iterable
from typing import Any, TypeVar

import instructor
import litellm
import structlog
from instructor.core.exceptions import InstructorRetryException
from litellm import APIError, AuthenticationError, RateLimitError, Timeout
from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings

logger = structlog.get_logger(__name__)

# USD por 1M de tokens. Los precios cambian: verificarlos periódicamente en las páginas oficiales
# de precios de OpenAI y Anthropic.
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
}

T = TypeVar("T", bound=BaseModel)

# Envuelve litellm.completion: añade response_model y reintentos con el error de validación.
# Es el único punto de entrada al LLM no-streaming (los tests lo sustituyen aquí).
_structured_client = instructor.from_litellm(litellm.completion)


def normalise_model_name(model: str) -> str:
    # LiteLLM exige prefijos como "anthropic/" para enrutar; la tabla de precios usa el nombre pelado
    return model.rpartition("/")[2]


def provider_from_model(model: str) -> str:
    name = normalise_model_name(model).lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "unknown"


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    pricing = MODEL_COSTS.get(normalise_model_name(model))
    if pricing is None:
        # 0.0 en vez de fallar: una tabla de precios desactualizada no debe tumbar una petición
        logger.warning("model_cost_unknown", model=model)
        return 0.0
    return round((tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1_000_000, 6)


class LLMCallMeta(BaseModel):
    latency_ms: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    model: str
    provider: str


def sum_call_metas(metas: Iterable[LLMCallMeta]) -> LLMCallMeta:
    """Agrega las llamadas de una misma petición (p. ej. actor + critic del ACB)."""
    metas = list(metas)
    # Si las llamadas usan modelos distintos se listan todos, en orden de aparición y sin repetir
    models = list(dict.fromkeys(m.model for m in metas))
    providers = list(dict.fromkeys(m.provider for m in metas))
    return LLMCallMeta(
        latency_ms=sum(m.latency_ms for m in metas),
        tokens_in=sum(m.tokens_in for m in metas),
        tokens_out=sum(m.tokens_out for m in metas),
        cost_usd=round(sum(m.cost_usd for m in metas), 6),
        model="+".join(models) or "none",
        provider="+".join(providers) or "unknown",
    )


def _meta(model: str, started_at: float, tokens_in: int, tokens_out: int) -> LLMCallMeta:
    return LLMCallMeta(
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=estimate_cost_usd(model, tokens_in, tokens_out),
        model=normalise_model_name(model),
        provider=provider_from_model(model),
    )


def _tokens_from_usage(usage: Any) -> tuple[int, int]:
    # LiteLLM puede no traer `usage`; con reintentos Instructor acumula los tokens de todos los intentos
    return getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0


def build_call_meta(completion: Any, model: str, started_at: float) -> LLMCallMeta:
    """`started_at` es un `time.perf_counter()` tomado justo antes de la llamada."""
    tokens_in, tokens_out = _tokens_from_usage(getattr(completion, "usage", None))
    # Se usa el modelo pedido, no completion.model: éste puede ser un snapshot fechado fuera de la tabla
    return _meta(model, started_at, tokens_in, tokens_out)


def error_type(error: BaseException | None) -> str:
    # Instructor envuelve el error original en InstructorRetryException: se clasifica por su causa
    if isinstance(error, AuthenticationError):
        return "auth"
    if isinstance(error, RateLimitError):
        return "rate_limit"
    if isinstance(error, Timeout):
        return "timeout"
    if isinstance(error, APIError):
        return "server_error"
    if isinstance(error, ValidationError):
        return "invalid_output"
    return "unknown"


def _api_key_for_model(model: str) -> str | None:
    settings = get_settings()
    return settings.ANTHROPIC_API_KEY if provider_from_model(model) == "anthropic" else settings.OPENAI_API_KEY


def _finish_reason(completion: Any) -> str | None:
    choices = getattr(completion, "choices", None)
    return getattr(choices[0], "finish_reason", None) if choices else None


def call_structured(
    messages: list[dict[str, Any]],
    response_model: type[T],
    model: str,
    *,
    max_retries: int,
    **log_context: Any,
) -> tuple[T, LLMCallMeta]:
    """Llamada estructurada con Instructor; `log_context` (purpose, session_id...) viaja a los logs.

    Si falla registra `llm_call_failed` y relanza la excepción original: cada llamador decide
    su política de fallo (error de negocio, fail-open, valor por defecto...).
    """
    started_at = time.perf_counter()
    try:
        result, completion = _structured_client.create_with_completion(
            model=model,
            api_key=_api_key_for_model(model),
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
        )
    except Exception as exc:
        cause = exc.__cause__ if isinstance(exc, InstructorRetryException) else exc
        tokens_in, tokens_out = _tokens_from_usage(getattr(exc, "total_usage", None))
        meta = _meta(model, started_at, tokens_in, tokens_out)
        logger.error(
            "llm_call_failed",
            **meta.model_dump(),
            error_type=error_type(cause),
            error=str(cause),
            attempts=getattr(exc, "n_attempts", 1),
            **log_context,
        )
        raise

    meta = build_call_meta(completion, model, started_at)
    logger.info("llm_call_completed", **meta.model_dump(), finish_reason=_finish_reason(completion), **log_context)
    return result, meta
