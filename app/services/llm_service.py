import itertools
import time

import instructor
import litellm
import structlog
from instructor.core.exceptions import InstructorRetryException
from litellm import APIError, AuthenticationError, RateLimitError, Timeout
from pydantic import ValidationError

from app.cache import build_cache_key, get_cached_estimation, set_cached_estimation
from app.config import estimate_cost_usd, get_settings
from app.logging_config import TRACE
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest, EstimationResult
from collections.abc import Iterator

settings = get_settings()
logger = structlog.get_logger(__name__)

# Envuelve litellm.completion: añade response_model y reintentos con el error de validación
_structured_client = instructor.from_litellm(litellm.completion)


class EstimationFailedError(Exception):
    """El LLM no devolvió una estimación válida (o el proveedor falló) tras los reintentos."""


def _model_name_for_provider(provider: str) -> str:
    if provider == "anthropic":
        return f"anthropic/{settings.ANTHROPIC_MODEL}"
    return settings.OPENAI_MODEL


def _api_key_for_provider(provider: str) -> str | None:
    return settings.ANTHROPIC_API_KEY if provider == "anthropic" else settings.OPENAI_API_KEY


def _other_provider(provider: str) -> str:
    return "anthropic" if provider == "openai" else "openai"


def _bare_model_name(model: str) -> str:
    # Quita el prefijo "anthropic/" que exige LiteLLM para enrutar, para que coincida con la tabla de precios
    return model.removeprefix("anthropic/")


def _shorten(text: str, max_chars: int = 25) -> str:
    # Solo añade "..." si realmente se recorta, para no sugerir texto omitido que no existe
    return text if len(text) <= max_chars else f"{text[:max_chars]}..."


def _error_type(error: BaseException | None) -> str:
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


# Endpoint de FastAPI sin streaming
def estimate_project(request: EstimationRequest) -> tuple[EstimationResult, int, int, str]:
    system_prompt, user_prompt = render_estimation_prompt(request)
    model = _model_name_for_provider(settings.LLM_PROVIDER)
    tokens_input_estimated = len(system_prompt + user_prompt) // 4
    logger.info(
        "llm_call_started",
        model_requested=model,
        provider=settings.LLM_PROVIDER,
        tokens_input_estimated=tokens_input_estimated,
        detail_level=request.detail_level,
        output_format=request.output_format,
        description=_shorten(request.description),
        prompt_version=settings.PROMPT_VERSION,
    )

    start = time.perf_counter()
    try:
        result, completion = _structured_client.create_with_completion(
            model=model,
            api_key=_api_key_for_provider(settings.LLM_PROVIDER),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=EstimationResult,
            max_retries=settings.STRUCTURED_OUTPUT_MAX_RETRIES,
        )
    except InstructorRetryException as exc:
        logger.error(
            "llm_call_failed",
            error_type=_error_type(exc.__cause__),
            error=str(exc.__cause__),
            attempts=exc.n_attempts,
        )
        raise EstimationFailedError("No valid estimation after retries") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    usage = completion.usage  # con reintentos Instructor acumula los tokens de todos los intentos
    tokens_output = usage.completion_tokens if usage else 0
    tokens_input = usage.prompt_tokens if usage else 0

    logger.info(
        "llm_call_completed",
        tokens_output=tokens_output,
        latency_ms=round(latency_ms, 2),
        cost_usd=estimate_cost_usd(_bare_model_name(model), tokens_input, tokens_output),
        finish_reason=completion.choices[0].finish_reason,
    )

    return (result, tokens_input, tokens_output, _bare_model_name(model))

# Versión con streaming, para la UI de Streamlit. Es un generador (yield en vez de return): la función no ejecuta nada hasta que alguien empieza a iterarla
def _simulate_stream_chunks(text: str, chunk_size: int = 20) -> Iterator[str]:
    # Trocea el texto ya cacheado para reproducir visualmente el mismo efecto que un streaming real
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
        time.sleep(0.01)

def _call_llm_stream(provider: str, system_prompt: str, user_prompt: str, stats: dict) -> Iterator[str]:
    model = _model_name_for_provider(provider)
    response = litellm.completion(
        model=model,
        api_key=_api_key_for_provider(provider),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in response:
        # Comprobado con logs reales: chunk.model siempre es el nombre pedido, no un snapshot fechado
        # distinto (a diferencia de lo que asumíamos antes sin verificarlo). LiteLLM no expone aquí
        # el snapshot exacto resuelto por el proveedor.
        stats["model_requested"] = _bare_model_name(model)
        stats["model_used"] = _bare_model_name(getattr(chunk, "model", model))

        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason:
                stats["finish_reason"] = finish_reason

        # A diferencia del SDK de OpenAI, el chunk de LiteLLM puede no tener "usage" en absoluto
        usage = getattr(chunk, "usage", None)
        if usage:
            stats["tokens_input"] = usage.prompt_tokens
            stats["tokens_output"] = usage.completion_tokens
        # Nivel TRACE (por debajo de DEBUG): oculto salvo que bajes LOG_LEVEL por debajo de DEBUG
        logger.log(TRACE, "raw_chunk", provider=provider, chunk=repr(chunk))


def _log_llm_call_completed(stats: dict, start: float) -> None:
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "llm_call_completed",
        tokens_input=stats.get("tokens_input", 0),
        tokens_output=stats.get("tokens_output", 0),
        latency_ms=round(latency_ms, 2),
        cost_usd=estimate_cost_usd(
            stats.get("model_requested", ""),
            stats.get("tokens_input", 0),
            stats.get("tokens_output", 0),
        ),
        finish_reason=stats.get("finish_reason"),
        provider_used=stats.get("provider_used"),
        fallback_used=stats.get("fallback_used", False),
    )


def _consume_and_cache(stream: Iterator[str], stats: dict, cache_key: str, start: float) -> Iterator[str]:
    full_response_parts: list[str] = []
    for chunk in stream:
        full_response_parts.append(chunk)
        yield chunk
    set_cached_estimation(cache_key, "".join(full_response_parts))
    _log_llm_call_completed(stats, start)


def estimate_project_stream(request: EstimationRequest, stats: dict) -> Iterator[str]:
    system_prompt, user_prompt = render_estimation_prompt(request)
    cache_key = build_cache_key(system_prompt, user_prompt)
    cached_estimation = get_cached_estimation(cache_key)
    cache_hit = cached_estimation is not None

    tokens_input_estimated = len(system_prompt + user_prompt) // 4
    logger.info(
        "llm_call_started",
        provider_requested=settings.LLM_PROVIDER,
        tokens_input_estimated=tokens_input_estimated,
        cache_hit=cache_hit,
        detail_level=request.detail_level,
        output_format=request.output_format,
        description=_shorten(request.description),
        prompt_version=settings.PROMPT_VERSION,
    )

    start = time.perf_counter()
    stats["cache_hit"] = cache_hit

    if cache_hit:
        yield from _simulate_stream_chunks(cached_estimation)
        logger.info(
            "llm_call_completed",
            tokens_input=0,
            tokens_output=0,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            cost_usd=0.0,
            finish_reason="cache_hit",
        )
        return

    primary_provider = settings.LLM_PROVIDER
    fallback_provider = _other_provider(primary_provider)

    for attempt in range(settings.LLM_MAX_RETRIES + 1):
        try:
            stream = _call_llm_stream(primary_provider, system_prompt, user_prompt, stats)
            first_chunk = next(stream)
        except AuthenticationError as exc:
            logger.warning(
                "llm_call_attempt_failed", attempt="primary_auth", provider=primary_provider, error=str(exc)
            )
            break
        except (RateLimitError, Timeout, APIError, StopIteration) as exc:
            logger.warning(
                "llm_call_attempt_failed",
                attempt=f"primary_retry_{attempt}",
                provider=primary_provider,
                error=str(exc),
            )
            continue
        else:
            stats["provider_used"] = primary_provider
            stats["fallback_used"] = False
            yield from _consume_and_cache(itertools.chain([first_chunk], stream), stats, cache_key, start)
            return

    try:
        stream = _call_llm_stream(fallback_provider, system_prompt, user_prompt, stats)
        first_chunk = next(stream)
    except (AuthenticationError, RateLimitError, Timeout, APIError, StopIteration) as exc:
        logger.error("llm_call_failed", error_type="all_providers_failed", error=str(exc))
        raise RuntimeError("Todos los proveedores fallaron") from exc

    stats["provider_used"] = fallback_provider
    stats["fallback_used"] = True
    yield from _consume_and_cache(itertools.chain([first_chunk], stream), stats, cache_key, start)