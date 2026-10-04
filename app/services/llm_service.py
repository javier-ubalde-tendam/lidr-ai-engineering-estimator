import time
from collections.abc import Iterator
from typing import Any, Literal

import instructor
import litellm
import structlog
from instructor.core.exceptions import InstructorRetryException
from litellm import APIError, AuthenticationError, RateLimitError, Timeout
from pydantic import ValidationError

from app.cache import build_cache_key, get_cached_estimation, set_cached_estimation
from app.config import estimate_cost_usd, get_settings
from app.dependencies import get_openai_client
from app.guardrails.input import check_input
from app.guardrails.output import enforce_scope_response
from app.logging_config import TRACE
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest, EstimationResult
from app.semantic_cache import semantic_cache_lookup, semantic_cache_store

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


def _totals_warnings(result: EstimationResult) -> list[str]:
    mismatch = result.totals_mismatch()
    if mismatch is None:
        return []
    logger.warning("validation_not_passed", validation="totals_match_phases", detail=mismatch)
    return [f"totals_match_phases: {mismatch}"]


# Endpoint de FastAPI sin streaming
def estimate_project(request: EstimationRequest) -> tuple[EstimationResult, int, int, str]:
    # Nunca se llama al LLM con un input que no ha pasado los guardrails
    check_input(request.description, openai_client=get_openai_client())
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

    _totals_warnings(result)  # sin reintento por totales: solo queda registrado en el log
    result = enforce_scope_response(result)  # red de seguridad si el model_validator no disparó
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

# "partial": estado provisional (campos None, números a medias); "complete": validado con los model_validator
StreamEvent = tuple[Literal["partial", "complete"], EstimationResult]

# Se crea una sola vez (cada Partial[...] cachea una clase nueva); Any porque los stubs ocultan model_from_chunks
_PartialEstimationResult: Any = instructor.Partial[EstimationResult]
_ESTIMATION_TOOL = instructor.openai_schema(EstimationResult).openai_schema


def _tool_json_fragments(stream: Iterator[Any], provider: str, model: str, stats: dict) -> Iterator[str]:
    for chunk in stream:
        # Comprobado con logs reales: chunk.model siempre es el nombre pedido, no un snapshot fechado
        # distinto (a diferencia de lo que asumíamos antes sin verificarlo). LiteLLM no expone aquí
        # el snapshot exacto resuelto por el proveedor.
        stats["model_requested"] = _bare_model_name(model)
        stats["model_used"] = _bare_model_name(getattr(chunk, "model", model))

        if chunk.choices:
            # Con tool calling el JSON viaja en tool_calls[0].function.arguments, no en delta.content
            tool_calls = chunk.choices[0].delta.tool_calls
            if tool_calls and tool_calls[0].function.arguments:
                yield tool_calls[0].function.arguments
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


def _drain_on_error(fragments: Iterator[str], states: Iterator[EstimationResult]) -> Iterator[EstimationResult]:
    try:
        yield from states
    except Exception:
        # El chunk con usage llega después del último fragmento JSON: se consume para no perder los tokens
        for _ in fragments:
            pass
        raise


def _open_partial_stream(
    provider: str, system_prompt: str, user_prompt: str, stats: dict
) -> tuple[EstimationResult, Iterator[EstimationResult]]:
    model = _model_name_for_provider(provider)
    # create_partial de Instructor 1.17 consume el stream entero antes de devolver nada: se llama a LiteLLM directo
    response = litellm.completion(
        model=model,
        api_key=_api_key_for_provider(provider),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{"type": "function", "function": _ESTIMATION_TOOL}],
        tool_choice={"type": "function", "function": {"name": _ESTIMATION_TOOL["name"]}},
        stream=True,
        stream_options={"include_usage": True},
    )
    fragments = _tool_json_fragments(response, provider, model, stats)
    states = _drain_on_error(fragments, _PartialEstimationResult.model_from_chunks(fragments))
    # Pedir el primer estado fuerza la conexión: los fallos de proveedor saltan aquí, antes de emitir nada
    return next(states), states


def _open_stream_with_fallback(
    system_prompt: str, user_prompt: str, stats: dict
) -> tuple[EstimationResult, Iterator[EstimationResult]]:
    primary_provider = settings.LLM_PROVIDER
    for attempt in range(settings.LLM_MAX_RETRIES + 1):
        try:
            first_state, states = _open_partial_stream(primary_provider, system_prompt, user_prompt, stats)
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
        stats["provider_used"] = primary_provider
        stats["fallback_used"] = False
        return first_state, states

    fallback_provider = _other_provider(primary_provider)
    try:
        first_state, states = _open_partial_stream(fallback_provider, system_prompt, user_prompt, stats)
    except (AuthenticationError, RateLimitError, Timeout, APIError, StopIteration) as exc:
        logger.error("llm_call_failed", error_type="all_providers_failed", error=str(exc))
        raise EstimationFailedError("All providers failed") from exc

    stats["provider_used"] = fallback_provider
    stats["fallback_used"] = True
    return first_state, states


def _get_cached_result(cache_key: str) -> EstimationResult | None:
    cached = get_cached_estimation(cache_key)
    if cached is None:
        return None
    try:
        return EstimationResult.model_validate_json(cached)
    except ValidationError:
        # Entradas del streaming anterior (texto libre) o de otra versión del schema: se tratan como miss
        logger.warning("cache_entry_invalid", cache_key=cache_key)
        return None


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


def estimate_project_stream_structured(request: EstimationRequest, stats: dict) -> Iterator[StreamEvent]:
    # Igual que en la versión bloqueante: el guardrail de input nunca se salta por cache ni por streaming
    check_input(request.description, openai_client=get_openai_client())
    system_prompt, user_prompt = render_estimation_prompt(request)
    cache_key = build_cache_key(system_prompt, user_prompt)
    cached_result = _get_cached_result(cache_key)
    exact_cache_hit = cached_result is not None
    semantic_cache_hit = False

    if cached_result is None:
        # Más caro que el exact-match (llama a la API de embeddings): solo se intenta si ya hubo miss
        cached_result = semantic_cache_lookup(request, settings.PROMPT_VERSION, get_openai_client())
        semantic_cache_hit = cached_result is not None
        if semantic_cache_hit:
            # Promociona el hit semántico a exact-match: si se repite este mismo texto literal,
            # la próxima vez se sirve sin volver a llamar a la API de embeddings
            set_cached_estimation(cache_key, cached_result.model_dump_json())

    cache_hit = exact_cache_hit or semantic_cache_hit

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
    stats["exact_cache_hit"] = exact_cache_hit
    stats["semantic_cache_hit"] = semantic_cache_hit

    if cached_result is not None:
        # Mismo parser que con el LLM real: el cliente recibe los mismos estados parciales
        replay = _PartialEstimationResult.model_from_chunks(_simulate_stream_chunks(cached_result.model_dump_json()))
        for state in replay:
            yield "partial", state
        logger.info(
            "llm_call_completed",
            tokens_input=0,
            tokens_output=0,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            cost_usd=0.0,
            finish_reason="cache_hit",
        )
        yield "complete", cached_result
        return

    first_state, states = _open_stream_with_fallback(system_prompt, user_prompt, stats)
    last_state = first_state
    yield "partial", first_state
    try:
        for last_state in states:
            yield "partial", last_state
        # Instructor no valida un JSON truncado; model_dump() fuerza la revalidación (una instancia no se revalida)
        result = EstimationResult.model_validate(last_state.model_dump())
    except (ValueError, AuthenticationError, RateLimitError, Timeout, APIError) as exc:
        # ValidationError (y los errores de JSON de jiter) heredan de ValueError
        logger.warning(
            "llm_call_attempt_failed",
            attempt="stream_final_validation",
            provider=stats["provider_used"],
            error_type=_error_type(exc),
            error=str(exc),
        )
        # Política de fallo: se descartan los parciales y se reintenta con Instructor bloqueante
        stats["blocking_fallback_used"] = True
        result, tokens_input, tokens_output, model = estimate_project(request)
        stats["tokens_input"] = stats.get("tokens_input", 0) + tokens_input
        stats["tokens_output"] = stats.get("tokens_output", 0) + tokens_output
        stats["model_requested"] = model

    # Solo se cachea un resultado sin avisos, para poder reintentar y obtener uno mejor
    if warnings := _totals_warnings(result):
        stats["validation_warnings"] = warnings
    else:
        set_cached_estimation(cache_key, result.model_dump_json())
        semantic_cache_store(request, result, settings.PROMPT_VERSION, get_openai_client())
    _log_llm_call_completed(stats, start)
    yield "complete", result
