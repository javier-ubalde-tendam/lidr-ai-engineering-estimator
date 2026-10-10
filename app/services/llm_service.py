import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import instructor
import litellm
import structlog
from instructor.core.exceptions import InstructorRetryException
from litellm import APIError, AuthenticationError, RateLimitError, Timeout
from pydantic import ValidationError

from app.attachments.extractor import enrich_transcript
from app.cache import build_cache_key, get_cached_estimation, set_cached_estimation
from app.config import get_settings
from app.dependencies import get_openai_client
from app.guardrails.input import check_input
from app.guardrails.output import enforce_scope_response
from app.logging_config import TRACE
from app.prompts.loader import (
    render_conversational_prompt,
    render_conversational_user,
    render_estimation_prompt,
)
from app.schemas.acb import BossTrace
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    ProjectType,
)
from app.semantic_cache import semantic_cache_lookup, semantic_cache_store
from app.services import critic as critic_service
from app.services.boss import Boss
from app.services.llm_wrapper import (
    LLMCallMeta,
    call_structured,
    error_type,
    estimate_cost_usd,
    normalise_model_name,
    provider_from_model,
    sum_call_metas,
)
from app.sessions.compression import apply_compression
from app.sessions.metadata_extractor import update_metadata
from app.sessions.models import Session
from app.sessions.tier_resolver import Tier, resolve_tier

settings = get_settings()
logger = structlog.get_logger(__name__)


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


def _shorten(text: str, max_chars: int = 25) -> str:
    # Solo añade "..." si realmente se recorta, para no sugerir texto omitido que no existe
    return text if len(text) <= max_chars else f"{text[:max_chars]}..."


def _totals_warnings(result: EstimationResult) -> list[str]:
    mismatch = result.totals_mismatch()
    if mismatch is None:
        return []
    logger.warning("validation_not_passed", validation="totals_match_phases", detail=mismatch)
    return [f"totals_match_phases: {mismatch}"]


def _call_estimation(
    messages: list[dict], model: str, *, purpose: str = "estimation", **log_context: Any
) -> tuple[EstimationResult, LLMCallMeta]:
    # messages es un array arbitrario: [system, user] en el flujo clásico, o
    # [system, resumen, anclas, ventana, user] en el conversacional
    try:
        return call_structured(
            messages,
            EstimationResult,
            model,
            max_retries=settings.STRUCTURED_OUTPUT_MAX_RETRIES,
            purpose=purpose,
            **log_context,
        )
    except InstructorRetryException as exc:
        # call_structured ya registró llm_call_failed con el detalle
        raise EstimationFailedError("No valid estimation after retries") from exc


# Endpoint de FastAPI sin streaming
def estimate_project(request: EstimationRequest) -> tuple[EstimationResult, int, int, str]:
    # Nunca se llama al LLM con un input que no ha pasado los guardrails
    check_input(request.description, openai_client=get_openai_client())
    system_prompt, user_prompt = render_estimation_prompt(request)
    model = _model_name_for_provider(settings.LLM_PROVIDER)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
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

    # Con reintentos Instructor acumula los tokens de todos los intentos en el meta
    result, meta = _call_estimation(messages, model)

    _totals_warnings(result)  # sin reintento por totales: solo queda registrado en el log
    result = enforce_scope_response(result)  # red de seguridad si el model_validator no disparó

    return (result, meta.tokens_in, meta.tokens_out, meta.model)


def _with_attachment_reference(transcript: str, attachment_filenames: list[str]) -> str:
    # El historial guarda una referencia a los nombres de los adjuntos, no su contenido completo:
    # repetirlo en cada turno sería caro en tokens y los hechos relevantes ya viven en el metadata
    if not attachment_filenames:
        return transcript
    names = ", ".join(attachment_filenames)
    return f"{transcript}\n\n(Attachments uploaded in this turn: {names})"


# --- Pasos compartidos por el flujo conversacional y el ACB ---


@dataclass
class _Turn:
    session: Session
    transcript: str  # lo que escribió el usuario, sin el texto de los adjuntos
    enriched_transcript: str  # transcript + adjuntos: es lo que ven guardrails, tier, prompt y metadata
    attachment_filenames: list[str]
    project_type: ProjectType
    detail_level: DetailLevel
    tier: Tier


def _prepare_turn(
    session: Session,
    transcript: str,
    attachments: list[tuple[str, str]],
    project_type: ProjectType,
    detail_level: DetailLevel,
    tier_override: Tier | None,
) -> _Turn:
    """Pasos 1-2: guardrail de input y tier."""
    enriched_transcript = enrich_transcript(transcript=transcript, attachments=attachments)
    # El contenido de un adjunto es input no confiable (posible prompt injection): mismo guardrail
    check_input(enriched_transcript, openai_client=get_openai_client())

    # Se resuelve ANTES de renderizar el prompt, con el metadata acumulado hasta el turno anterior
    tier, rule = resolve_tier(
        transcript=enriched_transcript, metadata=session.metadata, override=tier_override
    )
    session.last_resolved_tier = tier.value
    session.last_tier_rule = rule
    return _Turn(
        session=session,
        transcript=transcript,
        enriched_transcript=enriched_transcript,
        attachment_filenames=[filename for filename, _ in attachments],
        project_type=project_type,
        detail_level=detail_level,
        tier=tier,
    )


def _build_messages(turn: _Turn, critic_feedback: CriticFeedback | None) -> list[dict]:
    """Pasos 3-4: render del prompt y messages = system + resumen + anclas + ventana + user."""
    system_prompt, user_prompt = render_conversational_prompt(
        transcript=turn.enriched_transcript,
        project_type=turn.project_type.value,
        detail_level=turn.detail_level.value,
        metadata=turn.session.metadata,
        tier=turn.tier.value,
        critic_feedback=critic_feedback,
    )
    return [*turn.session.history.to_messages_list(system_prompt), {"role": "user", "content": user_prompt}]


def _commit_turn(turn: _Turn, result: EstimationResult) -> None:
    """Pasos 7-9: historial, compresión y metadata."""
    # Se guarda el user prompt SIN critic_feedback y con los adjuntos como referencia
    history_user_content = render_conversational_user(
        transcript=_with_attachment_reference(turn.transcript, turn.attachment_filenames),
        project_type=turn.project_type.value,
    )
    turn.session.history.append(user=history_user_content, assistant=result.model_dump_json())
    # append() ya no recorta: la compresión es la única que decide qué se olvida
    apply_compression(turn.session.history)
    turn.session.metadata = update_metadata(
        turn.session.metadata, turn.enriched_transcript, result, session_id=turn.session.session_id
    )


def estimate_conversational(
    session: Session,
    transcript: str,
    attachments: list[tuple[str, str]],
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    tier: Tier | None = None,
) -> tuple[EstimationResult, LLMCallMeta]:
    """Turno conversacional: usa el historial de la sesión y el project_metadata acumulado.

    A diferencia de estimate_project(), NO consulta ni escribe la cache exact-match ni la
    semántica: la respuesta depende del historial y del metadata de la sesión, y la clave de
    cache actual (system + user) no los tiene en cuenta, así que daría hits incorrectos entre
    sesiones distintas (o entre turnos de la misma sesión).
    """
    turn = _prepare_turn(session, transcript, attachments, project_type, detail_level, tier)
    messages = _build_messages(turn, critic_feedback=None)

    model = _model_name_for_provider(settings.LLM_PROVIDER)
    logger.info(
        "llm_call_started",
        model_requested=model,
        provider=settings.LLM_PROVIDER,
        tokens_input_estimated=sum(len(m["content"]) for m in messages) // 4,
        session_id=session.session_id,
        history_messages=len(messages) - 2,  # sin el system ni el user actual
        tier=turn.tier.value,
        detail_level=detail_level.value,
        output_format=output_format.value,
        prompt_version=settings.CONVERSATIONAL_PROMPT_VERSION,
    )

    result, meta = _call_estimation(messages, model, session_id=session.session_id)

    _totals_warnings(result)
    result = enforce_scope_response(result)
    _commit_turn(turn, result)
    return result, meta


def estimate_with_acb(
    session: Session,
    transcript: str,
    attachments: list[tuple[str, str]],
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    tier: Tier | None = None,
) -> tuple[EstimationResult, BossTrace, LLMCallMeta]:
    """Variante Actor-Critic-Boss del turno conversacional.

    Cuesta hasta 2 * BOSS_MAX_ITERATIONS llamadas (actor + critic por vuelta). Los borradores
    intermedios son desechables: la sesión solo recibe el resultado final, así que para el
    usuario el turno produce exactamente un mensaje del assistant.
    """
    turn = _prepare_turn(session, transcript, attachments, project_type, detail_level, tier)
    model = _model_name_for_provider(settings.LLM_PROVIDER)
    session_id = session.session_id
    logger.info(
        "acb_started",
        session_id=session_id,
        tier=turn.tier.value,
        model_requested=model,
        critic_model=settings.CRITIC_MODEL,
        max_iterations=settings.BOSS_MAX_ITERATIONS,
        output_format=output_format.value,
        prompt_version=settings.CONVERSATIONAL_PROMPT_VERSION,
    )

    actor_metas: list[LLMCallMeta] = []
    critic_metas: list[LLMCallMeta] = []

    def actor(critic_feedback: CriticFeedback | None) -> EstimationResult:
        # Se re-renderiza en cada vuelta para que el feedback del critic entre en el user prompt
        draft, meta = _call_estimation(
            _build_messages(turn, critic_feedback),
            model,
            purpose="acb_actor",
            session_id=session_id,
            iteration=len(actor_metas),
        )
        actor_metas.append(meta)
        return draft

    def critic(draft: EstimationResult) -> CriticFeedback:
        feedback, meta = critic_service.review(
            turn.enriched_transcript,
            session.metadata,
            turn.tier,
            draft,
            session_id=session_id,
            iteration=len(critic_metas),
        )
        if meta is not None:  # None = el critic falló y se aceptó el borrador (fail-open)
            critic_metas.append(meta)
        return feedback

    result, trace = Boss(settings.BOSS_MAX_ITERATIONS).run(actor, critic)
    total_meta = sum_call_metas([*actor_metas, *critic_metas])
    trace.llm_usage = total_meta

    _totals_warnings(result)
    result = enforce_scope_response(result)
    _commit_turn(turn, result)
    return result, trace, total_meta


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
        stats["model_requested"] = normalise_model_name(model)
        stats["model_used"] = normalise_model_name(getattr(chunk, "model", model))

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
    # Mismos seis campos que call_structured, para poder agregar coste y latencia de ambos caminos
    model = stats.get("model_requested", "")
    tokens_in = stats.get("tokens_input", 0)
    tokens_out = stats.get("tokens_output", 0)
    logger.info(
        "llm_call_completed",
        latency_ms=int((time.perf_counter() - start) * 1000),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=estimate_cost_usd(model, tokens_in, tokens_out),
        model=normalise_model_name(model),
        provider=stats.get("provider_used") or provider_from_model(model),
        purpose="estimation_stream",
        finish_reason=stats.get("finish_reason"),
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
            latency_ms=int((time.perf_counter() - start) * 1000),
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            model="cache",
            provider="cache",
            purpose="estimation_stream",
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
        result = enforce_scope_response(result)  # red de seguridad si el model_validator no disparó
    except (ValueError, AuthenticationError, RateLimitError, Timeout, APIError) as exc:
        # ValidationError (y los errores de JSON de jiter) heredan de ValueError
        logger.warning(
            "llm_call_attempt_failed",
            attempt="stream_final_validation",
            provider=stats["provider_used"],
            error_type=error_type(exc),
            error=str(exc),
        )
        # Política de fallo: se descartan los parciales y se reintenta con Instructor bloqueante
        stats["blocking_fallback_used"] = True
        # estimate_project() ya aplica enforce_scope_response internamente: no se repite aquí
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
