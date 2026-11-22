"""EstimationService - the conductor: the ONLY place where the generation siblings are composed.

Entry points (the routers in ``app/api`` only translate HTTP <-> these calls):

- ``estimate()``                 guardrails(in) -> prompt -> llm -> guardrails(out)
- ``estimate_stream()``          guardrails(in) -> cag exact -> cag semantic -> prompt -> llm stream
                                 -> guardrails(out) -> cag store (SSE endpoint)
- ``estimate_conversational()``  conversation (history + metadata + compression) -> prompt -> llm
                                 -> guardrails(out)
- ``estimate_with_acb()``        agentic (Boss orchestrates Actor + Critic over the conversation)
- ``estimate_with_agent()``      agentic (manual tool-calling loop; its ``search_budgets`` tool is
                                 a closure over RAG retrieval, built HERE so agentic never imports rag)

The blocking and the conversational paths do NOT read or write the caches on purpose: the
conversational answer depends on session state the cache key does not capture.
When RAG joins the request path, its step is added here, never as a cross-import between siblings.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import instructor
import structlog
from instructor.core.exceptions import InstructorRetryException
from litellm import APIError, AuthenticationError, RateLimitError, Timeout
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.domain.schemas.acb import ACBResponse, BossTrace
from app.domain.schemas.critic import CriticFeedback
from app.domain.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
    ObservedEstimationResponse,
    OutputFormat,
    ProjectType,
)
from app.domain.schemas.observation import TurnObservation
from app.foundation.attachments.extractor import enrich_transcript
from app.foundation.guardrails.input import check_input
from app.foundation.guardrails.output import enforce_scope_response
from app.foundation.llm.streaming import open_partial_stream, simulate_stream_chunks
from app.foundation.llm.wrapper import (
    LLMCallMeta,
    LLMWrapper,
    error_type,
    estimate_cost_usd,
    normalise_model_name,
    provider_from_model,
    sum_call_metas,
)
from app.foundation.prompts.loader import (
    render_conversational_prompt,
    render_conversational_user,
    render_estimation_prompt,
)
from app.generation.agentic import critic as critic_service
from app.generation.agentic.agent_loop import run_agent
from app.generation.agentic.agent_schemas import AgentResult, SearchBudgetsArgs
from app.generation.agentic.agent_tools import SearchBackend, build_tool_registry
from app.generation.agentic.boss import Boss
from app.generation.cag.exact import ExactCache
from app.generation.cag.semantic import SemanticCache
from app.generation.conversation.compression import apply_compression
from app.generation.conversation.metadata_extractor import update_metadata
from app.generation.conversation.models import Session
from app.generation.conversation.tier_resolver import Tier, resolve_tier
from app.generation.rag.schemas import RetrievalRequest, RetrievalResult
from app.generation.rag.task_hours import HISTORICAL_TASK_CHUNK_TYPE

logger = structlog.get_logger(__name__)

# "partial": estado provisional (campos None, números a medias); "complete": validado con los model_validator
StreamEvent = tuple[Literal["partial", "complete"], EstimationResult]

# Se crea una sola vez (cada Partial[...] cachea una clase nueva); Any porque los stubs ocultan model_from_chunks
_PartialEstimationResult: Any = instructor.Partial[EstimationResult]
_ESTIMATION_TOOL = instructor.openai_schema(EstimationResult).openai_schema


class EstimationFailedError(Exception):
    """El LLM no devolvió una estimación válida (o el proveedor falló) tras los reintentos."""


class AgentUnavailableError(Exception):
    """The agent cannot run: it needs the OpenAI Responses API (OPENAI_API_KEY) and retrieval."""


class BudgetRetriever(Protocol):
    """What the agent's search tool needs from RAG (``TranscriptEstimator`` implements it)."""

    async def retrieve_chunks(self, request: RetrievalRequest) -> RetrievalResult: ...


def build_budget_search_backend(retriever: BudgetRetriever, settings: Settings) -> SearchBackend:
    """``search_budgets`` backend: a thin closure over the existing RAG retrieval.

    It only translates the tool arguments into a ``RetrievalRequest`` (always restricted to the
    ``historical_task`` chunks, which carry engineer-hours) and the chunks into plain dicts.
    Hybrid search and reranking are decided by ``retrieve_chunks`` from the runtime config.
    """

    async def search(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        filters = args.filters
        request = RetrievalRequest.model_validate(
            {
                "query_text": args.query,
                "top_k": settings.AGENT_SEARCH_TOP_K,
                "distance_threshold": settings.AGENT_SEARCH_DISTANCE_THRESHOLD,
                "chunk_types": [HISTORICAL_TASK_CHUNK_TYPE],
                "sectors": (filters.sectors or None) if filters else None,
                "project_year_min": filters.project_year_min if filters else None,
                "project_year_max": filters.project_year_max if filters else None,
            }
        )
        result = await retriever.retrieve_chunks(request)
        return [
            {
                "id": chunk.id,
                "budget_id": chunk.budget_id,
                "sector": chunk.sector,
                "project_year": chunk.project_year,
                # Chunk text = parent-budget header + task detail: the task part is what helps
                "content_preview": chunk.content.split("\n\n", 1)[-1],
                "estimated_hours": chunk.estimated_hours,
                "distance": chunk.distance,
            }
            for chunk in result.chunks
        ]

    return search


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


def _with_attachment_reference(transcript: str, attachment_filenames: list[str]) -> str:
    # El historial guarda una referencia a los nombres de los adjuntos, no su contenido completo:
    # repetirlo en cada turno sería caro en tokens y los hechos relevantes ya viven en el metadata
    if not attachment_filenames:
        return transcript
    names = ", ".join(attachment_filenames)
    return f"{transcript}\n\n(Attachments uploaded in this turn: {names})"


@dataclass
class _Turn:
    session: Session
    transcript: str  # lo que escribió el usuario, sin el texto de los adjuntos
    enriched_transcript: str  # transcript + adjuntos: es lo que ven guardrails, tier, prompt y metadata
    attachment_filenames: list[str]
    project_type: ProjectType
    detail_level: DetailLevel
    tier: Tier


class EstimationService:
    """Single composition point of the estimation pipelines. Collaborators are injected."""

    def __init__(
        self,
        *,
        openai_client: Any,
        llm_wrapper: LLMWrapper | None = None,
        exact_cache: ExactCache | None = None,
        semantic_cache: SemanticCache | None = None,
        settings: Settings | None = None,
        async_openai_client: AsyncOpenAI | None = None,
        budget_retriever: BudgetRetriever | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.openai_client = openai_client
        self.async_openai_client = async_openai_client
        self.budget_retriever = budget_retriever
        self.llm_wrapper = llm_wrapper or LLMWrapper(self.settings)
        self.exact_cache = exact_cache or ExactCache()
        self.semantic_cache = semantic_cache or SemanticCache(openai_client)

    # ------------------------------------------------------------------
    # Blocking estimation (POST /api/v1/estimate)
    # ------------------------------------------------------------------

    def check_request(self, request: EstimationRequest) -> None:
        """Input guardrails only. Lets the stream endpoint answer a real 400 before streaming."""
        check_input(request.description, openai_client=self.openai_client)

    def estimate(self, request: EstimationRequest) -> EstimationResponse:
        result, *_ = self.estimate_with_usage(request)
        return EstimationResponse(result=result, prompt_version=self.settings.PROMPT_VERSION)

    def _call_estimation(
        self, messages: list[dict], model: str, *, purpose: str = "estimation", **log_context: Any
    ) -> tuple[EstimationResult, LLMCallMeta]:
        # messages es un array arbitrario: [system, user] en el flujo clásico, o
        # [system, resumen, anclas, ventana, user] en el conversacional
        try:
            return self.llm_wrapper.complete_structured_chat(
                messages,
                EstimationResult,
                model=model,
                max_retries=self.settings.STRUCTURED_OUTPUT_MAX_RETRIES,
                purpose=purpose,
                **log_context,
            )
        except InstructorRetryException as exc:
            # call_structured ya registró llm_call_failed con el detalle
            raise EstimationFailedError("No valid estimation after retries") from exc

    def estimate_with_usage(self, request: EstimationRequest) -> tuple[EstimationResult, int, int, str]:
        # Nunca se llama al LLM con un input que no ha pasado los guardrails
        self.check_request(request)
        system_prompt, user_prompt = render_estimation_prompt(request)
        model = self.llm_wrapper.primary_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(
            "llm_call_started",
            model_requested=model,
            provider=self.settings.LLM_PROVIDER,
            tokens_input_estimated=len(system_prompt + user_prompt) // 4,
            detail_level=request.detail_level,
            output_format=request.output_format,
            description=_shorten(request.description),
            prompt_version=self.settings.PROMPT_VERSION,
        )

        # Con reintentos Instructor acumula los tokens de todos los intentos en el meta
        result, meta = self._call_estimation(messages, model)

        _totals_warnings(result)  # sin reintento por totales: solo queda registrado en el log
        result = enforce_scope_response(result)  # red de seguridad si el model_validator no disparó
        return result, meta.tokens_in, meta.tokens_out, meta.model

    # ------------------------------------------------------------------
    # Conversational paths (shared steps)
    # ------------------------------------------------------------------

    def _prepare_turn(
        self,
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
        check_input(enriched_transcript, openai_client=self.openai_client)

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

    @staticmethod
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
        return [
            *turn.session.history.to_messages_list(system_prompt),
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _commit_turn(turn: _Turn, result: EstimationResult) -> None:
        """Pasos 7-9: historial, compresión y metadata."""
        # Se guarda el user prompt SIN critic_feedback y con los adjuntos como referencia
        history_user_content = render_conversational_user(
            transcript=_with_attachment_reference(turn.transcript, turn.attachment_filenames),
            project_type=turn.project_type.value,
        )
        turn.session.history.append(user=history_user_content, assistant=result.model_dump_json())
        turn.session.turn_count += 1
        # append() ya no recorta: la compresión es la única que decide qué se olvida
        apply_compression(turn.session.history)
        turn.session.metadata = update_metadata(
            turn.session.metadata,
            turn.enriched_transcript,
            result,
            session_id=turn.session.session_id,
        )

    @staticmethod
    def _observe_turn(
        turn: _Turn, attachments: list[tuple[str, str]], meta: LLMCallMeta
    ) -> TurnObservation:
        """Foto del turno ya comprometido (historial comprimido incluido); se emite y se devuelve."""
        history = turn.session.history
        observation = TurnObservation(
            turn_index=turn.session.turn_count,
            session_id=turn.session.session_id,
            enriched_transcript_chars=len(turn.enriched_transcript),
            attachments_total_chars=sum(len(text) for _, text in attachments),
            messages_in_window=len(history.messages),
            anchors_count=len(history.anchors),
            summary_chars=len(history.summary or ""),
            tokens_in=meta.tokens_in,
            tokens_out=meta.tokens_out,
            cost_usd=meta.cost_usd,
            latency_ms=meta.latency_ms,
            # El camino conversacional no consulta ninguna cache (ver estimate_conversational)
            cache_hit_kind="none",
            last_resolved_tier=turn.tier.value,
        )
        logger.info("turn_observed", **observation.model_dump())
        return observation

    def estimate_conversational(
        self,
        *,
        session: Session,
        transcript: str,
        attachments: list[tuple[str, str]],
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        tier: Tier | None = None,
    ) -> ObservedEstimationResponse:
        """Turno conversacional: usa el historial de la sesión y el project_metadata acumulado.

        A diferencia de estimate(), NO consulta ni escribe la cache exact-match ni la
        semántica: la respuesta depende del historial y del metadata de la sesión, y la clave de
        cache actual (system + user) no los tiene en cuenta, así que daría hits incorrectos entre
        sesiones distintas (o entre turnos de la misma sesión).
        """
        turn = self._prepare_turn(session, transcript, attachments, project_type, detail_level, tier)
        messages = self._build_messages(turn, critic_feedback=None)

        model = self.llm_wrapper.primary_model
        logger.info(
            "llm_call_started",
            model_requested=model,
            provider=self.settings.LLM_PROVIDER,
            tokens_input_estimated=sum(len(m["content"]) for m in messages) // 4,
            session_id=session.session_id,
            history_messages=len(messages) - 2,  # sin el system ni el user actual
            tier=turn.tier.value,
            detail_level=detail_level.value,
            output_format=output_format.value,
            prompt_version=self.settings.CONVERSATIONAL_PROMPT_VERSION,
        )

        result, meta = self._call_estimation(messages, model, session_id=session.session_id)

        _totals_warnings(result)
        result = enforce_scope_response(result)
        self._commit_turn(turn, result)
        return ObservedEstimationResponse(
            result=result,
            prompt_version=self.settings.CONVERSATIONAL_PROMPT_VERSION,
            cached=False,
            turn_observation=self._observe_turn(turn, attachments, meta),
            **meta.model_dump(),
        )

    def estimate_with_acb(
        self,
        *,
        session: Session,
        transcript: str,
        attachments: list[tuple[str, str]],
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        tier: Tier | None = None,
    ) -> ACBResponse:
        """Variante Actor-Critic-Boss del turno conversacional.

        Cuesta hasta 2 * BOSS_MAX_ITERATIONS llamadas (actor + critic por vuelta). Los borradores
        intermedios son desechables: la sesión solo recibe el resultado final, así que para el
        usuario el turno produce exactamente un mensaje del assistant.
        """
        turn = self._prepare_turn(session, transcript, attachments, project_type, detail_level, tier)
        model = self.llm_wrapper.primary_model
        session_id = session.session_id
        logger.info(
            "acb_started",
            session_id=session_id,
            tier=turn.tier.value,
            model_requested=model,
            critic_model=self.settings.CRITIC_MODEL,
            max_iterations=self.settings.BOSS_MAX_ITERATIONS,
            output_format=output_format.value,
            prompt_version=self.settings.CONVERSATIONAL_PROMPT_VERSION,
        )

        actor_metas: list[LLMCallMeta] = []
        critic_metas: list[LLMCallMeta] = []

        def actor(critic_feedback: CriticFeedback | None) -> EstimationResult:
            # Se re-renderiza en cada vuelta para que el feedback del critic entre en el user prompt
            draft, meta = self._call_estimation(
                self._build_messages(turn, critic_feedback),
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

        result, trace = Boss(self.settings.BOSS_MAX_ITERATIONS).run(actor, critic)
        total_meta = sum_call_metas([*actor_metas, *critic_metas])
        trace.llm_usage = total_meta

        _totals_warnings(result)
        result = enforce_scope_response(result)
        self._commit_turn(turn, result)
        return self._acb_response(result, trace, total_meta)

    def _acb_response(self, result: EstimationResult, trace: BossTrace, meta: LLMCallMeta) -> ACBResponse:
        return ACBResponse(
            result=result,
            prompt_version=self.settings.CONVERSATIONAL_PROMPT_VERSION,
            cached=False,
            acb=trace,
            **meta.model_dump(),
        )

    # ------------------------------------------------------------------
    # Tool-calling agent (POST /v1/estimate/agent)
    # ------------------------------------------------------------------

    async def estimate_with_agent(
        self,
        transcript: str,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_iterations: int | None = None,
    ) -> AgentResult:
        """Run the manual agent loop; ``None`` overrides fall back to the AGENT_* settings.

        Raises ``AgentUnavailableError`` without the async OpenAI client or the retriever, and
        ``AgentLoopError`` (from the loop) when the OpenAI call fails.
        """
        if self.async_openai_client is None:
            raise AgentUnavailableError("The agent requires OPENAI_API_KEY (OpenAI Responses API).")
        if self.budget_retriever is None:
            raise AgentUnavailableError("The agent requires the RAG retriever to search budgets.")

        settings = self.settings
        registry = build_tool_registry(build_budget_search_backend(self.budget_retriever, settings))
        model = model or settings.AGENT_MODEL
        effort = reasoning_effort or settings.AGENT_REASONING_EFFORT
        iterations = max_iterations or settings.AGENT_MAX_ITERATIONS
        logger.info(
            "agent_started",
            model=model,
            reasoning_effort=effort,
            max_iterations=iterations,
            transcript_chars=len(transcript),
        )
        return await run_agent(
            transcript,
            client=self.async_openai_client,
            registry=registry,
            model=model,
            effort=effort,
            max_iterations=iterations,
        )

    # ------------------------------------------------------------------
    # Streaming estimation (POST /api/v1/estimate/stream, used by Streamlit)
    # ------------------------------------------------------------------

    def _open_partial_stream(
        self, provider: str, system_prompt: str, user_prompt: str, stats: dict
    ) -> tuple[EstimationResult, Iterator[EstimationResult]]:
        return open_partial_stream(
            partial_model=_PartialEstimationResult,
            tool_schema=_ESTIMATION_TOOL,
            provider=provider,
            model=self.llm_wrapper.model_for_provider(provider),
            api_key=self.llm_wrapper.api_key_for_provider(provider),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stats=stats,
        )

    def _open_stream_with_fallback(
        self, system_prompt: str, user_prompt: str, stats: dict
    ) -> tuple[EstimationResult, Iterator[EstimationResult]]:
        primary_provider = self.settings.LLM_PROVIDER
        for attempt in range(self.settings.LLM_MAX_RETRIES + 1):
            try:
                first_state, states = self._open_partial_stream(
                    primary_provider, system_prompt, user_prompt, stats
                )
            except AuthenticationError as exc:
                logger.warning(
                    "llm_call_attempt_failed",
                    attempt="primary_auth",
                    provider=primary_provider,
                    error=str(exc),
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
            first_state, states = self._open_partial_stream(
                fallback_provider, system_prompt, user_prompt, stats
            )
        except (AuthenticationError, RateLimitError, Timeout, APIError, StopIteration) as exc:
            logger.error("llm_call_failed", error_type="all_providers_failed", error=str(exc))
            raise EstimationFailedError("All providers failed") from exc

        stats["provider_used"] = fallback_provider
        stats["fallback_used"] = True
        return first_state, states

    def _get_cached_result(self, cache_key: str) -> EstimationResult | None:
        cached = self.exact_cache.get(cache_key)
        if cached is None:
            return None
        try:
            return EstimationResult.model_validate_json(cached)
        except ValidationError:
            # Entradas del streaming anterior (texto libre) o de otra versión del schema: se tratan como miss
            logger.warning("cache_entry_invalid", cache_key=cache_key)
            return None

    @staticmethod
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

    def estimate_stream(self, request: EstimationRequest, stats: dict) -> Iterator[StreamEvent]:
        # Igual que en la versión bloqueante: el guardrail de input nunca se salta por cache ni por streaming
        self.check_request(request)
        system_prompt, user_prompt = render_estimation_prompt(request)
        cache_key = self.exact_cache.build_key(system_prompt, user_prompt)
        cached_result = self._get_cached_result(cache_key)
        exact_cache_hit = cached_result is not None
        semantic_cache_hit = False

        if cached_result is None:
            # Más caro que el exact-match (llama a la API de embeddings): solo se intenta si ya hubo miss
            cached_result = self.semantic_cache.lookup(request, self.settings.PROMPT_VERSION)
            semantic_cache_hit = cached_result is not None
            if semantic_cache_hit:
                # Promociona el hit semántico a exact-match: si se repite este mismo texto literal,
                # la próxima vez se sirve sin volver a llamar a la API de embeddings
                self.exact_cache.set(cache_key, cached_result.model_dump_json())

        cache_hit = exact_cache_hit or semantic_cache_hit

        logger.info(
            "llm_call_started",
            provider_requested=self.settings.LLM_PROVIDER,
            tokens_input_estimated=len(system_prompt + user_prompt) // 4,
            cache_hit=cache_hit,
            detail_level=request.detail_level,
            output_format=request.output_format,
            description=_shorten(request.description),
            prompt_version=self.settings.PROMPT_VERSION,
        )

        start = time.perf_counter()
        stats["cache_hit"] = cache_hit
        stats["exact_cache_hit"] = exact_cache_hit
        stats["semantic_cache_hit"] = semantic_cache_hit

        if cached_result is not None:
            # Mismo parser que con el LLM real: el cliente recibe los mismos estados parciales
            replay = _PartialEstimationResult.model_from_chunks(
                simulate_stream_chunks(cached_result.model_dump_json())
            )
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

        first_state, states = self._open_stream_with_fallback(system_prompt, user_prompt, stats)
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
            # estimate_with_usage ya aplica enforce_scope_response internamente: no se repite aquí
            result, tokens_input, tokens_output, model = self.estimate_with_usage(request)
            stats["tokens_input"] = stats.get("tokens_input", 0) + tokens_input
            stats["tokens_output"] = stats.get("tokens_output", 0) + tokens_output
            stats["model_requested"] = model

        # Solo se cachea un resultado sin avisos, para poder reintentar y obtener uno mejor
        if warnings := _totals_warnings(result):
            stats["validation_warnings"] = warnings
        else:
            self.exact_cache.set(cache_key, result.model_dump_json())
            self.semantic_cache.store(request, result, self.settings.PROMPT_VERSION)
        self._log_llm_call_completed(stats, start)
        yield "complete", result
