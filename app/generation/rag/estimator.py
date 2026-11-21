"""End-to-end RAG estimation from a transcript (Session 9).

``transcript -> query understanding -> retrieval -> augmentation -> generation`` producing a
grounded ``Estimate``. Every collaborator (LLM wrapper, runtime config, embedder, session
factory, stores, tokenizer) is injected into ``TranscriptEstimator``: this package never reaches
for the composition root (``app/dependencies.py`` builds the object).

Blocking calls (LLM, embedder, Redis) run in threads so the HTTP path never blocks the event
loop.
"""

import asyncio
from typing import cast, get_args
from uuid import uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.foundation.llm.runtime_config import RuntimeModelConfig, RuntimeRetrievalConfig
from app.foundation.llm.wrapper import LLMWrapper
from app.generation.rag.context_assembler import (
    TokenEncoder,
    build_context_block,
    truncate_to_token_budget,
)
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.errors import (
    GenerationError,
    MalformedEstimateError,
    RetrievalError,
    ensure_valid_vectors,
)
from app.generation.rag.idempotency import IdempotencyStore
from app.generation.rag.observability import log_stage
from app.generation.rag.prompt_builder import (
    build_structure_system_prompt,
    build_structure_user_message,
    build_system_prompt,
    build_user_message,
)
from app.generation.rag.quality.augmentation import augment_chunks
from app.generation.rag.quality.hallucination import gate_estimate
from app.generation.rag.query_reformulator import compose_search_text, reformulate_query
from app.generation.rag.retrieval.pipeline import resolve_search_options, retrieve
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import (
    Estimate,
    EstimationQuery,
    HallucinationReport,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
    SearchMode,
    Sector,
)
from app.generation.rag.store.repository import ChunkSearchStore
from app.generation.rag.validation import check_coherence, verify_citations

log = structlog.get_logger()

# Sectors present in the corpus: only filter retrieval when the reformulated brief names one of
# them (free-text sector values would otherwise over-filter to an empty result)
_KNOWN_SECTORS: frozenset[str] = frozenset(get_args(Sector))


def _insufficient(explanation: str) -> Estimate:
    """The canonical insufficient-context estimate (no numbers)."""
    return Estimate(
        total_engineer_days=None,
        duration_weeks=None,
        confidence="insufficient",
        reasoning="Retrieval did not surface enough relevant historical budgets.",
        insufficient_context_explanation=explanation,
    )


def _current_request_id() -> str:
    """Reuse the HTTP request id bound by the middleware, or mint one."""
    bound = structlog.contextvars.get_contextvars().get("request_id")
    return str(bound) if bound else str(uuid4())


class TranscriptEstimator:
    """Runs the transcript -> grounded estimate pipeline with injected collaborators."""

    def __init__(
        self,
        *,
        llm: LLMWrapper,
        models: RuntimeModelConfig,
        retrieval_config: RuntimeRetrievalConfig,
        embedder: OpenAIEmbedder,
        session_factory: async_sessionmaker[AsyncSession],
        store: ChunkSearchStore,
        idempotency: IdempotencyStore,
        token_encoder: TokenEncoder,
        settings: Settings,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._llm = llm
        self._models = models
        self._retrieval_config = retrieval_config
        self._embedder = embedder
        self._session_factory = session_factory
        self._store = store
        self._idempotency = idempotency
        self._token_encoder = token_encoder
        self._settings = settings
        self._reranker = reranker

    async def estimate_from_transcript(
        self, transcript: str, idempotency_key: str | None = None
    ) -> Estimate:
        """Run the full pipeline.

        Steps: idempotency lookup -> reformulate -> embed -> filtered retrieval (soft-fail
        short-circuits to an insufficient-context estimate) -> token-budget truncation ->
        optional augmentation -> context assembly -> generation -> citation validation (one
        corrective retry) -> optional hallucination gate -> coherence check -> cache. The two
        optional stages follow ``RuntimeRetrievalConfig`` (runtime override -> ``Settings``).

        With an ``idempotency_key`` a repeated call returns the cached estimate without
        re-running the pipeline (no LLM cost). Raises ``RagError`` subclasses on failure.
        """
        request_id = _current_request_id()

        if idempotency_key:
            cached = await asyncio.to_thread(self._idempotency.get, idempotency_key)
            if cached is not None:
                log.info("idempotency_hit", request_id=request_id, idempotency_key=idempotency_key)
                return cached

        estimate = await self._run(transcript, request_id)

        if idempotency_key:
            await asyncio.to_thread(self._idempotency.set, idempotency_key, estimate)
        return estimate

    async def _run(self, transcript: str, request_id: str) -> Estimate:
        settings = self._settings

        # 1. Query understanding
        reformulation_model = await asyncio.to_thread(self._models.effective, "REFORMULATION_MODEL")
        with log_stage("reformulation", request_id, model=reformulation_model):
            query = await reformulate_query(
                transcript,
                llm=self._llm,
                model=reformulation_model,
                max_tokens=settings.GENERATION_MAX_TOKENS,
            )

        # 2. Compose + embed the canonical search text
        search_text = compose_search_text(query)
        with log_stage("embedding", request_id):
            query_embedding = await self._embed(search_text)

        # 3. Metadata-filtered retrieval with soft-fail. Mode and reranking follow the runtime
        #    override (PUT /api/v1/config/retrieval) and fall back to the environment.
        search_mode = await asyncio.to_thread(self._retrieval_config.effective_search_mode)
        rerank = await asyncio.to_thread(self._retrieval_config.effective_rerank)
        sector = query.sector.lower().strip() if query.sector else None
        sectors = [sector] if sector in _KNOWN_SECTORS else None
        with log_stage(
            "retrieval", request_id, sectors=sectors, search_mode=search_mode, rerank=rerank
        ):
            retrieval = await self._retrieve(
                query_embedding, search_text, cast(SearchMode, search_mode), rerank, sectors
            )

        if retrieval.low_confidence:
            log.info(
                "retrieval_soft_fail",
                request_id=request_id,
                candidates=retrieval.candidates_evaluated,
            )
            return _insufficient(
                "No historical budgets crossed the relevance threshold for this project."
            )

        # 4. Truncate to the token budget (whole chunks only), augment and assemble the context.
        #    Augmentation (compress + edge-load reorder) keeps the ids, so the citation check
        #    below is unaffected.
        kept = truncate_to_token_budget(
            retrieval.chunks, settings.MAX_CONTEXT_TOKENS, self._token_encoder
        )
        if await asyncio.to_thread(self._retrieval_config.effective_augmentation):
            with log_stage("augmentation", request_id, chunks=len(kept)):
                kept = augment_chunks(
                    kept,
                    compress=settings.AUGMENTATION_COMPRESS,
                    reorder=settings.AUGMENTATION_REORDER,
                )
        context_block = build_context_block(kept)

        # 5. Generate the grounded estimate
        generation_model = await asyncio.to_thread(self._models.effective, "GENERATION_MODEL")
        with log_stage("generation", request_id, sources=len(kept), model=generation_model):
            estimate = await self._generate(context_block, query, generation_model)

        # 6. Verify per-line citations; one corrective retry on dangling ids
        retrieved_ids = {str(chunk.id) for chunk in kept}
        report = verify_citations(estimate, retrieved_ids)
        log.info(
            "citation_report",
            request_id=request_id,
            total_lines=report.total_lines,
            grounded_lines=report.grounded_lines,
            dangling_lines=report.dangling_lines,
            insufficient_lines=report.insufficient_lines,
            verified_citations=report.verified_citations,
            dangling_citations=report.dangling_citations,
        )
        if report.has_dangling:
            feedback = (
                f"your previous response cited chunk_ids absent from the context: "
                f"{report.dangling_citations}. Only cite the `id` attribute of <source> "
                "elements that appear in the <sources> block, and copy the matching "
                "document_id and a verbatim evidence span for each."
            )
            with log_stage("citation_retry", request_id, dangling=report.dangling_citations):
                estimate = await self._generate(
                    context_block, query, generation_model, feedback=feedback
                )
            report = verify_citations(estimate, retrieved_ids)
            if report.has_dangling:
                log.warning(
                    "citations_unrepaired",
                    request_id=request_id,
                    dangling_citations=report.dangling_citations,
                )
                estimate = estimate.model_copy(update={"confidence": "low"})

        # 6.5 Semantic gate: a real citation does not prove the number is entailed by it. A line
        #     claiming more than its evidence supports downgrades a confident estimate.
        if await asyncio.to_thread(self._retrieval_config.effective_hallucination_gate):
            estimate = await self._apply_hallucination_gate(estimate, kept, request_id)

        # 7. Coherence guard: one repair attempt, then reject
        if not check_coherence(estimate):
            feedback = (
                'when confidence is "insufficient", total_engineer_days and '
                "duration_weeks must be null, modules must be empty and "
                "insufficient_context_explanation must be filled; otherwise provide "
                "the modules, tasks and numbers."
            )
            with log_stage("coherence_repair", request_id):
                estimate = await self._generate(
                    context_block, query, generation_model, feedback=feedback
                )
            if not check_coherence(estimate):
                raise MalformedEstimateError(
                    "Estimate violates the insufficient-context coherence rule."
                )

        return estimate

    async def _apply_hallucination_gate(
        self, estimate: Estimate, chunks: list[RetrievedChunk], request_id: str
    ) -> Estimate:
        """Grade every line against its cited chunks; any degraded line caps the confidence."""
        judge_model = await asyncio.to_thread(self._models.effective, "HALLUCINATION_JUDGE_MODEL")
        with log_stage("hallucination_gate", request_id, model=judge_model):
            gate = await gate_estimate(
                estimate,
                chunks,
                llm=self._llm,
                tolerance=self._settings.HALLUCINATION_NUMERIC_TOLERANCE,
                judge_model=judge_model,
                max_tokens=self._settings.GENERATION_MAX_TOKENS,
            )
        log.info(
            "hallucination_report",
            request_id=request_id,
            total_lines=gate.total_lines,
            grounded_lines=gate.grounded_lines,
            degraded_lines=gate.degraded_lines,
            insufficient_lines=gate.insufficient_lines,
        )
        if gate.has_degraded and estimate.confidence in ("high", "medium"):
            return estimate.model_copy(update={"confidence": "low"})
        return estimate

    async def _embed(self, text: str) -> list[float]:
        try:
            vector = await asyncio.to_thread(self._embedder.embed_one, text)
            ensure_valid_vectors([vector], expected_count=1)
        except Exception as exc:
            raise RetrievalError("Embedding the search text failed.") from exc
        return vector

    async def _retrieve(
        self,
        query_embedding: list[float],
        search_text: str,
        search_mode: SearchMode,
        rerank: bool,
        sectors: list[str] | None,
    ) -> RetrievalResult:
        settings = self._settings
        async with self._session_factory() as session:
            return await retrieve(
                session,
                self._store,
                query_embedding=query_embedding,
                query_text=search_text,
                search_mode=search_mode,
                rerank=rerank,
                top_k=settings.RETRIEVAL_TOP_K,
                recall_k=settings.RETRIEVAL_RECALL_TOP_K,
                rerank_top_n=settings.RERANK_TOP_N,
                distance_threshold=settings.RETRIEVAL_DISTANCE_THRESHOLD,
                rrf_k=settings.RRF_K,
                lexical_match_mode=settings.LEXICAL_MATCH_MODE,
                sectors=sectors,
                reranker=self._reranker,
            )

    async def _generate(
        self,
        context_block: str,
        query: EstimationQuery,
        model: str,
        *,
        feedback: str | None = None,
        include_hours: bool = True,
    ) -> Estimate:
        """Single generation call; ``feedback`` appends a correction note for retries."""
        user_message = build_user_message(context_block, query)
        if feedback:
            user_message += f"\n\n<correction>\n{feedback}\n</correction>"

        try:
            estimate, _meta = await asyncio.to_thread(
                self._llm.complete_structured,
                system_prompt=build_system_prompt(include_hours=include_hours),
                user_message=user_message,
                response_model=Estimate,
                model=model,
                reasoning_effort=self._settings.GENERATION_REASONING_EFFORT,
                # Reasoning tokens count against max_tokens: a small ceiling is exhausted by
                # reasoning alone and truncates the JSON (see Settings.GENERATION_MAX_TOKENS)
                max_tokens=self._settings.GENERATION_MAX_TOKENS,
                purpose="rag_generation",
            )
        except Exception as exc:
            raise GenerationError("Grounded estimate generation failed.") from exc
        return estimate

    # ---- Single-stage entry points: the didactic /v1/estimate/stages/* endpoints reuse the same
    # ---- building blocks as ``_run`` (same models, settings and pure functions), one stage each.

    async def reformulate(self, transcript: str) -> EstimationQuery:
        model = await asyncio.to_thread(self._models.effective, "REFORMULATION_MODEL")
        return await reformulate_query(
            transcript,
            llm=self._llm,
            model=model,
            max_tokens=self._settings.GENERATION_MAX_TOKENS,
        )

    async def retrieve_chunks(self, request: RetrievalRequest) -> RetrievalResult:
        """Embed ``request.query_text`` and retrieve with the request's own knobs.

        ``search_mode`` / ``rerank`` follow request -> runtime override -> Settings, like
        ``POST /v1/retrieval/search``.
        """
        search_mode, rerank = await asyncio.to_thread(
            resolve_search_options,
            request.search_mode,
            request.rerank,
            self._settings,
            self._retrieval_config,
        )
        query_embedding = await self._embed(request.query_text)
        settings = self._settings
        async with self._session_factory() as session:
            return await retrieve(
                session,
                self._store,
                query_embedding=query_embedding,
                query_text=request.query_text,
                search_mode=search_mode,
                rerank=rerank,
                top_k=request.top_k,
                recall_k=settings.RETRIEVAL_RECALL_TOP_K,
                rerank_top_n=settings.RERANK_TOP_N,
                distance_threshold=request.distance_threshold,
                rrf_k=settings.RRF_K,
                lexical_match_mode=settings.LEXICAL_MATCH_MODE,
                sectors=[str(sector) for sector in request.sectors] if request.sectors else None,
                project_year_min=request.project_year_min,
                project_year_max=request.project_year_max,
                chunk_types=request.chunk_types,
                reranker=self._reranker,
            )

    async def generate(
        self, context_block: str, query: EstimationQuery, *, include_hours: bool = True
    ) -> Estimate:
        """One generation call, WITHOUT the corrective retries of the full pipeline."""
        model = await asyncio.to_thread(self._models.effective, "GENERATION_MODEL")
        return await self._generate(context_block, query, model, include_hours=include_hours)

    async def generate_structure(self, query: EstimationQuery) -> Estimate:
        """Free module -> task decomposition of the brief: no sources, no hours."""
        model = await asyncio.to_thread(self._models.effective, "GENERATION_MODEL")
        try:
            estimate, _meta = await asyncio.to_thread(
                self._llm.complete_structured,
                system_prompt=build_structure_system_prompt(),
                user_message=build_structure_user_message(query),
                response_model=Estimate,
                model=model,
                reasoning_effort=self._settings.GENERATION_REASONING_EFFORT,
                max_tokens=self._settings.GENERATION_MAX_TOKENS,
                purpose="rag_structure",
            )
        except Exception as exc:
            raise GenerationError("Structure generation failed.") from exc
        return estimate

    async def verify_estimate(
        self, estimate: Estimate, chunks: list[RetrievedChunk], *, use_judge: bool = True
    ) -> HallucinationReport:
        """The semantic hallucination gate over a finished estimate (no confidence downgrade)."""
        judge_model = await asyncio.to_thread(self._models.effective, "HALLUCINATION_JUDGE_MODEL")
        return await gate_estimate(
            estimate,
            chunks,
            llm=self._llm,
            tolerance=self._settings.HALLUCINATION_NUMERIC_TOLERANCE,
            judge_model=judge_model,
            use_judge=use_judge,
            max_tokens=self._settings.GENERATION_MAX_TOKENS,
        )
