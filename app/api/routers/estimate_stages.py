"""``POST /v1/estimate/stages/*``: the RAG pipeline, one stage at a time (didactic).

The full ``/v1/estimate/from-transcript`` pipeline hides its intermediate artifacts (the
reformulated brief, the retrieved chunks, the assembled context block). This router exposes each
stage as an independent, STATELESS endpoint: the caller keeps each stage's output and passes it
into the next one, so a stage can be re-run on its own (e.g. retrieval with other filters).
Every endpoint reuses the same building blocks as the orchestrator (``TranscriptEstimator`` and
the pure context/validation functions): no pipeline logic lives here.

Thin transport: validation in the request schemas (422), auth in ``require_estimate_key`` (401),
rate limits in the decorators (429), pipeline failures -> 502. Per-stage logs are correlated by
the ``X-Request-ID`` response header. ``/generate`` deliberately does NOT run the orchestrator's
corrective retries: it returns the grounding signals so a UI can show them.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.rate_limiting import limiter
from app.api.request_context import get_request_id
from app.api.security import require_estimate_key
from app.config import Settings, get_settings
from app.dependencies import get_token_encoder, get_transcript_estimator
from app.generation.rag.context_assembler import (
    TokenEncoder,
    build_context_block,
    truncate_to_token_budget,
)
from app.generation.rag.errors import RagError
from app.generation.rag.estimator import TranscriptEstimator
from app.generation.rag.observability import log_stage
from app.generation.rag.quality.augmentation import augment_chunks
from app.generation.rag.query_reformulator import compose_search_text
from app.generation.rag.schemas import (
    AssembleRequest,
    AssembleResult,
    GenerateRequest,
    GenerateResult,
    HallucinationReport,
    ReformulateRequest,
    ReformulationResult,
    RetrievalRequest,
    RetrievalResult,
    StructureRequest,
    VerifyRequest,
)
from app.generation.rag.validation import check_coherence, verify_citations

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate/stages", tags=["estimate-stages"])


def _stage_failed(stage: str, exc: Exception, detail: str) -> HTTPException:
    log.error("stage_failed", stage=stage, error_type=type(exc).__name__, error=str(exc)[:300])
    return HTTPException(status_code=502, detail=detail)


@router.post(
    "/reformulate",
    response_model=ReformulationResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("30/minute")
async def reformulate(
    request: Request,  # slowapi needs the Starlette request under this exact name
    payload: ReformulateRequest,
    estimator: TranscriptEstimator = Depends(get_transcript_estimator),
) -> ReformulationResult:
    """Stage 1: distill a transcript into a structured brief + the text to embed."""
    try:
        with log_stage("reformulation", get_request_id(request)):
            query = await estimator.reformulate(payload.transcript)
    except RagError as exc:
        raise _stage_failed("reformulation", exc, "Query reformulation failed.") from exc
    return ReformulationResult(query=query, search_text=compose_search_text(query))


@router.post(
    "/retrieve",
    response_model=RetrievalResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
async def retrieve_stage(
    request: Request,
    payload: RetrievalRequest,
    estimator: TranscriptEstimator = Depends(get_transcript_estimator),
) -> RetrievalResult:
    """Stage 2: embed the search text (sent as ``query_text``) and run the filtered retrieval.

    Same request/response as ``/v1/retrieval/search``, so the caller sees the soft-fail signals
    (``low_confidence`` / ``candidates_evaluated``)."""
    try:
        with log_stage("retrieval", get_request_id(request), sectors=payload.sectors):
            return await estimator.retrieve_chunks(payload)
    except RagError as exc:
        raise _stage_failed("retrieval", exc, "Retrieval failed.") from exc


@router.post(
    "/assemble",
    response_model=AssembleResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
async def assemble(
    request: Request,
    payload: AssembleRequest,
    settings: Settings = Depends(get_settings),
    encoder: TokenEncoder = Depends(get_token_encoder),
) -> AssembleResult:
    """Stage 3: truncate to the token budget (whole chunks), optionally augment, and build the
    ``<source>`` context block."""
    budget = payload.max_context_tokens or settings.MAX_CONTEXT_TOKENS
    with log_stage("augmentation", get_request_id(request), budget=budget, augment=payload.augment):
        kept = truncate_to_token_budget(payload.chunks, budget, encoder)
        if payload.augment:
            kept = augment_chunks(kept, compress=payload.compress, reorder=payload.reorder)
        context_block = build_context_block(kept)
        token_count = len(encoder.encode(context_block))

    return AssembleResult(
        context_block=context_block,
        kept_chunks=kept,
        dropped_count=len(payload.chunks) - len(kept),
        token_count=token_count,
        augmented=payload.augment,
    )


@router.post(
    "/structure",
    response_model=GenerateResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("15/minute")
async def structure(
    request: Request,
    payload: StructureRequest,
    estimator: TranscriptEstimator = Depends(get_transcript_estimator),
) -> GenerateResult:
    """Free module -> task decomposition of the brief (no retrieval, no sources, no hours).

    Same ``GenerateResult`` shape as ``/generate`` (citations always clean) so a client parses
    both alike."""
    try:
        with log_stage("structure", get_request_id(request)):
            estimate = await estimator.generate_structure(payload.query)
    except RagError as exc:
        raise _stage_failed("structure", exc, "Structure generation failed.") from exc
    return GenerateResult(estimate=estimate, fabricated_source_ids=[], coherent=True)


@router.post(
    "/generate",
    response_model=GenerateResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("15/minute")
async def generate(
    request: Request,
    payload: GenerateRequest,
    estimator: TranscriptEstimator = Depends(get_transcript_estimator),
) -> GenerateResult:
    """Stage 4: generate the grounded estimate and report its grounding signals.

    No automatic retry on fabricated citations or incoherence: ``fabricated_source_ids`` and
    ``coherent`` let the caller surface them."""
    try:
        with log_stage(
            "generation",
            get_request_id(request),
            sources=len(payload.kept_chunks),
            include_hours=payload.include_hours,
        ):
            estimate = await estimator.generate(
                payload.context_block, payload.query, include_hours=payload.include_hours
            )
    except RagError as exc:
        raise _stage_failed("generation", exc, "Estimate generation failed.") from exc

    report = verify_citations(estimate, {str(chunk.id) for chunk in payload.kept_chunks})
    return GenerateResult(
        estimate=estimate,
        fabricated_source_ids=report.dangling_citations,
        citation_report=report,
        coherent=check_coherence(estimate),
    )


@router.post(
    "/verify",
    response_model=HallucinationReport,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("15/minute")
async def verify(
    request: Request,
    payload: VerifyRequest,
    estimator: TranscriptEstimator = Depends(get_transcript_estimator),
) -> HallucinationReport:
    """Stage 5: semantic hallucination gate over a generated estimate.

    ``/generate`` reports REFERENTIAL grounding (a cited id was retrieved); this reports SEMANTIC
    grounding (a numeric anchor + a strict judge grade each line)."""
    try:
        with log_stage("hallucination_gate", get_request_id(request), use_judge=payload.use_judge):
            return await estimator.verify_estimate(
                payload.estimate, payload.kept_chunks, use_judge=payload.use_judge
            )
    except RagError as exc:
        raise _stage_failed("hallucination_gate", exc, "Hallucination gate failed.") from exc
