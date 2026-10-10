import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.attachments.extractor import (
    AttachmentExtractionError,
    UnsupportedAttachmentError,
    extract_text,
)
from app.config import get_settings
from app.dependencies import get_session_store
from app.guardrails.input import InputGuardrailViolation
from app.schemas.acb import ACBResponse
from app.schemas.estimation import (
    ConversationalEstimationResponse,
    DetailLevel,
    OutputFormat,
    ProjectType,
)
from app.services.llm_service import (
    EstimationFailedError,
    estimate_conversational,
    estimate_with_acb,
)
from app.sessions.models import ProjectMetadata, Session
from app.sessions.store import SessionNotFoundError, SessionStore
from app.sessions.tier_resolver import Tier

router = APIRouter(tags=["sessions"])
logger = structlog.get_logger(__name__)

# Form(...) exige min_length=20 igual que EstimationRequest.description, pero con un max_length
# holgado: con adjuntos transcritos el texto del turno puede crecer bastante más que 10000 chars
_TRANSCRIPT_MAX_LENGTH = 60_000


def _guardrail_violation_detail(exc: InputGuardrailViolation) -> dict:
    return {"reason": exc.reason, "message": exc.message}


@router.post("/sessions", status_code=201)
def create_session(store: SessionStore = Depends(get_session_store)) -> dict:
    session = store.create()
    return {"session_id": session.session_id}


class SessionInfoResponse(BaseModel):
    session_id: str
    message_count: int
    max_turns: int
    metadata: ProjectMetadata
    anchors_count: int
    summary_chars: int
    last_resolved_tier: str | None
    last_tier_rule: str | None


@router.get("/sessions/{session_id}", response_model=SessionInfoResponse)
def get_session(session_id: str, store: SessionStore = Depends(get_session_store)) -> SessionInfoResponse:
    try:
        session = store.get_or_404(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return SessionInfoResponse(
        session_id=session.session_id,
        message_count=len(session.history.messages),
        max_turns=session.history.max_turns,
        metadata=session.metadata,
        anchors_count=len(session.history.anchors),
        summary_chars=len(session.history.summary or ""),
        last_resolved_tier=session.last_resolved_tier,
        last_tier_rule=session.last_tier_rule,
    )


async def _load_session_and_attachments(
    session_id: str, attachments: list[UploadFile], store: SessionStore
) -> tuple[Session, list[tuple[str, str]]]:
    """Prelude común de /estimate y /estimate-acb: sesión (404) y texto de los adjuntos (415/422)."""
    try:
        session = store.get_or_404(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc

    settings = get_settings()
    extracted_attachments: list[tuple[str, str]] = []
    for upload in attachments:
        content = await upload.read()
        try:
            text = extract_text(filename=upload.filename or "", content=content, max_chars=settings.MAX_ATTACHMENT_CHARS)
        except UnsupportedAttachmentError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except AttachmentExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        extracted_attachments.append((upload.filename or "", text))
    return session, extracted_attachments


@router.post("/sessions/{session_id}/estimate", response_model=ConversationalEstimationResponse)
async def create_session_estimation(
    session_id: str,
    transcript: str = Form(..., min_length=20, max_length=_TRANSCRIPT_MAX_LENGTH),
    project_type: ProjectType = Form(...),
    detail_level: DetailLevel = Form(...),
    output_format: OutputFormat = Form(...),
    # Según los apuntes, en producción el tier debería venir de un canal autenticado, no del
    # cliente; aquí se acepta como override para demos y evals
    tier: Tier | None = Form(default=None),
    attachments: list[UploadFile] = File(default_factory=list),
    store: SessionStore = Depends(get_session_store),
) -> ConversationalEstimationResponse:
    session, extracted_attachments = await _load_session_and_attachments(session_id, attachments, store)

    try:
        result, meta = estimate_conversational(
            session=session,
            transcript=transcript,
            attachments=extracted_attachments,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier=tier,
        )
    except InputGuardrailViolation as exc:
        raise HTTPException(status_code=400, detail=_guardrail_violation_detail(exc)) from exc
    except EstimationFailedError as exc:
        raise HTTPException(status_code=502, detail="The LLM did not return a valid estimation") from exc

    return ConversationalEstimationResponse(
        result=result,
        prompt_version=get_settings().CONVERSATIONAL_PROMPT_VERSION,
        cached=False,
        **meta.model_dump(),
    )


@router.post("/sessions/{session_id}/estimate-acb", response_model=ACBResponse)
async def create_session_estimation_acb(
    session_id: str,
    transcript: str = Form(..., min_length=20, max_length=_TRANSCRIPT_MAX_LENGTH),
    project_type: ProjectType = Form(...),
    detail_level: DetailLevel = Form(...),
    output_format: OutputFormat = Form(...),
    tier: Tier | None = Form(default=None),  # override de demo/evals, ver /estimate
    attachments: list[UploadFile] = File(default_factory=list),
    store: SessionStore = Depends(get_session_store),
) -> ACBResponse:
    """Igual que /estimate, pero el resultado pasa por la revisión Actor-Critic-Boss.

    Misma respuesta más `acb` (traza de iteraciones). Es opcional y por endpoint porque cuesta
    hasta 2 * BOSS_MAX_ITERATIONS llamadas al LLM por turno.
    """
    session, extracted_attachments = await _load_session_and_attachments(session_id, attachments, store)

    try:
        result, trace, meta = estimate_with_acb(
            session=session,
            transcript=transcript,
            attachments=extracted_attachments,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier=tier,
        )
    except InputGuardrailViolation as exc:
        raise HTTPException(status_code=400, detail=_guardrail_violation_detail(exc)) from exc
    except EstimationFailedError as exc:
        raise HTTPException(status_code=502, detail="The LLM did not return a valid estimation") from exc

    return ACBResponse(
        result=result,
        prompt_version=get_settings().CONVERSATIONAL_PROMPT_VERSION,
        cached=False,
        acb=trace,
        **meta.model_dump(),
    )