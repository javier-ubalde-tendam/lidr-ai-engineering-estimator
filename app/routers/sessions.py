import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.attachments.extractor import (
    AttachmentExtractionError,
    UnsupportedAttachmentError,
    extract_text,
)
from app.config import get_settings
from app.dependencies import get_session_store
from app.guardrails.input import InputGuardrailViolation
from app.schemas.estimation import DetailLevel, EstimationResponse, OutputFormat, ProjectType
from app.services.llm_service import EstimationFailedError, estimate_conversational
from app.sessions.store import SessionNotFoundError, SessionStore

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


@router.get("/sessions/{session_id}")
def get_session(session_id: str, store: SessionStore = Depends(get_session_store)) -> dict:
    try:
        session = store.get_or_404(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return {
        "session_id": session.session_id,
        "message_count": len(session.history.messages),
        "max_turns": session.history.max_turns,
        "metadata": session.metadata.model_dump(),
    }


@router.post("/sessions/{session_id}/estimate", response_model=EstimationResponse)
async def create_session_estimation(
    session_id: str,
    transcript: str = Form(..., min_length=20, max_length=_TRANSCRIPT_MAX_LENGTH),
    project_type: ProjectType = Form(...),
    detail_level: DetailLevel = Form(...),
    output_format: OutputFormat = Form(...),
    attachments: list[UploadFile] = File(default_factory=list),
    store: SessionStore = Depends(get_session_store),
) -> EstimationResponse:
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

    try:
        result = estimate_conversational(
            session=session,
            transcript=transcript,
            attachments=extracted_attachments,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
    except InputGuardrailViolation as exc:
        raise HTTPException(status_code=400, detail=_guardrail_violation_detail(exc)) from exc
    except EstimationFailedError as exc:
        raise HTTPException(status_code=502, detail="The LLM did not return a valid estimation") from exc

    return EstimationResponse(
        result=result, prompt_version=settings.CONVERSATIONAL_PROMPT_VERSION, cached=False
    )
