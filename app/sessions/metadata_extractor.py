import structlog

from app.config import get_settings
from app.prompts.loader import render_metadata_extraction_prompt
from app.schemas.estimation import EstimationResult
from app.services.llm_wrapper import call_structured
from app.sessions.models import ProjectMetadata

logger = structlog.get_logger(__name__)


def update_metadata(
    previous: ProjectMetadata,
    transcript: str,
    result: EstimationResult,
    *,
    session_id: str | None = None,
) -> ProjectMetadata:
    """Segunda llamada (barata) con Instructor que extrae hechos del turno y los fusiona con lo ya sabido.

    Política de fallo: si la extracción falla por cualquier motivo (red, validación, proveedor),
    la conversación no debe caerse por ello. Se registra el fallo y se devuelve previous sin cambios.
    """
    settings = get_settings()
    system_prompt, user_prompt = render_metadata_extraction_prompt(
        previous_metadata=previous,
        transcript=transcript,
        estimation_summary=result.summary,
    )
    try:
        extracted, _meta = call_structured(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            ProjectMetadata,
            settings.METADATA_EXTRACTOR_MODEL,
            max_retries=settings.STRUCTURED_OUTPUT_MAX_RETRIES,
            purpose="metadata_extraction",
            session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001 - fallo intencionadamente amplio: nunca debe tumbar la conversación
        logger.warning("metadata_extraction_failed", error_type=type(exc).__name__, error=str(exc))
        return previous

    return previous.merge_with(extracted)
