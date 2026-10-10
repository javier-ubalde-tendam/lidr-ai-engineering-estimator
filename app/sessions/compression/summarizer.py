"""Resumen acumulativo de los turnos expulsados de la ventana.

Un único resumen por sesión: cada pasada integra el resumen previo con los turnos recién
expulsados y lo sustituye. Si el LLM falla se conserva el resumen previo: es preferible perder
una pasada de compresión a borrar lo que el modelo ya había asumido.
"""

import structlog
from pydantic import BaseModel, Field

from app.config import get_settings
from app.prompts.loader import render_conversation_summary_prompt
from app.services.llm_wrapper import call_structured
from app.sessions.models import Message

logger = structlog.get_logger(__name__)


class SummaryEnvelope(BaseModel):
    # Sobre mínimo para que Instructor devuelva un único campo de texto
    summary: str = Field(min_length=1, max_length=4000)


class CumulativeSummarizer:
    def __init__(self, *, model: str | None = None) -> None:
        self.model = model or get_settings().COMPRESSION_MODEL

    def summarize(self, *, previous_summary: str | None, evicted: list[Message]) -> str | None:
        if not evicted:
            return previous_summary

        system_prompt, user_prompt = render_conversation_summary_prompt(
            previous_summary=previous_summary, evicted=evicted
        )
        try:
            envelope, _meta = call_structured(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                SummaryEnvelope,
                self.model,
                max_retries=1,
                purpose="summary",
            )
        except Exception as exc:  # noqa: BLE001 - la compresión es best-effort: la conversación sigue
            logger.warning("summary_failed", error_type=type(exc).__name__, error=str(exc))
            return previous_summary
        return envelope.summary
