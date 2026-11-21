import re
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)

GuardrailReason = Literal["moderation", "prompt_injection", "pii"]


class InputGuardrailViolation(Exception):
    """El input no superó una capa del guardrail: nunca llegamos a llamar al LLM con él."""

    def __init__(self, reason: GuardrailReason, message: str) -> None:
        self.reason = reason
        self.message = message
        super().__init__(message)


# Heurísticas habituales de prompt injection; no son infalibles, pero cubren los intentos más comunes
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any )?previous instructions", re.IGNORECASE),
    re.compile(r"disregard (all |any )?(previous|prior|the above)", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"forget everything", re.IGNORECASE),
    re.compile(r"new instructions\s*:", re.IGNORECASE),
    re.compile(r"</?\s*(system|instructions)\s*>", re.IGNORECASE),
]

# Conservadoras a propósito: mejor dejar pasar algún PII real que bloquear descripciones legítimas de proyecto
_PII_PATTERNS = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE),  # email
    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),  # IBAN
    re.compile(r"\+\d{7,15}\b"),  # teléfono en formato internacional: +34612345678
    re.compile(r"\b\d{2,3}[-.\s]\d{3}[-.\s]\d{3,4}\b"),  # teléfono con separadores: 612 345 678
]


def _check_moderation(description: str, openai_client) -> None:
    if openai_client is None:
        # Sin cliente (tests de las capas de regex, o entorno sin credenciales): se salta esta capa
        return

    try:
        response = openai_client.moderations.create(input=description)
    except Exception as exc:
        # Fail closed: si no podemos verificar el input, no llamamos al LLM con él
        logger.warning("moderation_call_failed", error_type=type(exc).__name__, error=str(exc))
        raise InputGuardrailViolation(
            reason="moderation",
            message="Could not verify input safety (moderation service unavailable); request blocked",
        ) from exc

    result = response.results[0]
    if result.flagged:
        flagged_categories = [name for name, value in result.categories.model_dump().items() if value]
        logger.warning("moderation_flagged", categories=flagged_categories)
        raise InputGuardrailViolation(reason="moderation", message="Input flagged by content moderation")


def _check_prompt_injection(description: str) -> None:
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(description):
            logger.warning("prompt_injection_detected", pattern=pattern.pattern)
            raise InputGuardrailViolation(
                reason="prompt_injection", message="Input looks like a prompt injection attempt"
            )


def _check_pii(description: str) -> None:
    for pattern in _PII_PATTERNS:
        if pattern.search(description):
            logger.warning("pii_detected")
            raise InputGuardrailViolation(reason="pii", message="Input appears to contain personal data (PII)")


def check_input(description: str, *, openai_client=None) -> None:
    """Valida el input antes de llamar al LLM. Lanza InputGuardrailViolation si alguna capa falla."""
    _check_moderation(description, openai_client)
    _check_prompt_injection(description)
    _check_pii(description)
