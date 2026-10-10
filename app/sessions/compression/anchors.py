"""Detección de anclas: qué turnos deben sobrevivir a la expulsión de la ventana.

Un turno es ancla si lleva un compromiso duradero (NDA, contrato firmado, alcance congelado,
presupuesto cerrado, compliance...). El detector es deliberadamente conservador: un falso negativo
solo manda el turno al resumen, pero un falso positivo engorda el prompt para siempre porque las
anclas nunca se expulsan.

Dos estrategias: "heuristic" (regex, sin coste, determinista; por defecto) y "llm" (clasificador
binario, robusto a paráfrasis pero con una llamada por turno expulsado).
"""

import re
from dataclasses import dataclass, field
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.llm_wrapper import call_structured
from app.sessions.models import Message

logger = structlog.get_logger(__name__)

# Lista corta a propósito: preferimos perder una ancla a sobre-anclar. Cada regla lleva su variante en español
_HEURISTIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "nda",
        re.compile(
            r"\b(nda|non[- ]?disclosure|under embargo|legal hold"
            r"|acuerdo de confidencialidad|cl[aá]usula de confidencialidad)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "signed_contract",
        re.compile(
            r"\b(signed|countersigned)\s+(the\s+)?(contract|sow|msa|agreement)\b"
            r"|\b(contrato|acuerdo|sow|msa)\s+(ya\s+)?(est[aá]\s+)?firmado\b"
            r"|\bfirm(amos|aron|ado)\s+(ya\s+)?(el|un)\s+(contrato|acuerdo)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "scope_frozen",
        re.compile(
            r"\b(scope|backlog)\s+(is\s+)?(frozen|locked|final|fixed)\b"
            r"|\b(alcance|backlog)\s+(est[aá]\s+)?(congelado|cerrado|bloqueado|fijo|definitivo)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "budget_locked",
        re.compile(
            r"\b(budget|cap|ceiling)\s+(is\s+)?(locked|fixed|approved|capped)\s+at\b"
            r"|\bpresupuesto\s+(est[aá]\s+)?(cerrado|fijo|aprobado|limitado)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "compliance",
        re.compile(r"\b(hipaa|gdpr|rgpd|sox|pci[- ]?dss|fda|iso[- ]?27001)\b", re.IGNORECASE),
    ),
    (
        "deadline_hard",
        re.compile(
            r"\bhard\s+deadline\b|\bmust\s+go\s+live\s+by\b"
            r"|\bfecha\s+l[ií]mite\s+(innegociable|inamovible|estricta)\b"
            r"|\bdeadline\s+(innegociable|inamovible)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "contractual",
        re.compile(
            r"\bcontractually\s+(bound|required|obliged)\b"
            r"|\bobligad[oa]s?\s+(por|según)\s+contrato\b"
            r"|\bobligaci[oó]n\s+contractual\b",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit_commitment",
        re.compile(
            r"\b(we|the (client|customer))\s+(agreed|committed)\s+to\b"
            r"|\b(hemos|nos hemos)\s+(acordado|comprometido)\b"
            r"|\bel\s+cliente\s+(ha\s+)?(acord[oó]|se\s+compromet[ií]o|se\s+ha\s+comprometido)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass
class AnchorMatch:
    is_anchor: bool
    # Reglas que dispararon: permiten atribuir la decisión en el log `history_compressed`
    matched_rules: list[str] = field(default_factory=list)


class AnchorClassification(BaseModel):
    is_anchor: bool = Field(
        description=(
            "True if the user turn introduces a durable commitment, legal or compliance constraint, "
            "frozen scope or locked budget that the conversation must not forget."
        )
    )
    reason: str = Field(default="", max_length=200)


_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a binary classifier. Decide whether the user turn introduces a durable commitment about "
    "the project: a signed agreement, frozen scope, locked budget, hard deadline or legal/compliance "
    "context (NDA, HIPAA, GDPR...). Answer is_anchor=true only if forgetting the turn would harm the "
    "conversation; otherwise false. The turn may be in any language. Give a one-sentence reason."
)


class AnchorDetector:
    """Inspecciona solo el turno del user: la respuesta del assistant es función de él."""

    def __init__(self, *, mode: Literal["heuristic", "llm"] = "heuristic", model: str | None = None) -> None:
        self.mode = mode
        self.model = model or get_settings().COMPRESSION_MODEL

    def detect(self, message: Message) -> AnchorMatch:
        return self._detect_llm(message) if self.mode == "llm" else self._detect_heuristic(message)

    @staticmethod
    def _detect_heuristic(message: Message) -> AnchorMatch:
        matched = [name for name, pattern in _HEURISTIC_PATTERNS if pattern.search(message.content)]
        return AnchorMatch(is_anchor=bool(matched), matched_rules=matched)

    def _detect_llm(self, message: Message) -> AnchorMatch:
        try:
            classification, _meta = call_structured(
                [
                    {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"User turn:\n{message.content}"},
                ],
                AnchorClassification,
                self.model,
                max_retries=1,
                purpose="anchor_classification",
            )
        except Exception as exc:  # noqa: BLE001 - un fallo del clasificador no debe tumbar el turno
            # Se trata como no-ancla: lo peor es que el turno acabe resumido en vez de literal
            logger.warning("anchor_llm_failed", error_type=type(exc).__name__, error=str(exc))
            return AnchorMatch(is_anchor=False)
        if classification.is_anchor:
            return AnchorMatch(is_anchor=True, matched_rules=["llm_classifier"])
        return AnchorMatch(is_anchor=False)
