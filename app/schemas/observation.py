from typing import Literal

from pydantic import BaseModel, Field

CacheHitKind = Literal["none", "exact", "semantic"]


class TurnObservation(BaseModel):
    """Qué le costó al CAG un turno conversacional (evento `turn_observed`).

    `latency_ms`, `tokens_in`, `tokens_out` y `cost_usd` son los de la llamada de estimación;
    no incluyen las llamadas auxiliares del turno (extractor de metadata, resumen, anclas LLM).
    """

    turn_index: int = Field(ge=1, description="1-based dentro de la sesión")
    session_id: str
    enriched_transcript_chars: int = Field(ge=0, description="Transcript del turno + texto de los adjuntos")
    attachments_total_chars: int = Field(ge=0)
    messages_in_window: int = Field(ge=0, description="Mensajes de la ventana tras comprimir")
    anchors_count: int = Field(ge=0, description="Mensajes ancla (user + assistant) tras comprimir")
    summary_chars: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    cache_hit_kind: CacheHitKind
    last_resolved_tier: str | None
