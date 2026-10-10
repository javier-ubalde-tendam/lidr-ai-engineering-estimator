"""Política de compresión: qué se conserva literal, qué se resume y qué se descarta.

Es la única pieza que recorta el historial (append() ya no lo hace). Se ejecuta tras cada turno:
mientras la ventana supera `max_turns * 2` mensajes saca el par más antiguo; si el user es ancla
el par pasa a `anchors`, si no se encola para el resumen. Al final hace UNA sola llamada al
summarizer con todo lo encolado. Es idempotente: sin cambios en la ventana no hace nada.
"""

from typing import Literal

import structlog

from app.config import get_settings
from app.sessions.compression.anchors import AnchorDetector
from app.sessions.compression.summarizer import CumulativeSummarizer
from app.sessions.models import ConversationHistory, Message

logger = structlog.get_logger(__name__)


class CompressionPolicy:
    def __init__(self, *, anchor_detector: AnchorDetector, summarizer: CumulativeSummarizer) -> None:
        self.anchor_detector = anchor_detector
        self.summarizer = summarizer

    def apply(self, history: ConversationHistory) -> None:
        limit = history.max_turns * 2
        evicted_to_summary: list[Message] = []
        anchor_rules: list[str] = []
        promoted_anchors = 0

        while len(history.messages) > limit and len(history.messages) >= 2:
            # Con mensajes en pareja y alternados, los dos primeros son el par user/assistant más antiguo
            user_message, assistant_message = history.messages[0], history.messages[1]
            del history.messages[:2]

            match = self.anchor_detector.detect(user_message)
            if match.is_anchor:
                history.anchors += [user_message, assistant_message]
                anchor_rules += match.matched_rules
                promoted_anchors += 1
            else:
                evicted_to_summary += [user_message, assistant_message]

        if promoted_anchors == 0 and not evicted_to_summary:
            return

        if evicted_to_summary:
            history.summary = self.summarizer.summarize(
                previous_summary=history.summary, evicted=evicted_to_summary
            )

        logger.info(
            "history_compressed",
            promoted_anchors=promoted_anchors,
            anchor_rules=anchor_rules,
            evicted_to_summary=len(evicted_to_summary),
            summary_chars=len(history.summary or ""),
            anchors_count=len(history.anchors),
            recent_messages=len(history.messages),
        )


def apply_compression(
    history: ConversationHistory,
    *,
    anchor_detection_mode: Literal["heuristic", "llm"] | None = None,
    compression_model: str | None = None,
) -> None:
    """Monta el detector y el summarizer con la config y ejecuta la política una vez."""
    settings = get_settings()
    model = compression_model or settings.COMPRESSION_MODEL
    mode = anchor_detection_mode or settings.ANCHOR_DETECTION_MODE
    policy = CompressionPolicy(
        anchor_detector=AnchorDetector(mode=mode, model=model),
        summarizer=CumulativeSummarizer(model=model),
    )
    policy.apply(history)
