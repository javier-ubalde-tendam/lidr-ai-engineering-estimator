"""Memoria híbrida para conversaciones largas: anclas literales + resumen acumulativo + ventana reciente."""

from app.generation.conversation.compression.anchors import AnchorDetector, AnchorMatch
from app.generation.conversation.compression.policy import CompressionPolicy, apply_compression
from app.generation.conversation.compression.summarizer import CumulativeSummarizer

__all__ = [
    "AnchorDetector",
    "AnchorMatch",
    "CompressionPolicy",
    "CumulativeSummarizer",
    "apply_compression",
]
