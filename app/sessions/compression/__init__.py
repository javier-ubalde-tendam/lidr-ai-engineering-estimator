"""Memoria híbrida para conversaciones largas: anclas literales + resumen acumulativo + ventana reciente."""

from app.sessions.compression.anchors import AnchorDetector, AnchorMatch
from app.sessions.compression.policy import CompressionPolicy, apply_compression
from app.sessions.compression.summarizer import CumulativeSummarizer

__all__ = [
    "AnchorDetector",
    "AnchorMatch",
    "CompressionPolicy",
    "CumulativeSummarizer",
    "apply_compression",
]
