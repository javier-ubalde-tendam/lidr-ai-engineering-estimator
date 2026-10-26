from collections.abc import Sequence

from app.config import EMBEDDING_DIMENSION


class DocumentAlreadyIngestedError(Exception):
    def __init__(self, document_id: int) -> None:
        super().__init__(f"Document already ingested: {document_id}")
        self.document_id = document_id


class InvalidEmbeddingError(Exception):
    """El proveedor devolvió vectores que no encajan con el esquema vector(1536)."""


def ensure_valid_vectors(vectors: Sequence[Sequence[float]], expected_count: int) -> None:
    if len(vectors) != expected_count:
        raise InvalidEmbeddingError(
            f"Se esperaban {expected_count} vectores y llegaron {len(vectors)}"
        )
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSION:
            raise InvalidEmbeddingError(
                f"Dimensión {len(vector)} distinta de {EMBEDDING_DIMENSION}"
            )
