import io

import structlog
from docx import Document
from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = structlog.get_logger(__name__)


class AttachmentExtractionError(Exception):
    """No se pudo extraer texto del adjunto (fichero corrupto, vacío, etc.)."""

    def __init__(self, filename: str, message: str) -> None:
        self.filename = filename
        self.message = message
        super().__init__(f"{filename}: {message}")


class UnsupportedAttachmentError(AttachmentExtractionError):
    """La extensión del adjunto no está soportada (hoy: .pdf y .docx)."""

    def __init__(self, filename: str) -> None:
        super().__init__(filename, f"unsupported attachment type for {filename!r}")


def _extract_pdf_text(filename: str, content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
    except PyPdfError as exc:
        raise AttachmentExtractionError(filename, "could not open the PDF") from exc

    pages_text: list[str] = []
    for page_number, page in enumerate(reader.pages):
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - una página rota no debe tirar abajo el documento entero
            logger.warning(
                "attachment_page_extraction_failed",
                filename=filename,
                page_number=page_number,
                error=str(exc),
            )
    return "\n".join(pages_text)


def _extract_docx_text(filename: str, content: bytes) -> str:
    try:
        document = Document(io.BytesIO(content))
    except Exception as exc:
        raise AttachmentExtractionError(filename, "could not open the Word document") from exc
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text(*, filename: str, content: bytes, max_chars: int) -> str:
    """Extrae el texto de un adjunto soportado (.pdf, .docx), truncado a max_chars."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix == "pdf":
        text = _extract_pdf_text(filename, content)
    elif suffix == "docx":
        text = _extract_docx_text(filename, content)
    else:
        raise UnsupportedAttachmentError(filename)

    if len(text) > max_chars:
        logger.info("attachment_truncated", filename=filename, original_chars=len(text), max_chars=max_chars)
        text = text[:max_chars]
    return text


def enrich_transcript(*, transcript: str, attachments: list[tuple[str, str]]) -> str:
    """Concatena el transcript con el texto de cada adjunto, delimitado para que el LLM
    pueda distinguir qué viene de la transcripción y qué viene de un fichero adjunto."""
    parts = [transcript]
    for filename, text in attachments:
        parts.append(f"--- attachment: {filename} ---\n{text}\n--- end attachment ---")
    return "\n\n".join(parts)
