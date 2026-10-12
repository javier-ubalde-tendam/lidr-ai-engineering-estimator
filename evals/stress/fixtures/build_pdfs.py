"""Genera PDFs sintéticos con un tamaño de texto extraído conocido para el stress test del CAG.

    uv run python -m evals.stress.fixtures.build_pdfs
    uv run python -m evals.stress.fixtures.build_pdfs --sizes 5,20 --output-dir /tmp/pdfs

Los PDFs se escriben en `evals/stress/fixtures/generated/` (ignorado por git). Son reproducibles:
el texto sale de plantillas fijas sin aleatoriedad. Tras generarlos se mide con el MISMO
extractor que usa la app (`app.attachments.extractor.extract_text`) cuántos caracteres salen y
cuántos sobrevivirían a `MAX_ATTACHMENT_CHARS`. Necesita fpdf2 (extra `dev`: `uv sync --extra dev`).
"""

import argparse
from pathlib import Path

from pydantic import ValidationError

from app.attachments.extractor import extract_text

FIXTURES_DIR = Path(__file__).parent / "generated"
DEFAULT_SIZES_KB = (5, 20, 50, 100)
CHARS_PER_KB = 1000  # 1 KB = 1000 caracteres extraídos: tamaños redondos, no bytes del PDF
_TOLERANCE = 0.02
_NO_LIMIT = 10**9

# Plantillas de requisitos de un proyecto ficticio; se combinan por índice, sin azar
_SUBJECTS = (
    "The booking module",
    "The reporting service",
    "The notification engine",
    "The billing component",
    "The administration console",
    "The identity layer",
    "The search feature",
    "The import pipeline",
)
_BEHAVIOURS = (
    "must record every change together with the author and the time of the change",
    "should respond to common requests in under two seconds for typical data volumes",
    "needs to support several languages and regional date and currency formats",
    "has to keep working when a dependent service is temporarily unavailable",
    "must expose its data through a documented interface for other internal tools",
    "should let operators review and correct the data without developer support",
    "needs a clear error message for every validation rule that rejects an input",
    "has to be covered by automated tests before each release is published",
)
_NOTES = (
    "Operations will review this item during the quarterly planning meeting.",
    "The product owner asked to keep the first release as small as possible.",
    "This requirement was refined after the usability sessions with pilot users.",
    "Open question: whether the same rule applies to every customer segment.",
)


def build_document_text(target_chars: int) -> str:
    """Texto ASCII determinista de exactamente `target_chars` caracteres (sin PII ni patrones de guardrail)."""
    paragraphs: list[str] = []
    total = 0
    index = 0
    while total < target_chars:
        subject = _SUBJECTS[index % len(_SUBJECTS)]
        behaviour = _BEHAVIOURS[(index // len(_SUBJECTS)) % len(_BEHAVIOURS)]
        note = _NOTES[index % len(_NOTES)]
        paragraph = f"Requirement REQ-{index + 1:04d}. {subject} {behaviour}. {note}"
        paragraphs.append(paragraph)
        total += len(paragraph) + 1
        index += 1
    return "\n".join(paragraphs)[:target_chars]


def _render_pdf(text: str) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise SystemExit("fpdf2 is required to build the fixtures: run `uv sync --extra dev`") from exc

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for paragraph in text.split("\n"):
        pdf.multi_cell(0, 5, paragraph, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def extracted_chars(pdf_bytes: bytes, max_chars: int = _NO_LIMIT) -> int:
    return len(extract_text(filename="fixture.pdf", content=pdf_bytes, max_chars=max_chars))


def build_pdf(size_kb: int) -> tuple[bytes, int]:
    """PDF cuyo texto extraído mide ~size_kb * 1000 chars (±2%); devuelve (bytes, chars reales)."""
    target = size_kb * CHARS_PER_KB
    requested = target
    for _ in range(4):
        pdf_bytes = _render_pdf(build_document_text(requested))
        actual = extracted_chars(pdf_bytes)
        if abs(actual - target) <= target * _TOLERANCE:
            break
        # El extractor añade/quita saltos de línea: se corrige proporcionalmente y se reintenta
        requested = max(1, round(requested * target / max(actual, 1)))
    return pdf_bytes, actual


def fixture_path(size_kb: int, directory: Path | None = None) -> Path:
    return (directory or FIXTURES_DIR) / f"attachment_{size_kb}kb.pdf"


def _max_attachment_chars() -> int:
    from app.config import Settings, get_settings

    try:
        return get_settings().MAX_ATTACHMENT_CHARS
    except ValidationError:  # sin API key configurada: se usa el default de la clase
        return int(Settings.model_fields["MAX_ATTACHMENT_CHARS"].default)


def _parse_sizes(raw: str) -> list[int]:
    try:
        sizes = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"sizes must be integers separated by commas, got {raw!r}") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive (0 KB means 'no attachment', no PDF needed)")
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera los PDFs sintéticos del stress test del CAG.")
    parser.add_argument("--sizes", type=_parse_sizes, default=list(DEFAULT_SIZES_KB), help="KB de texto, p. ej. 5,20,50,100")
    parser.add_argument("--output-dir", type=Path, default=FIXTURES_DIR)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    limit = _max_attachment_chars()
    print(f"MAX_ATTACHMENT_CHARS = {limit}")
    print(f"{'size_kb':>7} {'pdf_bytes':>10} {'extracted':>10} {'kept_by_app':>12} {'truncated':>9}  file")
    for size_kb in args.sizes:
        pdf_bytes, raw_chars = build_pdf(size_kb)
        path = fixture_path(size_kb, args.output_dir)
        path.write_bytes(pdf_bytes)
        # Segunda extracción con el límite real de la app: lo que de verdad llegaría al prompt
        kept = extracted_chars(pdf_bytes, max_chars=limit)
        print(f"{size_kb:>7} {len(pdf_bytes):>10} {raw_chars:>10} {kept:>12} {kept < raw_chars!s:>9}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
