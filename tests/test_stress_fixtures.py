import pytest

from app.attachments.extractor import extract_text
from app.guardrails.input import check_input
from evals.stress.fixtures import build_pdfs


def test_the_document_text_is_deterministic_and_has_the_exact_length():
    first = build_pdfs.build_document_text(5000)

    assert first == build_pdfs.build_document_text(5000)
    assert len(first) == 5000
    assert build_pdfs.build_document_text(5000) != build_pdfs.build_document_text(6000)[:4000]


def test_the_document_text_does_not_trip_the_input_guardrails():
    # Si disparase PII o prompt injection, el stress test mediría el guardrail, no el CAG
    check_input(build_pdfs.build_document_text(60_000))


@pytest.mark.parametrize("size_kb", [1, 5])
def test_the_pdf_extracts_to_about_the_requested_size_with_the_app_extractor(size_kb):
    pdf_bytes, chars = build_pdfs.build_pdf(size_kb)

    target = size_kb * build_pdfs.CHARS_PER_KB
    assert abs(chars - target) <= target * 0.02
    # Mismo extractor que la app: la cifra devuelta es la que verá el prompt
    assert len(extract_text(filename="a.pdf", content=pdf_bytes, max_chars=10**9)) == chars


def test_the_pdf_build_is_reproducible_in_content():
    # Los bytes del PDF incluyen la fecha de creación; lo reproducible es el texto extraído
    first, _ = build_pdfs.build_pdf(1)
    second, _ = build_pdfs.build_pdf(1)

    def text(pdf_bytes):
        return extract_text(filename="a.pdf", content=pdf_bytes, max_chars=10**9)

    assert text(first) == text(second)


def test_max_attachment_chars_truncates_a_larger_extraction():
    pdf_bytes, chars = build_pdfs.build_pdf(100)

    kept = extract_text(filename="a.pdf", content=pdf_bytes, max_chars=60_000)

    assert chars == pytest.approx(100_000, rel=0.02)
    assert len(kept) == 60_000


def test_main_writes_the_files_and_reports_the_truncation(tmp_path, capsys):
    exit_code = build_pdfs.main(["--sizes", "1,70", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "attachment_1kb.pdf").exists()
    assert (tmp_path / "attachment_70kb.pdf").exists()
    lines = capsys.readouterr().out.splitlines()
    assert any("MAX_ATTACHMENT_CHARS = 60000" in line for line in lines)
    assert any(line.split()[:1] == ["70"] and line.split()[4] == "True" for line in lines if line.strip()[:1].isdigit())
    assert any(line.split()[:1] == ["1"] and line.split()[4] == "False" for line in lines if line.strip()[:1].isdigit())


@pytest.mark.parametrize("sizes", ["0", "-5", "abc", ""])
def test_main_rejects_invalid_sizes(sizes, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        build_pdfs.main(["--sizes", sizes, "--output-dir", str(tmp_path)])

    assert excinfo.value.code == 2


def test_fixture_path_uses_a_predictable_name():
    assert build_pdfs.fixture_path(20).name == "attachment_20kb.pdf"
    assert build_pdfs.fixture_path(20).parent == build_pdfs.FIXTURES_DIR
