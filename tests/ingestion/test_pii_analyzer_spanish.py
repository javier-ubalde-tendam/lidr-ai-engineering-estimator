"""Presidio analyzer wired to the real Spanish spaCy model.

Skipped when the model is not installed (it is a separate wheel, not a PyPI dependency): every
other PII test runs with a stub analyzer. Run them for real after installing the model, see the
README.
"""

import importlib.util

import pytest

from app.config import get_settings

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec(get_settings().PRESIDIO_SPACY_MODEL) is None,
    reason="spaCy Spanish model is not installed",
)


def test_spanish_names_and_project_identifiers_are_detected() -> None:
    from app.ingestion.pii import build_analyzer

    text = "Laura Fernández revisó BUDGET-2024-0001 del cliente CLI-0042."
    found = {
        (r.entity_type, text[r.start : r.end])
        for r in build_analyzer().analyze(text=text, language="es")
    }
    assert ("BUDGET_ID", "BUDGET-2024-0001") in found
    assert ("CLIENT_CODE", "CLI-0042") in found
    assert ("PERSON", "Laura Fernández") in found
