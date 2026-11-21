"""Tests for ConsistentPseudonymizer.

The real Presidio analyzer (spaCy ``es_core_news_md``) is NOT used here: loading it is slow and
needs a separately installed model. A stub analyzer emits canned results so the substitution logic,
the consistency contract and the mapping-store integration are exercised in isolation.
"""

from dataclasses import dataclass

import pytest

from app.ingestion.pii import ConsistentPseudonymizer, InMemoryMappingStore

pytestmark = pytest.mark.anyio


@dataclass
class _StubResult:
    """Minimal stand-in for presidio_analyzer.RecognizerResult."""

    entity_type: str
    start: int
    end: int
    score: float = 0.95


class StubAnalyzer:
    """Returns scripted results per input text."""

    def __init__(self, results_by_text: dict[str, list[_StubResult]]) -> None:
        self._results = results_by_text

    def analyze(self, *, text: str, language: str, entities: list[str] | None = None):
        return list(self._results.get(text, []))


def _span(text: str, value: str) -> tuple[int, int]:
    start = text.index(value)
    return start, start + len(value)


def _pseudonymizer(analyzer: StubAnalyzer, store: InMemoryMappingStore | None = None):
    return ConsistentPseudonymizer(
        analyzer=analyzer,  # type: ignore[arg-type]
        mapping_store=store or InMemoryMappingStore(),
        salt="unit-test-salt",
        faker_locale="es_ES",
        language="es",
    )


async def test_consistency_same_value_same_pseudonym_across_calls() -> None:
    text1 = "Habló con Laura Fernández ayer."
    text2 = "Laura Fernández firmó esta mañana."
    analyzer = StubAnalyzer(
        {
            text1: [_StubResult("PERSON", *_span(text1, "Laura Fernández"))],
            text2: [_StubResult("PERSON", *_span(text2, "Laura Fernández"))],
        }
    )
    pseudonymizer = _pseudonymizer(analyzer, InMemoryMappingStore())
    first = await pseudonymizer.pseudonymize(text1)
    second = await pseudonymizer.pseudonymize(text2)
    # Both invocations use the SAME pseudonym for "Laura Fernández"
    assert first.applied[0].pseudonym == second.applied[0].pseudonym
    assert "Laura Fernández" not in first.pseudonymized_text
    assert "Laura Fernández" not in second.pseudonymized_text


async def test_different_values_get_different_pseudonyms() -> None:
    text = "Laura Fernández y Javier Romero firmaron."
    analyzer = StubAnalyzer(
        {
            text: [
                _StubResult("PERSON", *_span(text, "Laura Fernández")),
                _StubResult("PERSON", *_span(text, "Javier Romero")),
            ]
        }
    )
    result = await _pseudonymizer(analyzer).pseudonymize(text)
    assert len({m.pseudonym for m in result.applied}) == 2


async def test_budget_id_gets_a_budget_shaped_pseudonym() -> None:
    text = "Revisando BUDGET-2024-0001 con el cliente."
    analyzer = StubAnalyzer({text: [_StubResult("BUDGET_ID", *_span(text, "BUDGET-2024-0001"))]})
    result = await _pseudonymizer(analyzer).pseudonymize(text)
    assert result.applied[0].entity_type == "BUDGET_ID"
    assert result.applied[0].pseudonym.startswith("BUDGET-")
    assert "BUDGET-2024-0001" not in result.pseudonymized_text


async def test_adjacent_entities_replaced_right_to_left() -> None:
    """Sorting by end offset descending keeps the earlier offsets valid."""
    text = "Carmen escribió a carmen.vidal@toledo-ind.es y firmó."
    analyzer = StubAnalyzer(
        {
            text: [
                _StubResult("PERSON", *_span(text, "Carmen")),
                _StubResult("EMAIL_ADDRESS", *_span(text, "carmen.vidal@toledo-ind.es")),
            ]
        }
    )
    result = await _pseudonymizer(analyzer).pseudonymize(text)
    assert "Carmen" not in result.pseudonymized_text.split()  # the whole word was replaced
    assert "carmen.vidal@toledo-ind.es" not in result.pseudonymized_text
    assert result.pseudonymized_text.endswith(" y firmó.")


async def test_only_the_salted_hash_reaches_the_store() -> None:
    text = "Habló Laura Fernández."
    analyzer = StubAnalyzer({text: [_StubResult("PERSON", *_span(text, "Laura Fernández"))]})
    store = InMemoryMappingStore()
    await _pseudonymizer(analyzer, store).pseudonymize(text)

    ((entity_type, original_hash),) = store._mappings
    assert entity_type == "PERSON"
    assert "Laura" not in original_hash
    assert len(original_hash) == 64  # hex HMAC-SHA256


async def test_forget_reports_whether_a_mapping_existed() -> None:
    text = "Habló Laura Fernández."
    analyzer = StubAnalyzer({text: [_StubResult("PERSON", *_span(text, "Laura Fernández"))]})
    store = InMemoryMappingStore()
    pseudonymizer = _pseudonymizer(analyzer, store)
    first = await pseudonymizer.pseudonymize(text)
    assert await store.forget("PERSON", first.applied[0].original_hash) is True
    assert await store.forget("PERSON", first.applied[0].original_hash) is False


async def test_no_pii_means_no_changes() -> None:
    text = "Una frase sin nada interesante."
    result = await _pseudonymizer(StubAnalyzer({text: []})).pseudonymize(text)
    assert result.pseudonymized_text == text
    assert result.applied == []
