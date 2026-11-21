"""Tests for the reparable cleaning + Pandera validation policy over the seed budgets."""

import json
from pathlib import Path

from app.ingestion.cleaning import clean_budget_records, validate_with_policy

SEED_BUDGETS = Path(__file__).resolve().parents[2] / "data" / "seed" / "budgets"


def _record(budget_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "budget_id": budget_id,
        "client_name": "Acme",
        "client_code": "CLI-0001",
        "currency": "EUR",
        "total_amount": 1000.0,
        "signed_at": "2024-03-01",
    }
    return base | overrides


def test_valid_rows_pass_untouched() -> None:
    result = validate_with_policy(clean_budget_records([_record("BUDGET-2024-0001")]))
    assert result.report["valid_rows"] == 1
    assert result.report["discarded_rows"] == 0
    assert result.report["quarantined_rows"] == 0


def test_negative_amount_is_discarded_and_unparseable_amount_is_quarantined() -> None:
    df = clean_budget_records(
        [
            _record("BUDGET-2024-0001"),
            _record("BUDGET-2024-0002", total_amount=-50000.0),
            _record("BUDGET-2024-0003", total_amount="not-a-number"),
        ]
    )
    result = validate_with_policy(df)
    assert list(result.valid["budget_id"]) == ["BUDGET-2024-0001"]
    assert list(result.discarded["budget_id"]) == ["BUDGET-2024-0002"]
    assert list(result.quarantined["budget_id"]) == ["BUDGET-2024-0003"]


def test_empty_input_yields_an_empty_report() -> None:
    result = validate_with_policy(clean_budget_records([]))
    assert result.report == {"input_rows": 0}


def test_seed_corpus_partition_matches_the_documented_story() -> None:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SEED_BUDGETS.glob("*.json"))]
    df = clean_budget_records(records)
    assert len(records) == 6
    assert len(df) == 5  # the two versions of BUDGET-2024-0005 collapse into one
    result = validate_with_policy(df)
    assert list(result.discarded["budget_id"]) == ["BUDGET-2024-0004"]
    assert result.report["valid_rows"] == 4
