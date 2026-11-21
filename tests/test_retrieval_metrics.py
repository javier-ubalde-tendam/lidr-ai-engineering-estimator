import json
from pathlib import Path

import pytest

from evals.retrieval_metrics import percentile, precision_at_k

ROOT = Path(__file__).resolve().parents[1]


def test_precision_counts_relevant_chunks_in_top_k() -> None:
    retrieved = ["A", "B", "A", "C", "D", "A"]

    assert precision_at_k(retrieved, {"A"}, k=5) == pytest.approx(2 / 5)


def test_precision_divides_by_k_even_with_fewer_results() -> None:
    assert precision_at_k(["A", "A"], {"A"}, k=5) == pytest.approx(2 / 5)
    assert precision_at_k([], {"A"}, k=5) == 0.0


def test_precision_ignores_chunks_without_budget_id() -> None:
    assert precision_at_k([None, "A"], {"A"}, k=2) == pytest.approx(0.5)


def test_precision_perfect_and_invalid_k() -> None:
    assert precision_at_k(["A", "B"] * 3, {"A", "B"}, k=5) == 1.0
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, k=0)


def test_percentile_nearest_rank() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]

    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == 5.0
    assert percentile(values, 100) == 5.0
    assert percentile([7.0], 50) == 7.0


@pytest.mark.parametrize("pct", [0, -1, 101])
def test_percentile_invalid(pct: float) -> None:
    with pytest.raises(ValueError):
        percentile([1.0], pct)
    with pytest.raises(ValueError):
        percentile([], 50)


def test_golden_set_shape_and_ids_exist_in_the_corpus() -> None:
    golden = json.loads((ROOT / "evals" / "golden_retrieval.json").read_text(encoding="utf-8"))
    corpus = json.loads((ROOT / "data" / "budgets_sample.json").read_text(encoding="utf-8"))
    budget_ids = {budget["budget_id"] for budget in corpus}
    sectors = {budget["client_metadata"]["sector"] for budget in corpus}

    queries = golden["queries"]
    assert len(queries) == 5
    assert len({q["id"] for q in queries}) == 5
    for query in queries:
        assert query["relevant_budget_ids"]
        assert set(query["relevant_budget_ids"]) <= budget_ids
        assert query["note"]
    # Every sector of the corpus is covered by at least one query
    assert {q["sector"] for q in queries} == sectors
