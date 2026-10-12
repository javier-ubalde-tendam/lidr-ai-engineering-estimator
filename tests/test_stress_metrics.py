import pytest

from evals.metrics import MetricResult
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric

SNAPSHOT = {
    "session_id": "s1",
    "summary": "The team chose PostgreSQL for the Northwind Booking backend.",
    "anchors": ["The budget is locked at 30000 EUR.", '{"summary": "assistant reply"}'],
    "metadata": {
        "project_name": "Northwind Booking",
        "assumed_team_size": 5,
        "mentioned_technologies": ["Stripe", "Auth0"],
        "agreed_scope": None,
    },
}


def test_metric_result_score_is_binary_and_derived_from_passed():
    assert MetricResult("m", True, "").score == 1.0
    assert MetricResult("m", False, "").score == 0.0


# --- LatencyBudgetMetric ---


@pytest.mark.parametrize(("latency", "passed"), [(100, True), (1000, True), (1001, False), (0, True)])
def test_latency_budget_is_inclusive(latency, passed):
    result = LatencyBudgetMetric(1000).evaluate({"latency_ms": latency})

    assert result.name == "latency_budget"
    assert result.passed is passed
    assert result.score == (1.0 if passed else 0.0)
    assert "budget_ms=1000" in result.details


@pytest.mark.parametrize("observation", [None, {}, {"latency_ms": None}, {"latency_ms": "fast"}, {"latency_ms": True}, "x"])
def test_latency_budget_fails_with_missing_or_invalid_data(observation):
    result = LatencyBudgetMetric(1000).evaluate(observation)

    assert not result.passed
    assert "missing or not a number" in result.details


def test_latency_budget_rejects_a_negative_budget():
    with pytest.raises(ValueError):
        LatencyBudgetMetric(-1)


# --- CostBudgetMetric ---


@pytest.mark.parametrize(("cost", "passed"), [(0.0, True), (0.005, True), (0.01, True), (0.0100001, False)])
def test_cost_budget_is_inclusive(cost, passed):
    result = CostBudgetMetric(0.01).evaluate({"cost_usd": cost})

    assert result.name == "cost_budget"
    assert result.passed is passed


def test_cost_budget_accepts_an_integer_cost_and_zero_budget():
    assert CostBudgetMetric(0).evaluate({"cost_usd": 0}).passed


@pytest.mark.parametrize("observation", [None, {}, {"cost_usd": None}, {"cost_usd": "0.1"}])
def test_cost_budget_fails_with_missing_or_invalid_data(observation):
    result = CostBudgetMetric(0.01).evaluate(observation)

    assert not result.passed
    assert "missing or not a number" in result.details


def test_cost_budget_rejects_a_negative_budget():
    with pytest.raises(ValueError):
        CostBudgetMetric(-0.01)


# --- MemoryDriftMetric ---


@pytest.mark.parametrize(
    ("fact", "where"),
    [
        ("PostgreSQL", ("summary",)),
        ("30000", ("anchors",)),
        ("Northwind Booking", ("metadata",)),
        ("Auth0", ("metadata",)),  # dentro de una lista anidada
        ("5", ("metadata",)),  # un valor numérico también es texto buscable
    ],
)
def test_memory_drift_passes_when_the_fact_is_in_the_requested_field(fact, where):
    result = MemoryDriftMetric(fact, where=where).evaluate(SNAPSHOT)

    assert result.name == "memory_drift"
    assert result.passed
    assert result.score == 1.0
    assert next(iter(where)) in result.details


def test_memory_drift_is_case_insensitive():
    assert MemoryDriftMetric("postgresql").evaluate(SNAPSHOT).passed
    assert MemoryDriftMetric("NORTHWIND booking", where=["metadata"]).evaluate(SNAPSHOT).passed


def test_memory_drift_defaults_to_summary_anchors_and_metadata():
    assert MemoryDriftMetric("Stripe").evaluate(SNAPSHOT).passed  # solo está en metadata
    assert MemoryDriftMetric("30000").evaluate(SNAPSHOT).passed  # solo está en anclas


def test_memory_drift_only_looks_in_the_requested_fields():
    result = MemoryDriftMetric("Stripe", where=("summary", "anchors")).evaluate(SNAPSHOT)

    assert not result.passed
    assert result.score == 0.0
    assert "not found" in result.details


def test_memory_drift_does_not_search_metadata_keys():
    assert not MemoryDriftMetric("project_name", where=("metadata",)).evaluate(SNAPSHOT).passed


def test_memory_drift_fails_when_the_fact_is_absent_everywhere():
    assert not MemoryDriftMetric("Django").evaluate(SNAPSHOT).passed


def test_memory_drift_is_literal_not_semantic():
    # El formato importa: "30,000" no es "30000". Limitación documentada, no un bug
    assert not MemoryDriftMetric("30,000").evaluate(SNAPSHOT).passed


def test_memory_drift_handles_empty_memory():
    empty = {"summary": None, "anchors": [], "metadata": {}}

    assert not MemoryDriftMetric("Northwind Booking").evaluate(empty).passed


@pytest.mark.parametrize("snapshot", [None, "text", 5, []])
def test_memory_drift_fails_when_the_snapshot_is_missing_or_invalid(snapshot):
    result = MemoryDriftMetric("Stripe").evaluate(snapshot)

    assert not result.passed
    assert "snapshot missing or invalid" in result.details


def test_memory_drift_tolerates_a_snapshot_without_some_fields():
    assert MemoryDriftMetric("Stripe").evaluate({"metadata": {"mentioned_technologies": ["Stripe"]}}).passed
    assert not MemoryDriftMetric("Stripe").evaluate({}).passed


@pytest.mark.parametrize("fact", ["", "   "])
def test_memory_drift_rejects_an_empty_fact(fact):
    with pytest.raises(ValueError):
        MemoryDriftMetric(fact)


@pytest.mark.parametrize("where", [(), ("summary", "window"), ("window",)])
def test_memory_drift_rejects_unknown_or_empty_fields(where):
    with pytest.raises(ValueError):
        MemoryDriftMetric("Stripe", where=where)


def test_memory_drift_accepts_a_single_field_name_as_string():
    assert MemoryDriftMetric("PostgreSQL", where="summary").evaluate(SNAPSHOT).passed
