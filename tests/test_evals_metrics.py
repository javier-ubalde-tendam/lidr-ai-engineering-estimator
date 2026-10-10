import pytest

from evals.dataset import GoldenCase, load_dataset
from evals.metrics import (
    ContentRecallMetric,
    CostBoundsMetric,
    SchemaAdherenceMetric,
    run_all_metrics,
)
from tests.sessions_helpers import GOOD_ESTIMATION_PAYLOAD


def make_case(**overrides) -> GoldenCase:
    fields = {
        "id": "case-test",
        "transcript": "A web portal for gym bookings with payments.",
        "project_type": "web_saas",
        "expected_in_summary": ["platform"],
        "expected_technologies_any_of": ["Stripe", "billing"],
        "expected_phase_count_range": [1, 3],
        "expected_cost_range_eur": [5000, 20000],
        "expected_duration_weeks_range": [2, 10],
    }
    return GoldenCase(**{**fields, **overrides})


OUT_OF_SCOPE_PAYLOAD = {
    **GOOD_ESTIMATION_PAYLOAD,
    "summary": "Out of scope: too vague to size.",
    "confidence_pct": 10,
    "phases": [{"name": "Not estimated", "duration_weeks": 1, "cost_eur": 0, "summary": "Cannot be sized without info."}],
    "total_duration_weeks": 1,
    "total_cost_eur": 0,
}


def test_the_golden_dataset_loads_all_sixteen_cases_with_their_tier_expectations():
    cases = load_dataset()

    assert len(cases) == 16
    assert {c.id for c in cases if c.expected_out_of_scope} == {"case-005-out-of-scope-vague", "case-006-out-of-scope-adversarial"}
    assert {c.expected_tier for c in cases} >= {None, "executive", "pm", "developer"}


def test_all_metrics_pass_for_a_good_result():
    results = run_all_metrics(make_case(), GOOD_ESTIMATION_PAYLOAD)

    assert [r.name for r in results] == ["schema_adherence", "cost_bounds", "content_recall"]
    assert all(r.passed for r in results), results


def test_schema_adherence_fails_when_the_payload_does_not_match_the_schema():
    broken = {**GOOD_ESTIMATION_PAYLOAD, "phases": []}

    result = SchemaAdherenceMetric().evaluate(make_case(), broken)

    assert not result.passed
    assert "EstimationResult" in result.details


def test_schema_adherence_checks_the_phase_count_range():
    result = SchemaAdherenceMetric().evaluate(make_case(expected_phase_count_range=[3, 6]), GOOD_ESTIMATION_PAYLOAD)

    assert not result.passed
    assert "2 phases" in result.details


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"expected_cost_range_eur": [20000, 30000]}, "cost 10000"),
        ({"expected_duration_weeks_range": [10, 20]}, "duration 5w"),
    ],
)
def test_cost_bounds_fails_outside_the_expected_ranges(overrides, fragment):
    result = CostBoundsMetric().evaluate(make_case(**overrides), GOOD_ESTIMATION_PAYLOAD)

    assert not result.passed
    assert fragment in result.details


def test_cost_bounds_for_out_of_scope_cases_only_needs_the_prefix():
    case = make_case(expected_out_of_scope=True)

    assert CostBoundsMetric().evaluate(case, OUT_OF_SCOPE_PAYLOAD).passed
    assert not CostBoundsMetric().evaluate(case, GOOD_ESTIMATION_PAYLOAD).passed


def test_content_recall_is_case_insensitive_and_looks_at_phases_too():
    case = make_case(expected_in_summary=["WEB PLATFORM"], expected_technologies_any_of=["design"])

    assert ContentRecallMetric().evaluate(case, GOOD_ESTIMATION_PAYLOAD).passed  # "Design" está en el nombre de una fase


def test_content_recall_reports_missing_terms_and_technologies():
    case = make_case(expected_in_summary=["kubernetes"], expected_technologies_any_of=["Okta", "DocuSign"])

    result = ContentRecallMetric().evaluate(case, GOOD_ESTIMATION_PAYLOAD)

    assert not result.passed
    assert "kubernetes" in result.details
    assert "Okta" in result.details


def test_content_recall_needs_only_one_of_the_expected_technologies():
    case = make_case(expected_in_summary=[], expected_technologies_any_of=["Okta", "billing"])

    assert ContentRecallMetric().evaluate(case, GOOD_ESTIMATION_PAYLOAD).passed


def test_content_recall_is_skipped_for_out_of_scope_cases():
    assert ContentRecallMetric().evaluate(make_case(expected_out_of_scope=True), OUT_OF_SCOPE_PAYLOAD).passed
