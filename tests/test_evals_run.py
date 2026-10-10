import json

from evals import run as evals_run
from evals.dataset import load_dataset
from tests.sessions_helpers import FakeOpenAIClient, use_fake_llm

# Dos casos del dataset real: uno normal con rangos holgados y uno con tier esperado "executive"
WIDE_RANGES = {
    "expected_cost_range_eur": [1000, 100000],
    "expected_duration_weeks_range": [1, 52],
    "expected_phase_count_range": [1, 8],
    "expected_in_summary": [],
    "expected_technologies_any_of": [],
}


def _cases():
    by_id = {case.id: case for case in load_dataset()}
    return [
        by_id["case-001-web-saas-crm"].model_copy(update=WIDE_RANGES),
        by_id["case-007-nda-executive"].model_copy(update=WIDE_RANGES),
    ]


def _setup(monkeypatch):
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    return use_fake_llm(monkeypatch)


def test_actor_mode_runs_the_cases_in_process_and_checks_the_tier(monkeypatch):
    fake = _setup(monkeypatch)

    rows = evals_run.run_cases(_cases(), "actor")

    assert [row["id"] for row in rows] == ["case-001-web-saas-crm", "case-007-nda-executive"]
    assert all("error" not in row for row in rows), rows
    assert all(row["passed"] for row in rows), rows
    nda_row = rows[1]
    assert nda_row["resolved_tier"] == "executive"
    assert "tier_match" in [m["name"] for m in nda_row["metrics"]]
    assert rows[0]["cost_usd"] > 0
    assert rows[0]["iterations"] is None  # el modo actor no tiene traza ACB
    assert len(fake.calls["EstimationResult"]) == 2
    assert fake.calls["CriticFeedback"] == []


def test_acb_mode_uses_the_acb_endpoint_and_reports_the_iterations(monkeypatch):
    fake = _setup(monkeypatch)

    rows = evals_run.run_cases(_cases(), "acb")

    assert all(row["passed"] for row in rows), rows
    assert all(row["iterations"] == 1 and row["final_decision"] == "accept" for row in rows)
    assert len(fake.calls["CriticFeedback"]) == 2


def test_each_case_gets_a_fresh_session_store(monkeypatch):
    from app.dependencies import get_session_store
    from app.main import app

    _setup(monkeypatch)

    evals_run.run_cases(_cases(), "actor")

    # El override temporal se retira al terminar: no contamina a otros tests ni al servidor
    assert get_session_store not in app.dependency_overrides


def test_a_tier_mismatch_fails_the_case(monkeypatch):
    _setup(monkeypatch)
    case = _cases()[0].model_copy(update={"expected_tier": "developer"})  # el resolver dará "default"

    (row,) = evals_run.run_cases([case], "actor")

    assert not row["passed"]
    tier_metric = next(m for m in row["metrics"] if m["name"] == "tier_match")
    assert "expected developer" in tier_metric["details"]


def test_a_guardrail_400_on_an_out_of_scope_case_counts_as_a_hit(monkeypatch):
    fake = _setup(monkeypatch)
    case = next(c for c in load_dataset() if c.id == "case-006-out-of-scope-adversarial")

    (row,) = evals_run.run_cases([case], "actor")

    assert row["passed"], row
    assert fake.calls["EstimationResult"] == []  # el guardrail cortó antes de llamar al actor


def test_a_http_error_is_reported_as_an_errored_row_not_a_crash(monkeypatch):
    fake = _setup(monkeypatch)
    fake.payloads["EstimationResult"] = [RuntimeError("provider down")]

    (row,) = evals_run.run_cases(_cases()[:1], "actor")

    assert "HTTP 502" in row["error"]


def test_main_prints_the_summary_and_writes_the_json_report(monkeypatch, tmp_path, capsys):
    _setup(monkeypatch)
    monkeypatch.setattr(evals_run, "load_dataset", _cases)
    output = tmp_path / "report.json"

    exit_code = evals_run.main(["--mode", "actor", "--limit", "2", "--output", str(output)])

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "2/2 cases passed" in printed
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "actor"
    assert report["summary"]["cases_passed"] == 2
    assert len(report["rows"]) == 2


def test_main_returns_a_failing_exit_code_when_a_case_fails(monkeypatch):
    _setup(monkeypatch)
    failing = _cases()[0].model_copy(update={"expected_cost_range_eur": [1, 2]})
    monkeypatch.setattr(evals_run, "load_dataset", lambda: [failing])

    assert evals_run.main(["--mode", "actor"]) == 1
