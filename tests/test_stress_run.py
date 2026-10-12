import csv
import json

import httpx
import pytest

from evals.stress import run as stress_run
from evals.stress.fixtures import build_pdfs
from evals.stress.scenarios import SCENARIOS
from tests.sessions_helpers import (
    FakeOpenAIClient,
    use_fake_estimation_llm,
    use_fake_llm,
    use_fake_metadata_llm,
)

GOOD_ESTIMATION = {
    "summary": "Web platform with billing and reporting.",
    "confidence_pct": 80,
    "phases": [{"name": "Build", "duration_weeks": 3, "cost_eur": 6000, "summary": "Build the platform."}],
    "total_duration_weeks": 3,
    "total_cost_eur": 6000,
}
METADATA_WITH_NAME = {
    "project_name": "Northwind Booking",
    "assumed_team_size": None,
    "mentioned_technologies": [],
    "agreed_scope": None,
}


@pytest.fixture
def fixtures_dir(tmp_path, monkeypatch):
    """Fixtures sintéticas pequeñas en un directorio temporal (mismo PDF para todos los tamaños)."""
    directory = tmp_path / "pdfs"
    directory.mkdir()
    pdf_bytes, _ = build_pdfs.build_pdf(1)
    for size in (5, 20, 50, 100):
        (directory / f"attachment_{size}kb.pdf").write_bytes(pdf_bytes)
    monkeypatch.setattr(build_pdfs, "FIXTURES_DIR", directory)
    return directory


def _setup(monkeypatch):
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    use_fake_estimation_llm(monkeypatch, [GOOD_ESTIMATION])
    use_fake_metadata_llm(monkeypatch, [METADATA_WITH_NAME])
    return use_fake_llm(monkeypatch)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_a_full_matrix_generates_at_least_fifty_valid_rows(monkeypatch, fixtures_dir, tmp_path):
    _setup(monkeypatch)
    output = tmp_path / "out" / "results.csv"

    exit_code = stress_run.main(
        [
            "--scenarios", "growing,pivot,contradiction",
            "--attachment-sizes", "0,5,20,50,100",
            "--repeats", "3",
            "--max-turns", "2",
            "--output", str(output),
        ]
    )  # fmt: skip

    rows = _read_csv(output)
    assert exit_code == 0
    assert len(rows) == 3 * 5 * 3 * 2 >= 50
    assert all(row["error"] == "" for row in rows)
    assert list(rows[0]) == stress_run.CSV_COLUMNS


def test_each_row_has_every_turn_observed_field_and_the_run_parameters(monkeypatch, fixtures_dir, tmp_path):
    _setup(monkeypatch)
    output = tmp_path / "results.csv"

    stress_run.main(
        ["--scenarios", "growing", "--attachment-sizes", "0,5", "--repeats", "1", "--max-turns", "3", "--output", str(output)]
    )

    rows = _read_csv(output)
    assert [(r["attachment_kb"], r["scenario_turn"], r["turn_index"]) for r in rows] == [
        ("0", "1", "1"), ("0", "2", "2"), ("0", "3", "3"),
        ("5", "1", "1"), ("5", "2", "2"), ("5", "3", "3"),
    ]  # fmt: skip
    for row in rows:
        assert row["scenario"] == "growing" and row["repeat"] == "1"
        for field in stress_run.OBSERVATION_FIELDS:
            assert row[field] != "" or field == "last_resolved_tier"
        assert row["cache_hit_kind"] == "none"
        assert row["wall_clock_ms"] != ""
    assert all(r["attachments_total_chars"] == "0" for r in rows[:3])
    assert all(int(r["attachments_total_chars"]) > 0 for r in rows[3:])
    assert len({r["session_id"] for r in rows}) == 2  # una sesión por (escenario, tamaño, repetición)


def test_metric_columns_use_the_budgets_and_the_memory_snapshot(monkeypatch, fixtures_dir, tmp_path):
    _setup(monkeypatch)
    output = tmp_path / "results.csv"

    # Presupuesto de coste 0 => todos fallan; presupuesto de latencia enorme => todos pasan
    stress_run.main(
        [
            "--scenarios", "growing", "--attachment-sizes", "0", "--max-turns", "2",
            "--latency-budget-ms", "1000000", "--cost-budget-usd", "0", "--output", str(output),
        ]
    )  # fmt: skip

    first, second = _read_csv(output)
    assert first["latency_budget_score"] == "1.0" and first["cost_budget_score"] == "0.0"
    # Turno 1: el hecho declarado es "Northwind Booking" y el extractor falso lo devuelve en metadata
    assert (first["memory_facts_tracked"], first["memory_facts_retained"], first["memory_recall"]) == ("1", "1", "1.0")
    assert first["memory_drift_score"] == "1.0"
    assert second["memory_facts_missing"] == ""


def test_lost_facts_are_reported_as_missing_and_lower_the_recall(monkeypatch, fixtures_dir, tmp_path):
    _setup(monkeypatch)
    output = tmp_path / "results.csv"

    # El extractor falso solo conoce el nombre: PostgreSQL (turno 3) nunca llega a la memoria
    stress_run.main(["--scenarios", "growing", "--attachment-sizes", "0", "--max-turns", "3", "--output", str(output)])

    third = _read_csv(output)[2]
    assert third["memory_facts_tracked"] == "2"
    assert third["memory_facts_missing"] == "PostgreSQL"
    assert third["memory_recall"] == "0.5"
    assert third["memory_drift_score"] == "0.0"


def test_stale_facts_still_in_memory_are_reported(monkeypatch, fixtures_dir, tmp_path):
    _setup(monkeypatch)
    use_fake_metadata_llm(
        monkeypatch,
        [{**METADATA_WITH_NAME, "project_name": "Atlas Ledger", "mentioned_technologies": ["Django"]}],
    )
    output = tmp_path / "results.csv"

    stress_run.main(["--scenarios", "pivot", "--attachment-sizes", "0", "--max-turns", "5", "--output", str(output)])

    rows = _read_csv(output)
    assert rows[3]["stale_facts_present"] == ""  # turno 4: Django todavía es vigente
    assert rows[4]["stale_facts_present"] == "Django"  # turno 5: reemplazado pero sigue en metadata
    assert rows[4]["memory_facts_missing"] == "NestJS"


def test_a_failing_turn_is_recorded_and_the_session_is_abandoned(monkeypatch, fixtures_dir, tmp_path, capsys):
    fake = _setup(monkeypatch)
    fake.payloads["EstimationResult"] = [GOOD_ESTIMATION, RuntimeError("provider down")]
    output = tmp_path / "results.csv"

    exit_code = stress_run.main(
        ["--scenarios", "growing", "--attachment-sizes", "0", "--max-turns", "5", "--output", str(output)]
    )

    rows = _read_csv(output)
    assert exit_code == 1
    assert len(rows) == 2  # turno 1 ok, turno 2 con error, y no se sigue
    assert rows[0]["error"] == ""
    assert "HTTP 502" in rows[1]["error"] and rows[1]["scenario_turn"] == "2"
    assert rows[1]["tokens_in"] == "" and rows[1]["wall_clock_ms"] != ""
    assert "1 error rows" in capsys.readouterr().out


def test_other_sessions_continue_after_one_is_abandoned(monkeypatch, fixtures_dir, tmp_path):
    fake = _setup(monkeypatch)
    # Falla solo la primera llamada de la primera sesión (Instructor no reintenta un RuntimeError)
    fake.payloads["EstimationResult"] = [RuntimeError("blip"), GOOD_ESTIMATION]
    output = tmp_path / "results.csv"

    stress_run.main(
        ["--scenarios", "growing", "--attachment-sizes", "0", "--repeats", "2", "--max-turns", "2", "--output", str(output)]
    )

    rows = _read_csv(output)
    assert [(r["repeat"], r["error"] != "") for r in rows] == [("1", True), ("2", False), ("2", False)]


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


def _run_session_with(handler, *, max_turns=3):
    rows: list[dict] = []
    failures = stress_run.run_session(
        _mock_client(handler),
        SCENARIOS["growing"],
        attachment_kb=0,
        repeat=1,
        latency_budget_ms=1000,
        cost_budget_usd=1,
        max_turns=max_turns,
        on_row=rows.append,
    )
    return failures, rows


def test_a_response_without_turn_observation_is_an_error_not_silent_data():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1"})
        return httpx.Response(200, json={"result": {}})  # servidor sin la instrumentación

    failures, rows = _run_session_with(handler)

    assert failures == 1 and len(rows) == 1
    assert "turn_observation" in rows[0]["error"]


def test_an_incomplete_observation_names_the_missing_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1"})
        return httpx.Response(200, json={"turn_observation": {"turn_index": 1}})

    failures, rows = _run_session_with(handler)

    assert failures == 1
    assert "latency_ms" in rows[0]["error"]


def test_a_failing_session_creation_is_reported():
    failures, rows = _run_session_with(lambda request: httpx.Response(500, text="boom"))

    assert failures == 1
    assert rows[0]["scenario_turn"] == 0 and "could not create the session" in rows[0]["error"]


def test_a_connection_error_during_a_turn_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1"})
        raise httpx.ConnectError("refused")

    failures, rows = _run_session_with(handler)

    assert failures == 1
    assert "request failed: ConnectError" in rows[0]["error"]


def test_a_failing_memory_snapshot_is_reported():
    observation = {name: 0 for name in stress_run.OBSERVATION_FIELDS}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1"})
        if request.url.path.endswith("/memory"):
            return httpx.Response(404, content=json.dumps({"detail": "session_not_found"}))
        return httpx.Response(200, json={"turn_observation": observation})

    failures, rows = _run_session_with(handler)

    assert failures == 1
    assert "memory snapshot" in rows[0]["error"]


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["--scenarios", "nope"], "unknown scenarios"),
        (["--attachment-sizes", "-1"], "non-negative"),
        (["--attachment-sizes", ""], "non-negative"),
        (["--repeats", "0"], "--repeats"),
        (["--max-turns", "0"], "--max-turns"),
        (["--cost-budget-usd", "-1"], "budgets"),
        (["--http", "localhost:8000"], "base URL"),
    ],
)
def test_invalid_arguments_exit_with_a_useful_message(argv, fragment, capsys, fixtures_dir):
    with pytest.raises(SystemExit) as excinfo:
        stress_run.main([*argv, "--output", "unused.csv"])

    assert excinfo.value.code == 2
    assert fragment in capsys.readouterr().err


def test_missing_fixtures_explain_how_to_generate_them(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(build_pdfs, "FIXTURES_DIR", tmp_path)  # directorio vacío

    with pytest.raises(SystemExit) as excinfo:
        stress_run.main(["--attachment-sizes", "0,5", "--output", str(tmp_path / "x.csv")])

    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "attachment_5kb.pdf" in err and "evals.stress.fixtures.build_pdfs" in err


def test_size_zero_needs_no_fixture(monkeypatch, tmp_path):
    _setup(monkeypatch)
    monkeypatch.setattr(build_pdfs, "FIXTURES_DIR", tmp_path)

    exit_code = stress_run.main(
        ["--scenarios", "pivot", "--attachment-sizes", "0", "--max-turns", "1", "--output", str(tmp_path / "x.csv")]
    )

    assert exit_code == 0


def test_an_unreachable_server_fails_fast_with_a_hint(monkeypatch, tmp_path):
    def unreachable(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(stress_run.httpx, "get", unreachable)

    with pytest.raises(SystemExit) as excinfo:
        stress_run.main(
            ["--http", "http://localhost:9", "--attachment-sizes", "0", "--output", str(tmp_path / "x.csv")]
        )

    assert "uvicorn" in str(excinfo.value)
