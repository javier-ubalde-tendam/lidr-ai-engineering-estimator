"""Runner de evals: pasa el dataset dorado por los endpoints conversacionales.

AVISO: hace llamadas REALES al LLM (y a la Moderation API), por lo que cuesta dinero y tarda.
El modo `acb` multiplica las llamadas por turno (hasta 2 * BOSS_MAX_ITERATIONS). NO forma parte
de `pytest`: se ejecuta a mano. Empieza siempre con `--limit` bajo.

    uv run python -m evals.run --mode actor --limit 3
    uv run python -m evals.run --mode acb --limit 3 --output report.json
    uv run python -m evals.run --mode actor --http http://localhost:8000

Por defecto la app corre en proceso con `TestClient` y un `SessionStore` nuevo por caso; con
`--http` se ataca un servidor ya levantado. Flujo por caso: crear sesión, POST /estimate (modo
actor) o /estimate-acb (modo acb) y GET /sessions/{id} para comparar el tier con el esperado.
"""

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import httpx

from evals.dataset import GoldenCase, load_dataset
from evals.metrics import MetricResult, run_all_metrics

Mode = Literal["actor", "acb"]
_HTTP_TIMEOUT_SECONDS = 180.0  # el modo acb encadena varias llamadas al LLM por turno


def _form_body(case: GoldenCase) -> dict[str, str]:
    body = {
        "transcript": case.transcript,
        "project_type": case.project_type,
        "detail_level": case.detail_level,
        "output_format": case.output_format,
    }
    if case.tier != "auto":
        body["tier"] = case.tier  # override explícito; "auto" = que decida el resolver
    return body


def _guardrail_payload(response: httpx.Response) -> dict[str, Any]:
    """Resultado sintético para un 400 de guardrail: es la forma de 'fuera de alcance' esperada."""
    detail = response.json().get("detail")
    reason = detail.get("reason", "unknown") if isinstance(detail, dict) else "unknown"
    return {
        "result": {
            "summary": f"Out of scope: blocked by the input guardrail ({reason}).",
            "confidence_pct": 0,
            "phases": [
                {
                    "name": "Not estimated",
                    "duration_weeks": 1,
                    "cost_eur": 0,
                    "summary": "Blocked by the input guardrail; no estimation produced.",
                }
            ],
            "total_duration_weeks": 1,
            "total_cost_eur": 0,
        }
    }


def run_case(client: httpx.Client, case: GoldenCase, mode: Mode) -> dict[str, Any]:
    """Ejecuta un caso contra cualquier cliente httpx (TestClient incluido) y devuelve su fila de informe."""
    session_id = client.post("/sessions").json()["session_id"]
    endpoint = f"/sessions/{session_id}/{'estimate-acb' if mode == 'acb' else 'estimate'}"

    started_at = time.perf_counter()
    response = client.post(endpoint, data=_form_body(case))
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    if response.status_code == 400 and case.expected_out_of_scope:
        payload = _guardrail_payload(response)  # el guardrail hizo de filtro de alcance: acierto
    elif response.is_error:
        return {"id": case.id, "latency_ms": latency_ms, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
    else:
        payload = response.json()

    metrics = run_all_metrics(case, payload["result"])
    resolved_tier = client.get(f"/sessions/{session_id}").json().get("last_resolved_tier")
    if case.expected_tier not in (None, "auto"):
        ok = resolved_tier == case.expected_tier
        metrics.append(MetricResult("tier_match", ok, f"expected {case.expected_tier}, got {resolved_tier}"))

    acb = payload.get("acb") or {}
    return {
        "id": case.id,
        "latency_ms": latency_ms,
        "cost_usd": payload.get("cost_usd"),
        "tokens_in": payload.get("tokens_in"),
        "tokens_out": payload.get("tokens_out"),
        "iterations": acb.get("iterations_run"),
        "final_decision": acb.get("final_decision"),
        "resolved_tier": resolved_tier,
        "passed": all(m.passed for m in metrics),
        "metrics": [{"name": m.name, "passed": m.passed, "details": m.details} for m in metrics],
    }


@contextmanager
def _client_for_case(http_base_url: str | None) -> Iterator[httpx.Client]:
    if http_base_url:
        with httpx.Client(base_url=http_base_url, timeout=_HTTP_TIMEOUT_SECONDS) as client:
            yield client
        return

    # Import perezoso: contra un servidor remoto no hace falta cargar la app ni su configuración
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.dependencies import get_session_store
    from app.main import app
    from app.sessions.store import SessionStore

    # Un store nuevo por caso: se crea UNA vez (el override se evalúa en cada request) para que
    # POST /sessions y las peticiones siguientes vean la misma sesión
    store = SessionStore(max_conversation_turns=get_settings().MAX_CONVERSATION_TURNS)
    app.dependency_overrides[get_session_store] = lambda: store
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_session_store, None)


def run_cases(cases: list[GoldenCase], mode: Mode, http_base_url: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            with _client_for_case(http_base_url) as client:
                row = run_case(client, case, mode)
        except Exception as exc:  # noqa: BLE001 - un caso roto no debe abortar el resto del informe
            row = {"id": case.id, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        rows.append(row)
        print(_format_row(row))
    return rows


def _format_row(row: dict[str, Any]) -> str:
    if "error" in row:
        return f"{row['id']:<36} ERROR {row['error']}"
    parts = [f"{row['id']:<36}", f"{row['latency_ms']:>7} ms"]
    if row.get("cost_usd") is not None:
        parts.append(f"${row['cost_usd']:.4f}")
    if row.get("iterations") is not None:
        parts.append(f"acb_iterations={row['iterations']}({row['final_decision']})")
    parts += [f"{m['name']}={'PASS' if m['passed'] else 'FAIL'}" for m in row["metrics"]]
    failures = [f"    - {m['name']}: {m['details']}" for m in row["metrics"] if not m["passed"]]
    return " | ".join(parts) + ("\n" + "\n".join(failures) if failures else "")


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if "error" not in row]
    per_metric: dict[str, list[bool]] = {}
    for row in scored:
        for metric in row["metrics"]:
            per_metric.setdefault(metric["name"], []).append(metric["passed"])
    return {
        "cases_total": len(rows),
        "cases_passed": sum(row["passed"] for row in scored),
        "cases_errored": len(rows) - len(scored),
        "total_cost_usd": round(sum(row.get("cost_usd") or 0 for row in scored), 6),
        "metrics": {name: f"{sum(results)}/{len(results)}" for name, results in per_metric.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evals contra los endpoints conversacionales (llama al LLM real).")
    parser.add_argument("--mode", choices=("actor", "acb"), required=True)
    parser.add_argument("--limit", type=int, default=None, help="Ejecuta solo los N primeros casos")
    parser.add_argument("--http", metavar="BASE_URL", default=None, help="Servidor real, p. ej. http://localhost:8000")
    parser.add_argument("--output", type=Path, default=None, help="Guarda el informe completo en JSON")
    args = parser.parse_args(argv)

    cases = load_dataset()[: args.limit]
    print(f"Running {len(cases)} cases, mode={args.mode}, target={args.http or 'in-process'}")
    rows = run_cases(cases, args.mode, args.http)

    summary = _summarise(rows)
    print(f"\n{summary['cases_passed']}/{summary['cases_total']} cases passed ({summary['cases_errored']} errored)")
    for name, score in summary["metrics"].items():
        print(f"  {name}: {score}")
    print(f"  total cost: ${summary['total_cost_usd']:.4f}")

    if args.output:
        args.output.write_text(
            json.dumps({"mode": args.mode, "summary": summary, "rows": rows}, indent=2), encoding="utf-8"
        )
        print(f"Report written to {args.output}")
    return 0 if summary["cases_passed"] == summary["cases_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
