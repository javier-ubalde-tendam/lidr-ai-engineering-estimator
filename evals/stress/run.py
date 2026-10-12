"""Runner del stress test del CAG: escenarios multi-turno x tamaños de adjunto x repeticiones.

AVISO: hace llamadas REALES al LLM (y a la Moderation API) y cuesta dinero: cada turno con adjunto
envía hasta MAX_ATTACHMENT_CHARS de texto. Empieza con `--repeats 1 --max-turns 5`. No forma parte de pytest.

    uv run python -m evals.stress.run --http http://localhost:8000 \
        --scenarios growing,pivot,contradiction --attachment-sizes 0,5,20,50,100 \
        --repeats 3 --output evals/stress/results.csv

Sin `--http` la app corre en proceso con `TestClient` (mismo patrón que `evals.run`). Por cada
(escenario, tamaño, repetición) crea una sesión, recorre los turnos con POST /estimate (el PDF se
adjunta en CADA turno: los adjuntos no se guardan en el historial, un cliente que sigue hablando del
documento tiene que volver a subirlo) y registra una fila CSV por turno. Si un turno falla, se
escribe la fila con el error y se abandona esa sesión: continuar sobre un historial al que le falta
un turno falsearía las métricas de memoria.
"""

import argparse
import csv
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from app.schemas.observation import TurnObservation
from evals.run import _client_for_case  # mismo cliente HTTP / in-process que los evals dorados
from evals.stress.fixtures.build_pdfs import DEFAULT_SIZES_KB, fixture_path
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric
from evals.stress.scenarios import SCENARIOS, Scenario

_HTTP_TIMEOUT_SECONDS = 180.0
DEFAULT_LATENCY_BUDGET_MS = 10_000  # presupuestos ilustrativos: se ajustan por CLI a lo que prometa el producto
DEFAULT_COST_BUDGET_USD = 0.01
_FORM_FIELDS = {"project_type": "web_saas", "detail_level": "summary", "output_format": "narrative"}

OBSERVATION_FIELDS = list(TurnObservation.model_fields)
CSV_COLUMNS = [
    "scenario",
    "attachment_kb",
    "repeat",
    "scenario_turn",
    *OBSERVATION_FIELDS,
    # Tiempo de la petición HTTP completa (adjunto, guardrails, estimación, metadata y compresión);
    # latency_ms es solo la llamada de estimación al LLM
    "wall_clock_ms",
    "latency_budget_score",
    "cost_budget_score",
    "memory_drift_score",  # 1.0 si TODOS los hechos esperados siguen en la memoria; vacío si no había ninguno
    "memory_recall",
    "memory_facts_tracked",
    "memory_facts_retained",
    "memory_facts_missing",
    "stale_facts_present",
    "error",
]

RowSink = Callable[[dict[str, Any]], None]


def _memory_columns(scenario: Scenario, turn: int, snapshot: dict[str, Any]) -> dict[str, Any]:
    tracked = scenario.expected_facts(turn)
    missing = [fact for fact in tracked if not MemoryDriftMetric(fact).evaluate(snapshot).passed]
    stale = [fact for fact in scenario.stale_facts(turn) if MemoryDriftMetric(fact).evaluate(snapshot).passed]
    columns: dict[str, Any] = {
        "memory_facts_tracked": len(tracked),
        "memory_facts_retained": len(tracked) - len(missing),
        "memory_facts_missing": "|".join(missing),
        "stale_facts_present": "|".join(stale),
    }
    if tracked:
        columns["memory_recall"] = round((len(tracked) - len(missing)) / len(tracked), 4)
        columns["memory_drift_score"] = 0.0 if missing else 1.0
    return columns


def _error_row(base: dict[str, Any], turn: int, message: str, **extra: Any) -> dict[str, Any]:
    return {**base, "scenario_turn": turn, "error": message, **extra}


def run_session(
    client: httpx.Client,
    scenario: Scenario,
    *,
    attachment_kb: int,
    repeat: int,
    latency_budget_ms: float,
    cost_budget_usd: float,
    max_turns: int | None,
    on_row: RowSink,
) -> int:
    """Recorre un escenario en una sesión nueva. Devuelve el número de filas con error (0 o 1)."""
    base = {"scenario": scenario.name, "attachment_kb": attachment_kb, "repeat": repeat}
    pdf = fixture_path(attachment_kb) if attachment_kb else None
    pdf_bytes = pdf.read_bytes() if pdf else None
    latency_metric = LatencyBudgetMetric(latency_budget_ms)
    cost_metric = CostBudgetMetric(cost_budget_usd)

    try:
        session_response = client.post("/sessions")
        session_response.raise_for_status()
        session_id = session_response.json()["session_id"]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        on_row(_error_row(base, 0, f"could not create the session: {type(exc).__name__}: {str(exc)[:200]}"))
        return 1

    for turn, transcript in enumerate(scenario.turns[:max_turns], start=1):
        files = [("attachments", (pdf.name, pdf_bytes, "application/pdf"))] if pdf and pdf_bytes else None
        try:
            started_at = time.perf_counter()
            response = client.post(
                f"/sessions/{session_id}/estimate", data={**_FORM_FIELDS, "transcript": transcript}, files=files
            )
            wall_clock_ms = int((time.perf_counter() - started_at) * 1000)
        except httpx.HTTPError as exc:
            on_row(_error_row(base, turn, f"request failed: {type(exc).__name__}: {str(exc)[:200]}"))
            return 1

        if response.is_error:
            on_row(
                _error_row(base, turn, f"HTTP {response.status_code}: {response.text[:200]}", wall_clock_ms=wall_clock_ms)
            )
            return 1

        try:
            observation = response.json()["turn_observation"]
            missing_fields = [name for name in OBSERVATION_FIELDS if name not in observation]
        except (ValueError, KeyError, TypeError):
            observation, missing_fields = None, ["turn_observation"]
        if missing_fields:
            on_row(
                _error_row(
                    base, turn, f"response without a valid turn_observation (missing {missing_fields}): old server?"
                )
            )
            return 1

        try:
            snapshot_response = client.get(f"/sessions/{session_id}/memory")
            snapshot_response.raise_for_status()
            snapshot = snapshot_response.json()
        except (httpx.HTTPError, ValueError) as exc:
            on_row(_error_row(base, turn, f"could not read the memory snapshot: {type(exc).__name__}: {str(exc)[:200]}"))
            return 1

        row = {
            **base,
            "scenario_turn": turn,
            **{name: observation[name] for name in OBSERVATION_FIELDS},
            "wall_clock_ms": wall_clock_ms,
            "latency_budget_score": latency_metric.evaluate(observation).score,
            "cost_budget_score": cost_metric.evaluate(observation).score,
            **_memory_columns(scenario, turn, snapshot),
        }
        on_row(row)
    return 0


def run_stress(
    http_base_url: str | None,
    scenarios: list[Scenario],
    sizes_kb: list[int],
    repeats: int,
    *,
    latency_budget_ms: float,
    cost_budget_usd: float,
    max_turns: int | None,
    on_row: RowSink,
) -> int:
    """Ejecuta toda la matriz; devuelve cuántas sesiones se abandonaron por error."""
    failed_sessions = 0
    with _client_for_case(http_base_url) as client:
        for scenario in scenarios:
            for size_kb in sizes_kb:
                for repeat in range(1, repeats + 1):
                    failed_sessions += run_session(
                        client,
                        scenario,
                        attachment_kb=size_kb,
                        repeat=repeat,
                        latency_budget_ms=latency_budget_ms,
                        cost_budget_usd=cost_budget_usd,
                        max_turns=max_turns,
                        on_row=on_row,
                    )
    return failed_sessions


def _int_list(raw: str) -> list[int]:
    try:
        return [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integers separated by commas, got {raw!r}") from exc


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Scenario]:
    names = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    unknown = [name for name in names if name not in SCENARIOS]
    if not names or unknown:
        parser.error(f"unknown scenarios {unknown or names}; available: {', '.join(SCENARIOS)}")
    if not args.attachment_sizes or any(size < 0 for size in args.attachment_sizes):
        parser.error("--attachment-sizes needs non-negative integers (0 = no attachment), e.g. 0,5,20,50,100")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.max_turns is not None and args.max_turns < 1:
        parser.error("--max-turns must be >= 1")
    if args.latency_budget_ms < 0 or args.cost_budget_usd < 0:
        parser.error("budgets must be >= 0")
    if args.http is not None and not args.http.startswith(("http://", "https://")):
        parser.error("--http must be a base URL such as http://localhost:8000")

    missing = [fixture_path(size) for size in args.attachment_sizes if size and not fixture_path(size).exists()]
    if missing:
        parser.error(
            "missing PDF fixtures: "
            + ", ".join(path.name for path in missing)
            + ". Generate them with `uv run python -m evals.stress.fixtures.build_pdfs`"
            + f" (default sizes: {','.join(map(str, DEFAULT_SIZES_KB))})"
        )
    return [SCENARIOS[name] for name in names]


def _check_server(base_url: str) -> None:
    try:
        httpx.get(f"{base_url.rstrip('/')}/health", timeout=5.0).raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"Cannot reach {base_url}/health ({type(exc).__name__}). Start it with "
            "`uv run uvicorn app.main:app --reload` or run in-process by omitting --http."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stress test del CAG (llama al LLM real; cuesta dinero).")
    parser.add_argument("--http", metavar="BASE_URL", default=None, help="Servidor real, p. ej. http://localhost:8000")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS), help="Lista separada por comas")
    parser.add_argument("--attachment-sizes", type=_int_list, default=[0, *DEFAULT_SIZES_KB], help="KB; 0 = sin adjunto")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=None, help="Recorta cada escenario a N turnos (pruebas baratas)")
    parser.add_argument("--latency-budget-ms", type=float, default=DEFAULT_LATENCY_BUDGET_MS)
    parser.add_argument("--cost-budget-usd", type=float, default=DEFAULT_COST_BUDGET_USD)
    parser.add_argument("--output", type=Path, default=Path("evals/stress/results.csv"))
    args = parser.parse_args(argv)
    scenarios = _validate(args, parser)
    if args.http:
        _check_server(args.http)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "error": 0}
    total_cost = 0.0
    print(
        f"Stress run: scenarios={[s.name for s in scenarios]} sizes_kb={args.attachment_sizes} "
        f"repeats={args.repeats} target={args.http or 'in-process'} -> {args.output}"
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, restval="")
        writer.writeheader()

        def on_row(row: dict[str, Any]) -> None:
            nonlocal total_cost
            writer.writerow(row)
            handle.flush()  # una ejecución interrumpida conserva lo medido hasta ese momento
            counts["error" if row.get("error") else "ok"] += 1
            total_cost += float(row.get("cost_usd") or 0)
            label = f"{row['scenario']} kb={row['attachment_kb']} r={row['repeat']} turn={row['scenario_turn']}"
            if row.get("error"):
                print(f"{label:<44} ERROR {row['error']}")
            else:
                print(f"{label:<44} {row['latency_ms']:>6} ms  tokens_in={row['tokens_in']:>6}  ${row['cost_usd']:.4f}")

        failed_sessions = run_stress(
            args.http,
            scenarios,
            args.attachment_sizes,
            args.repeats,
            latency_budget_ms=args.latency_budget_ms,
            cost_budget_usd=args.cost_budget_usd,
            max_turns=args.max_turns,
            on_row=on_row,
        )

    print(
        f"\n{counts['ok']} valid turn rows, {counts['error']} error rows "
        f"({failed_sessions} sessions abandoned). Estimation cost: ${total_cost:.4f}. CSV: {args.output}"
    )
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
