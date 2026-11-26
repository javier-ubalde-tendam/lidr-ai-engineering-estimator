"""Run the Session 13 estimation graph end to end and print a readable report.

    # Real run (Postgres checkpointer; run it inside Docker, see docs/sessions/13-graph.md)
    docker compose exec ai_service python scripts/run_graph_s13.py \\
        --out exercises/session-13/example_run_complex.txt

    # No Postgres: in-memory checkpointer
    uv run python scripts/run_graph_s13.py --memory

    # No Postgres and canned retrieval (still needs OPENAI_API_KEY: the LLM calls are real)
    uv run python scripts/run_graph_s13.py --memory --stub

Every run uses ``--estimation-id`` (default: a fresh uuid) as the LangGraph ``thread_id``. With
``LOGFIRE_TOKEN`` in ``.env`` the run exports one span per node to Logfire. Exit code: 0 when the
graph finishes (``validated`` or ``needs_review``), 1 when it raises.
"""

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Reuse the stub loader of the S12 script (scripts/ is sys.path[0] when this file is run)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from run_agent_s12 import load_stub_search, make_stub_backend

from app.api.graph_runtime import build_graph_dependencies
from app.config import get_settings
from app.domain.graph import GraphDependencies, build_graph
from app.foundation.observability import configure_logfire
from app.foundation.persistence.checkpointer import open_checkpointer

DEFAULT_TRANSCRIPT = ROOT / "exercises" / "session-12" / "sample_transcript_complex.txt"


def format_report(estimation_id: str, result: dict[str, Any]) -> str:
    """requirements -> components -> budget matches -> estimate -> status -> errors."""
    lines = [f"ESTIMATION {estimation_id}", "", "REQUIREMENTS"]
    lines += [f"  - {requirement}" for requirement in result.get("requirements", [])]
    lines += ["", "COMPONENTS"]
    lines += [f"  - {c['name']} [{c['category']}]" for c in result.get("components", [])]
    lines += ["", "BUDGET MATCHES (engineer-hours)"]
    lines += [
        f"  - {m['component']}: {m['reference_budget_id']} = {m['amount']:g} h "
        f"(distance {m['distance']:.2f})"
        for m in result.get("budget_matches", [])
    ]
    lines += ["", "ESTIMATE (engineer-hours)"]
    estimate = result.get("estimate")
    if estimate:
        lines += [
            f"  - {c['name']}: {c['hours']:g} h  refs={', '.join(c['reference_budget_ids']) or '-'}"
            for c in estimate["components"]
        ]
        lines += [f"  TOTAL: {estimate['total_hours']:g} h", f"  NOTES: {estimate['notes']}"]
    else:
        lines.append("  (none)")
    lines += ["", f"STATUS: {result.get('status')}", "", "ERRORS"]
    lines += [f"  - {error}" for error in result.get("errors", [])] or ["  (none)"]
    return "\n".join(lines)


@asynccontextmanager
async def _checkpointer(memory: bool) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    if memory:
        yield MemorySaver()
    else:
        async with open_checkpointer() as saver:
            yield saver


def _dependencies(stub: bool) -> GraphDependencies:
    deps = build_graph_dependencies()
    if not stub:
        return deps
    # Same models and wrapper, canned retrieval (no database)
    return GraphDependencies(
        llm_wrapper=deps.llm_wrapper,
        search_backend=make_stub_backend(load_stub_search()),
        settings=deps.settings,
    )


async def run(transcript: str, estimation_id: str, *, memory: bool, stub: bool) -> dict[str, Any]:
    deps = _dependencies(stub)
    async with _checkpointer(memory) as checkpointer:
        graph = build_graph(checkpointer, deps)
        config: Any = {"configurable": {"thread_id": estimation_id}}
        result: dict[str, Any] = await graph.ainvoke(
            {"transcript": transcript, "estimation_id": estimation_id}, config
        )
        return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Session 13 estimation graph.")
    parser.add_argument("transcript", type=Path, nargs="?", default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--estimation-id", help="LangGraph thread_id (default: a new uuid).")
    parser.add_argument("--memory", action="store_true", help="MemorySaver instead of Postgres.")
    parser.add_argument("--stub", action="store_true", help="Offline retrieval (S12 stub).")
    parser.add_argument("--out", type=Path, help="Also write the report to this file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Transcripts are Spanish/UTF-8: the Windows console default (cp1252) would choke
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    transcript = args.transcript.read_text(encoding="utf-8")
    estimation_id = args.estimation_id or str(uuid4())

    configure_logfire()
    # psycopg's async mode cannot run on Windows' default ProactorEventLoop: use the selector loop
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    try:
        result = asyncio.run(
            run(transcript, estimation_id, memory=args.memory, stub=args.stub),
            loop_factory=loop_factory,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure with a non-zero exit code
        print(f"Graph failed ({type(exc).__name__}): {str(exc)[:300]}", file=sys.stderr)
        return 1

    report = format_report(estimation_id, result)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nReport saved to {args.out}")
    print(f"\nthread_id: {estimation_id}")
    if get_settings().LOGFIRE_TOKEN:
        print("Trace: open your Logfire project (Live view) and look for the 'node: ...' spans.")
    else:
        print("No LOGFIRE_TOKEN set: spans were not exported (set it in .env to see the trace).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
