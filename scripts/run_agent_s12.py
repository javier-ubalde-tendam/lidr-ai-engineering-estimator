"""Run the Session 12 estimation agent and print its reasoning trace.

Default mode calls the running API (``POST /v1/estimate/agent``); ``--stub`` runs the loop in
process with the canned retrieval of ``exercises/session-12/reference_retrieval.py`` (no database,
no API server; it still needs OPENAI_API_KEY because the model calls are real).

    # Debug cheaply: small model, minimal reasoning, simple transcript
    uv run python scripts/run_agent_s12.py exercises/session-12/sample_transcript_simple.txt \\
        --model gpt-5-mini --effort minimal

    # Deliverable: real run on the complex transcript, trace saved to a file
    uv run python scripts/run_agent_s12.py exercises/session-12/sample_transcript_complex.txt \\
        --model gpt-5 --effort medium --out exercises/session-12/trace_complex.txt

    # Loop mechanics without the database or the API
    uv run python scripts/run_agent_s12.py exercises/session-12/sample_transcript_simple.txt \\
        --model gpt-5-mini --effort minimal --stub

API mode reads API_BASE_URL and ESTIMATE_API_KEY from the environment / .env. Every run makes paid
OpenAI calls. Exit code: 0 only when the agent ended with ``status == "done"``.
"""

import argparse
import asyncio
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.generation.agentic.agent_loop import format_trace, run_agent
from app.generation.agentic.agent_schemas import AgentResult, SearchBudgetsArgs
from app.generation.agentic.agent_tools import SearchBackend, build_tool_registry

STUB_PATH = ROOT / "exercises" / "session-12" / "reference_retrieval.py"
# One agent run is several sequential model turns: far longer than a normal API call
API_TIMEOUT_SECONDS = 600.0

StubSearch = Callable[[str, dict[str, Any] | None], list[dict[str, Any]]]


def load_stub_search(path: Path = STUB_PATH) -> StubSearch:
    """Import the standalone stub by file path (it is not part of any package)."""
    spec = importlib.util.spec_from_file_location("reference_retrieval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the retrieval stub from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    search: StubSearch = module.search_budgets_stub
    return search


def make_stub_backend(search: StubSearch) -> SearchBackend:
    """Adapt the sync stub to the async ``search_budgets`` backend contract."""

    async def backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        filters = args.filters.model_dump(exclude_none=True) if args.filters else None
        return search(args.query, filters)

    return backend


async def run_in_process(
    transcript: str, *, model: str, effort: str, max_iterations: int
) -> AgentResult:
    from openai import AsyncOpenAI

    api_key = get_settings().OPENAI_API_KEY
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required to run the agent.")
    registry = build_tool_registry(make_stub_backend(load_stub_search()))
    return await run_agent(
        transcript,
        client=AsyncOpenAI(api_key=api_key),
        registry=registry,
        model=model,
        effort=effort,
        max_iterations=max_iterations,
    )


def run_via_api(transcript: str, *, overrides: dict[str, Any]) -> AgentResult:
    settings = get_settings()
    if not settings.ESTIMATE_API_KEY:
        raise SystemExit("ESTIMATE_API_KEY is required to call the API.")
    url = f"{settings.API_BASE_URL.rstrip('/')}/v1/estimate/agent"
    body = {"transcript": transcript, **{k: v for k, v in overrides.items() if v is not None}}
    response = httpx.post(
        url,
        json=body,
        headers={"X-API-Key": settings.ESTIMATE_API_KEY},
        timeout=API_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise SystemExit(f"API error {response.status_code}: {response.text[:500]}")
    return AgentResult.model_validate(response.json())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Session 12 estimation agent.")
    parser.add_argument("transcript", type=Path, help="Text file with the meeting transcript.")
    parser.add_argument("--model", help="Model override (default: AGENT_MODEL).")
    parser.add_argument(
        "--effort",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort override (default: AGENT_REASONING_EFFORT).",
    )
    parser.add_argument("--max-iterations", type=int, help="Iteration cap (default: setting).")
    parser.add_argument("--stub", action="store_true", help="In-process run with the retrieval stub.")
    parser.add_argument("--out", type=Path, help="Also write the trace to this file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Transcripts and traces are Spanish/UTF-8: the Windows console default (cp1252) would choke
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    transcript = args.transcript.read_text(encoding="utf-8")

    if args.stub:
        settings = get_settings()
        result = asyncio.run(
            run_in_process(
                transcript,
                model=args.model or settings.AGENT_MODEL,
                effort=args.effort or settings.AGENT_REASONING_EFFORT,
                max_iterations=args.max_iterations or settings.AGENT_MAX_ITERATIONS,
            )
        )
    else:
        result = run_via_api(
            transcript,
            overrides={
                "model": args.model,
                "reasoning_effort": args.effort,
                "max_iterations": args.max_iterations,
            },
        )

    trace = format_trace(result)
    print(trace)
    if args.out:
        args.out.write_text(trace, encoding="utf-8")
        print(f"Trace saved to {args.out}")
    return 0 if result.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
