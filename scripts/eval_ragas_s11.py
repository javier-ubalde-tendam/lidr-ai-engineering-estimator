#!/usr/bin/env python3
"""RAGAS generation-quality baseline for the Session 11 golden set.

For each query in ``evals/golden_generation_s11.json`` this runs the REAL RAG generation pipeline
(reformulate -> embed -> retrieve -> assemble -> generate) and records the four inputs RAGAS needs:

    question      -> the estimation request (golden ``query``)
    answer        -> the estimate the pipeline generated, rendered as text
    contexts      -> the retrieved chunks placed in the context block
    ground_truth  -> the expert reference estimate (golden ``ground_truth``)

It then computes the four RAGAS metrics (faithfulness, answer_relevancy, context_precision,
context_recall) with an OpenAI chat model as judge and ``text-embedding-3-small`` for the
embedding-based metrics, and prints a table with one row per query plus an average row. The table
is the generation-quality BASELINE (see ``evals/RAGAS_BASELINE_S11.md``).

The collaborators come from the composition root (``app.dependencies``), exactly as the API builds
them: this script only drives the stages, it does not reimplement them.

Prerequisites (host run): a ``.env`` with OPENAI_API_KEY, the database up with the migrations
applied and the budget corpus ingested (``uv run python query_examples.py``), and the dev
dependencies installed (``uv sync --extra dev`` pulls ragas + datasets).

PAID: every run calls OpenAI (reformulation, generation, embeddings and the RAGAS judge).

Usage::

    uv run python scripts/eval_ragas_s11.py
    uv run python scripts/eval_ragas_s11.py --limit 2 --out evals/ragas_baseline_s11.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from app.generation.rag.schemas import Estimate

GOLDEN_PATH = ROOT / "evals" / "golden_generation_s11.json"
# Permissive threshold: the generator must see context for every query, so RAGAS grades
# GENERATION and not the soft-fail gate
GENERATION_THRESHOLD = 1.2
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def render_estimate_text(estimate: Estimate) -> str:
    """Render an ``Estimate`` as the plain-text 'answer' RAGAS scores."""
    if estimate.confidence == "insufficient":
        return (
            "Insufficient context to produce an estimate. "
            f"{estimate.insufficient_context_explanation or ''}"
        ).strip()

    lines: list[str] = []
    if estimate.total_engineer_days is not None:
        lines.append(f"Total: {estimate.total_engineer_days} engineer-days.")
    if estimate.duration_weeks is not None:
        lines.append(f"Duration: {estimate.duration_weeks} weeks.")
    lines.append(f"Confidence: {estimate.confidence}.")
    for module in estimate.modules:
        lines.append(f"Module {module.name}:")
        for task in module.tasks:
            days = "n/a" if task.engineer_days is None else f"{task.engineer_days}d"
            tag = "grounded" if task.grounded else "no-data"
            lines.append(f"  - {task.name}: {days} [{tag}]")
    if estimate.assumptions:
        lines.append("Assumptions: " + "; ".join(a.description for a in estimate.assumptions))
    lines.append(f"Reasoning: {estimate.reasoning}")
    return "\n".join(lines)


async def collect_sample(
    item: dict[str, Any], *, augment: bool = False, gate: bool = False
) -> dict[str, Any]:
    """Run the pipeline for one golden item and return a RAGAS-ready sample.

    ``augment`` / ``gate`` switch the Session 11 stages explicitly (the harness decides, not the
    runtime config), so a configuration is reproducible whatever Redis or .env say.
    """
    from app.config import get_settings
    from app.dependencies import (
        get_async_session_factory,
        get_chunk_store,
        get_embedder,
        get_llm_wrapper,
        get_reranker,
        get_runtime_config,
        get_token_encoder,
    )
    from app.generation.rag.context_assembler import build_context_block, truncate_to_token_budget
    from app.generation.rag.prompt_builder import build_system_prompt, build_user_message
    from app.generation.rag.quality.augmentation import augment_chunks
    from app.generation.rag.quality.hallucination import gate_estimate
    from app.generation.rag.query_reformulator import compose_search_text, reformulate_query
    from app.generation.rag.retrieval.pipeline import retrieve
    from app.generation.rag.schemas import Estimate
    from app.generation.rag.validation import verify_citations

    settings = get_settings()
    llm, models = get_llm_wrapper(), get_runtime_config()
    transcript = item.get("transcript") or item["query"]

    query = await reformulate_query(
        transcript,
        llm=llm,
        model=models.effective("REFORMULATION_MODEL"),
        max_tokens=settings.GENERATION_MAX_TOKENS,
    )
    search_text = compose_search_text(query)
    embedding = await asyncio.to_thread(get_embedder().embed_one, search_text)

    async with get_async_session_factory()() as session:
        retrieval = await retrieve(
            session,
            get_chunk_store(),
            query_embedding=embedding,
            query_text=search_text,
            search_mode="hybrid",
            rerank=True,
            top_k=settings.RETRIEVAL_TOP_K,
            recall_k=settings.RETRIEVAL_RECALL_TOP_K,
            rerank_top_n=settings.RERANK_TOP_N,
            distance_threshold=GENERATION_THRESHOLD,
            rrf_k=settings.RRF_K,
            lexical_match_mode=settings.LEXICAL_MATCH_MODE,
            reranker=get_reranker(),
        )

    kept = truncate_to_token_budget(
        retrieval.chunks, settings.MAX_CONTEXT_TOKENS, get_token_encoder()
    )
    if augment:
        kept = augment_chunks(
            kept, compress=settings.AUGMENTATION_COMPRESS, reorder=settings.AUGMENTATION_REORDER
        )
    context_block = build_context_block(kept)

    estimate, _meta = await asyncio.to_thread(
        llm.complete_structured,
        system_prompt=build_system_prompt(),
        user_message=build_user_message(context_block, query),
        response_model=Estimate,
        model=models.effective("GENERATION_MODEL"),
        reasoning_effort=settings.GENERATION_REASONING_EFFORT,
        max_tokens=settings.GENERATION_MAX_TOKENS,
        purpose="rag_generation",
    )
    citation = verify_citations(estimate, {str(chunk.id) for chunk in kept})
    hallucination = None
    if gate:
        hallucination = await gate_estimate(
            estimate,
            kept,
            llm=llm,
            tolerance=settings.HALLUCINATION_NUMERIC_TOLERANCE,
            judge_model=models.effective("HALLUCINATION_JUDGE_MODEL"),
            max_tokens=settings.GENERATION_MAX_TOKENS,
        )

    return {
        "id": item["id"],
        "question": item["query"],
        "answer": render_estimate_text(estimate),
        "contexts": [chunk.content for chunk in kept],
        "ground_truth": item["ground_truth"],
        "citation_report": citation.model_dump(),
        "hallucination_report": hallucination.model_dump() if hallucination else None,
    }


def require_openai_key() -> None:
    """Exit with an actionable message instead of failing deep inside the first OpenAI call."""
    from app.config import get_settings

    if not get_settings().OPENAI_API_KEY:
        raise SystemExit("ERROR: OPENAI_API_KEY is missing: set it in the repo-root .env file.")


def run_ragas(samples: list[dict[str, Any]], judge_model: str, embedding_model: str) -> Any:
    """Compute the four RAGAS metrics over the collected samples."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_list(
        [
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s["ground_truth"],
            }
            for s in samples
        ]
    )
    judge = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
    )


def _per_query_scores(result: Any, n: int) -> list[dict[str, float]]:
    """Per-query metric scores from a RAGAS result, tolerant to the library version."""
    df = result.to_pandas()
    return [
        {
            metric: float(df.iloc[i][metric]) if metric in df.columns else float("nan")
            for metric in METRIC_NAMES
        }
        for i in range(n)
    ]


def _markdown_table(ids: list[str], rows: list[dict[str, float]]) -> str:
    header = "| query | " + " | ".join(METRIC_NAMES) + " |"
    sep = "|" + "---|" * (len(METRIC_NAMES) + 1)
    body = []
    for qid, row in zip(ids, rows, strict=True):
        cells = " | ".join(f"{row[m]:.3f}" for m in METRIC_NAMES)
        body.append(f"| {qid} | {cells} |")
    avg = {m: sum(r[m] for r in rows) / len(rows) for m in METRIC_NAMES}
    avg_cells = " | ".join(f"{avg[m]:.3f}" for m in METRIC_NAMES)
    body.append(f"| **average** | {avg_cells} |")
    return "\n".join([header, sep, *body])


def score_samples(
    samples: list[dict[str, Any]], judge_model: str, embedding_model: str, out: str | None
) -> None:
    """Score pre-collected samples with RAGAS and print/persist the table.

    Split out so it can run in an ISOLATED venv (see ``score_ragas_s11.py``): ragas pins a
    langchain stack that can conflict with the app's.
    """
    result = run_ragas(samples, judge_model, embedding_model)
    rows = _per_query_scores(result, len(samples))
    table = _markdown_table([s["id"] for s in samples], rows)

    print("\n=== RAGAS generation baseline ===")
    print(table)

    if out:
        report = {
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "table_markdown": table,
            "per_query": [
                {"id": s["id"], **rows[i], "citation_report": s.get("citation_report")}
                for i, s in enumerate(samples)
            ],
        }
        Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {out}")


async def _amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N queries.")
    parser.add_argument("--out", type=str, default=None, help="Write the scored report as JSON.")
    parser.add_argument(
        "--collect-only",
        type=str,
        default=None,
        metavar="PATH",
        help="Only run the app pipeline and dump samples to PATH (no RAGAS). Use this in the "
        "project venv, then score the file in an isolated venv with --score-file.",
    )
    parser.add_argument(
        "--score-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Skip collection; load samples from PATH and only run RAGAS scoring.",
    )
    args = parser.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    judge_model = golden.get("judge_model", "gpt-4o-mini")
    embedding_model = golden.get("embedding_model", "text-embedding-3-small")

    if args.score_file:
        payload = json.loads(Path(args.score_file).read_text(encoding="utf-8"))
        samples = payload["samples"] if isinstance(payload, dict) else payload
        score_samples(samples, judge_model, embedding_model, args.out)
        return 0

    items = golden["queries"][: args.limit] if args.limit else golden["queries"]
    require_openai_key()

    print(f"Collecting {len(items)} samples through the generation pipeline...")
    samples = []
    for item in items:
        print(f"  - {item['id']}: {item['query'][:70]}...")
        samples.append(await collect_sample(item))

    if args.collect_only:
        Path(args.collect_only).write_text(
            json.dumps(
                {"judge_model": judge_model, "embedding_model": embedding_model, "samples": samples},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {len(samples)} samples to {args.collect_only} (run --score-file to score).")
        return 0

    print(f"\nScoring with RAGAS (judge={judge_model}, embeddings={embedding_model})...")
    score_samples(samples, judge_model, embedding_model, args.out)
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
