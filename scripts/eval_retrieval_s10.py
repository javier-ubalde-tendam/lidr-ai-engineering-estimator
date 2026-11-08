"""Session 10: measure precision@5 and latency of configurations A/B/C/D on the golden set.

    A = vector / no rerank (baseline)    B = hybrid / no rerank
    C = vector / rerank                  D = hybrid / rerank

Usage (stack up, corpus ingested, OPENAI_API_KEY in .env)::

    docker compose exec ai_service python scripts/eval_retrieval_s10.py [--config A|B|C|D|all] [--runs N]

Methodology:
* Every query goes through the SAME ``retrieve()`` the API uses.
* Each query is embedded ONCE; the embedding latency (an OpenAI round trip, identical for the
  four configurations) is measured and reported separately, not included in the per-config time.
* Permissive ``distance_threshold`` (2.0, the cosine maximum) so the threshold never truncates
  the top-5: we compare RANKING quality, not the soft-fail gate.
* ``top_k = rerank_top_n = 5``; recall width, RRF k and lexical mode come from Settings.
* The cross-encoder is loaded BEFORE measuring (its one-off load must not pollute latency).
* Per (query, configuration): 1 discarded warm-up run + N measured runs (default 5). The latency
  is the wall time of retrieve() including opening the DB session.
* precision@5 = chunks in the top-5 whose budget_id is in relevant_budget_ids / 5, averaged over
  the queries. Retrieval is deterministic, so precision comes from the last measured run.
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.db.session import dispose_engine, get_sessionmaker
from app.dependencies import get_chunk_store, get_embedder, get_reranker
from app.generation.rag.retrieval.pipeline import retrieve
from app.generation.rag.schemas import RetrievalResult, SearchMode
from evals.retrieval_metrics import percentile, precision_at_k

GOLDEN_PATH = ROOT / "evals" / "golden_retrieval.json"
OUTPUT_STEM = ROOT / "evals" / "retrieval_s10_results"
NO_THRESHOLD = 2.0
DEFAULT_RUNS = 5

# id -> (search label, rerank label, search_mode, rerank)
CONFIGS: dict[str, tuple[str, str, SearchMode, bool]] = {
    "A": ("Vectorial", "No", "vector", False),
    "B": ("Híbrida", "No", "hybrid", False),
    "C": ("Vectorial", "Sí", "vector", True),
    "D": ("Híbrida", "Sí", "hybrid", True),
}


class EvaluationAborted(Exception):
    pass


async def corpus_stats() -> tuple[int, int, set[str]]:
    async with get_sessionmaker()() as session:
        documents = await session.scalar(text("SELECT count(*) FROM documents"))
        chunks = await session.scalar(text("SELECT count(*) FROM chunks"))
        budget_ids = await session.scalars(
            text(
                "SELECT DISTINCT metadata->>'budget_id' FROM chunks "
                "WHERE metadata->>'budget_id' IS NOT NULL"
            )
        )
        return int(documents or 0), int(chunks or 0), {b for b in budget_ids if b}


def validate(queries: list[dict[str, Any]], chunks: int, budget_ids: set[str]) -> None:
    if chunks == 0:
        raise EvaluationAborted(
            "No chunks ingested. Run: docker compose run --rm ai_service python query_examples.py"
        )
    missing = {b for q in queries for b in q["relevant_budget_ids"]} - budget_ids
    if missing:
        raise EvaluationAborted(f"relevant_budget_ids not found in the corpus: {sorted(missing)}")


async def run_once(
    query: dict[str, Any], embedding: list[float], search_mode: SearchMode, rerank: bool, k: int
) -> tuple[RetrievalResult, float]:
    settings = get_settings()
    started = time.perf_counter()
    async with get_sessionmaker()() as session:
        result = await retrieve(
            session,
            get_chunk_store(),
            query_embedding=embedding,
            query_text=query["query"],
            search_mode=search_mode,
            rerank=rerank,
            top_k=k,
            recall_k=settings.RETRIEVAL_RECALL_TOP_K,
            rerank_top_n=k,
            distance_threshold=NO_THRESHOLD,
            rrf_k=settings.RRF_K,
            lexical_match_mode=settings.LEXICAL_MATCH_MODE,
            reranker=get_reranker() if rerank else None,
        )
    return result, (time.perf_counter() - started) * 1000


def device_info() -> str:
    # torch is only imported if a rerank config ran; never import it just to report
    torch = sys.modules.get("torch")
    if torch is None:
        return "cpu (torch not loaded)"
    return "cuda" if torch.cuda.is_available() else "cpu"


async def evaluate(config_ids: list[str], runs: int) -> dict[str, Any]:
    settings = get_settings()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    queries: list[dict[str, Any]] = golden["queries"]
    k = int(golden.get("k", 5))

    documents, chunks, budget_ids = await corpus_stats()
    validate(queries, chunks, budget_ids)
    print(f"Corpus: {documents} documents / {chunks} chunks. Queries: {len(queries)}")

    embedder = get_embedder()
    embeddings: dict[str, list[float]] = {}
    embedding_ms: list[float] = []
    for query in queries:
        started = time.perf_counter()
        embeddings[query["id"]] = await asyncio.to_thread(embedder.embed_one, query["query"])
        embedding_ms.append((time.perf_counter() - started) * 1000)

    if any(CONFIGS[c][3] for c in config_ids):
        print(f"Loading reranker {settings.RERANKER_MODEL} (excluded from timings)...")
        await asyncio.to_thread(get_reranker().load)

    results: dict[str, Any] = {}
    for config_id in config_ids:
        search_label, rerank_label, search_mode, rerank = CONFIGS[config_id]
        latencies: list[float] = []
        per_query: dict[str, Any] = {}
        for query in queries:
            embedding = embeddings[query["id"]]
            await run_once(query, embedding, search_mode, rerank, k)  # warm-up, discarded
            for _ in range(runs):
                result, elapsed_ms = await run_once(query, embedding, search_mode, rerank, k)
                latencies.append(elapsed_ms)
            top_budgets = [chunk.budget_id for chunk in result.chunks[:k]]
            per_query[query["id"]] = {
                "precision_at_k": precision_at_k(top_budgets, set(query["relevant_budget_ids"]), k),
                "top_budget_ids": top_budgets,
                "lexical_hits": result.lexical_hits,
                "vector_hits": result.vector_hits,
            }
        results[config_id] = {
            "search": search_label,
            "reranking": rerank_label,
            "precision_at_k": statistics.fmean(q["precision_at_k"] for q in per_query.values()),
            "latency_mean_ms": statistics.fmean(latencies),
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
            "mean_lexical_hits": statistics.fmean(q["lexical_hits"] for q in per_query.values()),
            "per_query": per_query,
        }
        print(f"Config {config_id} done")

    return {
        "metadata": {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "documents": documents,
            "chunks": chunks,
            "queries": len(queries),
            "k": k,
            "recall_k": settings.RETRIEVAL_RECALL_TOP_K,
            "rrf_k": settings.RRF_K,
            "lexical_match_mode": settings.LEXICAL_MATCH_MODE,
            "reranker_model": settings.RERANKER_MODEL,
            "distance_threshold": NO_THRESHOLD,
            "measured_runs_per_query": runs,
            "embedding_latency_mean_ms": statistics.fmean(embedding_ms),
            "device": device_info(),
            "cpu": f"{platform.machine()} x{os.cpu_count()} ({platform.system()})",
        },
        "queries": [q["id"] for q in queries],
        "results": results,
    }


def to_markdown(report: dict[str, Any]) -> str:
    meta, results = report["metadata"], report["results"]
    k = meta["k"]
    lines = [
        "# Resultados sesión 10: búsqueda híbrida + reranking",
        "",
        f"Generado por `scripts/eval_retrieval_s10.py` el {meta['date']}. No editar a mano.",
        "",
        f"| Config | Búsqueda | Reranking | Precision@{k} | Latencia media (ms) | p50 | p95 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for config_id, r in results.items():
        lines.append(
            f"| {config_id} | {r['search']} | {r['reranking']} | {r['precision_at_k']:.2f} "
            f"| {r['latency_mean_ms']:.1f} | {r['latency_p50_ms']:.1f} | {r['latency_p95_ms']:.1f} |"
        )
    lines += [
        "",
        (
            "Latencia media del embedding de la consulta (común a todas, no incluida arriba): "
            f"{meta['embedding_latency_mean_ms']:.1f} ms"
        ),
        "",
        f"## Precision@{k} por consulta",
        "",
        "| Consulta | " + " | ".join(results) + " |",
        "| --- |" + " --- |" * len(results),
    ]
    for query_id in report["queries"]:
        cells = [f"{results[c]['per_query'][query_id]['precision_at_k']:.2f}" for c in results]
        lines.append(f"| {query_id} | " + " | ".join(cells) + " |")

    hybrid = [c for c in results if CONFIGS[c][2] == "hybrid"]
    if hybrid:
        lines += ["", "## lexical_hits medios por consulta (configuraciones híbridas)", ""]
        lines += [f"- {c}: {results[c]['mean_lexical_hits']:.1f}" for c in hybrid]

    lines += ["", "## Metadatos", ""]
    lines += [f"- {key}: {value}" for key, value in meta.items()]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="measured runs per query")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    return args


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Hide the per-call rag_retrieve_done info logs: the report is the output
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    config_ids = list(CONFIGS) if args.config == "all" else [args.config]
    try:
        report = await evaluate(config_ids, args.runs)
    except EvaluationAborted as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 1
    finally:
        await dispose_engine()

    markdown = to_markdown(report)
    OUTPUT_STEM.with_suffix(".md").write_text(markdown, encoding="utf-8")
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print(markdown)
    print(f"Saved {OUTPUT_STEM.with_suffix('.md')} and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
