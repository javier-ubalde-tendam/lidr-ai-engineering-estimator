"""The task-corpus generator and its fixtures: valid, deterministic, ingestable, task-granular."""

import json
import subprocess
import sys
from pathlib import Path

from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.schemas import Budget, IngestRequest
from scripts.build_task_corpus import CHUNK_TYPE, DOCUMENT_TYPE, OUT_PATH, generate_corpus

ROOT = Path(__file__).resolve().parents[1]
ALL_SECTORS = {
    "finance",
    "ecommerce",
    "healthcare",
    "industrial",
    "logistics",
    "education",
    "media",
    "government",
}


def test_generated_projects_validate_as_budgets_with_a_module_per_task() -> None:
    budgets = [Budget.model_validate(project) for project in generate_corpus(count=6, seed=1)]

    assert len(budgets) == 6
    for budget in budgets:
        assert all(component.module for component in budget.components)
        assert budget.total_estimated_hours == sum(c.estimated_hours for c in budget.components)


def test_a_generated_project_is_accepted_by_the_ingest_contract() -> None:
    project = generate_corpus(count=1, seed=3)[0]

    request = IngestRequest.model_validate(
        {
            "source_path": f"data/task_corpus.json::{project['budget_id']}",
            "document_type": DOCUMENT_TYPE,
            "chunk_type": CHUNK_TYPE,
            "content": project,
        }
    )

    assert request.chunk_type == "historical_task"


def test_generation_is_deterministic_for_a_seed() -> None:
    assert generate_corpus(count=4, seed=7) == generate_corpus(count=4, seed=7)
    assert generate_corpus(count=4, seed=7) != generate_corpus(count=4, seed=8)


def test_budget_ids_are_unique_and_the_corpus_is_task_granular() -> None:
    corpus = generate_corpus(count=12, seed=90)

    ids = [project["budget_id"] for project in corpus]
    assert len(ids) == len(set(ids))
    assert sum(len(project["components"]) for project in corpus) >= 120
    assert len({c["module"] for project in corpus for c in project["components"]}) >= 8


def test_the_committed_corpus_is_the_default_generation() -> None:
    committed = json.loads(OUT_PATH.read_text(encoding="utf-8"))

    assert committed == generate_corpus()
    budgets = [Budget.model_validate(project) for project in committed]
    assert len(budgets) == 60 and sum(len(b.components) for b in budgets) >= 800
    assert {b.client_metadata.sector for b in budgets} == ALL_SECTORS


def test_each_task_becomes_one_chunk_carrying_its_module_and_hours() -> None:
    budget = Budget.model_validate(generate_corpus(count=1, seed=5)[0])

    chunks = JSONStructuralChunker().chunk([budget])

    assert len(chunks) == len(budget.components)
    for chunk, component in zip(chunks, budget.components, strict=True):
        assert f"Module: {component.module}\n" in chunk.text
        assert chunk.metadata["module"] == component.module
        assert chunk.metadata["estimated_hours"] == component.estimated_hours


def test_the_contradictions_fixture_validates_and_shares_the_corpus_tags() -> None:
    fixture = json.loads((ROOT / "data" / "task_corpus_contradictions.json").read_text("utf-8"))

    budgets = [Budget.model_validate(project) for project in fixture]

    hours = {c.estimated_hours for b in budgets for c in b.components}
    assert {40, 90} <= hours
    assert all(c.module for b in budgets for c in b.components)


def test_build_contradictions_script_no_longer_depends_on_a_missing_extra() -> None:
    # --help must exit 0 without touching the network or printing the old "8c" dependency message
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_contradictions_s11.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0
    assert "not part" not in result.stderr
    from scripts import build_contradictions_s11  # noqa: F401  (import must not raise)
