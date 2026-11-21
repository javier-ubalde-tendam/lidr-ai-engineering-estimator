"""Metadata filters (store SQL, retrieve forwarding) and request > runtime > Settings precedence."""

from typing import Any

import fakeredis
import pytest
from sqlalchemy.dialects import postgresql

from app.config import Settings
from app.foundation.llm.runtime_config import RuntimeRetrievalConfig
from app.generation.rag.retrieval.pipeline import resolve_search_options, retrieve
from app.generation.rag.store.repository import ChunkSearchStore
from tests.generation.rag.test_hybrid_retrieve import SESSION, VECTOR, FakeStore

pytestmark = pytest.mark.anyio


def runtime_with(settings: Settings, **overrides: str) -> RuntimeRetrievalConfig:
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    for key, value in overrides.items():
        redis_client.hset("estimator:runtime_retrieval", key, value)
    return RuntimeRetrievalConfig(redis_client, settings)


def test_precedence_is_request_then_runtime_then_settings() -> None:
    settings = Settings(
        OPENAI_API_KEY="sk-test", _env_file=None, RETRIEVAL_SEARCH_MODE="vector", RERANKER_ENABLED=False
    )
    runtime = runtime_with(settings, RETRIEVAL_SEARCH_MODE="hybrid", RERANKER_ENABLED="true")

    assert resolve_search_options(None, None, settings, runtime) == ("hybrid", True)
    assert resolve_search_options("vector", False, settings, runtime) == ("vector", False)
    assert resolve_search_options(None, False, settings, runtime) == ("hybrid", False)
    # No override stored: the runtime object falls back to Settings; without runtime too
    assert resolve_search_options(None, None, settings, runtime_with(settings)) == ("vector", False)
    assert resolve_search_options(None, None, settings) == ("vector", False)


def sql_of(filters: list[Any]) -> str:
    # literal_binds renders the values so the assertion reads like the real predicate
    from sqlalchemy import and_

    clause = and_(*filters)
    return str(clause.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


def test_no_filters_build_no_predicates() -> None:
    assert ChunkSearchStore._structural_filters(
        sectors=None, project_year_min=None, project_year_max=None, chunk_types=None
    ) == []
    assert ChunkSearchStore._structural_filters(
        sectors=[], project_year_min=None, project_year_max=None, chunk_types=[]
    ) == []


def test_each_filter_becomes_one_predicate_on_the_right_column() -> None:
    filters = ChunkSearchStore._structural_filters(
        sectors=["finance"], project_year_min=2022, project_year_max=2024, chunk_types=["x"]
    )

    sql = sql_of(filters)
    assert "(budget_chunks.metadata ->> 'client_sector') IN ('finance')" in sql
    assert "CAST((budget_chunks.metadata ->> 'year') AS INTEGER) >= 2022" in sql
    assert "CAST((budget_chunks.metadata ->> 'year') AS INTEGER) <= 2024" in sql
    assert "budget_chunks.chunk_type IN ('x')" in sql
    assert len(filters) == 4


async def test_retrieve_applies_the_same_filters_to_both_branches() -> None:
    store = FakeStore(VECTOR, lexical=VECTOR[:3])

    await retrieve(
        SESSION,
        store,
        query_embedding=[0.0],
        query_text="payments marketplace",
        search_mode="hybrid",
        sectors=["finance"],
        project_year_min=2022,
        chunk_types=["budget_component"],
    )

    expected = {
        "sectors": ["finance"],
        "project_year_min": 2022,
        "project_year_max": None,
        "chunk_types": ["budget_component"],
    }
    assert {k: store.vector_calls[0][k] for k in expected} == expected
    assert {k: store.lexical_calls[0][k] for k in expected} == expected
