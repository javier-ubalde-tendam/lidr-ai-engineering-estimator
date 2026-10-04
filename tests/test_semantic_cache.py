from app import semantic_cache
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    Phase,
    ProjectType,
)
from app.semantic_cache import build_semantic_bucket, semantic_cache_lookup, semantic_cache_store
from redisvl.exceptions import RedisVLError

GOOD_RESULT = EstimationResult(
    summary="Mobile booking app with online payments.",
    confidence_pct=80,
    phases=[Phase(name="Build", duration_weeks=3, cost_eur=6000, summary="Build the app.")],
    total_duration_weeks=3,
    total_cost_eur=6000,
)


class FakeEmbeddingResponse:
    def __init__(self, embedding: list[float]) -> None:
        self.data = [type("EmbeddingData", (), {"embedding": embedding})()]


class FakeEmbeddings:
    def create(self, model: str, input: str) -> FakeEmbeddingResponse:
        # El valor del vector no importa: FakeIndex.query devuelve resultados "enlatados"
        return FakeEmbeddingResponse([0.1] * semantic_cache.EMBEDDING_DIMS)


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


class FakeIndex:
    """Sustituye al SearchIndex real: sin Redis/RediSearch de por medio en estos tests."""

    def __init__(self, query_results: list[dict] | None = None) -> None:
        self._query_results = query_results if query_results is not None else []
        self.loaded: list[tuple[list[dict], int | None]] = []

    def query(self, query) -> list[dict]:
        return self._query_results

    def load(self, data: list[dict], ttl: int | None = None) -> list[str]:
        self.loaded.append((data, ttl))
        return ["fake-key"]


class FlakyIndex(FakeIndex):
    """Simula un índice ausente (p.ej. tras un reinicio del contenedor Redis) que se recupera al recrearlo."""

    def __init__(self, query_results: list[dict] | None = None) -> None:
        super().__init__(query_results)
        self.create_calls = 0
        self._attempts = 0

    def create(self) -> None:
        self.create_calls += 1

    def query(self, query) -> list[dict]:
        self._attempts += 1
        if self._attempts == 1:
            raise RedisVLError("Error while searching: No such index semantic_cache")
        return super().query(query)

    def load(self, data: list[dict], ttl: int | None = None) -> list[str]:
        self._attempts += 1
        if self._attempts == 1:
            raise RedisVLError("Error while searching: No such index semantic_cache")
        return super().load(data, ttl)


class AlwaysMissingIndex(FakeIndex):
    """El índice sigue sin existir incluso después de intentar recrearlo (Redis realmente caído)."""

    def create(self) -> None:
        pass

    def query(self, query) -> list[dict]:
        raise RedisVLError("Error while searching: No such index semantic_cache")

    def load(self, data: list[dict], ttl: int | None = None) -> list[str]:
        raise RedisVLError("Error while searching: No such index semantic_cache")


def make_request(**overrides) -> EstimationRequest:
    fields = {
        "description": "Mobile app with login, chat and push notifications.",
        "project_type": ProjectType.MOBILE_APP,
        "detail_level": DetailLevel.SUMMARY,
        "output_format": OutputFormat.NARRATIVE,
    }
    return EstimationRequest(**{**fields, **overrides})


def test_bucket_differs_with_output_format_even_for_the_same_description():
    narrative = make_request(output_format=OutputFormat.NARRATIVE)
    phases_table = make_request(output_format=OutputFormat.PHASES_TABLE)

    assert build_semantic_bucket(narrative, "v2") != build_semantic_bucket(phases_table, "v2")


def test_lookup_misses_when_the_bucket_is_empty(monkeypatch):
    monkeypatch.setattr(semantic_cache, "index", FakeIndex(query_results=[]))

    result = semantic_cache_lookup(make_request(), "v2", FakeOpenAIClient())

    assert result is None


def test_lookup_misses_when_similarity_is_below_threshold(monkeypatch):
    # distancia 0.5 -> similaridad 0.5, por debajo del threshold por defecto (0.92)
    monkeypatch.setattr(
        semantic_cache,
        "index",
        FakeIndex(query_results=[{"result_json": GOOD_RESULT.model_dump_json(), "vector_distance": "0.5"}]),
    )

    result = semantic_cache_lookup(make_request(), "v2", FakeOpenAIClient())

    assert result is None


def test_lookup_serves_the_hit_when_log_only_is_disabled(monkeypatch):
    monkeypatch.setattr(semantic_cache.settings, "SEMANTIC_CACHE_LOG_ONLY", False)
    monkeypatch.setattr(
        semantic_cache,
        "index",
        FakeIndex(query_results=[{"result_json": GOOD_RESULT.model_dump_json(), "vector_distance": "0.01"}]),
    )

    result = semantic_cache_lookup(make_request(), "v2", FakeOpenAIClient())

    assert result == GOOD_RESULT


def test_lookup_does_not_serve_the_hit_in_log_only_mode(monkeypatch):
    monkeypatch.setattr(semantic_cache.settings, "SEMANTIC_CACHE_LOG_ONLY", True)
    monkeypatch.setattr(
        semantic_cache,
        "index",
        FakeIndex(query_results=[{"result_json": GOOD_RESULT.model_dump_json(), "vector_distance": "0.01"}]),
    )

    result = semantic_cache_lookup(make_request(), "v2", FakeOpenAIClient())

    assert result is None  # log-only: se mide pero nunca se sirve


def test_store_saves_the_bucket_and_the_serialized_result(monkeypatch):
    fake_index = FakeIndex()
    monkeypatch.setattr(semantic_cache, "index", fake_index)
    request = make_request()

    semantic_cache_store(request, GOOD_RESULT, "v2", FakeOpenAIClient())

    [(data, ttl)] = fake_index.loaded
    assert data[0]["bucket"] == build_semantic_bucket(request, "v2")
    assert data[0]["result_json"] == GOOD_RESULT.model_dump_json()
    assert ttl == semantic_cache.settings.SEMANTIC_CACHE_TTL


# --- Auto-recuperación cuando el índice desaparece (p.ej. tras un reinicio del contenedor Redis) ---


def test_lookup_recreates_the_index_and_retries_when_it_is_missing(monkeypatch):
    monkeypatch.setattr(semantic_cache.settings, "SEMANTIC_CACHE_LOG_ONLY", False)
    flaky_index = FlakyIndex(
        query_results=[{"result_json": GOOD_RESULT.model_dump_json(), "vector_distance": "0.01"}]
    )
    monkeypatch.setattr(semantic_cache, "index", flaky_index)

    result = semantic_cache_lookup(make_request(), "v2", FakeOpenAIClient())

    assert result == GOOD_RESULT
    assert flaky_index.create_calls == 1  # se recreó una vez y el reintento tuvo éxito


def test_lookup_misses_softly_when_the_index_stays_missing(monkeypatch):
    monkeypatch.setattr(semantic_cache, "index", AlwaysMissingIndex())

    result = semantic_cache_lookup(make_request(), "v2", FakeOpenAIClient())

    assert result is None  # recrear tampoco ayudó (Redis realmente caído): se degrada a miss, no revienta


def test_store_recreates_the_index_and_retries_when_it_is_missing(monkeypatch):
    flaky_index = FlakyIndex()
    monkeypatch.setattr(semantic_cache, "index", flaky_index)

    semantic_cache_store(make_request(), GOOD_RESULT, "v2", FakeOpenAIClient())

    assert flaky_index.create_calls == 1
    assert len(flaky_index.loaded) == 1  # el reintento sí llegó a guardar el documento


def test_store_fails_softly_when_the_index_stays_missing(monkeypatch):
    monkeypatch.setattr(semantic_cache, "index", AlwaysMissingIndex())

    semantic_cache_store(make_request(), GOOD_RESULT, "v2", FakeOpenAIClient())  # no debe lanzar

