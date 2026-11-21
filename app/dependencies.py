"""Composition root: cached factories for every shared singleton and the per-request services.

Only ``app/api`` and the tests import this module. Tests replace pieces with
``app.dependency_overrides[get_xxx]``.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import anthropic
import tiktoken
from fastapi import Depends
from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.domain.estimation_service import EstimationService
from app.foundation.llm.runtime_config import RuntimeModelConfig, RuntimeRetrievalConfig
from app.foundation.llm.wrapper import LLMWrapper
from app.foundation.persistence.database import get_db_session, get_sessionmaker
from app.generation.cag.exact import ExactCache
from app.generation.cag.semantic import SemanticCache
from app.generation.conversation.store import SessionStore
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.chunking.strategies import (
    ContextualRetrievalChunker,
    FixedSizeChunker,
    HierarchicalChunker,
    PropositionalChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceWindowChunker,
)
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.context_assembler import TokenEncoder
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.estimator import TranscriptEstimator
from app.generation.rag.idempotency import IdempotencyStore
from app.generation.rag.index_jobs import IndexJobStore
from app.generation.rag.index_service import CorpusIndexService, SessionIngester
from app.generation.rag.ingest_service import IngestionService
from app.generation.rag.retrieval.reranker import CrossEncoderReranker, default_reranker
from app.generation.rag.retriever import Retriever
from app.generation.rag.store.repository import ChunkSearchStore
from app.generation.rag.task_hours import TaskHoursEstimator
from app.ingestion.catalog import DataCatalog, load_catalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.registry import ParserRegistry, default_registry

if TYPE_CHECKING:
    from app.ingestion.pii import ConsistentPseudonymizer


@lru_cache
def get_openai_client() -> OpenAI:
    # lru_cache sin argumentos = singleton perezoso: se construye una sola vez, en el primer uso
    settings = get_settings()
    return OpenAI(api_key=settings.OPENAI_API_KEY)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    # Sin settings fijos: lee la configuración vigente en cada llamada
    return LLMWrapper()


@lru_cache
def get_runtime_config() -> RuntimeModelConfig:
    # redis.from_url is lazy (no connection until the first command), so building it is safe
    # even when Redis is down; Redis is the source of truth, nothing is cached in the object
    settings = get_settings()
    return RuntimeModelConfig.from_url(settings.REDIS_URL, settings)


@lru_cache
def get_runtime_retrieval_config() -> RuntimeRetrievalConfig:
    settings = get_settings()
    return RuntimeRetrievalConfig.from_url(settings.REDIS_URL, settings)


@lru_cache
def get_cache() -> ExactCache:
    return ExactCache()


def get_semantic_cache(openai_client: OpenAI = Depends(get_openai_client)) -> SemanticCache:
    # No se cachea: es un envoltorio sin estado y debe seguir al cliente OpenAI (sustituible en tests)
    return SemanticCache(openai_client)


@lru_cache
def get_session_store() -> SessionStore:
    # Mismo patrón que get_openai_client: singleton perezoso, sustituible en tests vía
    # app.dependency_overrides[get_session_store]
    settings = get_settings()
    return SessionStore(max_conversation_turns=settings.MAX_CONVERSATION_TURNS)


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    # Singleton perezoso: el chunker carga el tokenizer una sola vez
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    # Reutiliza el cliente OpenAI compartido; el embedder solo conoce la interfaz del cliente
    return OpenAIEmbedder(client=get_openai_client(), model=get_settings().EMBEDDING_MODEL)


@lru_cache
def get_chunk_store() -> ChunkSearchStore:
    # Stateless read-only store (session 10): one instance for the whole app
    return ChunkSearchStore()


def get_reranker() -> CrossEncoderReranker:
    # Same singleton the retrieval pipeline falls back to; the model loads on the first rerank
    return default_reranker()


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    # The app-wide sessionmaker (singleton in foundation/persistence): for code that opens its
    # own sessions instead of receiving one per request
    return get_sessionmaker()


@lru_cache
def get_idempotency_store() -> IdempotencyStore:
    # Redis when reachable at first use, in-process dict otherwise (decided once per process)
    return IdempotencyStore.from_settings(get_settings())


@lru_cache
def get_token_encoder() -> TokenEncoder:
    # cl100k_base: the tokenizer of the context token budget (loaded once, lazily)
    return tiktoken.get_encoding("cl100k_base")


@lru_cache
def get_index_job_store() -> IndexJobStore:
    # Per-process registry of the corpus-expansion jobs (the polling client reads them back)
    return IndexJobStore()


@lru_cache
def get_catalog() -> DataCatalog:
    # Read once: re-reading would change the include/exclude decisions of a running process.
    # Rolling out a new catalog version means restarting the service
    return load_catalog(get_settings().CATALOG_PATH)


@lru_cache
def get_filesystem_loader() -> FileSystemLoader:
    return FileSystemLoader(data_root=get_settings().INGESTION_DATA_ROOT)


@lru_cache
def get_parser_registry() -> ParserRegistry:
    return default_registry()


def build_pseudonymizer(session: AsyncSession) -> "ConsistentPseudonymizer":
    """Pseudonymizer backed by the Postgres mapping table, on the caller's session.

    Not wired to any endpoint yet (the orchestrator does not pseudonymize): scripts and future
    callers build it here. Presidio/spaCy are imported lazily so the app starts without the
    spaCy model; the model only loads when ``build_analyzer`` runs.
    """
    from app.ingestion.pii import ConsistentPseudonymizer, PostgresMappingStore, build_analyzer

    settings = get_settings()
    return ConsistentPseudonymizer(
        analyzer=build_analyzer(),
        mapping_store=PostgresMappingStore(session),
        salt=settings.PSEUDONYM_HASH_SALT,
        faker_locale=settings.PSEUDONYM_FAKER_LOCALE,
    )


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic:
    # Only the Contextual Retrieval chunker needs it (prompt caching is not exposed by LLMWrapper)
    return anthropic.Anthropic(api_key=get_settings().ANTHROPIC_API_KEY)


# Chunking strategies of POST /embeddings/compare. The stateless ones are cheap to build; the
# ones that call an external provider receive their collaborators here and fail fast with a
# RuntimeError when the key is missing. None is cached: they are rebuilt per compare request so
# a runtime model override (RuntimeModelConfig) takes effect on the next call.
def get_semantic_chunker() -> SemanticChunker:
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("SemanticChunker requires OPENAI_API_KEY.")
    # Imported here: langchain-openai is only needed by this strategy
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
    return SemanticChunker(embeddings)


def get_propositional_chunker() -> PropositionalChunker:
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("PropositionalChunker requires OPENAI_API_KEY.")
    model = get_runtime_config().effective("PROPOSITIONAL_CHUNKER_MODEL")
    return PropositionalChunker(llm=get_llm_wrapper(), model=model)


def get_contextual_retrieval_chunker() -> ContextualRetrievalChunker:
    if not get_settings().ANTHROPIC_API_KEY:
        raise RuntimeError("ContextualRetrievalChunker requires ANTHROPIC_API_KEY.")
    model = get_runtime_config().effective("CONTEXTUAL_CHUNKER_MODEL")
    return ContextualRetrievalChunker(client=get_anthropic_client(), model=model)


# Registry: strategy name -> factory. ``structural`` reuses ``get_chunker`` (the ingestion one).
CHUNKER_FACTORIES: dict[str, Callable[[], Chunker]] = {
    "structural": get_chunker,
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "sentence_window": SentenceWindowChunker,
    "semantic": get_semantic_chunker,
    "propositional": get_propositional_chunker,
    "contextual_retrieval": get_contextual_retrieval_chunker,
    "hierarchical": HierarchicalChunker,
}
ALL_STRATEGIES = list(CHUNKER_FACTORIES)


def build_chunkers(names: list[str]) -> list[Chunker]:
    """Instantiate the requested chunkers by name.

    Raises ``KeyError`` (the unknown name) or ``RuntimeError`` (a provider key is missing).
    """
    chunkers: list[Chunker] = []
    for name in names:
        factory = CHUNKER_FACTORIES.get(name)
        if factory is None:
            raise KeyError(name)
        chunkers.append(factory())
    return chunkers


def get_chunker_builder() -> Callable[[list[str]], list[Chunker]]:
    # A dependency (instead of a direct import in the router) so tests swap the whole registry
    return build_chunkers


# Servicios por petición: comparten la AsyncSession (una por request) y los singletons anteriores.
# Los tests los sustituyen enteros o solo get_db_session/get_embedder.
def get_ingestion_service(
    session: AsyncSession = Depends(get_db_session),
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> IngestionService:
    return IngestionService(session=session, chunker=chunker, embedder=embedder)


def get_retriever(
    session: AsyncSession = Depends(get_db_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> Retriever:
    return Retriever(session=session, embedder=embedder)


def get_transcript_estimator(
    embedder: OpenAIEmbedder = Depends(get_embedder),
    store: ChunkSearchStore = Depends(get_chunk_store),
    reranker: CrossEncoderReranker = Depends(get_reranker),
) -> TranscriptEstimator:
    # Not cached: it only wires singletons, and per-request construction follows the test overrides
    return TranscriptEstimator(
        llm=get_llm_wrapper(),
        models=get_runtime_config(),
        retrieval_config=get_runtime_retrieval_config(),
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=store,
        idempotency=get_idempotency_store(),
        token_encoder=get_token_encoder(),
        settings=get_settings(),
        reranker=reranker,
    )


def get_task_hours_estimator(
    embedder: OpenAIEmbedder = Depends(get_embedder),
    store: ChunkSearchStore = Depends(get_chunk_store),
    reranker: CrossEncoderReranker = Depends(get_reranker),
) -> TaskHoursEstimator:
    return TaskHoursEstimator(
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=store,
        retrieval_config=get_runtime_retrieval_config(),
        settings=get_settings(),
        reranker=reranker,
    )


def get_corpus_index_service(
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> CorpusIndexService:
    return CorpusIndexService(SessionIngester(get_async_session_factory(), chunker, embedder))


def get_estimation_service(
    openai_client: OpenAI = Depends(get_openai_client),
    semantic_cache: SemanticCache = Depends(get_semantic_cache),
) -> EstimationService:
    # Ligero (solo cablea): se construye por petición para que respete los overrides de los tests
    return EstimationService(
        openai_client=openai_client,
        llm_wrapper=get_llm_wrapper(),
        exact_cache=get_cache(),
        semantic_cache=semantic_cache,
        settings=get_settings(),
    )
