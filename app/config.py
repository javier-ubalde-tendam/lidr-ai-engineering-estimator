from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Placeholder default of PSEUDONYM_HASH_SALT; the app logs a warning at startup while it is in use
DEV_PSEUDONYM_HASH_SALT = "dev-only-insecure-salt-change-me"

# Contrato de esquema de la sesión 08: chunks.embedding es vector(1536). Cambiar de modelo, aunque
# tenga la misma dimensión, obliga a re-embeber todo el corpus: nunca se mezclan modelos.
SUPPORTED_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
DEFAULT_DATABASE_URL = "postgresql+asyncpg://estimator:estimator@localhost:5432/estimator"


class DatabaseSettings(BaseSettings):
    """Solo la URL de la BD: Alembic no debe exigir claves de LLM para migrar."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = DEFAULT_DATABASE_URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    LLM_PROVIDER: str = "openai"
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_MODEL: str = "claude-haiku-4-5"
    LLM_MAX_RETRIES: int = 1
    STRUCTURED_OUTPUT_MAX_RETRIES: int = 2
    PROMPT_VERSION: str = "v2"
    CONVERSATIONAL_PROMPT_VERSION: str = "v4"
    # Pares user+assistant que conserva la ventana deslizante del historial conversacional
    MAX_CONVERSATION_TURNS: int = 6
    # Límite de caracteres por adjunto extraído (trunca, no rechaza)
    MAX_ATTACHMENT_CHARS: int = 60000
    # Modelo barato para la segunda llamada (Instructor) que extrae el project_metadata
    METADATA_EXTRACTOR_MODEL: str = "gpt-4o-mini"
    # "heuristic" = regex sin coste; "llm" = clasificador binario (una llamada barata por turno expulsado)
    ANCHOR_DETECTION_MODE: Literal["heuristic", "llm"] = "heuristic"
    # Modelo barato para el resumen acumulativo y el clasificador de anclas
    COMPRESSION_MODEL: str = "gpt-4o-mini"
    # Modelo del critic del Actor-Critic-Boss (auditor independiente del actor)
    CRITIC_MODEL: str = "gpt-4o-mini"
    # 1 borrador + 2 reintentos dirigidos; con 2 el actor a menudo no puede resolver todos los issues
    BOSS_MAX_ITERATIONS: int = 3
    APP_ENV: str = "development"
    LOG_LEVEL: str = "DEBUG"
    REDIS_URL: str = "redis://localhost:6379/0"
    # Fuera de Docker (host/Windows); dentro de Compose se sobrescribe con el host `postgres`
    DATABASE_URL: str = DEFAULT_DATABASE_URL
    API_BASE_URL: str = "http://localhost:8000"
    EMBEDDING_MODEL: str = SUPPORTED_EMBEDDING_MODEL
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    SEMANTIC_CACHE_TTL: int = 86400
    # Modo seguro de despliegue: se calcula embedding + lookup + se logea el score, pero nunca
    # se sirve el hit ni se salta el LLM. Pásalo a False en .env cuando confíes en los scores
    # observados en los logs (semantic_cache_hit_log_only) para tu caso de uso
    SEMANTIC_CACHE_LOG_ONLY: bool = True

    # Session 10 — retrieval (vector/hybrid) and cross-encoder reranking
    RETRIEVAL_TOP_K: int = 10
    # Cosine distance (0 = same direction): chunks farther than this are not "really close"
    RETRIEVAL_DISTANCE_THRESHOLD: float = 0.6
    RETRIEVAL_SEARCH_MODE: Literal["vector", "hybrid"] = "vector"
    RERANKER_ENABLED: bool = False
    # Multilingual, small, CPU-friendly cross-encoder
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Recall-then-rerank: recall wide (50) so the relevant chunk is in the pool, keep top 5
    RETRIEVAL_RECALL_TOP_K: int = Field(default=50, ge=1)
    RERANK_TOP_N: int = Field(default=5, ge=1)
    RRF_K: int = Field(default=60, ge=1)
    # "or" = any query lexeme matches (default); "and" = plainto_tsquery semantics (all lexemes)
    LEXICAL_MATCH_MODE: Literal["or", "and"] = "or"

    # Session 9 - RAG estimation (transcript -> grounded estimate). Models go through LLMWrapper.
    # Query understanding distills the transcript (small model); generation reasons over the
    # retrieved budgets (strongest model).
    REFORMULATION_MODEL: str = "gpt-5-mini"
    GENERATION_MODEL: str = "gpt-5"
    GENERATION_REASONING_EFFORT: Literal["minimal", "low", "medium", "high"] = "medium"
    # Cap for reasoning + output tokens of the RAG structured calls: gpt-5 spends reasoning tokens
    # from this same budget, so a small value truncates the JSON (finish_reason="length")
    GENERATION_MAX_TOKENS: int = 64000
    # Token budget of the assembled <source> context block (tiktoken cl100k_base)
    MAX_CONTEXT_TOKENS: int = 16384
    # Models selectable at runtime through PUT /api/v1/config/models, filtered by the configured keys
    AVAILABLE_MODELS: list[str] = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-5",
        "gpt-5-mini",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5",
    ]
    # Idempotency cache of POST /v1/estimate/from-transcript (seconds; 24h)
    IDEMPOTENCY_TTL: int = 86400
    # Keys of the secured routers. None/empty disables the router (401 on every request):
    # generate a local key and put it in your .env
    RETRIEVAL_API_KEY: str | None = None
    ESTIMATE_API_KEY: str | None = None

    # Session 10 live - advanced multi-index retrieval; each stage is independently switchable
    RETRIEVAL_ROUTING_ENABLED: bool = True
    QUERY_TRANSFORM_ENABLED: bool = True
    # Soft re-weight; off by default: turn it on only with evidence (no magic-number boosts)
    TEMPORAL_DECAY_ENABLED: bool = False
    # Small non-reasoning models: cheap, and no reasoning-token budget to starve the JSON
    ROUTER_MODEL: str = "gpt-4o-mini"
    QUERY_TRANSFORM_MODEL: str = "gpt-4o-mini"
    # weight = 0.5 ** (age / half_life); ~2.5 years because budgets age slowly
    TEMPORAL_DECAY_HALF_LIFE_DAYS: int = 900
    QUERY_MAX_SUBQUERIES: int = 4
    ROUTER_MAX_TARGETS: int = 3
    # Per-task hours (POST /v1/estimate/tasks/hours): neighbours per task and the cosine-distance
    # floor beyond which a task gets NO hours (red flag) instead of a low-confidence guess
    TASK_HOURS_TOP_K: int = 5
    TASK_HOURS_DISTANCE_THRESHOLD: float = 0.45

    # Session 11 live - generation quality
    # Hallucination gate: proves each cited number is entailed by its chunk (numeric anchor + judge)
    HALLUCINATION_GATE_ENABLED: bool = True
    HALLUCINATION_JUDGE_MODEL: str = "gpt-5-mini"
    # A grounded line whose hours deviate from the historical anchor by more than this fraction is degraded
    HALLUCINATION_NUMERIC_TOLERANCE: float = 0.5
    # Augmentation of the retrieved chunks BEFORE generation: compression + edge-loading reorder
    AUGMENTATION_ENABLED: bool = True
    AUGMENTATION_COMPRESS: bool = True
    AUGMENTATION_REORDER: bool = True
    AUGMENTATION_MODEL: str = "gpt-5-mini"
    # Synthesis: when the sources disagree beyond this coefficient of variation, emit an hour range
    SYNTHESIS_ENABLED: bool = True
    SYNTHESIS_CONTRADICTION_THRESHOLD: float = 0.35

    # Session 7 live - chunking strategies that call an LLM (POST /embeddings/compare)
    # Decomposes each component into atomic propositions (one call per component): a small model
    PROPOSITIONAL_CHUNKER_MODEL: str = "gpt-4o-mini"
    # Situates each chunk inside its parent budget (Anthropic prompt caching on the parent document)
    CONTEXTUAL_CHUNKER_MODEL: str = "claude-sonnet-4-5"

    # Session 6 - offline ingestion (catalog -> loaders -> parsers; PII pseudonymization)
    # YAML catalog of audited data sources, resolved relative to the working directory
    CATALOG_PATH: Path = Path("data/catalog/catalog.yaml")
    # Root the catalog ``location`` entries are resolved against
    INGESTION_DATA_ROOT: Path = Path("data/seed")
    # spaCy model of the Presidio analyzer. It must be Spanish: the default English pipeline
    # silently misses Hispanic names. The model is a separate wheel (see README / Dockerfile)
    PRESIDIO_SPACY_MODEL: str = "es_core_news_md"
    PSEUDONYM_FAKER_LOCALE: str = "es_ES"
    # Salt of the HMAC that keys pseudonym_mappings. This default is a DEV PLACEHOLDER: anyone who
    # knows it can recompute the hashes of known values. Set a long random secret in the real .env.
    # Rotating it orphans every existing mapping (known values get NEW pseudonyms afterwards)
    PSEUDONYM_HASH_SALT: str = DEV_PSEUDONYM_HASH_SALT

    @model_validator(mode="after")
    def validate_embedding_model(self) -> "Settings":
        if self.EMBEDDING_MODEL != SUPPORTED_EMBEDDING_MODEL:
            raise ValueError(
                f"EMBEDDING_MODEL={self.EMBEDDING_MODEL!r} no es compatible con el esquema "
                f"vector({EMBEDDING_DIMENSION}): solo se admite {SUPPORTED_EMBEDDING_MODEL!r}"
            )
        return self

    @model_validator(mode="after")
    def validate_provider_key(self) -> "Settings":
        if self.LLM_PROVIDER == "openai":
            if not self.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY debe estar seteada cuando LLM_PROVIDER=openai"
                )
        elif self.LLM_PROVIDER == "anthropic":
            if not self.ANTHROPIC_API_KEY:
                raise ValueError(
                    "ANTHROPIC_API_KEY debe estar seteada cuando LLM_PROVIDER=anthropic"
                )
        else:
            raise ValueError(f"LLM_PROVIDER no soportado: {self.LLM_PROVIDER}")
        return self


def get_settings() -> Settings:
    return Settings()
