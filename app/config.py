from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
