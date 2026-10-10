from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    API_BASE_URL: str = "http://localhost:8000"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    SEMANTIC_CACHE_TTL: int = 86400
    # Modo seguro de despliegue: se calcula embedding + lookup + se logea el score, pero nunca
    # se sirve el hit ni se salta el LLM. Pásalo a False en .env cuando confíes en los scores
    # observados en los logs (semantic_cache_hit_log_only) para tu caso de uso
    SEMANTIC_CACHE_LOG_ONLY: bool = True

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
