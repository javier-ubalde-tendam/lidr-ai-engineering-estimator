from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "DEBUG"

    @model_validator(mode="after")
    def validate_provider_key(self) -> "Settings":
        if self.LLM_PROVIDER == "openai":
            if not self.OPENAI_API_KEY or not self.OPENAI_API_KEY.startswith("sk-proj"):
                raise ValueError(
                    "OPENAI_API_KEY debe estar seteada y empezar por 'sk-proj' cuando LLM_PROVIDER=openai"
                )
        elif self.LLM_PROVIDER == "anthropic":
            if not self.ANTHROPIC_API_KEY or not self.ANTHROPIC_API_KEY.startswith("sk-ant"):
                raise ValueError(
                    "ANTHROPIC_API_KEY debe estar seteada y empezar por 'sk-ant' cuando LLM_PROVIDER=anthropic"
                )
        else:
            raise ValueError(f"LLM_PROVIDER no soportado: {self.LLM_PROVIDER}")
        return self


def get_settings() -> Settings:
    return Settings()

# Precios aproximados en USD por cada 1M de tokens. Referencia: pricing pages de OpenAI/Anthropic.
# Ojo: cambian con el tiempo, revisar periódicamente. Clave = alias de modelo (settings.LLM_MODEL).
MODEL_PRICING_USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def estimate_cost_usd(model: str, tokens_input: int, tokens_output: int) -> float | None:
    pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(model)
    if pricing is None:
        return None
    return (tokens_input * pricing["input"] + tokens_output * pricing["output"]) / 1_000_000