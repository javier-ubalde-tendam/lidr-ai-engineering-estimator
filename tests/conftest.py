import os

# Debe ir antes de importar app.*: llm_service lee la config al importarse.
# Asignación directa (no setdefault) para que los tests nunca usen una clave real.
os.environ["LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["PROMPT_VERSION"] = "v1"  # aísla los tests del PROMPT_VERSION del .env local
os.environ["CONVERSATIONAL_PROMPT_VERSION"] = "v4"  # idem para el camino conversacional
os.environ["ANCHOR_DETECTION_MODE"] = "heuristic"  # sin llamadas LLM extra en los tests
os.environ["BOSS_MAX_ITERATIONS"] = "3"
os.environ["STRUCTURED_OUTPUT_MAX_RETRIES"] = "2"
os.environ["LOG_LEVEL"] = "INFO"

from app.logging_config import configure_logging  # noqa: E402

# Sin configurar, structlog no filtra por nivel y logger.log(TRACE, ...) lanza KeyError
configure_logging()