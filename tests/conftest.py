import os

# Debe ir antes de importar app.*: llm_service lee la config al importarse.
# Asignación directa (no setdefault) para que los tests nunca usen una clave real.
os.environ["LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["PROMPT_VERSION"] = "v1"  # aísla los tests del PROMPT_VERSION del .env local
os.environ["STRUCTURED_OUTPUT_MAX_RETRIES"] = "2"