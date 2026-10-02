import os

# Debe ir antes de importar app.*: llm_service lee la config al importarse.
# Asignación directa (no setdefault) para que los tests nunca usen una clave real.
os.environ["LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "test-key"