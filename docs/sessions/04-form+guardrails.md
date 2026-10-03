# Session 04 - Form + Guardrails

## Funcionalidad incremental

Se añade:
- Formulario (en vez de chat) en la aplicación Streamlit + Schema en la petición (Pydantic)
- Prompts mediante templates Jinja2 versionados (v1/system.j2, user.j2, examples.j2) + loader + tests
     - system.j2   : rol del modelo, instrucciones generales, cómo formatear la salida
     - user.j2     : el bloque que envuelve la descripción del proyecto del usuario
     - examples.j2 : dos o tres ejemplos few-shot de estimaciones bien formadas
     - loader      : resuelve versión y renderiza
- JSON estructurado en la salida del LLM: EstimationResult + Instructor para salida estructurada
- Guardrails: input (moderation + injection), scope en prompt, validators de output, política de fallo declarada por guardrail
- Cache semántico con Redis/redisvl, bucket + embedding, solo tras guardrails, log-only primero


## Tecnologías

- Pydantic: define los modelos tipados (EstimationRequest, EstimationResult, Phase) que sirven a la vez de validación, JSON Schema para el LLM y documentación OpenAPI
- Jinja2: motor de plantillas para componer el prompt (system.j2, user.j2, examples.j2) separando estructura fija, variables y parámetros
- Instructor: envuelve el cliente del LLM (OpenAI/Anthropic/LiteLLM) para forzar que la respuesta cumpla un modelo Pydantic, con reintentos automáticos si falla la validación
- LiteLLM: agregador que normaliza la API de distintos proveedores (Mistral, Gemini, DeepSeek, etc.) bajo una interfaz común
- OpenAI Moderation API: clasifica el texto de entrada contra categorías de contenido tóxico; gratis y rápida (~50-100ms), primera capa del guardrail de input
- Guardrails AI: librería con un Hub de validators preconstruidos (PII, toxicidad, citas inventadas) y políticas de fallo (on_fail) para validar el output del LLM
- Redis: almacén en memoria usado como cache semántico (y exact-match de sesiones previas); se levanta como instancia aparte, normalmente con Docker
- redisvl: librería que abstrae índices vectoriales sobre Redis y ofrece la clase SemanticCache lista para usar (storage del vector, TTL, búsqueda por similaridad)
- LangCache: versión gestionada por Redis del cacheo semántico, sin infraestructura propia que mantener
- langchain.cache.RedisSemanticCache: alternativa dentro del ecosistema LangChain para el mismo patrón de cache semántico
- pgvector / Pinecone / Qdrant: vector stores alternativos sobre los que se podría implementar el cache semántico manualmente si el stack ya los incluye
- pytest: framework de testing usado para los tests de templates (prompt) y de contrato (schema), sin coste de llamadas a la API
- FastAPI: framework del servicio IA donde vive el endpoint `/estimate` y se define `EstimationRequest`/`EstimationResponse`
- JSON Schema: estándar en el que se apoya la salida estructurada; Pydantic lo genera automáticamente con `model_json_schema()`


## Cómo invocar

### Lanzar tests unitarios
uv run pytest -v

### Levantar docker redis (docker)
docker run -d --name lidr-redis -p 6379:6379 redis:7-alpine

### Levantar uvicorn (servidor web) para uso de FastAPI
uv run uvicorn app.main:app --reload

### Arrancar la aplicación streamlit (portal web con interfaz conversacional)
streamlit run streamlit_app.py