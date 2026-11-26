# Arquitectura del estimador

Este documento es el **contrato de arquitectura** del servicio (adaptado del de la solución de
referencia del profesor a este proyecto). El código de cada sesión nueva debe respetarlo. Si una
pieza no encaja en ninguna capa, primero se decide dónde vive (y se actualiza este documento), no se
crea otra carpeta suelta en la raíz. Las reglas de dependencias las **vigila un test**:
[tests/test_layering.py](tests/test_layering.py) recorre `app/` con `ast` y falla si un import las viola.

## 1. Las tres arquitecturas de IA (y por qué componen)

- **CAG — Cache-Augmented Generation** (`app/generation/cag/`). Responde sin tocar el LLM cuando ya
  hay una respuesta equivalente: primero un acierto exacto (SHA-256), luego uno por similitud vectorial.
- **RAG — Retrieval-Augmented Generation** (`app/generation/rag/`). Corpus de presupuestos,
  transcripciones y documentación técnica → chunks + embeddings en PostgreSQL/pgvector → recuperación
  (vectorial, híbrida, multi-índice, reranking) → estimación citable.
- **Agéntica** (`app/generation/agentic/`). Bucle Actor-Crítico-Boss que audita la estimación apoyado
  en la conversación multi-turno (`app/generation/conversation/`), y (Sesión 12) un **agente con tools**
  y bucle manual sobre la Responses API (`agent_loop.py`, `agent_tools.py`, `agent_schemas.py`).

**Tesis central**: estas tres capas no se conocen entre sí. **Componen únicamente a través del
conductor** (`app/domain/estimation_service.py`).

## 2. El pastel de capas

```
app/
├── main.py                 # factory FastAPI, lifespan, middleware X-Request-ID, rate limiter
├── config.py               # Settings (pydantic-settings)
├── dependencies.py         # composition root: factories cacheadas y servicios por petición
├── logging_config.py       # structlog (propio de este proyecto)
│
├── foundation/             # plomería sin opinión de arquitectura AI
│   ├── observability.py    #   configure_logfire(): Logfire no-op seguro (S13; no confundir con generation/rag/observability.py)
│   ├── llm/                #   wrapper.py (LLMWrapper + call_structured), streaming.py, runtime_config.py (overrides en Redis)
│   ├── prompts/            #   loader Jinja2 + plantillas versionadas (estimation/v1..v4, critic, ...)
│   ├── guardrails/         #   input (moderación + injection + PII) / output (filtro de scope)
│   ├── attachments/        #   extracción de texto de PDF/DOCX
│   └── persistence/        #   engine/sesiones async, Base declarativa, filas y repositorios de ingesta, checkpointer.py (LangGraph, S13)
│
├── domain/                 # contrato + conductor
│   ├── schemas/            #   EstimationRequest/Result/Response, ACBResponse, CriticFeedback, ...
│   ├── estimation_service.py  # EstimationService — ÚNICO punto de composición
│   └── graph/              #   (S13) grafo de LangGraph: state, schemas, nodes, build. Capa *conductor*: compone generation + foundation
│
├── generation/
│   ├── cag/                #   exact.py + semantic.py
│   ├── rag/                #   chunking/ embedding/ analysis/ store/ retrieval/ quality/ + estimator, validation, ...
│   ├── agentic/            #   boss.py + critic.py (ACB) · agent_loop.py + agent_tools.py + agent_schemas.py (agente S12)
│   └── conversation/       #   models, store, metadata_extractor, tier_resolver, compression/
│
├── ingestion/              # pipeline batch offline (catálogo, parsers, limpieza, PII)
│
└── api/                    # transporte: routers finos, sin lógica de negocio
    ├── estimations.py, sessions.py, embeddings.py, search.py, ingestion.py, config.py
    ├── security.py, rate_limiting.py, errors.py, request_context.py
    ├── graph_runtime.py    #   (S13) graph_lifespan: Logfire + checkpointer + grafo compilado en app.state.graph
    └── routers/            #   retrieval, retrieval_advanced, estimate, estimate_stages, estimate_tasks, estimate_agent, estimate_graph, corpus_index
```

## 3. Reglas de dependencias (MUST / MUST NOT)

| Capa | PUEDE importar | NO PUEDE importar |
|---|---|---|
| `config.py` | (nada interno) | todo |
| `foundation/*` | `config`, `logging_config` | `domain`, `generation`, `ingestion`, `api` |
| `domain/schemas/*` | `config`, `foundation` | `generation`, `api` |
| `generation/<x>/*` | `config`, `foundation`, `domain/schemas`, su propio paquete | `api`, `dependencies`, el conductor, **otro hermano de `generation`** |
| `domain/estimation_service.py` y `domain/graph/*` (CONDUCTOR) | todos los hermanos de `generation`, `foundation`, `schemas` | `api`, `dependencies` |
| `ingestion/*` | `config`, `foundation`, `domain/schemas`, `generation` | `api`, el conductor |
| `api/*` | `dependencies`, `domain`, `foundation`, tipos y servicios de `generation` | lógica de negocio |
| `dependencies.py` (COMPOSITION ROOT) | cualquier capa | (lo importan solo `api/` y los tests) |
| `main.py` | `api`, `foundation`, `config` | — |
| `foundation/observability.py`, `foundation/persistence/checkpointer.py` | solo `config` (y librerías externas) | el resto de capas |
| `api/graph_runtime.py` | lo que puede `api/*`, incl. `dependencies` | lógica de negocio |

**Sin ciclos entre conductores**: `domain/graph` puede importar de `domain/estimation_service` (misma capa),
pero `estimation_service.py` NO importa `domain/graph`.

**Aristas especiales**: `agentic` **puede** importar `conversation` (la inversa está prohibida); los
hermanos de `generation` se encuentran **solo** en el conductor.

**Excepciones documentadas** (lista corta en `ALLOWED_EXCEPTIONS` del test; cada una es deuda técnica):
`foundation.guardrails.output` y `foundation.prompts.loader` dependen de `domain.schemas`, y el loader
de `generation.conversation.models` (tipos de valor), igual que en la solución del profesor.

**Colaboradores inyectados**: dentro de `generation/` no se hace `from app.dependencies import ...`
(service locator). Los servicios reciben sus colaboradores por constructor/parámetro
(`TranscriptEstimator`, `TaskHoursEstimator`, `advanced_retrieve(...)`) y `dependencies.py` los cablea.

## 4. El conductor

`EstimationService` tiene cinco entradas:

- `estimate()` — guardrails(in) → prompt → llm → guardrails(out). Sin cachés (como antes).
- `estimate_stream()` — guardrails(in) → cag exacto → cag semántico → prompt → llm (stream) → guardrails(out) → cag.store.
- `estimate_conversational()` — conversation (historial + metadata + compresión) → prompt → llm → guardrails(out).
- `estimate_with_acb()` — agentic (Boss orquesta Actor + Critic sobre la conversación).
- `estimate_with_agent()` — agentic (agente con tools, Sesión 12). El conductor construye el backend de
  `search_budgets` como un closure sobre `TranscriptEstimator.retrieve_chunks` (`build_budget_search_backend`)
  y lo **inyecta** en el registro de tools; `dependencies.get_agent_estimation_service` le entrega el cliente
  `AsyncOpenAI` y el `TranscriptEstimator`.

### El agente de la Sesión 12 (`generation/agentic/`)

- `agent_schemas.py` (argumentos de las tools, `AgentEstimate`, traza), `agent_tools.py` (schemas planos
  `strict` que lee el modelo, `calculate_estimate` y `validate_estimate` deterministas, `search_budgets` con
  backend inyectado, `execute_tool`), `agent_loop.py` (`run_agent`, `SYSTEM_PROMPT`, `format_trace`).
- **Regla: las tools reciben sus backends inyectados desde el conductor.** El agente no importa
  `generation/rag`, `dependencies` ni `api`; la recuperación (híbrida + reranking según la runtime config) es la
  que ya existe, no se reimplementa. En tests y en `scripts/run_agent_s12.py --stub` el backend es un fake.
- **Excepción deliberada a "todo pasa por `LLMWrapper`"**: el bucle usa la Responses API cruda
  (`AsyncOpenAI.responses.create`) sin frameworks ni `tool_runner`, para poder ver y capturar cada paso. Por eso
  `get_async_openai_client()` es un cliente aparte (`None` sin `OPENAI_API_KEY`) y `get_openai_client` no cambia.
- Endpoint: `POST /v1/estimate/agent` (`api/routers/estimate_agent.py`, fino: auth, 5/min, validación de
  overrides, 503/502). Detalle en [docs/sessions/12-agent.md](docs/sessions/12-agent.md).

### El grafo de la Sesión 13 (`domain/graph/`)

- `state.py` (`EstimationState` con dos reducers acumuladores: `budget_matches` y `errors`), `schemas.py`
  (E/S de los nodos LLM), `nodes.py` (cinco nodos + `flag_for_review` + `route_after_validation`), `build.py`
  (`build_graph(checkpointer, deps)`, acepta `None`).
- **Inyección, no service locator**: los nodos son funciones de módulo `async def node(state, deps) -> dict` y
  reciben un `GraphDependencies` (`llm_wrapper`, `search_backend`, `settings`, dataclass congelado). `build.py`
  los enlaza con closures. Nada de `from app.dependencies import ...` dentro de `domain/graph`.
- **Arranque**: `main.py` no puede importar `dependencies`/`domain`, así que `api/graph_runtime.py` (`graph_lifespan`)
  construye las dependencias reales (el backend de `search_budgets` es el mismo closure sobre RAG que usa el agente
  S12), abre el checkpointer Postgres y deja el grafo en `app.state.graph`. Si Postgres no está disponible el
  servicio arranca igual con `graph = None` (el endpoint responde 503); `GRAPH_ENABLED=false` lo desactiva.
- Endpoint: `POST /v1/estimate/graph` (`api/routers/estimate_graph.py`, fino). Detalle en
  [docs/sessions/13-graph.md](docs/sessions/13-graph.md).

Los endpoints RAG de las sesiones 9-11 (`/v1/estimate/from-transcript`, `/v1/estimate/stages/*`,
`/v1/estimate/tasks/hours`, `/v1/retrieval/*`) no mezclan hermanos de `generation` (solo RAG): los
sirven directamente `TranscriptEstimator` / `TaskHoursEstimator` / `retrieve()`. **Cuando RAG entre en
el camino de `EstimationService`** (o lo haga un agente en la sesión 12) el paso se añade **en el
conductor**, nunca con un import cruzado entre hermanos.

## 5. Composition root

`dependencies.py` y `config.py` viven en la raíz, por encima de las capas. Factories cacheadas
(`get_openai_client`, `get_llm_wrapper`, `get_cache`, `get_session_store`, `get_embedder`,
`get_chunk_store`, `get_reranker`, `get_runtime_config`, ...) y servicios por petición
(`get_estimation_service`, `get_transcript_estimator`, `get_ingestion_service`, ...). Los tests
sustituyen piezas con `app.dependency_overrides[get_xxx] = lambda: Fake()`.

## 6. Camino de la petición principal

```
POST /api/v1/estimate
  └→ app/api/estimations.py                          (HTTP fino, mapeo de errores: 400/502/422)
       └→ EstimationService.estimate()
            1. foundation/guardrails/input.py        (moderación + injection + PII)
            2. foundation/prompts/loader.py          (Jinja2 versionado)
            3. foundation/llm/wrapper.py             (Instructor + validators + re-prompt)
            4. foundation/guardrails/output.py       (filtro de scope)
            5. return EstimationResponse(result, prompt_version)
```

## 7. ¿Dónde va mi código nuevo?

| Si añades… | Va en… |
|---|---|
| Backend LLM, plantilla de prompt, guardrail nuevo | `foundation/` |
| Retrieval, chunking, vector store, embeddings, calidad de generación | `generation/rag/` |
| Rol de agente, tools del agente o paso de orquestación | `generation/agentic/` |
| Nodo o arista nuevos del grafo de estimación | `domain/graph/` (los colaboradores entran por `GraphDependencies`) |
| Checkpointer, exportadores de trazas, plomería de observabilidad | `foundation/persistence/checkpointer.py`, `foundation/observability.py` |
| Cableado del grafo al arrancar la app | `api/graph_runtime.py` |
| Estrategia de cache | `generation/cag/` |
| Memoria conversacional | `generation/conversation/` |
| Endpoint HTTP | `api/` (fino) + factory en `dependencies.py` |
| Composición entre capas | método en `EstimationService` (`domain/`), **nunca** import cruzado |
| Fuente de datos / parser / limpieza offline | `ingestion/` |

## 8. Contratos públicos que NO se rompen

`POST /api/v1/estimate`, `POST /api/v1/estimate/stream`, `/sessions`, `/sessions/{id}`,
`/sessions/{id}/memory`, `/sessions/{id}/estimate`, `/sessions/{id}/estimate-acb`,
`POST /embeddings/ingest`, `POST /search`, `POST /v1/retrieval/search` (solo se añaden campos),
`GET /health`. Las formas JSON de `EstimationResponse`, `ACBResponse`, `SessionInfoResponse` y
`RetrievalResult` no cambian. `EstimationResult` mantiene el orden de campos que necesita Instructor y
sus `model_validator` (disparan el re-prompt).

## Apéndice — Mapa de migración de rutas (mi ruta anterior → ahora)

| Antes | Ahora |
|---|---|
| `app/services/llm_service.py` | `app/domain/estimation_service.py` (+ `app/foundation/llm/streaming.py`) |
| `app/services/llm_wrapper.py` | `app/foundation/llm/wrapper.py` |
| `app/services/boss.py`, `critic.py` | `app/generation/agentic/boss.py`, `critic.py` |
| `app/guardrails/*` | `app/foundation/guardrails/*` |
| `app/prompts/*` | `app/foundation/prompts/*` |
| `app/attachments/*` | `app/foundation/attachments/*` |
| `app/db/session.py` | `app/foundation/persistence/database.py` |
| `app/db/models.py` (`Base`) | `app/foundation/persistence/models.py` |
| `app/db/models.py` (`Document`, `Chunk`) | `app/generation/rag/store/models.py` |
| `app/schemas/*` | `app/domain/schemas/*` |
| `app/sessions/*` | `app/generation/conversation/*` |
| `app/cache.py` | `app/generation/cag/exact.py` |
| `app/semantic_cache.py` | `app/generation/cag/semantic.py` |
| `app/embedding_pipeline/chunker.py` | `app/generation/rag/chunking/structural.py` |
| `app/embedding_pipeline/embedder.py` | `app/generation/rag/embedding/embedder.py` |
| `app/embedding_pipeline/ingestion.py` | `app/generation/rag/ingest_service.py` |
| `app/embedding_pipeline/repository.py` + `generation/rag/store.py` | `app/generation/rag/store/repository.py` |
| `app/embedding_pipeline/retriever.py` | `app/generation/rag/retriever.py` |
| `app/embedding_pipeline/{schemas,errors}.py` | `app/generation/rag/{schemas,errors}.py` (fusionados con los de retrieval) |
| `app/embedding_pipeline/router.py` | `app/api/embeddings.py` (ingest, compare) y `app/api/search.py` (`/search`) |
| `app/routers/estimations.py`, `sessions.py` | `app/api/estimations.py`, `app/api/sessions.py` |
| `app/routers/retrieval.py` | `app/api/routers/retrieval.py` |
