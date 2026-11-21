# Alineación con la solución del profesor (estado Sesión 11 live)

Resumen del refactor y la migración que alinean este proyecto con `estimator/` de la solución del
profesor (`reference/ai-engineering-session_11_live/`, solo lectura). El diario completo, con cada
decisión y conflicto, está en `ALIGNMENT_LOG.md` (raíz, temporal). La arquitectura resultante se
describe en [ARCHITECTURE.md](../../ARCHITECTURE.md).

## 1. Estado por fase

| Fase | Estado | Notas |
|---|---|---|
| 0 Preparación y baseline | DONE | `reference/` ya estaba en `.gitignore` y `.dockerignore`; `include = ["app*", "evals*"]` lo ignora |
| 1 Reestructura por capas | DONE | + `tests/test_layering.py` (AST) y `ARCHITECTURE.md` |
| 2 Conductor | DONE | `EstimationService` con 4 entradas (`estimate`, `estimate_stream`, `estimate_conversational`, `estimate_with_acb`) |
| 3 Config y plomería | DONE | runtime config en Redis, `X-API-Key`, rate limit (slowapi), `X-Request-ID`, `log_stage` |
| 4 Sesión 9 | DONE | `POST /v1/estimate/from-transcript` + mejoras de `/v1/retrieval/search` |
| 5 Sesión 10 live | DONE | multi-índice, router, query transform, fusión round-robin, decay, `advanced-search`, migración `0003` |
| 6 Sesión 11 live | DONE | HNSW `halfvec` (`0004`), `quality/` (augmentation, hallucination, synthesis), evals/scripts portados (no ejecutados) |
| 8a Ingesta offline S6 | DONE | migración `0005`; **no** se añadieron `unstructured` ni `openpyxl` (nada los usa) |
| 8b Chunking S7 live | DONE | 7 estrategias + `POST /embeddings/compare` |
| 8c Etapas y horas por tarea | DONE | `/v1/estimate/stages/*`, `/v1/estimate/tasks/hours`, `/embeddings/index/*` (jobs en memoria) |
| 9 Verificación final | DONE | ver §9 |

Nada quedó `SKIPPED` ni `BLOCKED`, salvo el matiz de entorno: `uv sync --extra dev` falla en este
Windows (Python 3.14) por `scikit-network` (dependencia de `ragas`, necesita MSVC). Se usa
`uv sync --extra dev --no-install-package scikit-network`; el `uv.lock` está completo.

## 2. Ficheros movidos y nuevos

Mapa completo de rutas viejas → nuevas en el Apéndice de [ARCHITECTURE.md](../../ARCHITECTURE.md).
Resumen: `services/` → `foundation/llm` + `generation/agentic` + `domain/estimation_service.py`;
`guardrails/`, `prompts/`, `attachments/`, `db/session.py` → `foundation/`; `schemas/` → `domain/schemas/`;
`sessions/` → `generation/conversation/`; `cache.py`/`semantic_cache.py` → `generation/cag/`;
`embedding_pipeline/` + `generation/rag/store.py` → `generation/rag/`; `routers/` y el router de embeddings → `api/`.
No queda ningún fichero shim; se comprobó con búsqueda que no hay referencias a las rutas viejas fuera de
`reference/`, `arquitectura-actual.md` y las notas históricas de `docs/sessions/`.

Ficheros nuevos por capa:

- **foundation**: `llm/streaming.py`, `llm/runtime_config.py`, `persistence/models.py` (Base + filas de ingesta), `persistence/repositories/{jobs,mappings}.py`.
- **domain**: `estimation_service.py`, `schemas/ingestion.py`.
- **generation/cag**: clases `ExactCache` y `SemanticCache` (fachadas inyectables).
- **generation/rag**: `query_reformulator`, `context_assembler`, `prompt_builder`, `estimator` (`TranscriptEstimator`), `validation`, `idempotency`, `observability`, `task_hours`, `index_service`, `index_jobs`, `retrieval/{collections,router,query_transform,temporal,advanced_pipeline}`, `quality/{augmentation,hallucination,synthesis}`, `chunking/{base,strategies/*}`, `analysis/*`, `store/{models,repository}` multi-colección.
- **ingestion**: `catalog`, `loaders`, `parsers`, `cleaning`, `pii`, `documents`, `orchestrator`, `architecture`.
- **api**: `security`, `rate_limiting`, `config`, `errors`, `request_context`, `embeddings`, `search`, `ingestion`, `routers/{retrieval_advanced,estimate,estimate_stages,estimate_tasks,corpus_index}`.
- **datos/scripts/evals**: `data/{transcripts_sample,technical_docs_sample,test_queries,task_corpus,task_corpus_contradictions}.json`, `data/catalog/`, `data/seed/`, `evals/{golden_generation_s11,ragas_baseline_s11}.json`, `evals/RAGAS_BASELINE_S11.md`, scripts de S6/S7/S10/S11 (ninguno ejecutado con llamadas de pago).

## 3. Endpoints (antes → después)

Antes: `POST /api/v1/estimate`, `POST /api/v1/estimate/stream`, `/sessions*` (6 rutas), `POST /embeddings/ingest`,
`POST /search`, `POST /v1/retrieval/search`, `GET /health`. **Todos se conservan con el mismo contrato**
(`/v1/retrieval/search` solo gana campos opcionales y exige clave).

| Endpoint nuevo | Auth |
|---|---|
| `POST /v1/retrieval/search` (existente) | **`X-API-Key` = `RETRIEVAL_API_KEY`**, 120/min |
| `POST /v1/retrieval/advanced-search` | **`X-API-Key` = `RETRIEVAL_API_KEY`**, 120/min |
| `POST /v1/estimate/from-transcript` | **`X-API-Key` = `ESTIMATE_API_KEY`**, 10/min |
| `POST /v1/estimate/stages/{reformulate,retrieve,assemble,structure,generate,verify}` | **`ESTIMATE_API_KEY`** |
| `POST /v1/estimate/tasks/hours` | **`ESTIMATE_API_KEY`** |
| `GET/PUT /api/v1/config/{models,retrieval}` | sin clave (como el profesor) |
| `POST /embeddings/compare`, `/embeddings/index/{runs,jobs/{id},stats}` | sin clave ni límite (como el profesor): **riesgo de gasto** si la API es accesible |
| `POST /api/v1/ingestion/runs`, `GET /api/v1/ingestion/jobs/{id}` | sin clave (como el profesor) |

## 4. Migraciones

| Rev. | Qué hace | Prueba |
|---|---|---|
| `0003` | Renombra `chunks` → `budget_chunks` (índices, PK, FK y secuencia incluidos) y crea `transcript_chunks` y `technical_doc_chunks` | Desde `0002` con 25 filas: 25 filas tras `upgrade head`, `content_tsv` rellena, `downgrade` las devuelve a `chunks` |
| `0004` | HNSW `embedding::halfvec(1536)` `halfvec_cosine_ops` (m=16, ef_construction=128) en las 3 tablas | `pgvector/pgvector:pg16` soporta `halfvec`; la consulta del store usa la misma expresión y el planner usa el índice (EXPLAIN) |
| `0005` | `ingestion_jobs` y `pseudonym_mappings` | upgrade/downgrade con datos |

Comprobado en Postgres (contenedor temporal `pgvector/pgvector:pg16`, ya eliminado): `alembic upgrade head`
**desde cero**; **desde `0002` con datos** (mismo nº de filas); `downgrade -1` en cada paso hasta la base y
re-`upgrade`; `alembic check` sin drift (con `include_object` para ignorar los índices de expresión HNSW).
`0001` y `0002` no se tocaron.

## 5. Dependencias y variables de entorno

Dependencias nuevas: `slowapi`, `anthropic`, `langchain-experimental`, `langchain-openai`,
`langchain-text-splitters`, `nltk`, `presidio-analyzer`, `presidio-anonymizer`, `spacy`, `faker`, `pandas`,
`pandera`, `pyyaml`; dev: `fakeredis`, `ragas`, `datasets`. El modelo de spaCy `es_core_news_md` no es una
dependencia de PyPI: el `Dockerfile` lo instala salvo `--build-arg INSTALL_SPACY_MODEL=false`.

Variables nuevas para tu `.env` (ver `.env.example`): **`RETRIEVAL_API_KEY` y `ESTIMATE_API_KEY`** (genéralas con
`python -c "import secrets; print(secrets.token_urlsafe(32))"`; vacías = router deshabilitado), `PSEUDONYM_HASH_SALT`
(cámbiala), y opcionales: `REFORMULATION_MODEL`, `GENERATION_MODEL`, `GENERATION_REASONING_EFFORT`, `GENERATION_MAX_TOKENS`,
`MAX_CONTEXT_TOKENS`, `IDEMPOTENCY_TTL`, `RETRIEVAL_ROUTING_ENABLED`, `QUERY_TRANSFORM_ENABLED`, `TEMPORAL_DECAY_ENABLED`,
`ROUTER_MODEL`, `QUERY_TRANSFORM_MODEL`, `HALLUCINATION_*`, `AUGMENTATION_*`, `SYNTHESIS_*`, `TASK_HOURS_*`, `PROPOSITIONAL_CHUNKER_MODEL`,
`CONTEXTUAL_CHUNKER_MODEL`, `CATALOG_PATH`, `INGESTION_DATA_ROOT`, `PRESIDIO_SPACY_MODEL`, `PSEUDONYM_FAKER_LOCALE`.

## 6. Tests

| | Baseline | Final |
|---|---|---|
| `pytest -q` (unitarios) | 490 passed, 0 failed, 0 skipped (24 `integration` deseleccionados) | **878 passed**, 0 failed, 0 skipped (53 `integration` deseleccionados) |
| `pytest -m integration` | no ejecutado en el baseline (requiere Postgres) | **53 passed** contra Postgres real (BD `estimator_test` migrada a `head`) |
| `ruff check .` | 5 errores (4 autocorregibles + `PERF402`) | 1 error (el `PERF402` preexistente de `tests/test_llm_service.py`) |
| `mypy app --python-version 3.12` | 41 errores (con 3.11 aborta por los stubs de numpy) | 84 errores (no estaba en verde; la mayoría son huecos de tipado heredados de código movido, p. ej. `semantic.py`, `estimation_service.py`, redis en `runtime_config.py`) |

Tests nuevos: `test_layering`, runtime config, seguridad/rate limit/`X-Request-ID`, `from-transcript`
(auth, idempotencia, soft-fail), reformulador, assembler, validación de citas, router de colecciones, query transform,
fusión round-robin, decay temporal, `advanced_pipeline`, augmentation/hallucination/synthesis, estrategias de chunking,
`compare`, etapas, horas por tarea, índice de corpus, ingesta offline (+ repositorios en Postgres real), `ef_search` de HNSW.

## 7. Conflictos y decisiones (resumen)

- **Prevaleció mi código**: prompts `estimation/v1..v4`, `call_structured` y wrapper (se amplió con la clase `LLMWrapper` y `reasoning_effort`/`max_tokens`), Boss/Critic, guardrails, compresión de sesión, Streamlit (solo cambiaron imports), `SEMANTIC_CACHE_THRESHOLD` (0.92) y `PROMPT_VERSION`, `POST /search` (baseline exacto con `vector`, **sin** halfvec), mi RRF, mi chunker estructural (byte a byte idéntico).
- **Prevaleció el profesor**: pirámide de capas, conductor, runtime config, seguridad/rate limit, S9-S11, estrategias de chunking, ingesta S6.
- `GENERATION_REASONING_EFFORT`: el código del profesor usa `high`, el enunciado `medium`; se usó `medium`.
- `Sector` pasó de 4 a 8 valores (el corpus de tareas del profesor los usa); solo hace la validación más permisiva.
- `RetrievedChunk` ganó campos opcionales (`collection`, `source_id`, `relevance_score`, `document_date`, `estimated_hours`): el JSON actual solo se amplía.
- Overrides en runtime: solo los 6 modelos RAG (+2 de chunking); los modelos de conversación (critic, metadata, compresión) siguen siendo solo de `.env`.
- Capas: las excepciones documentadas en `tests/test_layering.py` (guardrail de salida y loader de prompts → `domain.schemas`, loader → `conversation.models`) son deuda heredada del profesor; el service locator `from app.dependencies import ...` dentro de `generation/` del profesor se sustituyó por inyección.
- Bug del profesor corregido en el port: fechas mixtas ISO/`dd/mm/yyyy` en `clean_budget_records`, `IndexRunRequest.chunk_type` vacío, patrón `API reference` del router de colecciones.

## 8. Lo que NO pude verificar (compruébalo a mano)

- Ningún camino real con OpenAI/Anthropic (todo con fakes): `from-transcript`, etapas, advanced-search, evals/ragas, estrategias de chunking de pago, `compare`, `/embeddings/index/runs`.
- El primer uso de tiktoken (`cl100k_base`) y de `punkt_tab` (nltk) necesita red.
- Rendimiento/recall de HNSW a escala (en tablas pequeñas el planner prefiere seq scan; se probó el uso del índice forzando `enable_seqscan=off`).
- Calidad de Presidio más allá de una transcripción de ejemplo.
- El `docker compose build` del backend sí funcionó (imagen `lidr-estimator:local`, import de `app.main` correcto), pero no se levantó el stack completo con claves reales ni se probó el frontend contra el nuevo backend.
- `mypy` no está en verde (ver §6).

## 9. Hallazgos de la Fase 9 para el ejercicio de la Sesión 12

- **`openai` y `client.responses.create`**: `uv.lock` fija `openai 3.3.0` (`pyproject`: `openai>=3.3.0`); se comprobó en el paquete instalado que `OpenAI(...).responses.create` existe (sin hacer ninguna llamada de red).
- **Función que envolvería `search_budgets`**: `TranscriptEstimator.retrieve_chunks(request: RetrievalRequest) -> RetrievalResult` (`app/generation/rag/estimator.py`, asíncrona): embebe el texto y llama a `retrieve(...)` (`generation/rag/retrieval/pipeline.py`) con `top_k`, umbral, `search_mode`/`rerank` (petición → override Redis → Settings) y filtros. Entrada: texto + filtros opcionales. Salida: `RetrievedChunk` con `content`, `sector`, `project_year`, `budget_id`, `chunk_type`, `distance` y `estimated_hours`.
- **Filtros del store hoy**: `sectors`, rango de años (`project_year_min/max`) y `chunk_types`; además los `HardFilters` del registro de colecciones (`technologies`, `sectors`, `min_date`, `version`) en el pipeline avanzado.
- **Qué faltaría**: no existe filtro por `component_type` (la metadata del chunk tiene `complexity`, `main_technology`, `module`, `component_id`) ni por rango de fechas más fino que el año; los chunks guardan **horas** (`estimated_hours`), no importes en euros, así que "importes" requeriría una tarifa/`calculate_estimate` o añadir el coste a la metadata. Habría que exponer esos filtros en `RetrievalRequest`/`_structural_filters` (o un wrapper tipado) y un esquema de salida pensado para tools (lista compacta).
- **Dónde vive el agente**: `app/generation/agentic/` (junto a `boss.py`/`critic.py`). No puede importar `generation/rag`; el conductor (`EstimationService`) construye las tools (closures sobre `retrieve_chunks`, calculadora) y las **inyecta** al agente por constructor/parámetro, igual que hoy se inyectan el LLM y los caches. `dependencies.py` cablea el `TranscriptEstimator` que el conductor recibiría.
- **Material de la rama `session_12` del profesor que debes traer tú** (no está en la referencia): `sample_transcript_simple.txt`, `sample_transcript_complex.txt`, `reference_retrieval.py`, `calculate_estimate_skeleton.py`.

## 10. Comandos para validar

```powershell
# 1. Claves locales (una vez) en .env: RETRIEVAL_API_KEY y ESTIMATE_API_KEY (ver README)
docker compose up --build -d                       # migra a head y arranca el stack
docker compose exec -T ai_service alembic current
docker compose exec -T ai_service alembic check

# 2. Tests
uv sync --extra dev --no-install-package scikit-network   # (Windows) evita compilar scikit-network
uv run ruff check .
uv run pytest -q
docker compose up -d postgres
$env:TEST_DATABASE_URL = "postgresql+asyncpg://estimator:estimator@localhost:5432/estimator_test"
uv run pytest -m integration -q

# 3. Contratos públicos (sin gasto): salud y documentación
curl http://localhost:8000/health
curl http://localhost:8000/docs

# 4. Endpoint protegido (la clave del .env)
curl -X POST http://localhost:8000/v1/retrieval/search -H "X-API-Key: <RETRIEVAL_API_KEY>" -H "Content-Type: application/json" -d '{"query_text": "mobile booking app with online payments"}'   # embebe: coste mínimo
```

Al terminar de revisar, borra `ALIGNMENT_LOG.md` (es temporal) y haz tú el commit.
