# LiDR AI Engineering Estimator

Herramienta de estimación de proyectos de software asistida por IA: a partir de una descripción o
una transcripción de reunión, un LLM genera una estimación estructurada (fases, duración, coste)
con guardrails de entrada/salida, caché y, desde la sesión 05, memoria conversacional y adjuntos.
Tras la clase en vivo se añadieron compresión del historial (anclas + resumen), tier de audiencia,
revisión Actor-Critic-Boss y evals.

Proyecto de aprendizaje del curso LIDR AI Engineering. Servicio IA en FastAPI (`app/`) + cliente
en Streamlit (`streamlit_app.py`).

## Cómo levantarlo

### Opción A: todo en Docker (recomendada)

Requisitos: Docker Desktop con **Docker Compose >= 2.20** (el `docker-compose.yml` raíz usa
`include`, introducido en 2.20; no se usa ninguna sintaxis más reciente). Comprueba la versión con
`docker compose version`. Los comandos valen tal cual en PowerShell.

1. Copia `.env.example` a `.env` y rellena `OPENAI_API_KEY` (y `ANTHROPIC_API_KEY` si
   `LLM_PROVIDER=anthropic`). Nunca subas `.env` al repositorio.

   > **`OPENAI_API_KEY` es obligatoria siempre**: los embeddings (ingesta, búsqueda y caché
   > semántica) usan OpenAI `text-embedding-3-small`, aunque el LLM de estimación sea Anthropic.

   ```powershell
   Copy-Item .env.example .env
   ```

2. Levanta el stack completo desde la raíz:

   ```powershell
   docker compose up --build -d
   ```

| Servicio | Qué es | Puerto | URL |
|---|---|---|---|
| `ai_service` | FastAPI. Al arrancar ejecuta `alembic upgrade head` y solo si termina bien lanza Uvicorn | 8000 | http://localhost:8000/docs |
| `frontend` | Streamlit | 8501 | http://localhost:8501 |
| `redis` | `redis/redis-stack-server` (RediSearch: lo exige `redisvl`) | 6379 | — |
| `postgres` | `pgvector/pgvector:pg16` (BD, usuario y password `estimator`) | 5432 (`POSTGRES_PORT` en `.env` si está ocupado) | — |

Organización de Compose (un único proyecto y red compartida; cada archivo resuelve sus rutas desde
su propio directorio):

```
docker-compose.yml                  orquestador: include de los tres siguientes
docker/infra/docker-compose.yml     redis, postgres y volúmenes (lidr-redis-data, lidr-postgres-data)
docker/backend/docker-compose.yml   ai_service
docker/frontend/docker-compose.yml  frontend
Dockerfile                          imagen única (uv + uv.lock), usada por ai_service y frontend
```

No se fija `name:` en el Compose: el proyecto sigue llamándose como el directorio, así se siguen
usando los volúmenes existentes (`lidr-ai-engineering-estimator_lidr-redis-data`).

Comandos habituales:

```powershell
docker compose ps                       # estado y healthchecks
docker compose logs -f ai_service       # logs del backend
docker compose down                     # elimina contenedores; NO borra volúmenes (no uses -v)
```

**URLs en el host frente a URLs en Docker:**

| Variable | Ejecución en el host (Windows, `uv run ...`) | Dentro de Docker (Compose la fija) |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | `redis://redis:6379/0` |
| `DATABASE_URL` | `postgresql+asyncpg://estimator:estimator@localhost:5432/estimator` | `postgresql+asyncpg://estimator:estimator@postgres:5432/estimator` |
| `API_BASE_URL` | `http://localhost:8000` | `http://ai_service:8000` |

Compose sobrescribe esas variables sobre el `.env`, así que el mismo `.env` sirve para ambos modos.
Entre contenedores nunca se usa `localhost`.

### Opción B: backend y Streamlit en el host

1. Infraestructura: `docker compose up -d redis postgres`
2. `uv sync --extra dev`
3. Migraciones: `uv run alembic upgrade head`
4. Backend: `uv run uvicorn app.main:app --reload`
5. Cliente: `uv run streamlit run streamlit_app.py`

### Migraciones (Alembic)

Las migraciones son la fuente de verdad del esquema (no se usa `create_all()`). Alembic solo necesita
`DATABASE_URL` (no claves de LLM).

```powershell
docker compose exec -T ai_service alembic current   # versión aplicada
docker compose exec -T ai_service alembic check     # el ORM y la migración coinciden
```

### Ejemplos de ingesta y búsqueda (sesión 08)

Con el stack levantado (hace llamadas reales de embeddings a OpenAI, coste mínimo):

```powershell
docker compose run --rm ai_service python query_examples.py
```

Ingesta `data/budgets_sample.json` presupuesto a presupuesto (idempotente: un 409 se cuenta como
"ya existente") y lanza cinco consultas con `k=5`. El contenedor efímero de `run --rm` desaparece,
así que para guardar el informe se usa el servicio en ejecución y se copia el fichero al host:

```powershell
docker compose exec -T ai_service python query_examples.py --output /app/output_examples.txt
docker compose cp ai_service:/app/output_examples.txt ./output_examples.txt
```

El informe real generado está en [output_examples.txt](output_examples.txt). Es de una ejecución
repetida (todo `409`); la primera ejecución sobre la BD vacía dejó 15 documents y 52 chunks y las
repeticiones no cambian esos recuentos.

Comprobar recuentos y que no hay índices vectoriales:

```powershell
docker compose exec -T postgres psql -U estimator -d estimator -c "SELECT (SELECT count(*) FROM documents) AS documents, (SELECT count(*) FROM chunks) AS chunks;"
docker compose exec -T postgres psql -U estimator -d estimator -c "SELECT indexname FROM pg_indexes WHERE indexdef ~* 'hnsw|ivfflat|diskann';"
```

## Cómo ejecutar los tests

```powershell
uv sync --extra dev
uv run pytest -v                  # unitarios: sin OpenAI, Redis ni Postgres
uv run pytest -m integration      # integración: PostgreSQL + pgvector reales
```

Los tests de integración necesitan `docker compose up -d postgres`. Usan una BD exclusiva
(`estimator_test`, que crean y migran con Alembic; `TEST_DATABASE_URL` permite cambiarla) y se niegan
a ejecutarse si el nombre de la BD no termina en `_test` o coincide con la de desarrollo, porque
hacen `TRUNCATE`. Si Postgres no está disponible fallan con el error real de conexión (no se omiten).

Lint (opcional):

```powershell
uv run ruff check .
```

## Estructura

```
app/
  main.py, config.py, dependencies.py     App FastAPI, settings (pydantic-settings), singletons
  routers/                                estimations.py (/api/v1/...) y sessions.py (/sessions/...)
  services/
    llm_service.py                        Estimación (bloqueante, streaming, conversacional, ACB)
    llm_wrapper.py                        call_structured(): latencia, tokens, coste, modelo, proveedor
    critic.py, boss.py                    Actor-Critic-Boss (critic fail-open; boss sin llamadas LLM)
  sessions/
    models.py, store.py                   Sesión, historial (anclas + resumen + ventana), metadata
    compression/                          Detector de anclas, resumen acumulativo, política de compresión
    tier_resolver.py                      Tier de audiencia por reglas explicables
    metadata_extractor.py                 Extractor LLM del project_metadata
  prompts/                                estimation/v1-v4, metadata_extraction, conversation_summary, critic
  schemas/                                estimation.py, critic.py, acb.py
  guardrails/, attachments/, cache.py, semantic_cache.py
  embedding_pipeline/                     Sesión 07-08: chunker, embedder, ingesta (servicio + repositorio), retriever y routers
  db/                                     Sesión 08: modelos SQLAlchemy (documents, chunks), engine/sesiones async
alembic/, alembic.ini                     Migraciones (0001: documents y chunks con pgvector)
docker/                                   Compose por servicio (infra, backend, frontend); Dockerfile en la raíz
query_examples.py                         Ingesta del corpus + 5 búsquedas de ejemplo (vía HTTP)
evals/                                    Dataset dorado, métricas binarias y runner CLI
data/                                     budgets_sample.json (presupuestos de ejemplo)
scripts/                                  compare.py (similitud coseno entre dos textos; utilidad histórica de la sesión 07)
docs/sessions/                            Notas de diseño por sesión
tests/                                    Tests sin llamadas reales (LLM falso); tests/integration con Postgres real
streamlit_app.py                          Cliente: formulario (stream) y conversación
```

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/v1/estimate` | Estimación de una sola vez (bloqueante), sin memoria entre llamadas |
| `POST` | `/api/v1/estimate/stream` | Igual que arriba, pero con streaming de la respuesta (SSE) |
| `POST` | `/sessions` | Crea una sesión conversacional nueva |
| `GET` | `/sessions/{session_id}` | Debug de la sesión: `message_count`, `max_turns`, `metadata`, `anchors_count`, `summary_chars`, `last_resolved_tier`, `last_tier_rule` |
| `GET` | `/sessions/{session_id}/memory` | Snapshot de depuración de lo que recuerda la sesión: `summary`, `anchors` (texto literal) y `metadata` (lo usan los evals de estrés) |
| `POST` | `/sessions/{session_id}/estimate` | Turno conversacional (multipart/form-data: adjuntos `.pdf`/`.docx` y `tier` opcional). Devuelve además `latency_ms`, `tokens_in`, `tokens_out`, `cost_usd`, `model`, `provider` y `turn_observation` (evento `turn_observed`) |
| `POST` | `/sessions/{session_id}/estimate-acb` | Igual que `/estimate`, pero revisado por Actor-Critic-Boss; añade `acb` (traza). Cuesta hasta `2 × BOSS_MAX_ITERATIONS` llamadas al LLM |
| `GET` | `/health` | Healthcheck |
| `POST` | `/embeddings/ingest` | Sesión 08: ingesta UN presupuesto en PostgreSQL (trocea, embebe y persiste en una transacción). `200`/`409`/`422` |
| `POST` | `/search` | Sesión 08: búsqueda semántica top-k por distancia coseno calculada en PostgreSQL |

## Variables de entorno

Ver `.env.example`. Las de la memoria conversacional y el ACB:

| Variable | Default | Qué controla |
|---|---|---|
| `CONVERSATIONAL_PROMPT_VERSION` | `v4` | Versión del prompt conversacional |
| `MAX_CONVERSATION_TURNS` | `6` | Pares user+assistant de la ventana reciente |
| `MAX_ATTACHMENT_CHARS` | `60000` | Límite de caracteres por adjunto |
| `METADATA_EXTRACTOR_MODEL` | `gpt-4o-mini` | Modelo del extractor de `project_metadata` |
| `ANCHOR_DETECTION_MODE` | `heuristic` | `heuristic` (regex) o `llm` (clasificador) |
| `COMPRESSION_MODEL` | `gpt-4o-mini` | Modelo del resumen acumulativo y del clasificador de anclas |
| `CRITIC_MODEL` | `gpt-4o-mini` | Modelo del critic |
| `BOSS_MAX_ITERATIONS` | `3` | Vueltas actor+critic (1 borrador + 2 reintentos) |

## Evals

> **Aviso:** hacen llamadas REALES al LLM (coste). No forman parte de `pytest`.

```
uv run python -m evals.run --mode actor --limit 3
uv run python -m evals.run --mode acb --limit 3
uv run python -m evals.run --mode actor --http http://localhost:8000 --output report.json
```

Detalle en [docs/sessions/05-memory+attachments.md](docs/sessions/05-memory+attachments.md#post-live-alinear-solución-con-el-profesor).

### Stress test del CAG (ejercicio previo a la sesión 6)

> **Aviso:** también llama al LLM real en cada turno (990 turnos la matriz completa). Empieza con `--max-turns 5`.

```
uv run python -m evals.stress.fixtures.build_pdfs
uv run python -m evals.stress.run --http http://localhost:8000 --scenarios growing,pivot,contradiction --attachment-sizes 0,5,20,50,100 --repeats 3 --output evals/stress/results.csv
```

Detalle y cómo leer el CSV en [docs/sessions/06-stress-cag.md](docs/sessions/06-stress-cag.md); el informe a rellenar está en [evals/stress/REPORT.md](evals/stress/REPORT.md).

## Documentación por sesión

- [docs/sessions/02-cag.md](docs/sessions/02-cag.md)
- [docs/sessions/03-streaming-chat+wrapper.md](docs/sessions/03-streaming-chat+wrapper.md)
- [docs/sessions/04-form+guardrails.md](docs/sessions/04-form+guardrails.md)
- [docs/sessions/05-memory+attachments.md](docs/sessions/05-memory+attachments.md)
  (incluye [Post-live: alinear solución con el profesor](docs/sessions/05-memory+attachments.md#post-live-alinear-solución-con-el-profesor))
- [docs/sessions/06-stress-cag.md](docs/sessions/06-stress-cag.md)
- [docs/sessions/07-embeddings+chunking.md](docs/sessions/07-embeddings+chunking.md) (contrato histórico)
- [docs/sessions/08-pgvector.md](docs/sessions/08-pgvector.md)

## Sesión 05: memoria + adjuntos

Se añadió un camino conversacional con memoria: historial (ventana deslizante) +
`project_metadata` acumulado (extraído por un segundo LLM barato tras cada turno, fusionado con lo
ya sabido). Los adjuntos (`.pdf`, `.docx`) se extraen **localmente** (Camino B) y se envían como
texto dentro del prompt, en vez de usar la Files API de un proveedor: así el resultado sigue
siendo compatible con el fallback OpenAI↔Anthropic ya existente y da más control (truncado, logs), además de dejarlo preparado para futuras ampliaciones a RAG.
Detalle completo de las decisiones de diseño en
[docs/sessions/05-memory+attachments.md](docs/sessions/05-memory+attachments.md).

## Sesión 07: pipeline de embeddings (histórico)

> **Histórico.** Este es el contrato de la sesión 07 y **ya no está vigente**: `POST /embeddings/ingest`
> ahora recibe un único presupuesto y persiste en PostgreSQL (ver "Sesión 08" más abajo).

**Objetivo (sesión 07):** convertir presupuestos históricos en chunks con embeddings, sin persistencia
ni retrieval.

Piezas que se conservan de `app/embedding_pipeline/`:
- `schemas.py`: `Budget`, `Chunk`, `EmbeddedChunk`...
- `chunker.py`: `JSONStructuralChunker`, un componente = un chunk con header contextual del presupuesto.
- `embedder.py`: `OpenAIEmbedder` (cliente inyectado, batches de 100, reintentos ante rate limit) y `estimate_cost_usd`.
- `SANITY_CHECK.md`: similitud coseno real de tres parejas de textos.

**Contrato histórico de `POST /embeddings/ingest` (sesión 07, sustituido):**
- Entrada: `{"budgets": [Budget, ...]}`; salida `200`: `{"chunks": [...con embedding...], "stats": {...}}`.

**Comparar dos textos (`scripts/compare.py`, utilidad histórica):**

```powershell
uv run python scripts/compare.py --text-a "..." --text-b "..."
docker compose exec -T ai_service python scripts/compare.py --text-a "..." --text-b "..."
```

Detalle en [docs/sessions/07-embeddings+chunking.md](docs/sessions/07-embeddings+chunking.md).

## Sesión 08: PostgreSQL + pgvector y búsqueda semántica (contrato vigente)

Los presupuestos se ingestan en PostgreSQL y se buscan por similitud semántica con SQL. Detalle y
comandos extensos en [docs/sessions/08-pgvector.md](docs/sessions/08-pgvector.md).

**`POST /embeddings/ingest`** (sin `/api/v1`): un presupuesto por petición.

```json
{"source_path": "data/budgets_sample.json::BUD-2024-001", "document_type": "historical_budget", "content": { "...Budget completo..." }}
```

- `200`: `{"document_id": 42, "chunks_created": 4, "embedding_dimension": 1536, "ingestion_time_ms": 1240}`
- `409`: `{"detail": "Document already ingested", "document_id": 42}` (mismo `source_path`)
- `422`: payload inválido; `500`: errores externos con mensaje genérico (`Embedding service error` / `Database error`)
- `source_path` es solo identidad/procedencia: el servidor nunca abre esa ruta.

**`POST /search`** (sin `/api/v1`): `{"query": "...", "k": 5}` (`k` entre 1 y 100). Devuelve `query`, `k`,
`search_time_ms` y `results` con `chunk_id`, `document_id`, `chunk_type`, `content`, `distance` (distancia
coseno: **menor = más cercano**) y `metadata`. No hay umbral de relevancia: una consulta fuera de dominio
también devuelve vecinos y hay que interpretar su distancia. Todavía no se usa en el prompt del estimador.

**Flujo:**

```mermaid
flowchart LR
  B[Presupuesto] --> C[Chunker] --> E[Embeddings OpenAI, 1 lote] --> T[(Transacción PostgreSQL: document + chunks)]
  Q[Query] --> QE[Embedding] --> S["SQL: ORDER BY embedding <=> query LIMIT k"] --> R[Top-k]
```

### Decisiones de la sesión 08

- **Dos tablas con FK `ON DELETE CASCADE`** (`documents` 1:N `chunks`): la procedencia se guarda una
  vez por presupuesto y borrar un documento no deja chunks huérfanos.
- **Columnas estables tipadas + `metadata` JSONB**: lo que siempre existe (tipo, contenido, vector,
  fechas) es columna con tipo e índice; lo variable por tipo de documento va en JSONB con índice GIN,
  sin migraciones por cada campo nuevo. La metadata de presupuesto vive en `documents` y la de
  componente en `chunks` (sin duplicar el JSON del presupuesto).
- **`vector(1536)` y `embedding` nullable**: 1536 es la dimensión de `text-embedding-3-small`, el único
  modelo admitido (`Settings` rechaza otros; cambiar de modelo obliga a re-embeber todo el corpus, no se
  mezclan). Es nullable por flexibilidad del esquema, aunque esta ingesta es atómica y siempre lo rellena;
  la búsqueda excluye los `NULL` solo para no devolver distancias nulas.
- **Coseno frente a L2 / producto interno**: la similitud entre textos depende de la dirección del
  vector, no de su norma. Con vectores normalizados el ranking de los tres coincide, pero coseno no
  depende de esa suposición y se alineará con `vector_cosine_ops` cuando haya índice vectorial.
- **Sin índice vectorial (baseline)**: ni HNSW ni IVFFlat. Con 52 chunks la búsqueda exacta es instantánea
  y sirve de referencia de calidad para medir después el coste de aproximar. Sí hay índices B-tree y GIN.
- **`ix_documents_source_path` es único** (el ejemplo del enunciado es no único): garantiza el `409` también
  con peticiones concurrentes; solo se interpreta como duplicado la violación de ese índice.
- **Una transacción por documento**: comprobar duplicado → insertar document (`flush`) → chunkear → embeber
  (en un hilo, el embedder es síncrono) → insertar chunks → commit. Cualquier fallo hace rollback de todo.
