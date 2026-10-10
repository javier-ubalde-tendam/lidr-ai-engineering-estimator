# LiDR AI Engineering Estimator

Herramienta de estimación de proyectos de software asistida por IA: a partir de una descripción o
una transcripción de reunión, un LLM genera una estimación estructurada (fases, duración, coste)
con guardrails de entrada/salida, caché y, desde la sesión 05, memoria conversacional y adjuntos.
Tras la clase en vivo se añadieron compresión del historial (anclas + resumen), tier de audiencia,
revisión Actor-Critic-Boss y evals.

Proyecto de aprendizaje del curso LIDR AI Engineering. Servicio IA en FastAPI (`app/`) + cliente
en Streamlit (`streamlit_app.py`).

## Cómo levantarlo

### 1. Redis (caché semántico y exact-match)

Necesita `redis-stack-server` (no `redis:7-alpine`: hace falta el módulo RediSearch).

```
docker compose up -d
```

### 2. Variables de entorno

Copia `.env.example` a `.env` y rellena al menos `OPENAI_API_KEY` (o `ANTHROPIC_API_KEY` si usas
`LLM_PROVIDER=anthropic`). Nunca subas `.env` al repositorio.

### 3. Dependencias

```
uv sync --extra dev
```

### 4. Servicio IA (FastAPI)

```
uv run uvicorn app.main:app --reload
```

### 5. Cliente (Streamlit)

```
streamlit run streamlit_app.py
```

## Cómo ejecutar los tests

```
uv run pytest -v
```

Lint (opcional):

```
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
evals/                                    Dataset dorado, métricas binarias y runner CLI
docs/sessions/                            Notas de diseño por sesión
tests/                                    Tests sin llamadas reales (LLM falso)
streamlit_app.py                          Cliente: formulario (stream) y conversación
```

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/v1/estimate` | Estimación de una sola vez (bloqueante), sin memoria entre llamadas |
| `POST` | `/api/v1/estimate/stream` | Igual que arriba, pero con streaming de la respuesta (SSE) |
| `POST` | `/sessions` | Crea una sesión conversacional nueva |
| `GET` | `/sessions/{session_id}` | Debug de la sesión: `message_count`, `max_turns`, `metadata`, `anchors_count`, `summary_chars`, `last_resolved_tier`, `last_tier_rule` |
| `POST` | `/sessions/{session_id}/estimate` | Turno conversacional (multipart/form-data: adjuntos `.pdf`/`.docx` y `tier` opcional). Devuelve además `latency_ms`, `tokens_in`, `tokens_out`, `cost_usd`, `model`, `provider` |
| `POST` | `/sessions/{session_id}/estimate-acb` | Igual que `/estimate`, pero revisado por Actor-Critic-Boss; añade `acb` (traza). Cuesta hasta `2 × BOSS_MAX_ITERATIONS` llamadas al LLM |
| `GET` | `/health` | Healthcheck |

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

## Documentación por sesión

- [docs/sessions/02-cag.md](docs/sessions/02-cag.md)
- [docs/sessions/03-streaming-chat+wrapper.md](docs/sessions/03-streaming-chat+wrapper.md)
- [docs/sessions/04-form+guardrails.md](docs/sessions/04-form+guardrails.md)
- [docs/sessions/05-memory+attachments.md](docs/sessions/05-memory+attachments.md)
  (incluye [Post-live: alinear solución con el profesor](docs/sessions/05-memory+attachments.md#post-live-alinear-solución-con-el-profesor))

## Sesión 05: memoria + adjuntos

Se añadió un camino conversacional con memoria: historial (ventana deslizante) +
`project_metadata` acumulado (extraído por un segundo LLM barato tras cada turno, fusionado con lo
ya sabido). Los adjuntos (`.pdf`, `.docx`) se extraen **localmente** (Camino B) y se envían como
texto dentro del prompt, en vez de usar la Files API de un proveedor: así el resultado sigue
siendo compatible con el fallback OpenAI↔Anthropic ya existente y da más control (truncado, logs), además de dejarlo preparado para futuras ampliaciones a RAG.
Detalle completo de las decisiones de diseño en
[docs/sessions/05-memory+attachments.md](docs/sessions/05-memory+attachments.md).
