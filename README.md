# LiDR AI Engineering Estimator

Herramienta de estimación de proyectos de software asistida por IA: a partir de una descripción o
una transcripción de reunión, un LLM genera una estimación estructurada (fases, duración, coste)
con guardrails de entrada/salida, caché y, desde la sesión 05, memoria conversacional y adjuntos.

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

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/v1/estimate` | Estimación de una sola vez (bloqueante), sin memoria entre llamadas |
| `POST` | `/api/v1/estimate/stream` | Igual que arriba, pero con streaming de la respuesta (SSE) |
| `POST` | `/sessions` | Crea una sesión conversacional nueva |
| `GET` | `/sessions/{session_id}` | Estado de una sesión: nº de mensajes, `max_turns`, `project_metadata` |
| `POST` | `/sessions/{session_id}/estimate` | Turno conversacional (multipart/form-data, admite adjuntos `.pdf`/`.docx`) |
| `GET` | `/health` | Healthcheck |

## Documentación por sesión

- [docs/sessions/02-cag.md](docs/sessions/02-cag.md)
- [docs/sessions/03-streaming-chat+wrapper.md](docs/sessions/03-streaming-chat+wrapper.md)
- [docs/sessions/04-form+guardrails.md](docs/sessions/04-form+guardrails.md)
- [docs/sessions/05-memory+attachments.md](docs/sessions/05-memory+attachments.md)

## Sesión 05: memoria + adjuntos

Se añadió un camino conversacional con memoria: historial (ventana deslizante) +
`project_metadata` acumulado (extraído por un segundo LLM barato tras cada turno, fusionado con lo
ya sabido). Los adjuntos (`.pdf`, `.docx`) se extraen **localmente** (Camino B) y se envían como
texto dentro del prompt, en vez de usar la Files API de un proveedor: así el resultado sigue
siendo compatible con el fallback OpenAI↔Anthropic ya existente y da más control (truncado, logs), además de dejarlo preparado para futuras ampliaciones a RAG.
Detalle completo de las decisiones de diseño en
[docs/sessions/05-memory+attachments.md](docs/sessions/05-memory+attachments.md).
