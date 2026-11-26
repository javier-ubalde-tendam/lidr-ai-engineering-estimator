# Sesión 13 — el flujo de estimación como grafo de LangGraph

El flujo transcripción → estimación se re-expresa como un **grafo con estado tipado**, persistido en el Postgres del
proyecto y observable con Logfire. Corre de principio a fin en secuencial y devuelve la estimación estructurada con
su `status`. Nada cambia de puertas afuera: `/v1/estimate/agent` y el resto de endpoints no se tocan.

Endpoint: `POST /v1/estimate/graph` (`X-API-Key` = `ESTIMATE_API_KEY`, 5/min). Las horas son **horas de ingeniero**
(el corpus guarda horas, no euros).

## El grafo

```
START → extract_requirements → classify_components → search_budgets
      → generate_estimate → validate_and_consolidate → (arista condicional)
          ├─ status == "validated"    → END
          └─ status == "needs_review" → flag_for_review → END
```

| Nodo | Qué hace |
|---|---|
| `extract_requirements` | LLM estructurado: transcripción → requisitos atómicos (`GRAPH_EXTRACTION_MODEL`). |
| `classify_components` | LLM estructurado: requisitos → componentes con categoría. |
| `search_budgets` | Una búsqueda por componente, **una tras otra** (sin `Send`), con el mismo backend RAG que el agente S12 (`historical_task`). Un fallo en un componente es *soft*: va a `errors` y se continúa. |
| `generate_estimate` | LLM estructurado (`AgentEstimate`, horas) anclado en la mediana de las horas históricas por componente; sin referencias → 0 h y se dice en `notes` (el total es un suelo). `GRAPH_GENERATION_MODEL`; si es `gpt-5*` se envían `reasoning_effort` y `max_tokens`. |
| `validate_and_consolidate` | Reutiliza `validate_estimate` de S12. `status = validated` si `ok`; si no, `needs_review`. Los `issues` van a `errors`; los `warnings` (p. ej. `no_reference`) no cambian el status. |
| `flag_for_review` | Solo en la rama de revisión: asegura `needs_review` y añade una nota a `errors`. |

## Dónde vive cada pieza

| Pieza | Ruta |
|---|---|
| Estado (dos reducers acumuladores: `budget_matches`, `errors`) | `app/domain/graph/state.py` |
| E/S de los nodos LLM | `app/domain/graph/schemas.py` |
| Nodos + `route_after_validation` | `app/domain/graph/nodes.py` |
| Cableado (`build_graph(checkpointer, deps)`) | `app/domain/graph/build.py` |
| Contrato HTTP | `app/domain/schemas/graph_estimation.py`, `app/api/routers/estimate_graph.py` |
| Checkpointer Postgres | `app/foundation/persistence/checkpointer.py` |
| Logfire (no-op seguro) | `app/foundation/observability.py` |
| Arranque (`app.state.graph`) | `app/api/graph_runtime.py` |
| Script | `scripts/run_graph_s13.py` |

Los colaboradores se **inyectan** con `GraphDependencies` (nada de `from app.dependencies import …` en los nodos);
ver [ARCHITECTURE.md](../../ARCHITECTURE.md).

## Persistencia

`AsyncPostgresSaver` (psycopg3) en el **mismo** Postgres (con pgvector); `setup()` crea sus tablas al arrancar y es
idempotente. `DATABASE_URL` (`postgresql+asyncpg://…`) se convierte a un DSN libpq. El `thread_id` de cada ejecución
es el `estimation_id` (el del body o un uuid nuevo): `await graph.aget_state({"configurable": {"thread_id": id}})`
recupera el estado. Un fallo al abrirlo **no es fatal**: el servicio arranca con `app.state.graph = None`, avisa
en el log y el endpoint responde 503. `GRAPH_ENABLED=false` ni lo intenta.

Ojo: `budget_matches` y `errors` son acumuladores. Reutilizar un `estimation_id` ya usado **añade** a lo que había
en ese hilo en vez de empezar de cero; usa un id nuevo para una estimación nueva.

## Cómo ejecutarlo

```powershell
# Offline parcial: sin Postgres y con retrieval simulado (las llamadas al LLM son reales: pide OPENAI_API_KEY)
uv run python scripts/run_graph_s13.py --memory --stub

# Real, dentro de Docker (Linux): Postgres con el corpus ingerido y checkpointer Postgres
docker compose exec ai_service python scripts/run_graph_s13.py --out exercises/session-13/example_run_complex.txt
```

El script imprime el informe (requisitos → componentes → `budget_matches` → estimación → status → errores) y el
`thread_id`. Los tests (`tests/domain/graph/`, `tests/api/test_estimate_graph.py`) usan fakes: sin red, sin Postgres.
Un test `integration` abre el checkpointer real (`uv run pytest -m integration tests/integration/test_checkpointer_db.py`).

**Windows**: el modo async de psycopg no funciona con el `ProactorEventLoop` por defecto de Windows (psycopg lo
rechaza con un error explícito). El script usa el selector loop en Windows, pero **el camino soportado es Docker
(Linux)**; `uvicorn` en el host Windows arrancará con el grafo desactivado (503) si no usa ese loop.

## Logfire

1. Crea un token de **escritura** en tu proyecto de Logfire y ponlo solo en tu `.env` local (`LOGFIRE_TOKEN=…`);
   nunca en git. `Settings` lee `.env`, pero Logfire lee el entorno: por eso `configure_logfire` pasa el token
   explícitamente (`send_to_logfire="if-token-present"`: sin token, nada se exporta).
2. Ejecuta el script en Docker (arriba) o llama al endpoint. Cada nodo abre un span `node: <nombre>`
   (`extract_requirements`, `classify_components`, `search_budgets`, `generate_estimate`,
   `validate_and_consolidate`, y `flag_for_review` en la rama); además se instrumentan FastAPI, `httpx` (llamadas a
   OpenAI) y `asyncpg` (retrieval). No se capturan cuerpos de petición (las transcripciones pueden llevar datos
   personales).
3. Abre el *Live view* de Logfire y busca los spans `node: …`.

## Qué queda fuera (lo del directo)

Paralelismo con `Send`, reintentos/backoff/fallback/timeouts/circuit breakers, intervención humana con
`interrupt()` y optimización a partir de trazas. La rama `flag_for_review` solo marca; no reintenta ni pregunta.

## Decisión: LangGraph, no LangChain

El problema es un **flujo con estado, ramas y persistencia**, no una cadena de prompts. LangGraph da exactamente eso
(estado tipado con reducers, aristas condicionales, checkpointer por `thread_id`) y deja los nodos como funciones
Python normales que siguen usando nuestro `LLMWrapper` y nuestro retrieval; LangChain añadiría abstracciones de
modelos/prompts/retrievers que ya tenemos y que no necesitamos duplicar.
