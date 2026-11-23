# Sesión 12 — resumen de la implementación del agente

Agente con bucle manual sobre la Responses API: recibe una transcripción, usa `search_budgets` y
`calculate_estimate` (+ `validate_estimate`), itera y devuelve una estimación estructurada con su traza.
Diseño detallado en [12-agent.md](12-agent.md).

## 1. Estado por fase

| Fase | Estado | Notas |
|---|---|---|
| 0 Comprobaciones previas | DONE | Kit completo; `openai 3.3.0` con `responses.create` y todos los parámetros; `RetrievalRequest`/`RetrievedChunk` y filtro `historical_task` verificados |
| 1 Configuración y cliente async | DONE | `AGENT_*` en `Settings`, `.env.example`, `get_async_openai_client()` |
| 2 Esquemas y tools | DONE | `agent_schemas.py`, `agent_tools.py` |
| 3 Bucle | DONE | `agent_loop.py` (`run_agent`, `SYSTEM_PROMPT`, `format_trace`) |
| 4 Conductor, endpoint, script | DONE | `estimate_with_agent`, `POST /v1/estimate/agent`, `scripts/run_agent_s12.py` |
| 5 Tests | DONE | Todos con fakes; sin red |
| 6 Documentación | DONE | `ARCHITECTURE.md`, `README.md`, `12-agent.md`, este resumen |

Ninguna fase BLOCKED.

## 2. Ficheros nuevos y modificados

**Nuevos**
- `app/generation/agentic/agent_schemas.py`, `agent_tools.py`, `agent_loop.py` — el agente (capa `generation/agentic`).
- `app/api/routers/estimate_agent.py` — router fino (capa `api`).
- `scripts/run_agent_s12.py` — ejecución por API o `--stub` en proceso.
- `tests/agent_helpers.py`, `tests/generation/agentic/{test_agent_tools,test_agent_loop,test_stub_flow}.py`,
  `tests/test_estimate_with_agent.py`, `tests/api/test_estimate_agent.py`.
- `docs/sessions/12-agent.md`, `docs/sessions/12-agent-summary.md`.

**Modificados**
- `app/config.py` (variables `AGENT_*`), `app/dependencies.py` (`get_async_openai_client`,
  `get_agent_estimation_service`), `app/domain/estimation_service.py` (conductor: `estimate_with_agent`,
  `build_budget_search_backend`, `AgentUnavailableError`, dos colaboradores opcionales), `app/main.py` (router).
- `.env.example`, `docker/backend/docker-compose.yml` (solo un comentario: el compose ya pasa todo `.env`
  con `env_file`), `ARCHITECTURE.md`, `README.md`.
- `exercises/session-12/` (el kit) queda sin tocar.

## 3. Endpoint nuevo

`POST /v1/estimate/agent`
- **Auth**: cabecera `X-API-Key` = `ESTIMATE_API_KEY` (401 si falta o es otra clave).
- **Límite**: 5/min por clave (429).
- **Petición**: `{transcript (100-50000 car.), model?, reasoning_effort? (minimal|low|medium|high), max_iterations? (1-20)}`.
  `model` debe ser un `gpt-5*` de `AVAILABLE_MODELS` (422 si no).
- **Respuesta 200**: `{status: "done"|"max_steps_exceeded"|"error", estimate|null, trace[], usage, iterations, error|null}`
  (superconjunto de `{status, estimate, trace, usage}`). Cada paso de `trace`: `step`, `reasoning`, `actions[]`
  (`tool`, `call_id`, `arguments`, `observation`, `duration_ms`, `error`), `usage`, `duration_ms`, `validation_error`.
- **Errores**: 503 sin `OPENAI_API_KEY`; 502 si falla la API de OpenAI (mensaje genérico); 422 validación.
- Ningún endpoint existente cambia.

## 4. Diseño de las tools (descripciones finales, tal cual)

**`search_budgets(query, filters)`** — filtros `sectors`, `project_year_min`, `project_year_max` (todos aplicados):
> Search the historical database for past engineering TASKS similar to ONE task of the project and return the engineer-hours each one actually took. The items are functional tasks (a feature or a piece of a module, roughly 16 to 72 hours each): not whole systems, and not small chores or product-specific details. Break a component down to that level before searching. Make one call per task, each with its own query; do not mix unrelated tasks in a single search. Independent searches, including the rephrasings of a round, can be issued together in the same turn. Returns `items` (id, budget_id, sector, project_year, content_preview, estimated_hours = engineer-hours, distance = cosine distance: lower is more similar), `count`, a `summary`, the `filters_applied` and `low_confidence`. When `low_confidence` is true or `count` is 0 the evidence is weak: rephrase with more generic wording (what the task does, without product or protocol names) or drop the filters. If two rephrasings are still weak, the history has no equivalent for that task: report it as unbudgeted instead of searching further. The `estimated_hours` of the relevant items are the `reference_amounts` of that task in `calculate_estimate`.

**`calculate_estimate(components[{name, tasks[{name, reference_amounts}]}])`**
> Deterministic cost model (no LLM) that turns historical hours into an estimate. Pass the project components, each with its `tasks`; every task has a `name` and `reference_amounts`: the `estimated_hours` of the relevant items that `search_budgets` returned for that task (an empty list when nothing relevant was found). A task costs the median of its references plus a fixed 15% contingency; a component is the SUM of its tasks; the project total is the sum of the components. Returns per component its `estimated_hours` and the per-task breakdown, plus `total_hours` and a `summary`. A task without references costs 0 h and is listed in `unbudgeted_tasks` (hours are never guessed): search again for it, or report it as unbudgeted. Its output is the basis of the final answer.

**`validate_estimate(components[{name, tasks[{name, estimated_hours, reference_amounts}]}], total_hours)`**
> Sanity-check a draft estimate before answering. Pass every component with its tasks (each with `name`, `estimated_hours` as returned by `calculate_estimate`, and `reference_amounts`), plus the `total_hours` you intend to report. Returns `ok`, a list of `issues` (task without references, task hours far outside the range of its references, total that does not match the sum of the tasks, non-positive or disproportionate total) and a `summary`. When `ok` is false, fix the cause (search again or recompute) and validate again.

Por qué así: es lo único que lee el modelo, así que cada descripción dice qué hace, cuándo usarla (una búsqueda por
tarea, agrupando las independientes en un mismo turno), qué devuelve y cómo encadenarla (las horas de
`search_budgets` son los `reference_amounts` de cada tarea en `calculate_estimate`); dice explícitamente que el corpus
son tareas (la primera ejecución real se atascó buscando "sistemas enteros"); la `query` pide ≥ 10 caracteres y
específica (contrato de retrieval); `low_confidence` + `hint` permiten reformular; no se dicta cuántas búsquedas
hacer. Los schemas son planos (`type/name/description/parameters/strict`), `strict: true`, todo en `required`
(opcionales con `["tipo","null"]`) y `additionalProperties: false` en cada objeto (lo comprueba un test).

**Convergencia (prompt + bucle):** el prompt informa del presupuesto de vueltas y de que las últimas están reservadas
(`build_system_prompt(max_iterations)`), fija una regla de parada (tras dos reformulaciones sin buen resultado: tarea
sin referencias, a `notes`, y seguir) y pide agrupar también las reformulaciones. Como el texto solo no bastó en la
segunda ejecución real, el **bucle** lo hace cumplir: de la vuelta `max_iterations - 2` en adelante `tool_choice`
restringe a `calculate_estimate`/`validate_estimate` y la última va con `tool_choice="none"`; cada continuación lleva
un aviso "turn N of M". Los resultados vacíos o flojos de `search_budgets` incluyen `filters_applied` y una salida
explícita ("sin equivalente en el histórico: sin presupuesto"). Sigue sin fijar el número de búsquedas ni nombrar
componentes.

## 5. Variables de entorno nuevas

| Variable | Default | Para tu `.env` |
|---|---|---|
| `AGENT_MODEL` | `gpt-5` | opcional |
| `AGENT_REASONING_EFFORT` | `medium` | opcional (`minimal`/`low`/`medium`/`high`) |
| `AGENT_MAX_ITERATIONS` | `10` | opcional |
| `AGENT_SEARCH_TOP_K` | `5` | opcional |
| `AGENT_SEARCH_DISTANCE_THRESHOLD` | `0.6` | opcional |

Obligatorias (ya existentes): `OPENAI_API_KEY` (el agente necesita OpenAI aunque `LLM_PROVIDER` sea otro) y
`ESTIMATE_API_KEY`. Tras cambiar `.env` en Docker: `docker compose up -d --force-recreate ai_service`.

## 6. Tests

| | Baseline (S11) | Final |
|---|---|---|
| `uv run pytest -q` | 878 passed, 53 deselected | **956 passed**, 53 deselected (+78) |
| `uv run pytest -m integration -q` | — | 53 passed (Postgres local en Docker) |
| `uv run ruff check .` | 1 error (`PERF402`, `tests/test_llm_service.py`) | el mismo único error, sin nuevos |
| `tests/test_layering.py` | verde | verde, sin añadir `ALLOWED_EXCEPTIONS` |

Nuevos: schemas de tools (planos, `strict`, `required`/`additionalProperties` a todos los niveles, formato final
estricto); bucle (llamada única, paralelas en una sola continuación, `call_id` correcto, encadenado
`previous_response_id` + `instructions`/`tools` reenviadas, tool que falla / args inválidos / JSON roto / tool
desconocida, `max_iterations` agotado, final inválido → un turno correctivo → `error`, sin resúmenes de razonamiento,
uso de tokens, fallo de OpenAI, instrucciones con el presupuesto de vueltas y regla de parada); `calculate_estimate`
(mediana por tarea, contingencia, componente = suma de sus tareas, tarea sin referencias, determinismo, redondeos);
`validate_estimate` (cada incidencia); `search_budgets` (forma compacta, `low_confidence`, filtros hasta el backend);
backend real con retriever falso (`RetrievalRequest` con `chunk_types=["historical_task"]`, `top_k`, umbral y filtros);
`format_trace`; endpoint (401, 200, `max_steps_exceeded`, 422, 503, 502, 429); y una comprobación offline con
`sample_transcript_complex.txt` + el stub del kit y un modelo guionizado (4 búsquedas, una por componente).

## 7. Decisiones y desviaciones respecto al enunciado

1. **Un único punto de `responses.create` dentro del `for`** (en vez de "primera llamada fuera del bucle"): evita duplicar argumentos; mismo comportamiento.
2. **`for … else`**: ruff `PLW0120` lo prohíbe sin `break`; el `return max_steps_exceeded` va justo después del bucle.
3. **`text.format` estricto en todas las vueltas** + validación Pydantic (no `responses.parse`), para no esconder el bucle.
4. **Sin `component_type`** como filtro (no existe en el store); filtros `sectors` y rango de años, todos aplicados.
5. **`status="error"` y `max_steps_exceeded` → HTTP 200** con traza; 503/502 solo para fallos de transporte.
6. **Modelos del endpoint**: solo `gpt-5*` de `AVAILABLE_MODELS` (subconjunto de lo pedido: `reasoning` no existe en gpt-4o*/Claude).
7. **Docker**: los compose ya pasan todo `.env`; no se redeclaran variables (sería pisarlas con defaults); solo se actualizó un comentario.
8. **Conductor**: `EstimationService` recibe dos colaboradores opcionales y hay un factory propio (`get_agent_estimation_service`) para no arrastrar embedder/reranker a los demás endpoints.
9. **Elementos sin horas** se descartan de `items` (no sirven como `reference_amounts`). `low_confidence` si no hay resultados o el mejor supera 0,45 de distancia.
10. **Preview** de cada item: texto del chunk sin la cabecera del presupuesto padre (primer bloque hasta la línea en blanco).
11. **Sin guardrails de entrada** en el endpoint (igual que `from-transcript`).
12. La lista de sectores en la descripción de la tool es estática (el backend valida y devuelve el error).
13. **Estimación por tareas** (tras la primera ejecución real, que agotó las 10 vueltas buscando "el backend entero" sin calcular): el corpus son tareas de 12-63 h, así que `calculate_estimate` y `validate_estimate` trabajan con `components[{name, tasks[...]}]`; tarea = mediana de sus referencias + 15 %, componente = **suma** de sus tareas, total = suma de los componentes. Cambio de contrato respecto al esqueleto del kit (que daba una mediana por componente).
14. **Convergencia en el bucle** (tras la segunda ejecución real, `trace_complex_02.txt`: 24 búsquedas en paralelo en la vuelta 1, luego una por vuelta y 10 vueltas sin calcular): `SYSTEM_PROMPT` → `build_system_prompt(max_iterations)`; vueltas finales reservadas (`min(3, max_iterations // 3)`) con `tool_choice` = `allowed_tools` (solo calcular/validar) y última vuelta con `"none"`; aviso de turno en cada continuación. Esto **cambia** la semántica pedida ("solo `max_steps_exceeded` al agotar vueltas"): ahora el agente casi siempre llega a una respuesta, y `max_steps_exceeded` queda como red de seguridad.
15. **Granularidad y vacíos**: el prompt pide tareas funcionales de ~16-72 h (3 a 6 por componente) y la tool devuelve `filters_applied` y una salida explícita ante resultados vacíos o flojos. El corpus sintético de `build_task_corpus.py` no contiene SAP/ERP/IDoc/OpenAPI, así que esas tareas acabarán como "sin presupuesto" en `notes` (resultado legítimo) salvo que amplíes las plantillas y reingieras.

## 8. NO VERIFICADO

- **Tres ejecuciones reales**
  - Las dos primeras (`trace_complex.txt`, `trace_complex_02.txt`) `max_steps_exceeded`. El comportamiento con la convergencia forzada del bucle y el nuevo prompt **no está medido**: hay que repetir la ejecución real. Con tareas habrá bastantes búsquedas por componente: si 10 vueltas no bastan, sube `AGENT_MAX_ITERATIONS` (el endpoint admite hasta 20).
  - La tercera (`trace_complex_03_OK.txt`) finaliza pero no es del todo fiable.
- **No contrastado con la API**: `tool_choice` como `allowed_tools` (`mode: "auto"`, lista de funciones por nombre) y como `"none"` en una petición con `previous_response_id`, `text.format` estricto y reasoning. Está en el SDK instalado (`ToolChoiceAllowedParam`) y en la guía oficial, pero no se ha ejecutado. Si la API lo rechazase (400), el síntoma sería un `502` en la vuelta 8.
- Verificado contra la guía oficial *Function calling*: tools planas con `strict`, `function_call_output` con `call_id`, varias llamadas por respuesta, `previous_response_id` con `instructions` reenviadas, `["tipo","null"]` para opcionales.
- **No contrastado con la API**: (a) combinar `tools` y `text.format` json_schema estricto en la misma petición con gpt-5/gpt-5-mini; (b) que acepte `type: ["object","null"]` con `properties` anidadas en `filters` (el patrón documentado es `["string","null"]`); (c) que `reasoning={"summary": "auto"}` esté disponible en tu organización (puede exigir verificación): si da 400, pasa `reasoning_summary=None` a `run_agent` y la traza usará `(no reasoning summary available)`; (d) que los `reasoning` items se mantengan entre vueltas solo con `previous_response_id` (es lo que documenta la guía).
- Que el corpus de tareas ingerido devuelva distancias < 0,6 para las consultas del agente (depende de tu BD).
- `mypy` no se ha ejecutado sobre el código nuevo (ya no estaba verde en la S11).

## 9. Comandos para validar y generar la traza del entregable

```powershell
# 0. Variables: OPENAI_API_KEY y ESTIMATE_API_KEY en .env (opcional AGENT_*). Tras tocar .env en Docker:
docker compose up -d --force-recreate ai_service

# 1. Stack arriba + corpus de tareas ingerido (las horas salen de los chunks historical_task)
docker compose up --build -d
uv run python scripts/build_task_corpus.py --ingest

# 2. Comprobación sin gasto
uv run ruff check .
uv run pytest -q
uv run pytest -m integration -q

# 3. Depurar barato (API en marcha): gpt-5-mini + minimal + transcripción simple
uv run python scripts/run_agent_s12.py exercises/session-12/sample_transcript_simple.txt --model gpt-5-mini --effort minimal

# 3b. Solo la mecánica del bucle (sin BD ni API; llamadas reales al modelo)
uv run python scripts/run_agent_s12.py exercises/session-12/sample_transcript_simple.txt --model gpt-5-mini --effort minimal --stub

# 4. Ejecución real del entregable: gpt-5 medium sobre la compleja
uv run python scripts/run_agent_s12.py exercises/session-12/sample_transcript_complex.txt --model gpt-5 --effort medium --out exercises/session-12/trace_complex.txt
```

Criterio de aceptación a mirar en la traza: más de una búsqueda (una por componente), alguna reformulación si hay
`low_confidence`, `calculate_estimate` y `validate_estimate` usadas, `status: done`.

## 10. Cómo ver trazas del servicio en funcionamiento

```powershell
# Todo el backend en tiempo real
docker compose logs -f ai_service

# Solo los eventos del agente
docker compose logs -f ai_service | Select-String "agent_"

# Empezar desde las últimas líneas, sin el histórico
docker compose logs -f --tail 0 ai_service
```