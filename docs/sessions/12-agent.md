# Sesión 12 — agente con bucle manual sobre la Responses API

Un agente recibe una transcripción, la descompone en componentes y estos en tareas, busca tareas históricas
**por tarea**, calcula con una función determinista (componente = suma de las medianas de sus tareas), valida el
borrador y devuelve una estimación estructurada **más la traza** de lo que pensó y observó en cada vuelta.

Código: `app/generation/agentic/agent_schemas.py`, `agent_tools.py`, `agent_loop.py`; conductor:
`EstimationService.estimate_with_agent`; endpoint: `POST /v1/estimate/agent`; script:
`scripts/run_agent_s12.py`. Las horas son **horas de ingeniero** (el corpus guarda horas, no euros).

## Flujo

```
POST /v1/estimate/agent  (api/routers/estimate_agent.py: auth, 5/min, overrides validados)
  └→ EstimationService.estimate_with_agent()          (conductor: construye el backend y el registro)
       └→ run_agent()                                  (generation/agentic/agent_loop.py)
            vuelta N: responses.create → ¿function_call?
              sí → ejecuta TODAS en paralelo (asyncio.gather) → function_call_output → vuelta N+1
              no → respuesta final → valida contra AgentEstimate → AgentResult
       search_budgets → backend inyectado → TranscriptEstimator.retrieve_chunks(RetrievalRequest(
                          chunk_types=["historical_task"], top_k, distance_threshold, filtros))
```

## Decisiones de diseño

**El bucle es nuestro.** Sin `tool_runner`, SDK de agentes ni LangChain. Cada vuelta es una llamada a
`client.responses.create`. La primera envía la transcripción; las siguientes solo los items nuevos y se encadenan
con `previous_response_id` (el servidor conserva los items previos, incluidos los de razonamiento, por eso **no**
se usa `store=False`). `instructions` **no** se arrastra con `previous_response_id`: se reenvía en cada vuelta junto
con las `tools`. La llamada vive en un único punto dentro del `for` (en vez de una primera llamada fuera del bucle
y otra dentro) para no duplicar los argumentos; el comportamiento es el mismo.

**Paralelismo.** Una vuelta puede traer varios `function_call` (p. ej. una búsqueda por tarea). Se ejecutan
con `asyncio.gather` y **todos** los `function_call_output` vuelven en **una sola** petición, cada uno con el
`call_id` de su llamada. En la traza, un STEP con varias acciones muestra un par action/observation por llamada.

**Los errores son observaciones.** Argumentos inválidos (Pydantic), JSON roto, tool desconocida o excepción en el
backend no revientan el bucle: `execute_tool` devuelve `{"error": "..."}` y el modelo lo lee, corrige y reintenta.
Solo un fallo de la propia API de OpenAI (`AgentLoopError`, con la traza parcial) corta la ejecución.

**Salvaguarda de paradas.** Máximo `AGENT_MAX_ITERATIONS` vueltas del modelo, con las últimas reservadas para cerrar
(ver "Convergencia" más abajo). Si se agotan →
`status="max_steps_exceeded"` con la traza parcial y sin más llamadas (ruff `PLW0120` prohíbe `for … else` sin
`break`, así que el "else" es el `return` tras el bucle; semántica idéntica).

**Salida final estructurada.** Se usa `client.responses.create` en todas las vueltas con
`text={"format": {"type": "json_schema", "strict": true, ...}}`, el schema generado desde `AgentEstimate`
(`strict_json_schema`: `required` completo y `additionalProperties: false` en cada objeto) y validación posterior
con Pydantic (`total_hours` = suma de componentes ±0,5 h). El formato solo condiciona el mensaje final, nunca las
llamadas a funciones. Si no valida, **un** turno correctivo (cuenta como iteración); si vuelve a fallar →
`status="error"` con la traza. Se descartó `responses.parse(text_format=...)` para no esconder la semántica del bucle.

**Filtros expuestos de `search_budgets`** (todos aplicados de verdad): `sectors`, `project_year_min`,
`project_year_max` (el "date_range" del enunciado). **No se expone `component_type`**: no existe en el store (la
metadata tiene `complexity`, `main_technology`, `module`, `component_id`) y filtrar por texto libre vaciaría los
resultados; la tarea se describe en la propia `query` y lo resuelve la similitud semántica. La tool siempre
restringe a `chunk_type="historical_task"` (los únicos chunks con horas). Un sector inexistente o un rango de años
invertido vuelven al modelo como error legible.

**`search_budgets` no reimplementa la recuperación.** Su backend (un closure del conductor) solo traduce los
argumentos a un `RetrievalRequest`; híbrido y reranking los decide `retrieve_chunks` (petición → override Redis →
Settings). La salida es compacta: `items` (id, budget_id, sector, project_year, `content_preview` ≤160, horas,
distancia), `count`, `summary` y `low_confidence`. Es `low_confidence` si no hay resultados o el mejor está a más de
0,45 de distancia (el umbral de `TASK_HOURS_DISTANCE_THRESHOLD`): entonces añade un `hint` para reformular.

**Estimación por tareas: componente = suma de las medianas de sus tareas.** El corpus histórico son
**tareas sueltas** (de unas decenas de horas), no sistemas. La primera ejecución real sobre la transcripción compleja
lo dejó claro: el agente se pasó 9 vueltas buscando "el backend entero" (los ítems devueltos eran de 12-63 h), nunca
encontró nada comparable y agotó las 10 vueltas sin calcular. Y, aunque hubiera terminado, la mediana de tareas por
componente habría dado ~25-50 h para un backend completo. Por eso:

- `search_budgets` dice que los ítems son tareas y pide **una búsqueda por tarea**; el prompt hace descomponer cada
  componente hasta ese nivel antes de buscar.
- `calculate_estimate` recibe `components[{name, tasks[{name, reference_amounts}]}]`. Cada **tarea** cuesta la
  **mediana** de sus referencias más la contingencia fija (`CONTINGENCY_FACTOR = 0.15`); el **componente** es la
  **suma** de sus tareas y el total la suma de los componentes. Los redondeos se encadenan (tarea → componente →
  total) para que cada nivel coincida con lo que se muestra. Una tarea sin referencias cuesta 0 h y aparece en
  `unbudgeted_tasks` (no se inventan horas).
- Mediana y no media porque, con pocas referencias, una tarea atípica (una muy holgada o una hecha con prisas)
  arrastra la media; la mediana la ignora.

**`validate_estimate` (extensión opcional).** Determinista y por tarea: tarea sin referencias, horas fuera de
`[0,5·min(refs), 1,5·max(refs)]`, horas no positivas, total que no cuadra con la suma de las tareas, total no
positivo o desproporcionado (> 20 000 h). Cada incidencia lleva `where` ("Componente / Tarea"). Devuelve
`{ok, issues, summary}`.

**System prompt.** Define el rol y un **método** (descomponer en componentes y tareas funcionales → buscar por tarea,
agrupando en el mismo turno las búsquedas independientes y también las reformulaciones de una ronda → reformular si
la evidencia es débil → calcular → validar), sin fijar cuántas búsquedas hacer ni nombrar componentes: si fuera
directivo, el criterio "más de una búsqueda" dejaría de medir al agente. Vive en `agent_loop.py`
(`build_system_prompt(max_iterations)`). Pide una **granularidad** alineada con el corpus: tareas funcionales de
~16-72 h (de 3 a 6 por componente), no tareas menores (documentación, CI, configuración) ni detalles de producto.

**Convergencia (segunda ejecución real, `trace_complex_02.txt`).** Con tareas el agente lanzó 24 búsquedas en
paralelo en la vuelta 1, pero después reformuló **de una en una** y agotó las 10 vueltas sin calcular; el presupuesto
de vueltas y la regla de parada del prompt no los aplicó. Además, el corpus sintético no tiene nada de SAP/ERP/OpenAPI,
así que esas reformulaciones no podían tener éxito. Un texto en el prompt no basta, así que el **bucle** fuerza la
convergencia:

- Las últimas vueltas están **reservadas** (`wrap_up_turns = min(3, max_iterations // 3)`): con 10 vueltas, de la 8 a
  la 9 `tool_choice` es `allowed_tools` (`auto`) restringido a `calculate_estimate` y `validate_estimate` (la lista
  `tools` no cambia, para no invalidar la caché del prompt) y la **10** va con `tool_choice="none"`: solo puede
  contestar. Con `max_iterations` ≤ 2 no se reserva ninguna.
- Cada continuación lleva un aviso corto (`[Turn N of M; K left ... Searching is possible until turn S]`) y el prompt
  anuncia el mismo reparto, así el modelo planifica en lugar de descubrirlo al final.
- `max_steps_exceeded` se mantiene como red de seguridad (p. ej. un fake que siga llamando a tools, o una respuesta
  final inválida en la última vuelta).
- Si el modelo no llega a calcular, la respuesta final de la última vuelta sale sin `calculate_estimate`: el validador
  de `AgentEstimate` sigue exigiendo que el total sea la suma de los componentes y las lagunas van a `notes`.

**Salida de `search_budgets` ante resultados vacíos o flojos.** Antes decía solo "prueba otros términos", lo que
invitaba a insistir. Ahora devuelve `filters_applied` y un `hint` con salida: reformular con lenguaje más genérico (o
sin filtros, una vez) y, si sigue vacío, tratar la tarea como **sin equivalente en el histórico** (0 h, a `notes`).
Con resultados flojos (> 0,45) dice que se usen solo los ítems claramente del mismo tipo y que se reformule como
máximo dos veces.

**Fuera de `LLMWrapper`.** El agente usa `AsyncOpenAI` (`get_async_openai_client`, `None` sin `OPENAI_API_KEY`).
Es la excepción deliberada para poder ver el bucle; los demás caminos siguen pasando por `LLMWrapper`.

**Endpoint.** Los modelos admitidos son los `gpt-5*` de `AVAILABLE_MODELS` (la API de razonamiento no está
disponible en `gpt-4o*` ni en Claude). `max_steps_exceeded` y `error` son estados de negocio: HTTP 200 con la
traza; falta de clave → 503; fallo de OpenAI → 502 genérico (el detalle solo va a los logs). No pasa por los
guardrails de entrada: la transcripción va al modelo como mensaje de usuario, igual que en `from-transcript`.

## Cómo leer una traza

```
STEP 1
  reasoning:   Four components, each split into tasks; I will search each task separately.
  action:      search_budgets(query="OAuth2 authentication backend...", filters=null)
  observation: 4 historical items; hours 380-420 (median 400); closest distance 0.31.
  action:      search_budgets(query="SAP ERP integration middleware...", filters=null)
  observation: No historical items found for '...'.
STEP 2
  ...
RESULT
  status:      done
  iterations:  5
  tokens:      input=... output=... reasoning=... total=...
  total hours: 3010.5
  component:   Backend: 1322.5 h (refs: TASK-2023-0007, ...)
```

- Un **STEP** es una vuelta del modelo. `reasoning` es el resumen de razonamiento que devuelve la API (si no hay,
  `(no reasoning summary available)` o el texto que escribió el modelo antes de llamar a las tools).
- Cada `action` es una llamada a una tool con sus argumentos; su `observation` es el `summary` que devolvió (o
  `ERROR: ...` si falló). En JSON (`/v1/estimate/agent`) cada acción incluye además `observation` completa,
  `duration_ms` y `error`, y cada paso su `usage` de tokens.
- Señales útiles: varias acciones en el mismo STEP = llamadas paralelas; una búsqueda repetida con otra `query`
  tras una observación `low_confidence` = el agente **reformuló**; un STEP sin acciones = la respuesta final
  (`observation: final answer rejected (...)` indica el turno correctivo).
- El bloque `RESULT` resume estado, vueltas, tokens agregados, total de horas, componentes con los `budget_id`
  que los respaldan y las `notes` (huecos de evidencia, supuestos).
