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
- `calculate_estimate` recibe `components[{name, tasks[{name, references[{budget_id, estimated_hours}]}]}]`
  (las referencias llevan su id: ver «Referencias con id»). Cada **tarea** cuesta la
  **mediana** de sus referencias más la contingencia fija (`CONTINGENCY_FACTOR = 0.15`); el **componente** es la
  **suma** de sus tareas y el total la suma de los componentes. Los redondeos se encadenan (tarea → componente →
  total) para que cada nivel coincida con lo que se muestra. Una tarea sin referencias cuesta 0 h y aparece en
  `unbudgeted_tasks` (no se inventan horas).
- Mediana y no media porque, con pocas referencias, una tarea atípica (una muy holgada o una hecha con prisas)
  arrastra la media; la mediana la ignora.

**`validate_estimate` (extensión opcional).** Determinista y por tarea, con dos severidades (ver «Validador con
`warnings`»). `issues` (rompen `ok`): horas fuera de `[0,5·min(refs), 1,5·max(refs)]`, horas no positivas, total que
no cuadra con la suma de las tareas, total no positivo o desproporcionado (> 20 000 h). `warnings` (informativos):
tarea sin referencias. Cada hallazgo lleva `where` ("Componente / Tarea"). Devuelve `{ok, issues, warnings, summary}`.

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

## Evolución y estado actual

### Evolución del agente (una frase por traza)

1. `trace_complex_01_FAIL`: el agente solo buscó (13 búsquedas largas, con consultas a nivel de componente) y agotó las 10 iteraciones sin llegar a calcular ni responder (`max_steps_exceeded`, ~80k tokens).
2. `trace_complex_02_FAIL`: se añadió más paralelismo (33 búsquedas), pero el agente siguió reformulando consultas con jerga muy específica (IDoc, DEBMAS/MATMAS, RabbitMQ) que el corpus no tiene, y volvió a agotar las 10 iteraciones sin respuesta (~163k tokens).
3. `trace_complex_03_OK`: con el cierre forzado y las consultas más funcionales, el agente por fin convergió y devolvió un estimado, pero era una cota inferior sin avisar, con referencias reconstruidas de memoria, una validación contradictoria y un cálculo desactualizado respecto a las últimas búsquedas.
4. `trace_complex_04`: tras el refuerzo, el agente recalcula después de buscar, usa referencias por id válidas, el sistema informa de cobertura (15 de 20), cota inferior y tareas sin presupuestar, y el validador distingue avisos de errores. Queda un fallo: validó con solo 2 de 4 componentes, y la validación final con `total_mismatch` se ignoró.

### Estado actual y limitación conocida (`trace_complex_04.txt`)

La última ejecución real (gpt-5 medium, transcripción compleja) termina con `status: done` en 10 vueltas, ~318 k
tokens y 563,7 h. `RESULT` muestra la cobertura (**15 de 20 tareas con referencias, 75 %**), `total is a lower bound:
yes` y las 5 tareas sin presupuesto; la primera validación da «0 issues, 5 warnings: no_reference×5» (ya no hay falsos
errores) y los `budget_id` citados son ids reales devueltos por las búsquedas.

**Fallo pendiente (no corregido):** el agente validó con solo **2 de los 4 componentes**, y la validación final
devolvió `total_mismatch` (1 issue, 3 warnings), que ignoró. El total entregado sí coincide con la suma de los cuatro
componentes del último cálculo, así que el número no está mal; lo que falla es que la validación no cubrió el borrador
completo y el bucle no lo detecta. Posibles refuerzos (no implementados): que el bucle compare los componentes de
`validate_estimate` con los del último `calculate_estimate` y avise (`partial_validation`), o que trate un `issue`
sin resolver en la última validación como aviso en `AgentResult.warnings`.

## Endurecimiento tras la ejecución real `trace_complex_03_OK.txt`

La tercera ejecución real terminó (`status: done`) pero la traza mostró cinco defectos. Los cambios siguientes los
corrigen sin tocar el umbral de recuperación ni el nivel de tareas; la cuarta ejecución (`trace_complex_04.txt`) los
confirma en la práctica (ver arriba).

**Nivel de tareas (diseño, no cambia).** El corpus (`historical_task`) son tareas de 12-71 h (mediana 30 h), por
eso el prompt pide tareas funcionales de ~16-72 h, «de 3 a 6 por componente», y la tool de búsqueda dice lo mismo.
Es parte del diseño: la estimación es **suma de tareas** y solo es comparable con el histórico a ese nivel.

**Invariante de frescura (vive en el bucle, no en la tool).** La estimación final debe salir de un
`calculate_estimate` hecho **después de la última `search_budgets` que devolvió evidencia**. `calculate_estimate` es
una función pura, sin acceso a la conversación, y el enunciado admite cálculos parciales; por eso no se le añade
ninguna precondición: la garantía la da `run_agent`, que recuerda el turno de la última búsqueda con ítems y el de
la última calculadora correcta.
- Una búsqueda en el **mismo** turno que el cálculo cuenta como posterior (el cálculo no pudo verla). Una búsqueda
  sin resultados no es evidencia nueva y no obliga a recalcular.
- Si el modelo **responde** con una búsqueda posterior al último cálculo → turno correctivo («llegó evidencia nueva
  tras tu último cálculo: recalcula»), con el mismo mecanismo que las respuestas inválidas (más el aviso de turno).
- Si llama a `validate_estimate` con un cálculo desfasado no falla: la observación recibe un aviso determinista
  `stale_calculation` en `warnings` (y en el `summary`).
- Se comprueba también que `total_hours` coincide con el del último cálculo (±`TOTAL_TOLERANCE_HOURS`).
- **Caso límite**: el último turno va con `tool_choice="none"` y no puede recalcular. El turno correctivo solo se da
  si **queda un turno posterior en el que `calculate_estimate` está permitida y otro después para responder**
  (`_can_recalculate`, que reutiliza `_tool_choice`/`wrap_up_turns`). Si no, la respuesta se **acepta** y el aviso
  `stale_calculation` queda en `AgentResult.warnings`, en `AgentStep.warnings` y en la traza. Un descuadre de total
  solo necesita un turno para responder de nuevo. Los turnos correctivos están acotados (`MAX_ANSWER_CORRECTIONS = 2`
  y siempre ≤ `max_iterations`): pasado ese límite se acepta con aviso, nunca hay bucles.

**Referencias con id (segundo cambio de contrato respecto al kit del profesor).** El primer cambio fue el nivel
`tasks` (el kit daba una mediana por componente). El segundo: `calculate_estimate` y `validate_estimate` ya no
reciben `reference_amounts: number[]` sino `references: [{budget_id, estimated_hours}]` (schemas estrictos: todo en
`required`, `additionalProperties: false` en cada nivel). Antes el modelo reconstruía de memoria los
`reference_budget_ids` al final (ids mezclados entre componentes). Ahora:
- `calculate_estimate` devuelve `reference_budget_ids` por tarea y por componente (sin duplicados, orden estable) y
  la mediana se calcula sobre `estimated_hours` como siempre.
- El bucle guarda los `budget_id` de todas las observaciones de `search_budgets` de la ejecución y **valida** el JSON
  final: un id desconocido, o repetido dentro de un componente, provoca turno correctivo (misma regla de turnos).
- Si el conjunto de ids de un componente difiere del último cálculo, solo se registra un aviso
  `reference_ids_differ` en la traza (no se rechaza: elegir un subconjunto puede ser legítimo).

**Cobertura calculada por el sistema.** El bucle guarda el último resultado de `calculate_estimate` y deriva, sin
intervención del modelo, `unbudgeted_tasks` («Componente / Tarea»), `coverage` (`budgeted_tasks` / `total_tasks` /
`ratio`, p. ej. 11 de 21) y `total_is_lower_bound` (hay tareas sin presupuesto: valen 0 h, así que el total es una
**cota inferior**, no una estimación completa). Van en `AgentResult` (y por tanto en la respuesta de
`POST /v1/estimate/agent`, solo **añadiendo** campos, junto a `warnings`) y en el bloque `RESULT` de la traza.
`AgentEstimate`, lo que rellena el modelo, no cambia.

**Validador con `warnings`.** `no_reference` pasa de `issue` a `warning` y no rompe `ok`: cuando ya no se puede
buscar, una tarea sin referencias es un hueco ya declarado (coste 0 h, a `notes`), no algo que corregir; tratarla
como error empujaba al modelo a dudar en dos turnos («10 issue(s) found»). `ok` es `true` si no hay `issues`. El
`summary` detalla recuentos y códigos («0 issues, 10 warnings: no_reference×10»). La descripción de la tool y el
paso 5 del prompt se cambiaron **a la vez** (si no, el modelo seguiría viendo la contradicción): los `issues` se
corrigen; las tareas sin presupuesto ya declaradas son aceptables y se informan en `notes`.

**Prompt (método, no pasos).** Calcular cuando termina la ronda de búsquedas; si se busca después de calcular,
recalcular; preferir **una ronda agrupada de reformulaciones** (en paralelo) antes de calcular. Sin números fijos de
búsquedas; se mantienen «como mucho dos reformulaciones por tarea» y los turnos reservados.

**Por qué NO se sube el umbral de distancia** (`AGENT_SEARCH_DISTANCE_THRESHOLD = 0,6`, `WEAK_MATCH_DISTANCE = 0,45`).
Las 10 tareas a 0 h de la traza (tarifas, offline, firma/foto, SAP/ERP, maestros, facturas) no tienen equivalente en
el corpus: 0 coincidencias por palabra clave en las 1 543 tareas. Subir el umbral solo añadiría análogos
irrelevantes y haría la estimación parecer más completa de lo que es. La respuesta correcta a una laguna del
histórico es mostrarla (`coverage`, `unbudgeted_tasks`, `total_is_lower_bound`), no taparla.

**Traza.** Los `action:` de `calculate_estimate` y `validate_estimate` ya no se truncan (se imprimen en varias líneas:
un componente por bloque, una tarea por línea con `budget_id=horas`). El resto se recorta a ≥ 600 caracteres con la
marca explícita `…[truncated]`. Las observaciones de `search_budgets` añaden, tras el resumen, los 3 ítems más
cercanos (`budget_id`, horas, distancia) para auditar qué referencias eligió el agente. Los avisos
(`stale_calculation`, `reference_ids_differ`, …) y la cobertura salen en la traza.

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
  que los respaldan, las `notes` (huecos de evidencia, supuestos), la **cobertura** (`11 of 21 tasks have
  historical references`), si el total es una **cota inferior**, las tareas sin presupuesto y los `warning`.
- Para darla por buena: más de una búsqueda; `calculate_estimate` llamada **después** de la última búsqueda;
  `status: done`; cobertura visible; los ids de `RESULT` existen en los `closest:` de las búsquedas.
