# Session 06 - Stress test del CAG (ejercicio previo)

## Funcionalidad incremental

Ejercicio previo a la sesión 6: **medir con datos dónde se degrada el CAG actual** bajo
conversaciones largas y adjuntos grandes, antes de introducir RAG. No se cambia ninguna política del
sistema (compresión, límites, modelo): el baseline mide lo que hay.

Se añade:
- Una observación estructurada por turno, `turn_observed`, emitida con structlog y devuelta en la
  respuesta de `POST /sessions/{session_id}/estimate` (campo `turn_observation`).
- `GET /sessions/{session_id}/memory`: snapshot de lo que la sesión recuerda (resumen, anclas,
  metadata). Endpoint de depuración/evals, no pensado para producción. Se usa para preguntarle a la aplicación qué datos recuerda y poder tomar medidas en el test de stress (si olvida, si contradice, etc.) según el escenario.
- `evals/stress/scenarios.py`: tres escenarios de 22 turnos (`growing`, `pivot`, `contradiction`) con
  un fact-tracker de hechos literales.
- `evals/stress/fixtures/build_pdfs.py`: PDFs sintéticos de ~5, 20, 50 y 100 KB de texto extraído.
- `evals/stress/metrics.py`: `LatencyBudgetMetric`, `CostBudgetMetric` y `MemoryDriftMetric`.
- `evals/stress/run.py`: runner CLI que escribe un CSV con una fila por turno.
- `evals/stress/REPORT.md`: informe preparado para rellenar con los resultados reales.

## Tecnologías

- httpx / `TestClient`: el runner ataca un servidor real (`--http`) o la app en proceso, igual que `evals.run`
- fpdf2 (dev-dependency, ya existente): genera los PDFs sintéticos; no se añade ninguna dependencia nueva
- pypdf: el mismo extractor de la app (`extract_text`) mide cuántos caracteres salen de cada PDF
- structlog: emite el evento `turn_observed`
- `csv` de la biblioteca estándar: formato de salida; sin base de datos ni librerías de gráficos

## Conceptos

### Qué es CAG y cómo se construye hoy el contexto

CAG (como en la sesión 02) consiste en meter en el prompt todo lo que el modelo
necesita saber, sin recuperar nada dinámicamente: el contexto lo decide el código, no una búsqueda.
En el camino conversacional (`estimate_conversational()` en `app/services/llm_service.py`), el array de
`messages` de cada turno se construye así:

| Orden | Pieza | De dónde sale | Crece con... |
|---|---|---|---|
| 1 | `system`: reglas, ejemplos, `<project_metadata>` y `<audience>` (tier) | Prompt `v4` + metadata acumulado + tier resuelto antes de renderizar | El metadata (tecnologías acumuladas) |
| 2 | Resumen (mensaje `user` con prefijo fijo) | `CumulativeSummarizer`, turnos expulsados de la ventana que no eran ancla | Las expulsiones; lo recorta el propio resumen |
| 3 | Anclas | Pares user/assistant con un compromiso duradero (NDA, presupuesto cerrado...) | Cada ancla: **nunca se expulsan** y no hay máximo |
| 4 | Ventana reciente | Últimos `MAX_CONVERSATION_TURNS` (6) pares literales | Tope fijo |
| 5 | `user`: transcript del turno **+ texto de los adjuntos** | `enrich_transcript()` | El tamaño del adjunto (hasta `MAX_ATTACHMENT_CHARS`) |

Detalles que importan para interpretar las medidas:
- Los adjuntos **no se guardan en el historial**: el turno solo conserva una referencia al nombre del
  fichero. Un cliente que sigue hablando del documento tiene que volver a subirlo, y cada subida
  vuelve a pagar su texto completo. Por eso el runner adjunta el PDF en cada turno.
- El texto enriquecido (transcript + adjuntos) también lo ven los guardrails, el resolutor de tier y
  el extractor de metadata. Un adjunto grande encarece además esas llamadas auxiliares.
- El camino conversacional **no usa ninguna caché** (ver la sesión 05): `cache_hit_kind` siempre es `none`.
- La compresión actúa tras cada turno: expulsa de la ventana lo que supera `MAX_CONVERSATION_TURNS`
  pares, lo promueve a ancla o lo resume. El metadata se actualiza con una llamada extra.

### Por qué se mide el baseline antes de introducir RAG

RAG añade piezas (chunking, embeddings, índice, recuperación, reranking) que solo compensan si el CAG
tiene un problema real que resolver. Sin una medida previa no se puede saber si RAG mejora algo, ni
cuánto, ni a qué precio. Este ejercicio fija el punto de comparación: con qué adjunto y en qué turno el
CAG incumple un presupuesto o pierde hechos. Ese es el número contra el que se juzgará cualquier
alternativa.

### Qué mide cada bloque

| Bloque | Qué mide | Cómo se interpreta |
|---|---|---|
| `turn_observed` | Tamaño del contexto (`enriched_transcript_chars`, `attachments_total_chars`, `messages_in_window`, `anchors_count`, `summary_chars`), consumo (`tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`), caché y tier | Es la *causa*: qué le pasó al contexto en cada turno |
| `LatencyBudgetMetric(budget_ms)` | `latency_ms <= budget_ms` | Contrato de latencia por turno |
| `CostBudgetMetric(budget_usd)` | `cost_usd <= budget_usd` | Contrato de coste por turno |
| `MemoryDriftMetric(fact, where=...)` | El hecho (subcadena literal, sin mayúsculas) aparece en `summary`, `anchors` o `metadata` | Es la *consecuencia*: qué sobrevivió |
| Fact-tracker de cada escenario | Qué hechos debería recordar la memoria tras el turno N | El recall de un turno = hechos encontrados / hechos esperados |

Los tres escenarios:
- `growing`: añade requisitos poco a poco. Mide el crecimiento del contexto y del coste acumulado, y si
  sobrevive el nombre del proyecto y las tecnologías declaradas a lo largo de la conversación.
- `pivot`: cambia de Django a NestJS en el turno 5. Responde si la metadata sustituye el stack anterior
  o acumula ambos (`stale_facts_present`).
- `contradiction`: presupuesto de `30000` EUR en el turno 2 (dicho como "locked", así que el detector de
  anclas lo promueve) contradicho por `80000` EUR en el turno 8. Permite ver qué persiste y dónde.

Los hechos son literales sin separador de miles (`30000`, no `30.000`): la búsqueda es de subcadena y no
entiende de formato. Si el LLM reescribe la cifra, el recall bajará aunque la memoria *conserve* el
dato; es una limitación asumida de las métricas deterministas.

### Cómo leer las tres curvas

1. **Latencia vs tokens de entrada.** Si la latencia sube con `tokens_in`, el contexto es el cuello de
   botella. Si no lo hace, el problema es otro (el proveedor, las llamadas auxiliares...).
2. **Coste acumulado vs índice del turno.** Una recta indica coste constante por turno; un cambio de
   pendiente señala el turno donde entra algo nuevo. Con `MAX_CONVERSATION_TURNS=6`, el candidato obvio es
   el turno 7: empieza la compresión.
3. **Recall vs número de turnos.** Es la curva que casi nunca se mira y la más importante: dice si la
   memoria sigue siendo fiable.

### Fallo explícito frente a degradación silenciosa del recall

Un fallo explícito (timeout, 502, 400 de guardrail, validación) se ve: hay un código de error, un log, una
alerta. Una degradación silenciosa del recall no se ve: el endpoint responde 200, el JSON es válido, el
coste está dentro del presupuesto... y el modelo ya no recuerda el nombre del proyecto o sigue usando un
presupuesto obsoleto. Nada falla, simplemente el sistema es peor. Por eso la memoria necesita su propia
métrica (`MemoryDriftMetric`): si no se mide, no se puede detectar. El truncado de adjuntos es el mismo
tipo de problema: `extract_text` recorta a `MAX_ATTACHMENT_CHARS` y solo deja un log informativo.

### Presupuestos de latencia y coste como contratos de diseño

Un presupuesto no es un objetivo aspiracional: es una promesa con consecuencias de diseño. "El turno
responde en menos de X segundos" o "ningún turno cuesta más de Y" decide cuántos mensajes caben en la
ventana, si compensa una llamada extra al extractor, o si se permite un adjunto de 100 KB. Sin presupuesto
cada mejora de calidad parece gratis. Los defaults del runner (10000 ms y 0.01 USD por turno) son
**ilustrativos**: hay que fijarlos según lo que prometa el producto (`--latency-budget-ms`,
`--cost-budget-usd`) y anotarlos en el informe.

## Decisiones de diseño y límites

- **La observación viaja en la respuesta**, no en el log: el runner no tiene que parsear la consola ni
  reconciliar logs. Es un campo nuevo y opcional (`turn_observation`) en una subclase de la respuesta
  (`ObservedEstimationResponse`), solo para `/estimate`; el ACB (`/estimate-acb`) no cambia. El mismo
  objeto se emite con structlog como `turn_observed`.
- **`turn_index`** sale de `Session.turn_count`, que solo avanza cuando un turno se compromete en el
  historial: un turno rechazado (400, 502) no consume índice.
- **`estimate_conversational()` devuelve ahora tres valores** (resultado, meta, observación). Solo lo
  llama el router.
- **Los campos de consumo son los de la llamada de estimación.** No suman el extractor de metadata, el
  resumen ni el clasificador de anclas LLM, así que el coste y la latencia reales de un turno son mayores.
  `wall_clock_ms` en el CSV es la petición HTTP completa (con adjunto, guardrails, estimación, metadata y
  compresión); no es comparable con `latency_ms`.
- **`MetricResult` ganó una propiedad `score`** (1.0 / 0.0, derivada de `passed`): los consumidores y
  tests existentes no cambian.
- **El adjunto se envía en cada turno**, no una vez. Es el peor caso y refleja cómo funciona hoy la app.
- **Una sesión con error se abandona.** El turno fallido no se confirma en el historial; seguir
  falsearía las métricas de memoria. La fila con el error queda en el CSV y el runner sale con código 1.
- **Los fixtures no llevan hechos verificables**: se mide el *tamaño* y el *truncado* del adjunto, no cuánto
  contenido útil del documento se pierde.
- **Sin RAG, sin optimizar, sin gráficos, sin base de datos.** Solo se mide.

## Cómo invocar

Todos los comandos se ejecutan desde la raíz del repositorio (en PowerShell, para curl hay que usar
`curl.exe`).

### 1. Dependencias

```
uv sync --extra dev
```

`fpdf2` está en el extra `dev`; es lo único que necesita el generador de PDFs.

### 2. Generar los PDFs

```
uv run python -m evals.stress.fixtures.build_pdfs
```

Escribe `evals/stress/fixtures/generated/attachment_{5,20,50,100}kb.pdf` (ignorado por git) y muestra, con
el extractor de la app, cuántos caracteres salen de cada uno y cuántos conserva la app. Con
`MAX_ATTACHMENT_CHARS=60000`, el de 100 KB extrae 100000 caracteres y la app se queda con 60000: se trunca
sin error. El caso de 0 KB no necesita fichero. 1 KB equivale aquí a 1000 caracteres extraídos, no a bytes
del PDF.

### 3. Arrancar la aplicación

Redis solo es necesario para las cachés de `/api/v1/*`; el camino conversacional no las usa, pero el
servidor intenta conectarse al arrancar el índice semántico y degrada sin romperse si no está.

```
docker compose up -d
uv run uvicorn app.main:app --reload
```

Hace falta un `.env` con `OPENAI_API_KEY` (o `ANTHROPIC_API_KEY` con `LLM_PROVIDER=anthropic`) y
`CONVERSATIONAL_PROMPT_VERSION=v4`. `--reload` no vigila `.env`: reinicia uvicorn si lo cambias. Ejecuta el
stress test con el **mismo modelo y los mismos límites** que quieras medir y anótalos en el informe.

### 4. Comprobar que está disponible

```
curl.exe http://localhost:8000/health
```

### 5. Ejecutar el runner

> **Aviso de coste:** llama al LLM real en cada turno, y cada turno con adjunto envía hasta 60000
> caracteres de texto. La matriz completa son 3 escenarios × 5 tamaños × 3 repeticiones × 22 turnos
> (990 turnos). Empieza con una ejecución pequeña.

Prueba barata (un escenario, sin adjunto y uno de 5 KB, 5 turnos):

```
uv run python -m evals.stress.run --http http://localhost:8000 --scenarios growing --attachment-sizes 0,5 --repeats 1 --max-turns 5 --output evals/stress/results.csv
```

Ejecución completa:

```
uv run python -m evals.stress.run --http http://localhost:8000 --scenarios growing,pivot,contradiction --attachment-sizes 0,5,20,50,100 --repeats 3 --output evals/stress/results.csv
```

Sin `--http` el runner arranca la app en proceso con `TestClient` (mismo patrón que `evals.run`): es lo que
usan los tests con un LLM falso. Antes de empezar valida los argumentos y, con `--http`, comprueba
`/health`; si faltan fixtures indica el comando para generarlas. El CSV se escribe fila a fila: una
ejecución interrumpida conserva lo medido.

Otros parámetros: `--max-turns N`, `--latency-budget-ms`, `--cost-budget-usd`.

### 6. Tests

```
uv run pytest -v
```

Los tests del stress test no llaman a ningún LLM: usan el LLM falso de `tests/sessions_helpers.py`.

## Cómo leer el CSV y completar el informe

Una fila por turno, con estas columnas (`evals/stress/run.py`, `CSV_COLUMNS`):

| Grupo | Columnas |
|---|---|
| Parámetros | `scenario`, `attachment_kb`, `repeat`, `scenario_turn` |
| `turn_observed` | `turn_index`, `session_id`, `enriched_transcript_chars`, `attachments_total_chars`, `messages_in_window`, `anchors_count`, `summary_chars`, `tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`, `cache_hit_kind`, `last_resolved_tier` |
| Tiempo de pared | `wall_clock_ms` (petición HTTP completa; distinto de `latency_ms`) |
| Métricas | `latency_budget_score`, `cost_budget_score` (1.0 pasa / 0.0 incumple), `memory_drift_score` (1.0 si **todos** los hechos esperados siguen; vacío si no había ninguno), `memory_recall`, `memory_facts_tracked`, `memory_facts_retained`, `memory_facts_missing`, `stale_facts_present` |
| Error | `error` (no vacío: el turno falló y la sesión se abandonó; las columnas de medida quedan vacías) |

Para completar `evals/stress/REPORT.md`:
1. Descartar las filas con `error` no vacío al calcular percentiles y medias, y contarlas aparte
   (sección 1 del informe).
2. Tablas por escenario y tamaño: agrupar por `scenario` y `attachment_kb`. P50/P95 sobre `latency_ms`;
   coste acumulado = suma de `cost_usd` por sesión (`session_id`), promediada entre `repeat`; hit rates sobre
   `cache_hit_kind`; recall medio sobre `memory_recall`.
3. Curvas: latencia por tramos de `tokens_in`; coste acumulado por `turn_index`; recall por `turn_index`.
4. Anotar el modelo, los presupuestos, `MAX_CONVERSATION_TURNS` y `MAX_ATTACHMENT_CHARS` usados.
5. Redactar la lectura y la conclusión **después** de ver los números; el informe ya recoge los criterios
   fijados de antemano.

### El límite de extracción de adjuntos

`MAX_ATTACHMENT_CHARS` (60000) trunca, no rechaza. En los datos se ve como `attachments_total_chars ==
60000` para los adjuntos de 100 KB (y como `attachment_truncated` en el log del servidor). El tamaño de
contexto del caso de 100 KB es por tanto **igual al de ~60 KB**: no es una escala nueva, es el tope. Si
las curvas de 50 y 100 KB coinciden, es por esto. Esa pérdida de contenido no aparece en ninguna métrica
de recall (los fixtures no llevan hechos).

## Limitaciones

- Los resultados dependen del modelo, de los precios de `MODEL_COSTS` (si el modelo no está en la tabla,
  `cost_usd` es 0.0 y la métrica de coste pasa siempre) y del estado del proveedor en ese momento.
- `tokens_in`, `tokens_out`, `cost_usd` y `latency_ms` no incluyen las llamadas auxiliares del turno.
- El recall es literal y sensible al formato; el fact-tracker no distingue entre un hecho "recordado" y
  uno recordado *incorrectamente* (por eso `stale_facts_present` se registra aparte).
- Los hechos de `growing` son nombres de tecnologías: el extractor de metadata suele conservarlas por
  construcción (unión sin duplicados), así que `growing` mide sobre todo el nombre del proyecto y el
  crecimiento del contexto. `pivot` y `contradiction` son los que exponen la información obsoleta.
- Con `ANCHOR_DETECTION_MODE=heuristic`, el único turno que se promueve a ancla es el presupuesto cerrado de
  `contradiction` (comprobado en los tests de escenarios). Con `llm` el comportamiento puede cambiar.
- `SessionStore` es un dict en memoria: el servidor debe correr con un solo worker.
- Un adjunto grande puede ser rechazado por la Moderation API o por el proveedor; el runner lo registrará
  como error de esa sesión, no lo ocultará.
