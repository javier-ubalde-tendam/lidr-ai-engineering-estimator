# Informe del stress test del CAG (ejercicio previo a la sesión 6)

> **Estado: pendiente de ejecución.** Este informe está preparado, pero **no contiene resultados
> reales**: el stress test no se ha ejecutado contra un LLM real (requiere servidor levantado,
> credenciales y coste). Las celdas con `—` y los apartados marcados como *pendiente* se rellenan a
> partir de `evals/stress/results.csv` tras la ejecución. Cómo reproducirlo:
> [docs/sessions/06-stress-cag.md](../../docs/sessions/06-stress-cag.md).

## 1. Parámetros de ejecución

Rellenar tras la ejecución (los valores de "Configuración" salen del código actual; el resto, de la ejecución).

| Parámetro | Valor |
|---|---|
| Fecha de la ejecución | — |
| Commit / rama evaluada | — |
| Modo (`--http` o in-process) | — |
| Comando exacto | — |
| Escenarios | `growing`, `pivot`, `contradiction` (22 turnos cada uno) |
| Tamaños de adjunto (KB de texto extraído) | 0 (sin adjunto), 5, 20, 50, 100 |
| Repeticiones | — |
| Presupuesto de latencia (`--latency-budget-ms`) | — (default del runner: 10000) |
| Presupuesto de coste por turno (`--cost-budget-usd`) | — (default del runner: 0.01) |
| Modelo de estimación (`OPENAI_MODEL` / `ANTHROPIC_MODEL`) | — |
| `MAX_CONVERSATION_TURNS` | 6 (default; verificar el `.env` de la ejecución) |
| `MAX_ATTACHMENT_CHARS` | 60000 (default; verificar el `.env` de la ejecución) |
| `ANCHOR_DETECTION_MODE` | `heuristic` (default; verificar) |
| Filas válidas / filas con error | — / — |

Notas de lectura del CSV que condicionan todas las tablas:

- El adjunto se envía en **cada turno** de la sesión; el de 100 KB se trunca a 60000 caracteres
  (`attachment_truncated` en el log del servidor y `attachments_total_chars == 60000` en el CSV).
- `latency_ms`, `tokens_in`, `tokens_out` y `cost_usd` son los de **la llamada de estimación**. No
  incluyen el extractor de metadata ni el resumen acumulativo, que también se ejecutan en cada turno.
  `wall_clock_ms` es la petición HTTP completa.
- `cache_hit_kind` vale siempre `none`: el camino conversacional no usa ninguna caché. Las columnas de
  hit rate son una comprobación de que sigue siendo así, no una medida de rendimiento.
- Ignorar las filas con `error` no vacío al calcular percentiles y medias; contarlas aparte.

## 2. Resultados por escenario y tamaño de adjunto

Definiciones: **P50/P95** de `latency_ms` sobre todas las filas válidas del grupo (percentil por rango
más cercano); **coste acumulado** = suma de `cost_usd` de los turnos de una sesión, promediada entre
repeticiones; **hit rate** = proporción de filas con `cache_hit_kind` igual a `exact` / `semantic`;
**recall medio** = media de `memory_recall` (filas con `memory_facts_tracked > 0`).

| Escenario | Adjunto (KB) | P50 latencia (ms) | P95 latencia (ms) | Coste acumulado (USD) | Hit rate exacto | Hit rate semántico | Recall medio |
|---|---|---|---|---|---|---|---|
| growing | 0 | — | — | — | — | — | — |
| growing | 5 | — | — | — | — | — | — |
| growing | 20 | — | — | — | — | — | — |
| growing | 50 | — | — | — | — | — | — |
| growing | 100 | — | — | — | — | — | — |
| pivot | 0 | — | — | — | — | — | — |
| pivot | 5 | — | — | — | — | — | — |
| pivot | 20 | — | — | — | — | — | — |
| pivot | 50 | — | — | — | — | — | — |
| pivot | 100 | — | — | — | — | — | — |
| contradiction | 0 | — | — | — | — | — | — |
| contradiction | 5 | — | — | — | — | — | — |
| contradiction | 20 | — | — | — | — | — | — |
| contradiction | 50 | — | — | — | — | — | — |
| contradiction | 100 | — | — | — | — | — | — |

Complemento útil: % de filas con `latency_budget_score == 0.0` y con `cost_budget_score == 0.0` por
grupo (cuántos turnos incumplen cada contrato de diseño).

## 3. Curvas

### 3.1 Latencia frente a tokens de entrada

Agrupar todas las filas válidas por tramo de `tokens_in`. Cada fila de la tabla es un tramo; la pendiente
de la P50 indica cuánto cuesta en latencia cada token extra de contexto.

| Tramo de `tokens_in` | Nº de turnos | P50 latencia (ms) | P95 latencia (ms) |
|---|---|---|---|
| < 2000 | — | — | — |
| 2000 – 4999 | — | — | — |
| 5000 – 9999 | — | — | — |
| 10000 – 19999 | — | — | — |
| ≥ 20000 | — | — | — |

### 3.2 Coste acumulado frente al índice del turno

Media entre escenarios y repeticiones del coste acumulado de la sesión hasta ese turno, por tamaño de adjunto.

| `turn_index` | 0 KB | 5 KB | 20 KB | 50 KB | 100 KB |
|---|---|---|---|---|---|
| 1 | — | — | — | — | — |
| 5 | — | — | — | — | — |
| 6 | — | — | — | — | — |
| 7 | — | — | — | — | — |
| 10 | — | — | — | — | — |
| 15 | — | — | — | — | — |
| 20 | — | — | — | — | — |
| 22 | — | — | — | — | — |

Los turnos 6 y 7 se incluyen porque con `MAX_CONVERSATION_TURNS=6` la ventana se llena en el 6 y la
compresión (resumen y anclas) empieza a actuar en el 7: ahí se espera el primer cambio de pendiente.

### 3.3 Recall frente al número de turnos

Media de `memory_recall` por escenario (entre tamaños y repeticiones). Un recall de 1.0 significa que
todos los hechos declarados hasta ese turno siguen en resumen, anclas o metadata.

| `turn_index` | growing | pivot | contradiction |
|---|---|---|---|
| 1 | — | — | — |
| 5 | — | — | — |
| 6 | — | — | — |
| 7 | — | — | — |
| 10 | — | — | — |
| 15 | — | — | — |
| 20 | — | — | — |
| 22 | — | — | — |

Además, para `pivot` y `contradiction`, anotar en qué proporción de filas posteriores al cambio
`stale_facts_present` no está vacío (`Django`, `30000`): es información obsoleta que la memoria sigue
sirviendo.

## 4. Lectura de los resultados

> **Pendiente.** Los dos párrafos siguientes se redactan con los números de las secciones 2 y 3.
> Hasta entonces solo se listan las preguntas a las que deben responder y las hipótesis que salen del
> diseño del código (no son resultados).

**Dónde y por qué empieza a degradarse el CAG.** *(pendiente)* Responder con el `turn_index` y el tamaño
de adjunto concretos donde aparece el primer cambio de pendiente o el primer incumplimiento de
presupuesto. Hipótesis a contrastar, derivadas del código: (a) el contexto crece con el adjunto, que se
reenvía íntegro en cada turno, hasta el tope de 60000 caracteres; (b) a partir del turno 7 entra la
compresión, y con ella las llamadas auxiliares del resumen; (c) las anclas no se expulsan nunca, así que
en `contradiction` el contexto no vuelve a su tamaño inicial.

**Qué dimensión domina: latencia, coste o pérdida de memoria.** *(pendiente)* Comparar, para cada
tamaño de adjunto, qué contrato se rompe primero (`latency_budget_score`, `cost_budget_score` o un recall
por debajo del umbral que se decida) y cuál lo hace con más margen. Tener presente que la pérdida de
memoria no produce ningún error: solo aparece en `memory_recall` y `stale_facts_present`.

## 5. Conclusión: qué observación justificaría comparar con RAG

*(pendiente de datos; criterios decididos de antemano para no ajustarlos a los resultados)*

Comparar con RAG está justificado si, en la ejecución real, se cumple al menos una de estas condiciones:

1. La latencia o el coste por turno incumplen su presupuesto **solo con adjuntos grandes** y el
   crecimiento sigue a `tokens_in` (el contexto es el problema, no la conversación).
2. Con adjuntos de 100 KB el CAG descarta contenido sin avisar (truncado a 60000 caracteres): lo que
   queda fuera del límite no se puede preguntar. Este stress test mide el truncado, pero **no** mide
   cuánto contenido del adjunto se pierde (los fixtures no llevan hechos verificables).
3. El recall cae bajo el umbral elegido a partir del turno en que entra la compresión, sin que ningún
   presupuesto de latencia o coste se haya roto antes.

Si, por el contrario, los presupuestos se cumplen y el recall se mantiene, la evidencia no justifica
todavía añadir un retriever: el coste de otra pieza (índice, chunking, reranking) no estaría compensado.
