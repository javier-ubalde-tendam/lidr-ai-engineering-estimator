# Stress test del CAG — sesión 6

## 1. Parámetros y cobertura

| Parámetro | Resultado |
|---|---|
| Escenarios | `growing`, `pivot`, `contradiction` |
| Turnos por sesión (por cada prueba) | 22 |
| Texto extraído del adjunto | 0, 5.000, 20.000, 50.000 y 60.000 caracteres (objetivo: 0, 5, 20, 50 y 100 KB) |
| Repeticiones | 3 |
| Filas de turnos | 990 observaciones sin errores registrados: 3 escenarios × 5 tamaños × 3 repeticiones × 22 turnos |
| Presupuesto de latencia | 10.000 ms, valor por defecto documentado del runner |
| Presupuesto de coste por turno | 0,01 USD, valor por defecto documentado del runner |
| Modelo, proveedor, fecha | gpt-4o-mini, openai, 2026-10-11 |

El adjunto de 100 KB se limita a 60.000 caracteres por `MAX_ATTACHMENT_CHARS`. Por ello, los casos de 50 y 100 KB no representan el doble de contenido procesado: el segundo alcanza el límite de extracción.

La latencia, los tokens y el coste corresponden a la llamada de estimación según la instrumentación del runner. `wall_clock_ms` mide el tiempo total de la petición HTTP. Las cachés exacta y semántica no registraron hits: `cache_hit_kind` fue `none` en los 990 turnos.

Para reproducir la ejecución con la configuración documentada:

```bash
uv run python -m evals.stress.fixtures.build_pdfs
docker compose up --build -d
curl -sf http://localhost:8000/health
uv run python -m evals.stress.run \
  --http http://localhost:8000 \
  --scenarios growing,pivot,contradiction \
  --attachment-sizes 0,5,20,50,100 \
  --repeats 3 \
  --latency-budget-ms 10000 \
  --cost-budget-usd 0.01 \
  --output evals/stress/results.csv
```

El CSV no conserva el modelo, proveedor, commit ni los argumentos exactos usados. Conviene anotarlos junto con futuras ejecuciones para que los resultados sean reproducibles.

## 2. Resumen por escenario y tamaño de adjunto

Recordatorio de los tres escenarios:
- `growing`: añade requisitos poco a poco. Mide el crecimiento del contexto y del coste acumulado y si
  sobrevive el nombre del proyecto y las tecnologías declaradas a lo largo de la conversación.
- `pivot`: cambia de Django a NestJS en el turno 5. Responde si la metadata sustituye el stack anterior
  o acumula ambos (`stale_facts_present`).
- `contradiction`: presupuesto de `30000` EUR en el turno 2 (dicho como "locked", así que el detector de
  anclas lo promueve) contradicho por `80000` EUR en el turno 8. Permite ver qué persiste y dónde.

P50 y P95 se calculan por rango más cercano sobre las filas de cada grupo. El coste acumulado es la suma de los 22 turnos, promediada entre las tres repeticiones. El recall medio es la media de `memory_recall` en las filas con hechos bajo seguimiento.

El recall medio es la proporción media de hechos conocidos que la memoria de la sesión sigue conservando después de cada turno. Mide qué fracción de lo que el usuario ya dijo sigue disponible para el modelo fuera de la ventana reciente.

Nota: el camino conversacional en el test de stree nunca consulta las cachés. Por eso los hit son siempre None.

| Escenario | Adjunto (KB) | P50 latencia (ms) | P95 latencia (ms) | Coste acumulado medio (USD) | Hit exacto | Hit semántico | Recall medio |
|---|---:|---:|---:|---:|---:|---:|---:|
| growing | 0 | 2.231 | 3.452 | 0,017165 | 0% | 0% | 88,6% |
| growing | 5 | 2.713 | 3.812 | 0,020252 | 0% | 0% | 88,6% |
| growing | 20 | 2.681 | 4.365 | 0,030400 | 0% | 0% | 88,6% |
| growing | 50 | 2.894 | 6.268 | 0,053224 | 0% | 0% | 88,6% |
| growing | 100 | 2.740 | 4.518 | 0,056810 | 0% | 0% | 88,6% |
| pivot | 0 | 2.281 | 3.341 | 0,016385 | 0% | 0% | 100% |
| pivot | 5 | 2.607 | 3.119 | 0,020605 | 0% | 0% | 100% |
| pivot | 20 | 2.582 | 4.787 | 0,030799 | 0% | 0% | 100% |
| pivot | 50 | 2.830 | 6.073 | 0,058881 | 0% | 0% | 100% |
| pivot | 100 | 2.761 | 4.771 | 0,055205 | 0% | 0% | 100% |
| contradiction | 0 | 2.146 | 3.086 | 0,015587 | 0% | 0% | 59,1% |
| contradiction | 5 | 2.640 | 3.751 | 0,020062 | 0% | 0% | 54,5% |
| contradiction | 20 | 2.712 | 3.986 | 0,029413 | 0% | 0% | 52,3% |
| contradiction | 50 | 2.753 | 4.809 | 0,048542 | 0% | 0% | 62,9% |
| contradiction | 100 | 2.861 | 4.487 | 0,054814 | 0% | 0% | 73,5% |

## 3. Curvas

### 3.1 Latencia frente a tokens de entrada

| Tramo de `tokens_in` | Turnos | P50 latencia (ms) | P95 latencia (ms) |
|---|---:|---:|---:|
| < 2.000 | 0 | — | — |
| 2.000–4.999 | 246 | 2.286 | 3.297 |
| 5.000–9.999 | 341 | 2.672 | 3.884 |
| 10.000–19.999 | 377 | 2.796 | 4.743 |
| ≥ 20.000 | 26 | 5.810 | 7.304 |

### 3.2 Coste acumulado frente al turno (sin adjunto)

Coste acumulado medio en USD entre las tres repeticiones. Usar 0 KB permite aislar el crecimiento de la conversación.

| Turno | `growing` | `pivot` | `contradiction` |
|---:|---:|---:|---:|
| 1 | 0,000508 | 0,000509 | 0,000506 |
| 5 | 0,002935 | 0,002948 | 0,002684 |
| 6 | 0,003636 | 0,003655 | 0,003311 |
| 7 | 0,004374 | 0,004403 | 0,003967 |
| 10 | 0,006689 | 0,006711 | 0,006160 |
| 15 | 0,010626 | 0,010561 | 0,010057 |
| 20 | 0,014715 | 0,014474 | 0,013971 |
| 22 | 0,017165 | 0,016385 | 0,015587 |

### 3.3 Recall de memoria frente al número de turnos

Media por escenario entre los cinco tamaños y las tres repeticiones. Se omite el turno 1 porque todavía no mide la conservación de hechos de turnos anteriores. Cada punto posterior contiene 15 observaciones.

| Turno N | `growing` | `pivot` | `contradiction` |
|---:|---:|---:|---:|
| 2 | 100% | 100% | 73,3% |
| 3 | 50% | 100% | 73,3% |
| 5 | 50% | 100% | 66,7% |
| 6 | 66,7% | 100% | 66,7% |
| 7 | 66,7% | 100% | 63,3% |
| 8 | 66,7% | 100% | 60% |
| 9 | 100% | 100% | 50% |
| 10 | 100% | 100% | 50% |
| 15 | 100% | 100% | 56,7% |
| 20 | 100% | 100% | 56,7% |
| 22 | 100% | 100% | 56,7% |

`memory_recall` es la proporción de hechos seguidos que aparecen en el snapshot. Puede bajar cuando se añaden hechos nuevos al seguimiento y luego recuperarse; por tanto, no siempre representa que el sistema haya olvidado un hecho concreto. Las columnas `memory_facts_missing` y `stale_facts_present` ayudan a identificar qué se perdió y qué dato obsoleto continúa presente.

## 4. Lectura: dónde empieza a degradarse el CAG

El efecto más consistente del tamaño del adjunto aparece en el coste. El coste acumulado medio de una sesión pasa de **0,0156–0,0172 USD sin adjunto** a **0,0485–0,0589 USD con 50 KB**, aproximadamente entre 3 y 3,5 veces más. El límite de extracción frena el crecimiento entre los casos de 50 y 100 KB. La latencia aumenta con los tokens: para más de 20.000 tokens, la P50 es 5.810 ms y la P95 7.304 ms. Con el presupuesto configurado de 10 segundos, 3 de 990 turnos lo superan (máximo: 21.511 ms); los 990 turnos cumplen el presupuesto de coste de 0,01 USD. La latencia no crece de forma estrictamente monotónica con los KB, lo que aconseja interpretar los percentiles con cautela: solo hay tres repeticiones por combinación.

La degradación más preocupante es la memoria silenciosa ante cambios. En `growing`, `PostgreSQL` falta entre los turnos 3 y 8, aunque el recall vuelve al 100% desde el turno 9 y el nombre del proyecto se conserva. En `pivot`, el recall de hechos esperados permanece en 100%, pero `Django` sigue apareciendo como dato obsoleto en las **255 de 255 observaciones posteriores al pivote**. En `contradiction`, el presupuesto antiguo de 30.000 EUR persiste en las **210 de 210 observaciones posteriores al cambio**, mientras que el nuevo valor de 80.000 EUR falta en **192 de 210 (91,4%)**; el recall medio de ese escenario queda en 59,1% sin adjunto y entre 52,3% y 73,5% según el tamaño. Esto muestra que una puntuación de recall alta puede coexistir con información obsoleta, por lo que conviene leerla junto a `stale_facts_present`.

## 5. Qué justificaría comparar con RAG

Los resultados justifican comparar el CAG con una estrategia de recuperación, especialmente para resolver hechos contradictorios o modificados: el valor antiguo permanece y el nuevo se omite con frecuencia, sin que la llamada falle. Los adjuntos también elevan el coste y el tamaño del contexto; el caso de 100 KB se trunca a 60.000 caracteres. Sin embargo, este test no incluye hechos verificables dentro de los PDFs, así que mide el coste y el truncado, pero no el recall de su contenido. Una comparación con RAG debería medir específicamente si recupera el presupuesto vigente y los datos relevantes de adjuntos grandes con menos contexto.

## 6. Validación y límites

- El CSV contiene 990 filas de turnos y ninguna fila con error.
- `latency_budget_score`: 987 aprobados y 3 fallidos.
- `cost_budget_score`: 990 aprobados.
- `memory_drift_score`: 639 aprobados y 351 fallidos.
- `cache_hit_kind` fue `none` en todos los turnos. El camino conversacional en estos tests nunca consulta las cachés. 
- El CSV contiene los campos de observación por turno, pero no demuestra por sí solo que cada petición haya emitido el evento `turn_observed` en los logs.
- El CSV tampoco acredita el resultado de los tests unitarios ni el estado del repositorio; esas comprobaciones deben verificarse aparte.
- Modelo, proveedor y fecha: gpt-4o-mini, openai, 2026-10-11.
