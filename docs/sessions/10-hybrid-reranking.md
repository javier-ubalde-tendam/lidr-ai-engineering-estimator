# Session 10 - Búsqueda híbrida + reranking (ejercicio previo)

Alcance: búsqueda full-text en PostgreSQL, fusión RRF con la rama vectorial, reranking con un
cross-encoder (recall-then-rerank) y medición de las cuatro configuraciones A/B/C/D sobre un golden
set de 5 consultas. **Fuera de alcance** (pasos posteriores): expansión/reformulación de consultas,
routing multi-índice, filtros por metadatos de cara al cliente, generación de estimaciones con citas,
autenticación, rate limiting, idempotencia, `X-Request-ID`, Streamlit. `/search` (sesión 08) no cambia.

## Piezas

| Pieza | Fichero |
| --- | --- |
| Migración `content_tsv` (generada) + índice GIN | [alembic/versions/0002_chunks_content_tsv_fts.py](../../alembic/versions/0002_chunks_content_tsv_fts.py) |
| Modelo ORM (`Computed`, índice GIN, `TEXT_SEARCH_CONFIG`) | [app/db/models.py](../../app/db/models.py) |
| Store de solo lectura (k-NN con umbral + rama léxica) | [app/generation/rag/store.py](../../app/generation/rag/store.py) |
| RRF (función pura) | [app/generation/rag/retrieval/fusion.py](../../app/generation/rag/retrieval/fusion.py) |
| Cross-encoder (carga perezosa) | [app/generation/rag/retrieval/reranker.py](../../app/generation/rag/retrieval/reranker.py) |
| Puerta de entrada del reranker | [app/generation/rag/retrieval/verify_reranker.py](../../app/generation/rag/retrieval/verify_reranker.py) |
| `retrieve()` (las 4 configuraciones) | [app/generation/rag/retrieval/pipeline.py](../../app/generation/rag/retrieval/pipeline.py) |
| Endpoint `POST /v1/retrieval/search` | [app/routers/retrieval.py](../../app/routers/retrieval.py) |
| Golden set + revisión humana | [evals/golden_retrieval.json](../../evals/golden_retrieval.json), [evals/GOLDEN_SET_REVIEW.md](../../evals/GOLDEN_SET_REVIEW.md) |
| Métrica Precision@5 y percentiles | [evals/retrieval_metrics.py](../../evals/retrieval_metrics.py) |
| Script de medición | [scripts/eval_retrieval_s10.py](../../scripts/eval_retrieval_s10.py) |
| Resultados (generados, no editar) | [evals/retrieval_s10_results.md](../../evals/retrieval_s10_results.md), [.json](../../evals/retrieval_s10_results.json) |

## Decisiones de diseño

- **Configuración FTS `english`.** El enunciado dice que el dataset está en español, pero
  `data/budgets_sample.json` está en **inglés** («Faceted search», «Order lifecycle»…). Con
  `english` el stemming (`promotions` → `promot`) y las stop-words mejoran el recall léxico; con
  `spanish` las palabras inglesas no se normalizarían. El nombre es idéntico en la migración, en
  `TEXT_SEARCH_CONFIG` (modelo y consulta) y un test unitario lo comprueba: si no coincidiera,
  Postgres no usaría el índice GIN y el matching cambiaría sin avisar.
- **Columna generada `STORED`.** Postgres recalcula `content_tsv` en cada insert/update: no hay
  trigger que mantener y la migración rellenó sola los 60 chunks ya ingeridos (verificado:
  `count(content_tsv) = 60`). En el ORM es de solo lectura (`Computed`) y `deferred`.
- **Semántica OR en la rama léxica (`LEXICAL_MATCH_MODE=or`).** `plainto_tsquery` une los lexemas
  con AND. Las consultas reales son frases largas: exigir todos los lexemas en un mismo chunk devuelve
  casi siempre 0 filas y la «búsqueda híbrida» degenera en vectorial sin que se note. Medido en la BD
  de desarrollo con *«Online shop with shopping cart, checkout and card payments»*: **AND = 0 filas,
  OR = 10**. Se reescribe `&` → `|` sobre el `tsquery` (`replace(plainto_tsquery(...)::text, ' & ',
  ' | ')::tsquery`); `ts_rank_cd` sigue premiando a los chunks que cubren más términos. Una consulta
  solo de stop-words da un `tsquery` vacío y devuelve `[]`. La solución de referencia del profesor usa
  AND; `LEXICAL_MATCH_MODE=and` permite compararlo. Contrapartida observada: con OR la rama léxica
  devuelve de media **26,6 de 60 chunks** por consulta (poco selectiva en un corpus pequeño y homogéneo).
- **RRF con `k = 60`.** Se fusiona por **posición**, no por score: distancia coseno (menor = mejor) y
  `ts_rank_cd` (mayor = mejor, sin cota) no son comparables. `k` grande ⇒ hay que rankear bien en las
  dos ramas; `k` pequeño ⇒ un #1 en una sola rama domina. Empates ⇒ id ascendente (determinista).
- **Recall 50 → top 5.** Si va a haber fusión o reranking se recupera ancho (`RETRIEVAL_RECALL_TOP_K`);
  si no, solo `top_k`. El pool fusionado se corta a `recall_k` antes del cross-encoder, que deja
  `RERANK_TOP_N`. Sus scores crudos no se usan como umbral. Se ejecuta con `asyncio.to_thread`.
- **Distancia centinela `1.0`.** Un chunk que solo encuentra la rama léxica no tiene distancia coseno;
  se le asigna 1.0 («ortogonal») solo para mostrar. Ni RRF ni el reranker la usan. Si el chunk está en
  las dos ramas, gana su distancia real.
- **Precedencia de los conmutadores:** campo de la petición (`search_mode`, `rerank`) → variables
  `RETRIEVAL_SEARCH_MODE` / `RERANKER_ENABLED` → defaults (`vector`, `false`).
- **Soft-fail vs error.** Sin resultados ⇒ `200` con `low_confidence=true`. Fallo de BD o del
  reranker ⇒ `RetrievalError` ⇒ `502` con mensaje genérico (al log solo tipo y mensaje truncado).
- **Infra del modelo.** `torch` CPU desde el índice de PyTorch solo en Linux (`[tool.uv.sources]`);
  en Windows los wheels de PyPI ya son CPU. Caché de Hugging Face en el volumen `lidr-hf-cache`
  (`HF_HOME=/home/app/.cache/huggingface`, propiedad de `app`): la carga del modelo bajó de 22 s
  (descarga) a ~3 s al recrear el contenedor. `uv sync --no-cache` en el `Dockerfile`: la imagen pasó
  de 4,99 GB a 2,89 GB.

## Cómo invocar A/B/C/D

| Config | `search_mode` | `rerank` |
| --- | --- | --- |
| A (línea base) | `vector` | `false` |
| B | `hybrid` | `false` |
| C | `vector` | `true` |
| D | `hybrid` | `true` |

1. **Body de la petición** (prevalece sobre el entorno). PowerShell:

   ```powershell
   $body = @{ query_text = "Online shop with shopping cart, card checkout and order email"; top_k = 5; search_mode = "hybrid"; rerank = $true } | ConvertTo-Json
   Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/retrieval/search -ContentType "application/json" -Body $body
   ```

   curl:

   ```bash
   curl -s -X POST http://localhost:8000/v1/retrieval/search -H "Content-Type: application/json" \
     -d '{"query_text": "Online shop with shopping cart, card checkout and order email", "top_k": 5, "search_mode": "hybrid", "rerank": true}'
   ```

2. **Variables de entorno** (si el body no trae `search_mode`/`rerank`): en `.env`
   `RETRIEVAL_SEARCH_MODE=hybrid` y `RERANKER_ENABLED=true`, y después
   `docker compose up -d --force-recreate ai_service` (`Settings` se lee al arrancar el contenedor).
3. **Script de medición:** `docker compose exec ai_service python scripts/eval_retrieval_s10.py --config D`
   (o `--config all`, el valor por defecto).

La respuesta incluye `search_mode`, `rerank`, `search_time_ms` (sin el embedding), `vector_hits`,
`lexical_hits` y `low_confidence`. La primera petición con reranking carga el modelo (~3 s).

Puerta de entrada del reranker (exit 0 = listo, 1 = no carga, 2 = ranking incorrecto):

```powershell
docker compose exec ai_service python -m app.generation.rag.retrieval.verify_reranker
```

## Resultados (ejecución real, 2026-11-08)

`docker compose exec ai_service python scripts/eval_retrieval_s10.py` (5 consultas, 1 calentamiento
+ 5 ejecuciones medidas por consulta y configuración; CPU x86_64, 12 vCPU de Docker Desktop; 17
documentos / 60 chunks; `recall_k=50`, `rrf_k=60`, `LEXICAL_MATCH_MODE=or`, umbral 2.0). Copiado tal
cual de [evals/retrieval_s10_results.md](../../evals/retrieval_s10_results.md):

| Config | Búsqueda | Reranking | Precision@5 | Latencia media (ms) | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| A | Vectorial | No | 0.88 | 6.8 | 5.2 | 20.1 |
| B | Híbrida | No | 0.80 | 9.3 | 8.6 | 13.2 |
| C | Vectorial | Sí | 0.84 | 5514.0 | 6170.5 | 6541.7 |
| D | Híbrida | Sí | 0.84 | 6216.1 | 6171.7 | 6695.0 |

Embedding de la consulta (común a las cuatro, no incluido): 274,8 ms de media.

| Consulta | A | B | C | D |
| --- | --- | --- | --- | --- |
| q1-ecommerce-checkout-trap | 1.00 | 0.80 | 0.80 | 0.80 |
| q2-finance-lending | 0.80 | 0.80 | 0.80 | 0.80 |
| q3-healthcare-telemedicine | 0.80 | 0.80 | 0.80 | 0.80 |
| q4-industrial-iot-maintenance | 0.80 | 0.60 | 1.00 | 1.00 |
| q5-logistics-warehouse | 1.00 | 1.00 | 0.80 | 0.80 |

Lectura por consulta (top-5 en el `.json`):

- q2 y q3 están en su **techo de 0,80** en las cuatro configuraciones (un único presupuesto relevante
  de 4 chunks): no discriminan.
- La trampa de q1 (BUD-2024-003, pasarela de pagos) **no entró en el top-5 de ninguna configuración**.
  Lo que sí coló: BUD-2024-006 (marketplace) por la rama léxica en B, y BUD-2024-008 (devoluciones de
  *moda*, por «clothing») vía el reranker en C/D.
- q4: el reranker sube BUD-2024-015 (relevante) y llega a 1,00; la rama léxica de B mete un segundo
  chunk de BUD-2024-010 (wearables) y baja a 0,60.
- q5: el reranker mete en el top-5 el chunk de BUD-2024-016 de 1.768 tokens (migración de core
  bancario), que el modelo trunca a 512.
- C y D devuelven **el mismo top-5** en las 5 consultas: con 60 chunks y recall 50, ambos pools son
  casi todo el corpus y decide el cross-encoder.
- Latencia del reranking: ~2,1 s para 50 pares sin BUD-2024-016 frente a ~6,2 s con él (diagnóstico
  aparte, mismo contenedor). Ese único chunk de 1.768 tokens obliga a rellenar el lote a 512 tokens y
  triplica el coste.

## Conclusiones

En este proyecto usaría la **configuración A (vectorial, sin reranking)** como valor por defecto y
mantendría los conmutadores para volver a medir cuando cambie el corpus. Con los números obtenidos,
A tiene la mejor Precision@5 (0,88) con 6,8 ms de media. La híbrida B baja a 0,80 sin ganar nada a
cambio. Con OR su rama léxica casa ~27 de 60 chunks por consulta y, sobre un corpus pequeño y
homogéneo, aporta más ruido que señal. El reranking (C/D, 0,84) **no justifica su latencia en este
caso de uso**: cuesta unos 5,5–6,2 s por consulta en CPU (≈800× la búsqueda y ≈20× el embedding) y,
en conjunto, pierde 0,04 frente a A. Gana en q4 (+0,20), pero pierde en q1 y q5 (−0,20 cada una),
con errores visibles: «clothing» ↔ moda, y un chunk sobredimensionado truncado.

Limitaciones: corpus de 60 chunks, así que un recall de 50 equivale casi a «todo el corpus» y RRF
apenas cambia el pool. Solo hay 5 consultas, y dos de ellas no pueden pasar de 0,80: una diferencia
de 0,04 es **un chunk de 25 posiciones**, por debajo de cualquier umbral de significancia. La
anotación es de un LLM y está pendiente de revisión humana. Además, Precision@5 cuenta chunks, no
presupuestos distintos, y la latencia se ha medido en CPU (Docker Desktop) con un único chunk atípico
que la domina. Con un corpus mayor, consultas con vocabulario exacto (códigos, nombres de producto) o
GPU, la conclusión podría cambiar. Por eso todo sigue siendo conmutable sin tocar código.

## Tests

- Unitarios (`uv run pytest`): RRF ([test_rrf_fusion.py](../../tests/generation/rag/test_rrf_fusion.py)),
  reranker con modelo falso ([test_reranker.py](../../tests/generation/rag/test_reranker.py)),
  composición de A/B/C/D y precedencia ([test_hybrid_retrieve.py](../../tests/generation/rag/test_hybrid_retrieve.py)),
  endpoint ([test_retrieval_endpoint.py](../../tests/test_retrieval_endpoint.py)), métrica y forma del
  golden set ([test_retrieval_metrics.py](../../tests/test_retrieval_metrics.py)) y coherencia de la
  configuración FTS entre migración y modelo ([test_db_layer.py](../../tests/test_db_layer.py)).
- Integración (`uv run pytest -m integration`, BD `estimator_test`):
  [test_lexical_search_db.py](../../tests/integration/test_lexical_search_db.py). Comprueba la columna
  generada y el índice GIN, el relleno automático, que **OR devuelve filas donde AND no**, el orden por
  cobertura, el `tsquery` vacío, el umbral de `search_filtered` y `downgrade`/`upgrade` sobre una tabla
  con datos.
