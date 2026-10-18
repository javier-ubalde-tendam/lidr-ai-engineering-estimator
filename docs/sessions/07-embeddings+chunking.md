# Session 07 - Embeddings y chunking (ejercicio previo)

## Funcionalidad incremental

Ejercicio previo a la sesión 7: un **pipeline mínimo** que convierte presupuestos históricos en
chunks con embeddings, base para el RAG de las siguientes sesiones. No se toca la lógica de
estimación existente.

Se añade, en `app/embedding_pipeline/`:
- `schemas.py`: modelos Pydantic de entrada (`Budget`, `BudgetComponent`...) y salida (`Chunk`,
  `EmbeddedChunk`, `IngestStats`, `IngestResponse`).
- `chunker.py`: `JSONStructuralChunker`, un componente del presupuesto = un chunk.
- `embedder.py`: `OpenAIEmbedder` (batches, reintentos ante rate limit, logging) y
  `estimate_cost_usd()`.
- `router.py`: `POST /embeddings/ingest`.
- `SANITY_CHECK.md`: similitud coseno real de tres parejas de textos.
- `scripts/compare.py`: CLI que compara dos textos con similitud coseno.
- Datos de ejemplo en `data/budgets_sample.json` (15 presupuestos, 52 componentes).

## Tecnologías

- OpenAI Embeddings (`text-embedding-3-small`, 1536 dimensiones): el modelo ya estaba en `EMBEDDING_MODEL`
- tiktoken: cuenta tokens con el tokenizer del modelo (dependencia declarada, antes solo transitiva)
- Pydantic v2 / FastAPI / structlog: igual que el resto del servicio
- Biblioteca estándar (`math`, `argparse`): similitud coseno a mano, sin numpy

## Conceptos

### Embeddings y similitud coseno

Un embedding es un vector de números que representa el significado de un texto: textos con
significado parecido quedan en direcciones parecidas. La similitud coseno mide el ángulo entre dos
vectores: `a·b / (|a|·|b|)`. Vale 1 para la misma dirección, 0 para ortogonales (sin relación).
Se usa el ángulo y no la distancia porque lo importante es la dirección, no la longitud.

### Chunking estructural

Los presupuestos ya vienen estructurados (JSON), así que no hace falta adivinar dónde cortar: cada
componente es una unidad con sentido propia. Un componente = un chunk, sin overlap ni división de
descripciones largas. Es la estrategia más simple posible; sirve de base para comparar otras.

### Contextual chunk headers

Un componente suelto ("Backend services") es ambiguo. Cada chunk empieza con un header del
presupuesto padre (proyecto, sector, año, tecnología principal) para que el embedding herede ese
contexto. Formato exacto del texto embebido:

```
[Project: {project_summary}]
[Client sector: {sector} | Year: {year} | Main tech: {main_technology}]

Component: {name}
Description: {description}
Tech stack: {a, b, c}
Complexity: {complexity}
Estimated hours: {estimated_hours}
```

### Metadata filtrable frente a texto embebido

Lo que va en `text` se convierte en vector (búsqueda por significado). Lo que va en `metadata`
(`budget_id`, `component_id`, `client_sector`, `main_technology`, `year`, `complexity`,
`estimated_hours`) **no se embebe**: se guarda aparte para filtrar de forma exacta
("solo sector finance, año 2024"). Los filtros exactos no deben delegarse a la similitud.

### Batching

La API admite una lista de textos por llamada. `OpenAIEmbedder` envía batches de 100
(`BATCH_SIZE`): 250 chunks = 3 llamadas (100/100/50) en vez de 250. Reduce latencia y riesgo de
rate limit, y el orden de entrada se conserva en la salida.

### Coste

`estimated_cost_usd = tokens / 1.000.000 × 0.02` (constante `PRICE_PER_MILLION_TOKENS_USD`, tomada
del enunciado; hay que verificarla en la página oficial de precios porque cambia). Con el sample
completo: 4.126 tokens ≈ 0,00008 USD. Los embeddings son baratos frente a las llamadas al LLM.

## Decisiones de diseño y límites

- **Proveedor desacoplado:** `OpenAIEmbedder` recibe el cliente `OpenAI` por constructor; no lo
  crea. Los tests usan un cliente falso y no hacen red.
- **Reintentos solo para `RateLimitError`:** 3 reintentos con esperas de 1, 2 y 4 s (4 llamadas
  como máximo); si se agotan, se relanza. Cualquier otro error se propaga sin reintento. La
  función de espera es inyectable (`sleep`) para que los tests no esperen.
- **Endpoint síncrono (`def`):** el cliente OpenAI es bloqueante; FastAPI ejecuta los `def` en un
  threadpool y no bloquea el event loop.
- **Errores:** ante `openai.OpenAIError` se loguea el tipo y el mensaje truncado, y al cliente solo
  llega `500 "Embedding service error"` (nunca detalles ni la API key).
- **`Literal` cerrados** para `sector` y `complexity`: valores del dataset real (4 sectores,
  3 complejidades). Un sector nuevo exige ampliar el `Literal`.
- **Tokenizer:** `tiktoken.encoding_for_model(EMBEDDING_MODEL)` se resuelve una vez por instancia del
  chunker (singleton vía `get_chunker()`). La primera vez puede descargar el vocabulario.
- **`cosine_similarity` vive en `scripts/compare.py`**: solo la usa ese script; sin abstracción
  prematura. Los tests la cargan con `importlib`.

## Cómo invocar

Servicio levantado con `uv run uvicorn app.main:app`:

- Swagger: `http://localhost:8000/docs` → `POST /embeddings/ingest` → *Try it out* con
  `{"budgets": [...]}`.
- curl con el sample:
  Primero generamos un fichero con el cuerpo de la llamada:
  `'{"budgets": ' + (Get-Content data\budgets_sample.json -Raw -Encoding UTF8) + '}' | Set-Content -Encoding utf8 body.json`
  Después invocamos para obtener todos los vecores:
  `curl.exe -X POST http://localhost:8000/embeddings/ingest -H "Content-Type: application/json" --data-binary "@body.json"`
- Comparar dos textos: `uv run python scripts/compare.py --text-a "..." --text-b "..."`
  (con `OPENAI_API_KEY` en el `.env` de la raíz).

Resultado real con el sample: `total_budgets=15`, `total_chunks=52`, `total_tokens=4126`,
`estimated_cost_usd≈0.0000825`.

## Limitaciones (qué queda para la sesión en vivo y la 08)

- Sin persistencia: los vectores se devuelven en la respuesta; no hay base de datos vectorial.
- Sin retrieval ni métricas de retrieval (recall, MRR...).
- Una sola estrategia de chunking; sin overlap, recursive, semantic, hierarchical ni late chunking.
- Sin comparativa de modelos de embeddings ni reducción de dimensiones.
- Sin enriquecimiento de chunks con LLM (contextual retrieval).
- Un presupuesto enorme genera una respuesta enorme (1536 floats por chunk); no hay paginación.
