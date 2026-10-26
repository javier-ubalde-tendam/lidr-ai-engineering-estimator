# Session 08 - PostgreSQL + pgvector y búsqueda semántica (ejercicio previo)

Alcance previo a la sesión 08: persistir los presupuestos en PostgreSQL con pgvector y buscarlos por
distancia coseno con SQL. Todo el stack (FastAPI, Streamlit, Redis Stack, PostgreSQL) corre en Docker.
No se conecta `/search` al prompt del estimador (integración RAG: otra sesión) ni hay índices
vectoriales, filtros por metadata, búsqueda híbrida ni reranking.

## Arranque y comprobaciones (PowerShell + Docker Desktop, Compose >= 2.20)

```powershell
Copy-Item .env.example .env            # y rellena OPENAI_API_KEY (obligatoria: embeddings)
docker compose config --quiet
docker compose up --build -d
docker compose ps
docker compose exec -T postgres psql -U estimator -d estimator -c "SELECT version();"
docker compose exec -T ai_service alembic current
docker compose exec -T ai_service alembic check
docker compose exec -T postgres psql -U estimator -d estimator -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
docker compose run --rm ai_service python query_examples.py
docker compose exec -T postgres psql -U estimator -d estimator -c "SELECT (SELECT count(*) FROM documents) AS documents, (SELECT count(*) FROM chunks) AS chunks;"
```

Informe a fichero (el contenedor de `run --rm` se destruye, por eso se usa `exec` + `cp`):

```powershell
docker compose exec -T ai_service python query_examples.py --output /app/output_examples.txt
docker compose cp ai_service:/app/output_examples.txt ./output_examples.txt
```

Parar sin perder datos: `docker compose down` (nunca `-v`: borraría los volúmenes `lidr-postgres-data` y
`lidr-redis-data`).

## Compose jerárquico

`docker-compose.yml` solo contiene `include` de `docker/infra`, `docker/backend` y `docker/frontend`.
Cada archivo incluido resuelve sus rutas relativas desde su propio directorio (por eso el build usa
`context: ../..` y el `env_file` es `../../.env`). Todos comparten la red por defecto del proyecto, y
no se fija `name:` para mantener el volumen `lidr-ai-engineering-estimator_lidr-redis-data` existente.

- `ai_service`: `alembic upgrade head && exec uvicorn ...` (Uvicorn solo arranca si la migración termina
  bien); espera a `redis` y `postgres` sanos; healthcheck con `urllib` (la imagen no trae `curl`).
- `frontend`: Streamlit con `API_BASE_URL=http://ai_service:8000`; espera al backend sano.
- La imagen no define `ENTRYPOINT`, así que `python query_examples.py` se ejecuta tal cual con `run`.
- El virtualenv vive en `/opt/venv` (fuera de `/app`), el vocabulario de tiktoken se descarga al construir
  la imagen y el proceso corre como usuario no root `app` (`/app` escribible por él).
- El healthcheck de Postgres usa `-h 127.0.0.1`: el servidor temporal de inicialización solo escucha por
  socket Unix y un `pg_isready` sin host daría "sano" antes de tiempo.

## Esquema

Ver la migración `alembic/versions/0001_documents_chunks_pgvector.py` y los modelos `app/db/models.py`
(el atributo `metadata_` se mapea a la columna `metadata`, porque `metadata` está reservado en SQLAlchemy
declarativo). `alembic check` no detecta diferencias entre modelo y migración. El `downgrade` no elimina
la extensión `vector`, que puede ser compartida.

## Tests de integración

`uv run pytest -m integration` (requiere `docker compose up -d postgres`). Usan la BD `estimator_test`
(se crea si no existe y se migra con Alembic), embeddings simulados de 1536 dimensiones y PostgreSQL
real: cubren rollback tras fallos del embedder / de una escritura posterior al `flush`, carrera de
duplicados resuelta por el índice único, `ON DELETE CASCADE`, ranking coseno, `LIMIT k` y exclusión de
`NULL`. Para no tocar datos de desarrollo, el fixture rechaza cualquier URL cuya BD no termine en `_test`
o coincida con la de `DATABASE_URL`.

## Notas de la implementación

- El `chunk_id` lógico (`budget_id::component_id`) se conserva en `chunks.metadata`; no se confunde con el
  `id` BIGINT de la tabla, que es el que devuelve `/search`.
- El embedder síncrono se ejecuta con `asyncio.to_thread`: al hilo solo viajan los textos, nunca la
  `AsyncSession`. Se mantiene el batching de 100 (un documento con <=100 chunks = una llamada).
- `search_time_ms` e `ingestion_time_ms` se miden con `time.perf_counter()`; el primero incluye el embedding
  de la query y la consulta SQL.
- La consulta de `/search` es `SELECT ... ORDER BY embedding <=> :query LIMIT :k`: el ranking lo hace
  PostgreSQL con parámetros enlazados.
- Cambiar `EMBEDDING_MODEL` a otro modelo falla al arrancar la configuración con un mensaje explícito.

## Cómo invocar

### 1. Desde Swagger (más cómodo)

- Abre http://localhost:8000/docs
- POST /search: pulsa Try it out y envía:
{"query": "REST API development with JWT authentication for financial sector", "k": 5}
- POST /embeddings/ingest: pega un presupuesto de budgets_sample.json dentro de content:
{"source_path": "mi-prueba::1", "document_type": "historical_budget", "content": { ...un Budget... }}
La primera vez da 200; repetirlo da 409 con 

### 2. Desde PowerShell

- Recuerda usar curl.exe, porque curl es un alias de Invoke-WebRequest.
'{"query":"integration with external system","k":3}' | Set-Content -Encoding utf8 q.json
curl.exe -s -X POST http://localhost:8000/search -H "Content-Type: application/json" --data-binary "@q.json"
- Si aparece {"detail":...} con 422, revisa que query no esté vacío y que k esté entre 1 y 100.

### 3. Script de ejemplo (ingesta + 5 consultas)

`docker compose run --rm ai_service python query_examples.py`
Es idempotente: una segunda ejecución muestra 15 × 409.

### 4. Streamlit

Abre http://localhost:8501 para la UI de estimación. Esta no usa /search todavía; esa integración queda para otra sesión.

### 5. Inspeccionar la base de datos

#### 5.1. Usando docker

`docker compose exec -T postgres psql -U estimator -d estimator -c "SELECT (SELECT count(*) FROM documents) AS documents, (SELECT count(*) FROM chunks) AS chunks;"`

`docker compose exec -T postgres psql -U estimator -d estimator -c "SELECT id, document_id, left(content, 60), metadata->>'chunk_id' FROM chunks LIMIT 5;"`

#### 5.2. Cliente JDBC (p.ej. DBEaver)

Postgres publica el puerto 5432 en tu máquina mientras el stack esté levantado.

Host: localhost
Puerto: 5432 (o el POSTGRES_PORT de tu .env, si lo has cambiado)
Base de datos: estimator
Usuario: estimator
Password: estimator
URL JDBC equivalente: jdbc:postgresql://localhost:5432/estimator

#### 5.3. SQL de ejemplo

- Primero debemos obtener el emmbedding del texto a buscar:

`docker compose exec -T ai_service python -c "import json; from app.dependencies import get_embedder as g; print(json.dumps(g().embed_one('REST API with OAuth authentication for fintech sector')))" > "C:\Temp\embedding.json"`

- SQL de ejemplo (recorto el valor del embedding)

`@set query_vector = '[-0.0501708984375, -0.02337646484375, ...]'`
`@set k = 5`
`SELECT c.id, c.content, c.chunk_type,`
`       c.embedding <=> CAST(:query_vector AS vector) AS distance,`
`       d.metadata->'client_metadata'->>'sector' AS sector,`
`       d.ingested_at`
`FROM chunks c`
`JOIN documents d ON d.id = c.document_id`
`WHERE d.metadata->'client_metadata'->>'sector' = 'finance'`
`  AND d.ingested_at > NOW() - INTERVAL '24 months'`
`  AND (d.metadata->>'total_estimated_hours')::int BETWEEN 300 AND 700`
`ORDER BY c.embedding <=> CAST(:query_vector AS vector)`
`LIMIT :k;`

### 6. Tests automáticos

uv run pytest                    # unitarios, sin OpenAI/Redis/Postgres
uv run pytest -m integration     # requiere: docker compose up -d postgres

### 7. Logs si algo falla

`docker compose logs -f ai_service`

Un 500 con Embedding service error suele indicar un problema con OPENAI_API_KEY; Database error indica un problema con Postgres.

