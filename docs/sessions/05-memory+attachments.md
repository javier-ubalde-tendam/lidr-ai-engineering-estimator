# Session 05 - Memory + Attachments

## Funcionalidad incremental

Se añade:
- Memoria conversacional en el servicio IA: sesiones en memoria del proceso (`app/sessions/`),
  con un historial de mensajes (ventana deslizante) y un `project_metadata` acumulado.
- Adjuntos (`.pdf`, `.docx`) extraídos localmente (Camino B) y enviados como texto dentro del prompt.
- Nuevo endpoint `POST /sessions/{session_id}/estimate` (multipart/form-data, bloqueante) que
  usa el historial + metadata de la sesión para mantener coherencia entre turnos.
- `GET /sessions/{session_id}` para inspeccionar el estado (nº de mensajes, metadata) y
  `POST /sessions` para crear una sesión nueva.
- Extracción de `project_metadata` mediante un LLM extractor: una segunda llamada barata con
  Instructor, después de cada turno, cuyo resultado se fusiona (`merge_with`) con lo ya sabido.
- Prompt `v3` (copia de `v2` + bloque `<project_metadata>` e instrucciones de conversación) y
  prompt `metadata_extraction/v1` para el extractor.
- Modo conversacional en Streamlit (`streamlit_app.py`), junto al formulario existente, con
  historial de chat, subida de adjuntos y panel lateral con la metadata y el tamaño del historial.

## Tecnologías

- pypdf: extrae el texto de un PDF página a página; si una página falla, se registra en el log y se ignora (no tira abajo el documento entero)
- python-docx: extrae los párrafos no vacíos de un `.docx`
- python-multipart: dependencia que FastAPI necesita internamente para poder parsear `Form`/`UploadFile` (`multipart/form-data`)
- fpdf2 (dev-dependency): genera PDFs mínimos válidos en los tests, sin necesitar un fichero de ejemplo en el repo
- Pydantic: modela `Message`, `ConversationHistory`, `ProjectMetadata` y `Session`, igual que ya modelaba `EstimationRequest`/`EstimationResult`
- Instructor: se reutiliza para la llamada de estimación (ahora con un array de `messages` arbitrario) y para la nueva llamada del extractor de metadata (`response_model=ProjectMetadata`)
- FastAPI `Depends` + `app.dependency_overrides`: `get_session_store()` es un singleton (como `get_openai_client()`), sustituible en los tests

## Decisiones de diseño

### Historial vs memoria

Se mantienen dos conceptos separados y complementarios:
- **Historial** (`ConversationHistory`): un array de mensajes `user`/`assistant`, con una ventana
  deslizante de `MAX_CONVERSATION_TURNS` pares. Es la memoria "a corto plazo", literal: lo que se
  dijo en los últimos turnos, en su forma original.
- **Memoria** (`ProjectMetadata`): hechos destilados (`project_name`, `assumed_team_size`,
  `mentioned_technologies`, `agreed_scope`), inyectados en el `system_prompt` de cada turno vía un
  bloque `<project_metadata>`. Es memoria "a largo plazo": sobrevive aunque el turno donde se dijo
  ese hecho ya haya salido de la ventana del historial.

El `system_prompt` nunca se guarda en el historial: se regenera en cada turno a partir del
metadata actual, así que siempre refleja el estado más reciente aunque la conversación sea larga.

### Por qué una ventana deslizante como estrategia por defecto

Es la opción más simple y barata: cada turno cuesta tokens proporcionales a
`1 (system) + 2 * MAX_CONVERSATION_TURNS (historial) + 1 (turno actual)`, con un tope fijo y
predecible. El riesgo es perder hechos antiguos que ya no están en la ventana — por eso existe el
`project_metadata` en paralelo: los hechos importantes se "rescatan" del historial antes de que se
recorten.

Qué empujaría a sustituirla por algo más sofisticado (p. ej. resumir los turnos descartados en vez
de tirarlos, o un historial por tópicos):
- El coste de tokens crece demasiado si `MAX_CONVERSATION_TURNS` necesita subir mucho para no
  perder contexto relevante.
- Se pierden matices que el extractor de metadata no captura (no todo es un hecho estructurado:
  una aclaración o una broma recurrente no caben en `ProjectMetadata`).

### Camino B para adjuntos (extracción local) y por qué

Se extrae el texto del PDF/Word localmente con `pypdf`/`python-docx` y se envía como texto plano
dentro del prompt, en vez de usar la Files API de un proveedor. Motivos:
- **Independiente del proveedor**: compatible con el fallback OpenAI↔Anthropic que ya tiene el
  proyecto. Con la Files API, el fichero solo existiría en el proveedor donde se subió; si el
  fallback cambia de proveedor a mitad de conversación, el fichero ya no sería accesible.
- **Más control**: se puede truncar (`MAX_ATTACHMENT_CHARS`), registrar en el log cuántos
  caracteres se extrajeron/truncaron, y tratar errores por página sin perder el documento entero.
- **Prepara el chunking de RAG** del módulo 3: ya tenemos una función que convierte un fichero en
  texto plano, el primer paso antes de trocearlo en chunks para un índice vectorial.

### Cómo se extrae el `project_metadata`: LLM extractor

Después de cada turno, una segunda llamada (barata, `METADATA_EXTRACTOR_MODEL`) con Instructor
recibe el metadata previo, el transcript enriquecido del turno y el resumen de la estimación, y
devuelve un `ProjectMetadata` con solo los hechos anclados en esos textos (null si no está claro).
El resultado se fusiona con `previous.merge_with(extracted)`: los escalares se sobrescriben solo si
el update trae un valor no nulo, y las tecnologías se unen sin duplicados (comparación
case-insensitive).

Trade-off frente a una heurística (regex/keywords):
- **Coste y latencia**: una llamada extra por turno, con su coste y tiempo, aunque usando un
  modelo barato y una respuesta corta.
- **Robustez**: un LLM entiende paráfrasis ("el equipo seremos cinco personas" → `assumed_team_size=5`)
  que una heurística basada en patrones fijos no captura.

Política de fallo: si la extracción falla por cualquier motivo (red, validación, proveedor caído),
se registra `metadata_extraction_failed` en el log y se devuelve el metadata anterior sin cambios.
La conversación nunca se cae por un fallo en esta llamada secundaria.

### Caches desactivadas en el camino conversacional

`estimate_conversational()` no usa ni la caché exact-match ni la semántica. La clave actual de
caché se construye a partir de `system_prompt` + `user_prompt`; en el camino conversacional, el
resultado también depende del historial y del `project_metadata` de la sesión, que esa clave no
tiene en cuenta. Dos sesiones distintas (o dos turnos de la misma sesión) podrían generar el mismo
`system_prompt` + `user_prompt` por una coincidencia textual, mientras el contexto real es distinto
→ el hit sería incorrecto. Queda documentado en el código y deliberadamente sin usar; una caché
correcta para este camino necesitaría una clave que incluya el estado de la sesión.

### Memoria volátil en proceso y sus límites

`SessionStore` es un `dict[str, Session]` en memoria del proceso, sin Redis ni base de datos. Se
acepta perder las sesiones al reiniciar el proceso (son conversaciones de una estimación, no datos
de negocio) a cambio de no añadir infraestructura todavía. Importante: **solo funciona con un único
worker de uvicorn**; con `--workers N>1` cada proceso tendría su propio diccionario, y una petición
podría caer en un worker que nunca vio esa sesión. La persistencia real (Redis/BBDD) se abordaría
más adelante si hiciera falta soportar varios workers o sobrevivir a un reinicio.

### Variables de entorno nuevas

| Variable | Default | Qué controla |
|---|---|---|
| `MAX_CONVERSATION_TURNS` | `6` | Pares user+assistant que conserva la ventana deslizante del historial |
| `MAX_ATTACHMENT_CHARS` | `60000` | Límite de caracteres por adjunto extraído (trunca, no rechaza) |
| `METADATA_EXTRACTOR_MODEL` | `gpt-4o-mini` | Modelo de la segunda llamada que extrae el `project_metadata` |
| `CONVERSATIONAL_PROMPT_VERSION` | `v3` | Versión del prompt usada por `estimate_conversational()` |

## Cómo invocar

### Lanzar tests unitarios
```
uv run pytest -v
```

### Levantar docker redis (necesita redis-stack, no redis:7-alpine, para el cache semántico/RediSearch)
```
docker compose up -d
```

### Levantar uvicorn (servidor web) para uso de FastAPI
```
uv run uvicorn app.main:app --reload
```

### Arrancar la aplicación streamlit (portal web con interfaz conversacional)
```
streamlit run streamlit_app.py
```

### Probar el flujo conversacional por línea de comandos (2 turnos + 1 adjunto)

Los ejemplos usan `curl.exe` (en PowerShell, `curl` es un alias de `Invoke-WebRequest`: hay que
llamar al binario real explícitamente) en una sola línea para que funcionen igual en PowerShell y
en una shell POSIX.

Crear una sesión:
```
curl.exe -X POST http://localhost:8000/sessions
```

Primer turno (sustituye `<session_id>` por el valor devuelto arriba):
```
curl.exe -X POST http://localhost:8000/sessions/<session_id>/estimate -F "transcript=We are building Acme Portal, a web SaaS for gym bookings." -F "project_type=web_saas" -F "detail_level=summary" -F "output_format=narrative"
```

Segundo turno, con un adjunto:
```
curl.exe -X POST http://localhost:8000/sessions/<session_id>/estimate -F "transcript=Here is the budget breakdown from the client, see the attached file." -F "project_type=web_saas" -F "detail_level=summary" -F "output_format=narrative" -F "attachments=@budget.pdf;type=application/pdf"
```

Inspeccionar el estado de la sesión (historial + metadata):
```
curl.exe http://localhost:8000/sessions/<session_id>
```
