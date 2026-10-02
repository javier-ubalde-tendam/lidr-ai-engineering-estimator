# Session 03 - Wrapper

## Arquitectura

          ┌─────────────────────────┐
          │   Interfaz Streamlit    │
          └───────────┬─────────────┘
                      │
                      │POST
                      │
          ┌───────────▼─────────────┐
          │   Endpoint FastAPI      │
          │   (ver. streaming)      │
          └───────────┬─────────────┘
                      │
          ┌───────────▼─────────────┐
          │   LLM Wrapper           │
          │   - Abstracción         │ # No importamos openai ni anthopic, es transparente
          │   - Fallback            │ # Si falla el LLM solicitado, pasa al siguiente
          │   - Logging (structlog) │
          └─────────────┬───────────┘
                        │
                 hit    │ 
                 ┌──────┴──────┐
                 ▼             │ no hit
            ┌─────────┐        │
            │  Cache  │        │
            │  Redis  │        │
            │ (exact  │        │
            │  match) │        │
            └─────────┘        │
                               │
                               │
                    ┌──────────┴─────────┐
                    ▼                    ▼
               ┌─────────┐ Falback ┌──────────┐
               │ OpenAI  │ <-----> │Anthropic │
               └─────────┘         └──────────┘


## Funcionalidad incremental

Modelo de "dos procesos": esta sesión introduce por primera vez que hacen falta dos servidores corriendo a la vez (Uvicorn + Streamlit) comunicándose por red — es un cambio de arquitectura respecto a sesiones anteriores donde Streamlit llamaba todo en el mismo proceso.

Se añade:
- Interfaz gráfica (web) conversacional con Streamlit. Incluye un sidebar con la información de modelo, tokens, tiempo de respuesta, coste estimado (usd) y hit de Cache.
- Streaming. La respuesta se muestra conforme el LLM la genera, no al final de golpe
- Caché. Si el input es el mismo se devuelve respuesta Caché (Redis). Cacheo exact match (una coma cambia la clave y no hace hit)
- Logging. Trazar por consola con datos estructurados
- Wrapper LLM. Abstraer la llamada a los LLM usando un agrupador (LiteLLM)
- Fallback. Se reinenta fallo con el LLM (no todos, AuthenticationError no se reintenta con el mismo proveedor y salta directo al fallback) y en caso de fallo final (tras LLM_MAX_RETRIES) se hace switch al otro proveedor. Se pueden usar openai y anthopic


## Tecnologías

- Server-Sent Events (SSE) de FastAPI: usamos para devolver los chunk poco a poco al cliente desde el endpoint de FastAPI
- LiteLLM (wrapper/agregador de LLMs): librería Python open source que expone una interfaz para abstraer proveedores de LLM.
- Streamlit: framework Python para interfaz de aplicaciones de IA. Genera interfaz web con chat, sesión, etc. La aplicación en esta sesión es una aplicación streamlit (punto principal de entrada)
- httpx: es la pieza que permite a Streamlit consumir el SSE del backend
- structlog: logging estructurado
- redis: Cache (en una instancia aparte, nos tenemos que conectar). La levantamos con docker


## Cómo invocar

### Levantar docker redis (docker)
docker run -d --name lidr-redis -p 6379:6379 redis:7-alpine

### Levantar uvicorn (servidor web) para uso de FastAPI
uv run uvicorn app.main:app --reload

### Arrancar la aplicación streamlit (portal web con interfaz conversacional)
streamlit run streamlit_app.py

### Lanzar petición
- Acceder a la local URL: http://localhost:8501
- Interactuar con el chat