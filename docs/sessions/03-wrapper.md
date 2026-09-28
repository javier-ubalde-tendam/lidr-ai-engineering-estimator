# Session 03 - Wrapper


## Funcionalidad incremental

Se añade:
- Interfaz gráfica (web) conversacional con Streamlit
- Streaming → la respuesta se muestra conforme el LLM la genera, no al final de golpe
- Caché → si el input es el misma se devuelve respuesta Caché (Redis)
- Wrapper LLM
- Logging



## Tecnologías

- structlog: logging estructurado
- redis (en docker): Cache


## Cómo invocar

### Levantar docker redis (docker)
docker run -d --name lidr-redis -p 6379:6379 redis:7-alpine

### Arrancar la aplicación
streamlit run streamlit_app.py

### Lanzar petición
- Acceder a la local URL: http://localhost:8501
- Interactuar con el chat