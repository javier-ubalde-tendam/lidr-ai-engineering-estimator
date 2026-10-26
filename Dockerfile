# Imagen unica reutilizable por ai_service (FastAPI + Alembic + scripts) y frontend (Streamlit).
# El comando de arranque lo define cada servicio en su Compose: sin ENTRYPOINT, para que
# `docker compose run ai_service python query_examples.py` no quede atrapado por Uvicorn.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

# El venv vive fuera de /app para que ningun bind mount sobre /app lo oculte
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache

WORKDIR /app

# Capa de dependencias: solo se invalida si cambian pyproject.toml o uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# El vocabulario BPE se descarga al construir la imagen, no en el primer arranque
RUN python -c "import tiktoken; tiktoken.encoding_for_model('text-embedding-3-small')"

RUN useradd --create-home --uid 1000 app && chown app:app /app
COPY --chown=app:app app ./app
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app alembic.ini streamlit_app.py query_examples.py ./
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app data ./data

USER app

EXPOSE 8000 8501
CMD ["python", "--version"]
