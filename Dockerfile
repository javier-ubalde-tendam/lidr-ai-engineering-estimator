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
    TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# Capa de dependencias: solo se invalida si cambian pyproject.toml o uv.lock
COPY pyproject.toml uv.lock ./
# --no-cache: uv's download cache (~1.6 GB with torch) must not stay baked into this layer
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# El vocabulario BPE se descarga al construir la imagen, no en el primer arranque
RUN python -c "import tiktoken; tiktoken.encoding_for_model('text-embedding-3-small')"

# Sentence tokenizer of the sentence-window chunker (POST /embeddings/compare). Best effort: if the
# download fails here, the chunker fetches it lazily on first use
ENV NLTK_DATA=/opt/nltk_data
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/opt/nltk_data', quiet=True)"

# Spanish spaCy model of the Presidio analyzer (offline ingestion, PII pseudonymization). It is a
# wheel hosted on GitHub, not on PyPI, so it is not part of uv.lock. ~40 MB; build with
# `--build-arg INSTALL_SPACY_MODEL=false` to skip it (the app still starts; only the PII
# analyzer needs it, and it fails there with a clear "model not found" error). The wheel version
# must match the locked spaCy minor (3.8.x). Not a best-effort step: a failed download fails the build
ARG INSTALL_SPACY_MODEL=true
ARG SPACY_MODEL_URL=https://github.com/explosion/spacy-models/releases/download/es_core_news_md-3.8.0/es_core_news_md-3.8.0-py3-none-any.whl
RUN if [ "$INSTALL_SPACY_MODEL" = "true" ]; then \
        VIRTUAL_ENV=/opt/venv uv pip install --no-cache "$SPACY_MODEL_URL"; \
    fi

RUN useradd --create-home --uid 1000 app && chown app:app /app
# Cross-encoder weights cache (session 10). Created here owned by `app` so the named volume
# mounted on top inherits these permissions on first use (the model is not baked into the image)
RUN mkdir -p "$HF_HOME" && chown -R app:app /home/app/.cache
COPY --chown=app:app app ./app
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app alembic.ini streamlit_app.py query_examples.py ./
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app data ./data
COPY --chown=app:app evals ./evals
# Sample transcripts used by scripts/run_agent_s12.py and scripts/run_graph_s13.py
COPY --chown=app:app exercises ./exercises

USER app

EXPOSE 8000 8501
CMD ["python", "--version"]
