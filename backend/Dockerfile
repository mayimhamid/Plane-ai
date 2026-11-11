# syntax=docker/dockerfile:1.7
# Plane AI API image. Contains only the FastAPI backend; LLM runtimes
# (Ollama / llama.cpp) run in their own containers.

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
# Install dependencies from pyproject.toml only, so this layer is cached
# until dependencies change (not on every code edit).
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > requirements.txt \
 && pip install -r requirements.txt

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PLANE_AI_APP_ENV=production \
    PLANE_AI_API_HOST=0.0.0.0 \
    PLANE_AI_API_PORT=8100

RUN groupadd --system --gid 10001 planeai \
 && useradd --system --uid 10001 --gid planeai --no-create-home --shell /usr/sbin/nologin planeai

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Code stays root-owned (read-only for the app user).
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

USER 10001:10001
EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-m", "app.healthcheck"]

# Uvicorn handles SIGTERM by draining requests, then running shutdown hooks.
STOPSIGNAL SIGTERM
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "app"]
