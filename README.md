# Plane AI

A local AI Copilot for [Plane](https://plane.so) project management. All inference runs on your own
hardware through a local LLM runtime (Ollama or llama.cpp). No data leaves the machine.

> Status: infrastructure scaffolding. There's no application logic yet.

## Layout

| Path | Purpose |
|------|---------|
| `backend/` | Python backend (`app/core/config.py` holds the centralized settings) |
| `runtimes/ollama/` | Ollama runtime setup |
| `runtimes/llama-cpp/` | llama.cpp server runtime setup |
| `orchestration/model-init/` | Model download/initialization jobs |
| `configs/` | Shared non-secret configuration files |
| `benchmarks/` | Runtime/model benchmarks (results are git-ignored) |
| `tests/` | Test suite |
| `docs/` | Documentation |

## Configuration

All settings are environment variables with the `PLANE_AI_` prefix. See [`.env.example`](.env.example).

```bash
cp .env.example .env
```

Resolution order (highest priority first): process environment, `.env.<APP_ENV>`, `.env`, then the
built-in defaults. `PLANE_AI_APP_ENV` selects `development` (default), `production`, or `test`.
In production, startup fails if `DEBUG` is on or no database password is set.

## Model weights

Model weights are **never** committed. They live in `PLANE_AI_MODEL_STORAGE_PATH` (default `./models`,
git-ignored). `*.gguf`, `*.safetensors`, and similar files are git-ignored everywhere.

## Running with Docker

Plane AI runs as its own Compose project (`plane-ai`), next to Plane, with its own PostgreSQL.
Plane's database is never touched.

```bash
cp .env.example .env   # then set PLANE_AI_DB_PASSWORD
docker compose up -d --build
curl http://127.0.0.1:8100/health
curl http://127.0.0.1:8100/ready
```

- `plane-ai-db` (official `postgres:16-alpine`): data in the `plane-ai-db-data` volume, healthchecked,
  only on the internal network `plane-ai-internal`, with no published port.
- `plane-ai-api`: runs as a non-root user, runs `alembic upgrade head` on start (`PLANE_AI_RUN_MIGRATIONS`),
  and is published on `127.0.0.1:8100` by default.

Development override: publishes the DB on `127.0.0.1:5433` and sets `APP_ENV=development`.

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

## API

| Endpoint | Meaning |
|----------|---------|
| `GET /health` | Liveness: the process is up. Doesn't check dependencies. |
| `GET /ready` | Readiness: the database is reachable. Returns `503` if it isn't. |
| `GET /docs` | OpenAPI UI (disabled in production) |

## Development

```bash
pip install -e "backend[dev]"
pytest tests

# API locally (with the DB from the dev compose override)
cd backend && python -m app

# Migrations
cd backend && alembic revision --autogenerate -m "describe change"
cd backend && alembic upgrade head
```
