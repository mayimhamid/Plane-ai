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

## Model

Target model: **`Qwen3.5-2B-Q4_K_M.gguf`** (about 1.2 GiB) from `unsloth/Qwen3.5-2B-GGUF`. The download
URL is pinned to a specific commit and checked against a known SHA256.

- Weights are **never** committed. `*.gguf`, `models/`, and similar paths are git-ignored.
- In Docker the model lives in the `plane-ai-models` volume, or in a host path outside the repo
  set by `PLANE_AI_MODEL_VOLUME`.
- `plane-ai-model-init` is a one-shot job. It downloads the file only if it's missing,
  resumes interrupted downloads, verifies the SHA256, and atomically moves the file into place.
  If the file is already verified, it exits right away, so restarts don't re-download.
- **Both runtimes use the same file.** llama.cpp reads it directly from the volume
  (read-only). Ollama can't serve a GGUF in place, so on first start it imports the same file
  into its own store (one extra copy of about 1.2 GiB in `plane-ai-ollama-data`). It re-imports
  only when the file or the context length changes.

## Running with Docker

Plane AI runs as its own Compose project (`plane-ai`) next to Plane, with its own PostgreSQL.
Plane's database is never touched.

```
plane-ai-db (healthy) ──────────────────────────────────────┐
plane-ai-model-init (exits 0) ─> ollama | llama-cpp (healthy) ┴─> plane-ai-api
```

| Service | Image | Network | Published |
|---------|-------|---------|-----------|
| `plane-ai-db` | `postgres:16-alpine` | internal | no |
| `plane-ai-model-init` | built from `orchestration/model-init` | edge (needs internet) | no |
| `plane-ai-ollama` (profile `ollama`) | `ollama/ollama` | internal, alias `plane-ai-llm:11434` | no |
| `plane-ai-llama-cpp` (profile `llama-cpp`) | `ghcr.io/ggml-org/llama.cpp:server` | internal, alias `plane-ai-llm:8080` | no |
| `plane-ai-api` | built from `backend/` | internal + edge | `127.0.0.1:8100` |

`plane-ai-internal` is a Docker `internal` network, with no internet access and no published ports.
Runtime CPU and memory are capped by `PLANE_AI_LLM_CPUS` / `PLANE_AI_LLM_MEMORY`.

### Selecting the runtime

Exactly one runtime runs at a time. It's selected by a single variable in `.env`:

```dotenv
PLANE_AI_LLM_RUNTIME=ollama          # or: llama-cpp
COMPOSE_PROFILES=${PLANE_AI_LLM_RUNTIME}
```

Never put both profiles in `COMPOSE_PROFILES`.

### Production

```bash
cp .env.example .env
```

Edit `.env`: set `PLANE_AI_DB_PASSWORD`, choose `PLANE_AI_LLM_RUNTIME`, and pin the runtime images. Then:

```bash
docker compose up -d --build
```

```bash
docker compose logs -f plane-ai-model-init
```

```bash
docker compose ps
```

```bash
curl http://127.0.0.1:8100/ready
```

The first start downloads the model (about 1.2 GiB). Later starts reuse it.

### Switching runtime

```bash
docker compose down
```

Then change `PLANE_AI_LLM_RUNTIME` in `.env` and start again:

```bash
docker compose up -d
```

### Development

Publishes the DB on `127.0.0.1:5433`, the runtime on `127.0.0.1:11434` or `:8080`, and sets
`APP_ENV=development`:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

Only the database, with the API run locally via `cd backend && python -m app`:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d plane-ai-db
```

### Useful commands

Re-run the model check by hand:

```bash
docker compose run --rm plane-ai-model-init
```

Stop the stack (volumes and the model are kept):

```bash
docker compose down
```

Validate the config without starting anything:

```bash
docker compose config --services
```

### Verifying the infrastructure

Static checks (git, compose), no downloads:

```bash
bash tests/infra/verify_stack.sh --static
```

Full end-to-end check on the server (build, model download once, both runtimes, health):

```bash
bash tests/infra/verify_stack.sh
```

See [docs/infrastructure-verification.md](docs/infrastructure-verification.md).

## API

| Endpoint | Meaning |
|----------|---------|
| `GET /health` | Liveness: the process is up. Doesn't check dependencies. |
| `GET /ready` | Readiness: the database is reachable and the selected runtime serves the model (`ok`, `unavailable`, `loading`, `model_missing`). Returns `503` if not. |
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
