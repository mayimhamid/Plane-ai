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

## Development

```bash
pip install -e "backend[dev]"
pytest tests
```
