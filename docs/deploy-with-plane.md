# Running Plane AI next to Plane on one server

Plane AI is its own compose project. Plane's `api` container reaches it over an external Docker network,
`plane-ai-link`; nothing needs a published port. Plane's side of the setup (its `plane.env` and compose file) is in
Plane's `docs/team-ops/AI.md`.

```bash
docker network create plane-ai-link            # once
cd /opt/smartfactoryteam/plane-ai
cp .env.example .env                            # first time: set PLANE_AI_DB_PASSWORD and PLANE_AI_SERVICE_TOKEN
docker compose -f docker-compose.yml -f docker-compose.plane.yml up -d --build
docker compose logs -f plane-ai-model-init      # first start: the model download
curl http://127.0.0.1:8100/ready
```

Updating after a `git pull`:

```bash
docker compose -f docker-compose.yml -f docker-compose.plane.yml up -d --build
```

Migrations run when the API container starts (`PLANE_AI_RUN_MIGRATIONS=true`). Jobs that were running during a
restart start again by themselves.

## Sizing on a shared server

The model runs on the CPU. On a machine that also runs other products, keep it from taking everything:

| `.env` | Suggested | Why |
| --- | --- | --- |
| `PLANE_AI_LLM_CPUS` / `PLANE_AI_LLM_THREADS` | 4 | the runtime's hard CPU cap and the threads used at High effort |
| `PLANE_AI_LLM_MEMORY` | 6g | hard memory cap of the runtime (the 2B Q4 model needs ~1.5 GB plus the context) |
| `PLANE_AI_LLM_CONTEXT_LENGTH` | 8192 | the largest context; Low effort uses 2048 of it |
| `PLANE_AI_KV_CACHE_TYPE` | q8_0 | 8-bit attention cache: about half the memory per token of context |
| `PLANE_AI_LLM_IDLE_SECONDS` | 300 | idle this long, the runtime unloads the model and frees its memory (llama.cpp sleeps, Ollama unloads); the next request loads it again. `-1` keeps it loaded. A workspace turning AI Off also stops its running jobs |
| `PLANE_AI_LLM_MAX_CONCURRENCY` | 1 | one answer at a time; others queue (two at once are both slower on a CPU) |

In Plane, Effort Low uses 1-2 threads and short answers: slower per answer, least load for the other products.

## Optional speed-ups

- **llama.cpp** as the runtime (`PLANE_AI_LLM_RUNTIME=llama-cpp`, `COMPOSE_PROFILES=llama-cpp`) reuses the prompt
  cache between requests and supports a **draft model** for speculative decoding: put a small GGUF of the same
  family in the model volume, set `PLANE_AI_DRAFT_MODEL_FILE`, and add `-f runtimes/llama-cpp/speculative.yaml`.
- A **LoRA adapter** trained from the exported ratings: see [fine-tuning.md](fine-tuning.md).
