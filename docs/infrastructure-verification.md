# Infrastructure verification report (stage 10)

Date: 2026-11-12. Scope: Plane AI infrastructure only. No chatbot features.

Verification runs in two places:

- **Local** (developer machine): unit tests and static checks. No images are built, nothing is
  pulled, and no model is downloaded.
- **Server**: `tests/infra/verify_stack.sh` runs the real stack end to end. It hasn't been run yet.

## Results

| # | Check | Local | Server (`verify_stack.sh`) |
|---|-------|-------|----------------------------|
| 1 | Docker builds successfully | Not run (would pull base images). Dockerfiles reviewed. | `docker compose build`, api image runs as uid 10001 |
| 2 | Compose configuration is valid | **Pass**: prod, dev override, both profiles | Same checks, repeated |
| 3 | PostgreSQL becomes healthy | Not run | Waits for the `pg_isready` healthcheck |
| 4 | Model downloads only once | **Pass**: test with fake server (3 runs, 1 request) | 2nd runtime start and a manual re-run log "present and verified", with no download |
| 5 | SHA256 validation works | **Pass**: mismatch, corrupt file, quarantine, cleanup | Wrong hash on a 3 KB probe file must exit non-zero |
| 6 | Ollama can load the model | Client logic: **pass** (mocked) | `ollama run` + `ollama ps`, file SHA256 inside the container |
| 7 | llama.cpp can load the same model | Client logic: **pass** (mocked) | `/v1/models` serves the alias, same file SHA256 |
| 8 | API health checks work | **Pass**: `/health`, `/ready` (db + runtime states) | `curl /health` and `/ready` for each runtime |
| 9 | Runtime selection works | **Pass**: each profile resolves to exactly one runtime | Only the selected runtime container is running |
| 10 | No model weights in Git | **Pass**: no weight files, nothing over 5 MB, `*.gguf`/`models/`/`.env` ignored | Same checks, repeated |

Local unit tests: **45 passed** (config, health and readiness, runtime clients, model initializer).

## Issues found and fixed

1. **API would crash on startup in Docker.** Compose passes unset optional variables as empty
   strings (`PLANE_AI_LOG_LEVEL: ""`), which the settings rejected as an invalid log level.
   Fixed with `env_ignore_empty=True`, so empty means "use the default". Added a regression test.
2. **`/ready` ignored the LLM runtime.** It now checks that the selected runtime is up *and*
   serves the configured model. Statuses: `ok`, `unavailable`, `loading`, `model_missing`.
   Otherwise it returns 503.
3. **Misleading API env var.** `PLANE_AI_MODEL_STORAGE_PATH=/models` was set on the API container,
   which doesn't mount the model volume. Removed.

## Architecture notes

- **API to runtime:** `app/llm/runtime.py` has one small client per runtime, reached at
  `http://plane-ai-llm:<port>` (a network alias both runtimes share, so the API only needs
  `PLANE_AI_LLM_RUNTIME`). It only checks connectivity and that the model is available, with no
  generation. `httpx` is now a runtime dependency because it is the async HTTP client for FastAPI.
- **One model file:** llama.cpp reads `/models/<file>` read-only. Ollama imports the same file once
  (about 1.2 GiB copy in `plane-ai-ollama-data`) because it can't serve a GGUF in place. The
  script proves both see the identical file by SHA256.

## Known limits / to confirm on the server

- Runtime images default to `latest`. Pin `PLANE_AI_OLLAMA_IMAGE` / `PLANE_AI_LLAMA_CPP_IMAGE` to
  versions that support the Qwen3.5 architecture.
- `PLANE_AI_LLM_CPUS` (default 4) must not exceed the server's CPU count, or Docker refuses to start.
- A bind-mounted `PLANE_AI_MODEL_VOLUME` must be writable by uid 10001.
- CPU only. There's no GPU configuration yet.

## How to run on the server

```bash
cp .env.example .env
```

Set `PLANE_AI_DB_PASSWORD` in `.env`, then:

```bash
bash tests/infra/verify_stack.sh --static
```

```bash
bash tests/infra/verify_stack.sh 2>&1 | tee verify-$(date +%F).log
```

The full run downloads the model once (about 1.2 GiB), pulls both runtime images, and leaves the
runtime selected in `.env` running. It never removes volumes.
