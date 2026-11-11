#!/usr/bin/env bash
# Plane AI infrastructure verification (stage 10).
#
#   tests/infra/verify_stack.sh --static   # git + compose checks only (no images, no downloads)
#   tests/infra/verify_stack.sh            # full run on the server: builds, pulls runtime images,
#                                          # downloads the model ONCE, starts each runtime in turn
#
# Uses the repo-root .env. Never deletes volumes, so the model and the database
# survive. At the end, the runtime selected in .env is started again.
#
# Env overrides: RUNTIMES="ollama llama-cpp"  WAIT_TIMEOUT=1800  API_PORT=8100

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="full"
[ "${1:-}" = "--static" ] && MODE="static"
RUNTIMES="${RUNTIMES:-ollama llama-cpp}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"

env_value() {  # env_value NAME DEFAULT  -> value from .env, else default
    local v
    v="$(grep -E "^$1=" .env 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
    echo "${v:-$2}"
}
API_PORT="${API_PORT:-$(env_value PLANE_AI_API_PORT 8100)}"
MODEL_FILE="$(env_value PLANE_AI_LLM_MODEL_FILE Qwen3.5-2B-Q4_K_M.gguf)"
MODEL_NAME="$(env_value PLANE_AI_LLM_MODEL_NAME qwen3.5-2b)"
MODEL_SHA="$(env_value PLANE_AI_MODEL_SHA256 aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223)"
# Tiny file from the same pinned HF commit: proves a bad checksum is rejected
# without touching the real model.
PROBE_URL="https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/.gitattributes"
ZERO_SHA="0000000000000000000000000000000000000000000000000000000000000000"

RESULTS=()
FAILED=0
pass() { RESULTS+=("PASS  $1"); echo "  [PASS] $1"; }
fail() { RESULTS+=("FAIL  $1${2:+ -- $2}"); FAILED=$((FAILED + 1)); echo "  [FAIL] $1${2:+ -- $2}"; }
skip() { RESULTS+=("SKIP  $1${2:+ -- $2}"); echo "  [SKIP] $1${2:+ -- $2}"; }
step() { echo; echo "== $*"; }
check() { local name="$1"; shift; if "$@" >/dev/null 2>&1; then pass "$name"; else fail "$name"; fi; }

other_runtime() { if [ "$1" = "ollama" ]; then echo "llama-cpp"; else echo "ollama"; fi; }
cid() { docker compose ps -a -q "$1" 2>/dev/null | head -n1; }

wait_for() {  # wait_for <service> healthy|exited
    local svc="$1" want="$2" deadline=$((SECONDS + WAIT_TIMEOUT)) c state
    while [ "$SECONDS" -lt "$deadline" ]; do
        c="$(cid "$svc")"
        if [ -n "$c" ]; then
            if [ "$want" = "exited" ]; then
                state="$(docker inspect -f '{{.State.Status}}' "$c")"
                [ "$state" = "exited" ] && return 0
            else
                state="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$c")"
                [ "$state" = "healthy" ] && return 0
            fi
        fi
        sleep 5
    done
    return 1
}

# ================================================================== static checks
step "Git: no model weights committed"
if git rev-parse --git-dir >/dev/null 2>&1; then
    weights="$(git ls-files | grep -Ei '\.(gguf|ggml|safetensors|bin|pt|pth|onnx)$' || true)"
    if [ -z "$weights" ]; then pass "no weight files tracked by git"; else fail "weight files tracked" "$weights"; fi
    big=""
    while IFS= read -r -d '' f; do
        if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 5000000 ]; then big="$big $f"; fi
    done < <(git ls-files -z)
    if [ -z "$big" ]; then pass "no tracked file larger than 5 MB"; else fail "large tracked files" "$big"; fi
    check "models/$MODEL_FILE is git-ignored" git check-ignore -q "models/$MODEL_FILE"
    check "*.gguf anywhere is git-ignored" git check-ignore -q "runtimes/x/$MODEL_FILE"
    check ".env is git-ignored" git check-ignore -q .env
else
    skip "git checks" "not a git repository"
fi

step "Compose configuration"
if [ -f .env ]; then pass ".env exists"; else fail ".env exists" "cp .env.example .env and set PLANE_AI_DB_PASSWORD"; fi
for rt in ollama llama-cpp; do
    services="$(COMPOSE_PROFILES=$rt PLANE_AI_LLM_RUNTIME=$rt docker compose config --services 2>&1 | sort | tr '\n' ' ')"
    expected="$(printf '%s\n' plane-ai-api plane-ai-db plane-ai-model-init "plane-ai-$rt" | sort | tr '\n' ' ')"
    if [ "$services" = "$expected" ]; then
        pass "profile $rt -> exactly one runtime: $services"
    else
        fail "profile $rt services" "$services"
    fi
done
check "dev override is valid" docker compose -f docker-compose.yml -f docker-compose.dev.yml config --quiet
db_ports="$(docker compose config --format json 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["services"]["plane-ai-db"].get("ports", []))' 2>/dev/null || echo '?')"
if [ "$db_ports" = "[]" ]; then pass "production: database has no published port"; else fail "production db ports" "$db_ports"; fi

if [ "$MODE" = "static" ]; then
    step "Summary (static)"
    printf '%s\n' "${RESULTS[@]}"
    [ "$FAILED" -eq 0 ] && exit 0 || exit 1
fi

# ================================================================== build & pull
step "Build images"
check "docker compose build (api + model-init)" docker compose --profile ollama --profile llama-cpp build
check "pull runtime images" docker compose --profile ollama --profile llama-cpp pull --ignore-buildable --quiet
api_user="$(docker run --rm --entrypoint id "plane-ai-api:${PLANE_AI_IMAGE_TAG:-latest}" -u 2>/dev/null)"
if [ "$api_user" = "10001" ]; then pass "api image runs as non-root (uid 10001)"; else fail "api image user" "$api_user"; fi

# ================================================================== per runtime
first_run=1
for rt in $RUNTIMES; do
    other="$(other_runtime "$rt")"
    step "Runtime: $rt"
    docker compose --profile ollama --profile llama-cpp down --remove-orphans >/dev/null 2>&1
    export COMPOSE_PROFILES="$rt" PLANE_AI_LLM_RUNTIME="$rt"
    if ! docker compose up -d >/dev/null 2>&1; then
        fail "[$rt] docker compose up"
        continue
    fi

    if wait_for plane-ai-db healthy; then pass "[$rt] postgres healthy"; else fail "[$rt] postgres healthy"; fi

    if wait_for plane-ai-model-init exited; then
        code="$(docker inspect -f '{{.State.ExitCode}}' "$(cid plane-ai-model-init)")"
        logs="$(docker compose logs --no-log-prefix plane-ai-model-init 2>&1)"
        if [ "$code" = "0" ]; then pass "[$rt] model-init exited 0"; else fail "[$rt] model-init exit code" "$code"; fi
        if [ "$first_run" -eq 0 ]; then
            if echo "$logs" | grep -q "present and verified" && ! echo "$logs" | grep -qE "downloading|resuming"; then
                pass "[$rt] model NOT downloaded again (verified file reused)"
            else
                fail "[$rt] model reuse" "model-init downloaded again"
            fi
        fi
    else
        fail "[$rt] model-init finished" "timeout ${WAIT_TIMEOUT}s"
    fi

    if wait_for "plane-ai-$rt" healthy; then pass "[$rt] runtime container healthy"; else fail "[$rt] runtime container healthy"; fi
    if wait_for plane-ai-api healthy; then pass "[$rt] api container healthy"; else fail "[$rt] api container healthy"; fi

    running="$(docker compose ps --services --status running | tr '\n' ' ')"
    other_c="$(docker ps -q --filter label=com.docker.compose.project=plane-ai --filter "label=com.docker.compose.service=plane-ai-$other")"
    if echo "$running" | grep -q "plane-ai-$rt" && [ -z "$other_c" ]; then
        pass "[$rt] only the $rt runtime is running"
    else
        fail "[$rt] runtime selection" "running: $running"
    fi

    # Same model file, byte for byte, in both runtimes.
    sum="$(docker compose exec -T "plane-ai-$rt" sha256sum "/models/$MODEL_FILE" 2>/dev/null | cut -d' ' -f1)"
    if [ "$sum" = "$MODEL_SHA" ]; then
        pass "[$rt] sees /models/$MODEL_FILE with the expected SHA256"
    else
        fail "[$rt] model file checksum" "${sum:-not found}"
    fi

    # Load the model into memory (a one-word completion, just to force a load).
    if [ "$rt" = "ollama" ]; then
        load_cmd="timeout 600 ollama run \"$MODEL_NAME\" \"Reply with OK\" >/dev/null && ollama ps | grep -q \"$MODEL_NAME\""
        if docker compose exec -T plane-ai-ollama sh -c "$load_cmd" >/dev/null 2>&1; then
            pass "[ollama] model loads (ollama ps lists $MODEL_NAME)"
        else
            fail "[ollama] model load"
        fi
    else
        # llama-server's /health is 200 only after the model is loaded.
        if docker compose exec -T plane-ai-llama-cpp curl -fsS http://localhost:8080/v1/models 2>/dev/null | grep -q "\"$MODEL_NAME\""; then
            pass "[llama-cpp] model loaded and served as $MODEL_NAME"
        else
            fail "[llama-cpp] model load"
        fi
    fi

    health="$(curl -fsS "http://127.0.0.1:$API_PORT/health" 2>/dev/null)"
    if echo "$health" | grep -q '"status":"ok"'; then pass "[$rt] GET /health"; else fail "[$rt] GET /health" "$health"; fi
    ready="$(curl -sS "http://127.0.0.1:$API_PORT/ready" 2>/dev/null)"
    if echo "$ready" | grep -q '"status":"ready"' && echo "$ready" | grep -q "\"runtime\":\"$rt\""; then
        pass "[$rt] GET /ready (database + $rt runtime ok)"
    else
        fail "[$rt] GET /ready" "$ready"
    fi

    first_run=0
done

# ================================================================== model-init behaviour
step "Model initializer"
first_rt="${RUNTIMES%% *}"
export COMPOSE_PROFILES="$first_rt" PLANE_AI_LLM_RUNTIME="$first_rt"
out="$(docker compose run --rm --no-deps plane-ai-model-init 2>&1)"
if echo "$out" | grep -q "present and verified"; then
    pass "re-run is a no-op (idempotent)"
else
    fail "idempotent re-run" "$(echo "$out" | tail -n1)"
fi

out="$(docker compose run --rm --no-deps \
    -e PLANE_AI_MODEL_STORAGE_PATH=/tmp/sha-probe \
    -e PLANE_AI_LLM_MODEL_FILE=probe.bin \
    -e PLANE_AI_MODEL_URL="$PROBE_URL" \
    -e PLANE_AI_MODEL_SHA256="$ZERO_SHA" \
    plane-ai-model-init 2>&1)"
code=$?
if [ "$code" -ne 0 ] && echo "$out" | grep -q "checksum mismatch"; then
    pass "wrong SHA256 is rejected (exit $code, partial file deleted)"
else
    fail "SHA256 rejection" "exit $code"
fi

# ================================================================== graceful shutdown
step "Graceful shutdown"
start=$SECONDS
docker compose stop -t 30 plane-ai-api >/dev/null 2>&1
elapsed=$((SECONDS - start))
if docker compose logs --no-log-prefix plane-ai-api 2>&1 | grep -q "shutdown complete" && [ "$elapsed" -lt 25 ]; then
    pass "api stops cleanly on SIGTERM (${elapsed}s)"
else
    fail "api graceful shutdown" "${elapsed}s"
fi

# ================================================================== restore
step "Restore the runtime selected in .env"
unset COMPOSE_PROFILES PLANE_AI_LLM_RUNTIME
docker compose --profile ollama --profile llama-cpp down --remove-orphans >/dev/null 2>&1
check "docker compose up -d (runtime from .env)" docker compose up -d

step "Summary"
printf '%s\n' "${RESULTS[@]}"
echo
if [ "$FAILED" -eq 0 ]; then echo "ALL CHECKS PASSED"; exit 0; fi
echo "$FAILED CHECK(S) FAILED"
exit 1
