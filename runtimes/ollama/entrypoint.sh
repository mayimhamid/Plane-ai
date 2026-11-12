#!/bin/sh
# Start Ollama and register the shared GGUF file from /models as a local model.
# Ollama can't serve a GGUF in place: `ollama create` copies it into its own
# store (/root/.ollama) once. It's re-imported only when the source file or
# the context length changes.
set -eu

MODEL_PATH="/models/${PLANE_AI_LLM_MODEL_FILE:?}"
MODEL_NAME="${PLANE_AI_LLM_MODEL_NAME:?}"
CTX="${PLANE_AI_LLM_CONTEXT_LENGTH:-8192}"
STAMP_FILE="/root/.ollama/.plane-ai-import"
ADAPTER="${PLANE_AI_LORA_ADAPTER:-}"
ADAPTER_PATH=""
if [ -n "$ADAPTER" ]; then
    ADAPTER_PATH="/models/$ADAPTER"
    if [ ! -f "$ADAPTER_PATH" ]; then
        echo "ollama-entrypoint: LoRA adapter not found: $ADAPTER_PATH (starting without it)" >&2
        ADAPTER_PATH=""
    fi
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ollama-entrypoint: model file not found: $MODEL_PATH (did model-init run?)" >&2
    exit 1
fi

ollama serve &
SERVE_PID=$!
# Forward stop signals to the server and exit when it does.
trap 'kill -TERM "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID"; exit 0' TERM INT

i=0
until ollama list >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 60 ]; then echo "ollama-entrypoint: server did not start" >&2; exit 1; fi
    sleep 1
done

STAMP="$MODEL_NAME $(stat -c '%s %Y' "$MODEL_PATH") ctx=$CTX adapter=${ADAPTER_PATH:+$(stat -c '%s %Y' "$ADAPTER_PATH")}"
if [ "$(cat "$STAMP_FILE" 2>/dev/null || true)" != "$STAMP" ] || ! ollama show "$MODEL_NAME" >/dev/null 2>&1; then
    echo "ollama-entrypoint: importing $MODEL_PATH as $MODEL_NAME (num_ctx=$CTX)"
    printf 'FROM %s\nPARAMETER num_ctx %s\n' "$MODEL_PATH" "$CTX" > /tmp/Modelfile
    if [ -n "$ADAPTER_PATH" ]; then
        echo "ollama-entrypoint: with LoRA adapter $ADAPTER_PATH"
        printf 'ADAPTER %s\n' "$ADAPTER_PATH" >> /tmp/Modelfile
    fi
    ollama create "$MODEL_NAME" -f /tmp/Modelfile
    echo "$STAMP" > "$STAMP_FILE"
else
    echo "ollama-entrypoint: $MODEL_NAME already imported"
fi

wait "$SERVE_PID"
