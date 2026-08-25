#!/bin/sh
set -eu

ollama serve &
pid=$!

echo "Waiting for Ollama..."
for i in $(seq 1 30); do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

MODEL="${OLLAMA_MODEL:-gemma4:e4b}"
echo "Pulling ${MODEL} (this can take 10-20 minutes on first start)..."
ollama pull "${MODEL}"

echo "Gemma ready."
wait "$pid"
