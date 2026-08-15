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

echo "Pulling ${OLLAMA_MODEL:-gemma2:9b}..."
ollama pull "${OLLAMA_MODEL:-gemma2:9b}"

echo "Gemma ready."
wait "$pid"
