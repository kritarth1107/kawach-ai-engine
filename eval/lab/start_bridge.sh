#!/bin/bash
# Start the Saheli Lab bridge on localhost and write a fresh token for the bots.  eval/lab/start_bridge.sh [port]
set -euo pipefail
cd "$(dirname "$0")/../.."
LAB_DIR="${SAHELI_LAB_DIR:-/home/m4dm4x/OpenBot/Shared/saheli-lab}"
PORT="${1:-8787}"
TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
printf 'LAB_TOKEN=%s\nLAB_URL=http://127.0.0.1:%s\n' "$TOKEN" "$PORT" > "$LAB_DIR/lab.env"
chmod 600 "$LAB_DIR/lab.env"
export LAB_TOKEN="$TOKEN" PYTHONPATH=. DEBUG=false DATABASE_URL="${SIM_DATABASE_URL:-postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_sim}"
setsid nohup .venv/bin/python -m uvicorn eval.lab.bridge:app --host 127.0.0.1 --port "$PORT" > "$LAB_DIR/bridge.log" 2>&1 &
echo $! > "$LAB_DIR/bridge.pid"
sleep 3 && eval/lab/labctl.sh status && echo "bridge up on :$PORT (pid $(cat "$LAB_DIR/bridge.pid"))"
