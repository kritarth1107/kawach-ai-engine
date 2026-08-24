#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Starting local pgvector Postgres on :5433 ..."
docker compose up -d db

echo "Waiting for Postgres ..."
for i in $(seq 1 30); do
  if docker compose exec -T db pg_isready -U postgres >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Creating tables + pgvector ..."
export DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_ai'
PYTHONPATH=. .venv/bin/python scripts/init_db.py

echo
echo "Local Saheli memory is ready."
echo "1. Restart the AI engine (Ctrl+C in that terminal, then):"
echo "     cd /home/noooblien/kawach/kawach-ai-engine"
echo "     PYTHONPATH=. .venv/bin/python scripts/run_dev.py"
echo "2. Keep backend + dashboard running, then seed the demo thread:"
echo "     cd /home/noooblien/kawach/kavach-backend-api"
echo "     npx tsx scripts/yc-demo-seed.ts"
echo "3. Open http://localhost:3000  — login as Kritarth, Ask Saheli about Vasundara."
