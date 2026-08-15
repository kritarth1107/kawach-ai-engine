#!/usr/bin/env bash
# Truncate all kawach-ai-engine Postgres tables (fresh AI memory).
# Usage: ./scripts/reset-ai-postgres.sh
set -euo pipefail

if [[ -z "${DATABASE_URL:-}" ]]; then
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL not set"
  exit 1
fi

# Convert asyncpg URL to psql-compatible libpq URL
PSQL_URL="${DATABASE_URL/postgresql+asyncpg/postgresql}"
PSQL_URL="${PSQL_URL//\?ssl=require/?sslmode=require}"
PSQL_URL="${PSQL_URL//&ssl=require/&sslmode=require}"

psql "$PSQL_URL" -v ON_ERROR_STOP=1 <<'SQL'
TRUNCATE TABLE
  memory_snippets,
  document_chunks,
  messages,
  conversations,
  memory_documents,
  elders,
  family_members,
  users,
  families
RESTART IDENTITY CASCADE;
SQL

echo "Postgres kawach_ai truncated."
