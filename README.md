# Kawach AI Engine

Standalone AI microservice for **Kavach CareOS** — Grok chat, RAG over labs/conversations, LangGraph Saheli.

Called by [`kavach-backend-api`](../kavach-backend-api/) (or any HTTP client). Not tied to the dashboard directly.

## Stack

| Layer | Tech |
|-------|------|
| API | FastAPI (:8000) |
| DB | PostgreSQL + pgvector |
| Agent | LangGraph |
| LLM | Grok / xAI (`langchain-xai`) |
| Fallback | Grok CLI (`grok -p`) |

## Quick start

```bash
cd kawach-ai-engine
cp .env.example .env
docker compose up -d db
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/init_db.py
python scripts/run_dev.py
```

Health: http://localhost:8000/health  
Docs: http://localhost:8000/docs

## Auth

All `/v1/*` routes require:

```http
X-Kavach-Secret: <KAWACH_API_SECRET>
```

## Key routes

- `POST /v1/families` — create tenant
- `POST /v1/families/{id}/elders` — add elder
- `POST /v1/chat` — Saheli (RAG + Grok)
- `POST /v1/documents/ingest` — lab/report RAG ingest
- `POST /v1/documents/search` — vector search

See [TENANT_ISOLATION.md](./TENANT_ISOLATION.md) for multi-family isolation rules.
