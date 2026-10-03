from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api import brain, care_brief, dash, chat, documents, doctor_brief, families, health, memory
from app.core.config import get_settings
from app.db.migrate import run_instinct_migrations
from app.db.session import Base, engine
from app.care import models as care_models  # noqa: F401
from app.tasks import models as task_models  # noqa: F401
from app.specialists import channels as specialist_channels  # noqa: F401
from app.llm import spend as llm_spend
from app.care import baselines as care_baselines  # noqa: F401
from app.learn import models as learn_models  # noqa: F401
from app.care import memory_index as care_memory_index  # noqa: F401
from app.models import entities  # noqa: F401


async def ensure_database_schema() -> None:
    async with engine.begin() as conn:
        await run_instinct_migrations(conn)
        await conn.run_sync(Base.metadata.create_all)
        # Indexes added after a table already exists (create_all only makes new tables).
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_turns_thread_id ON turns (family_id, thread_id, id)"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_database_schema()
    from app.db.session import SessionLocal

    llm_spend.configure(SessionLocal)
    yield
    await llm_spend.flush()  # write the last ledger rows before the instance stops
    from app.agents.tool_client import close_clients

    await close_clients()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Kavach AI Engine",
        description="Standalone multi-family memory — RAG + LangGraph Saheli (Grok)",
        version="0.1.0",
        lifespan=lifespan,
    )
    origins = (
        ["*"]
        if settings.cors_origins.strip() == "*"
        else [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {
            "service": "kawach-ai-engine",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    app.include_router(health.router)
    app.include_router(families.router, prefix="/v1")
    app.include_router(chat.router, prefix="/v1")
    app.include_router(documents.router, prefix="/v1")
    app.include_router(memory.router, prefix="/v1")
    app.include_router(care_brief.router, prefix="/v1")
    app.include_router(doctor_brief.router, prefix="/v1")
    app.include_router(brain.router)
    app.include_router(dash.router)
    return app


app = create_app()
