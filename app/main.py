from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api import care_brief, chat, documents, doctor_brief, families, health, memory
from app.core.config import get_settings
from app.db.migrate import run_instinct_migrations
from app.db.session import Base, engine
from app.models import entities  # noqa: F401


async def ensure_database_schema() -> None:
    async with engine.begin() as conn:
        await run_instinct_migrations(conn)
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_database_schema()
    yield


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
    return app


app = create_app()
