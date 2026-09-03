from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import care_brief, chat, documents, families, health
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    app.include_router(care_brief.router, prefix="/v1")
    return app


app = create_app()
