from fastapi import APIRouter

from app.llm.grok import llm_provider_label
from app.rag.embeddings import embedding_provider_label

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "kawach-ai-engine",
        "llm": llm_provider_label(),
        "embeddings": embedding_provider_label(),
    }
