"""Embedding clients — Azure Foundry preferred, xAI/OpenAI optional fallback."""

from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings

XAI_API_BASE = "https://api.x.ai/v1"


def _valid_api_key(key: str) -> bool:
    k = key.strip()
    if not k:
        return False
    if k in {"xai-...", "sk-...", "sk-your-key"}:
        return False
    if k.endswith("..."):
        return False
    return True


def embeddings_available() -> bool:
    settings = get_settings()
    return (
        _valid_api_key(settings.azure_openai_api_key)
        and bool(settings.azure_embedding_deployment)
    ) or _valid_api_key(settings.xai_api_key) or _valid_api_key(settings.openai_api_key)


def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()

    if _valid_api_key(settings.azure_openai_api_key) and settings.azure_embedding_deployment:
        return OpenAIEmbeddings(
            model=settings.azure_embedding_model,
            api_key=settings.azure_openai_api_key,
            base_url=settings.azure_openai_base_url,
            deployment=settings.azure_embedding_deployment,
        )

    if _valid_api_key(settings.xai_api_key):
        return OpenAIEmbeddings(
            model=settings.grok_embedding_model,
            api_key=settings.xai_api_key,
            base_url=XAI_API_BASE,
        )

    if _valid_api_key(settings.openai_api_key):
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    raise RuntimeError(
        "RAG embeddings need Azure Foundry (AZURE_OPENAI_API_KEY + AZURE_EMBEDDING_DEPLOYMENT) "
        "or XAI_API_KEY / OPENAI_API_KEY as fallback."
    )


def embedding_provider_label() -> str:
    settings = get_settings()
    if _valid_api_key(settings.azure_openai_api_key) and settings.azure_embedding_deployment:
        return f"azure:{settings.azure_embedding_deployment}"
    if _valid_api_key(settings.xai_api_key):
        return f"xai:{settings.grok_embedding_model}"
    if _valid_api_key(settings.openai_api_key):
        return f"openai:{settings.openai_embedding_model}"
    return "none"


async def embed_text(text: str) -> list[float]:
    embeddings = get_embeddings()
    return await embeddings.aembed_query(text)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = get_embeddings()
    return await embeddings.aembed_documents(texts)
