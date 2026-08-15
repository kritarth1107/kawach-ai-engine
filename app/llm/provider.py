"""Unified chat provider — Azure Foundry, self-hosted Gemma (Ollama), or xAI fallback."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.llm import grok as grok_provider


def _azure_chat_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.azure_chat_deployment,
        api_key=settings.azure_openai_api_key,
        base_url=settings.azure_openai_base_url,
        temperature=0.6,
        max_retries=2,
    )


def _ollama_chat_llm() -> ChatOpenAI:
    settings = get_settings()
    base = settings.ollama_base_url.rstrip("/")
    return ChatOpenAI(
        model=settings.ollama_model,
        api_key="ollama",
        base_url=f"{base}/v1",
        temperature=0.6,
        max_retries=2,
    )


async def chat_invoke(system: str, user: str) -> str:
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()

    if provider == "ollama" and settings.ollama_base_url:
        llm = _ollama_chat_llm()
    elif provider == "azure" and settings.azure_openai_api_key and settings.azure_chat_deployment:
        llm = _azure_chat_llm()
    elif settings.ollama_base_url:
        llm = _ollama_chat_llm()
    elif settings.azure_openai_api_key and settings.azure_chat_deployment:
        llm = _azure_chat_llm()
    else:
        return await grok_provider.chat_invoke(system, user)

    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = response.content
    return content if isinstance(content, str) else str(content)


def llm_provider_label() -> str:
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()

    if provider == "ollama" and settings.ollama_base_url:
        return f"ollama:{settings.ollama_model}"
    if provider == "azure" and settings.azure_chat_deployment:
        return f"azure:{settings.azure_chat_deployment}"
    if settings.ollama_base_url:
        return f"ollama:{settings.ollama_model}"
    if settings.azure_chat_deployment:
        return f"azure:{settings.azure_chat_deployment}"
    return grok_provider.llm_provider_label()
