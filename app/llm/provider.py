"""Unified chat provider — Vertex AI (GCP), Azure Foundry, Ollama, or xAI fallback."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.llm import grok as grok_provider
from app.llm import vertex as vertex_provider


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


def get_caregiver_chat_llm():
    settings = get_settings()
    if vertex_provider.vertex_configured():
        return vertex_provider.get_vertex_caregiver_llm()
    if settings.azure_openai_api_key and settings.azure_chat_deployment:
        return _azure_chat_llm()
    if settings.ollama_base_url:
        return _ollama_chat_llm()
    raise RuntimeError("No LLM provider configured for caregiver agent with tools")


async def chat_invoke_messages(messages: list[BaseMessage]) -> AIMessage:
    settings = get_settings()
    if vertex_provider.vertex_configured():
        result = await vertex_provider.chat_invoke_messages(messages)
        return result if isinstance(result, AIMessage) else AIMessage(content=str(result))
    llm = _azure_chat_llm() if settings.azure_openai_api_key else _ollama_chat_llm()
    response = await llm.ainvoke(messages)
    return response if isinstance(response, AIMessage) else AIMessage(content=str(response))


async def chat_invoke(system: str, user: str) -> str:
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()

    if provider == "vertex" and vertex_provider.vertex_configured():
        return await vertex_provider.chat_invoke(system, user)
    if provider == "ollama" and settings.ollama_base_url:
        llm = _ollama_chat_llm()
    elif provider == "azure" and settings.azure_openai_api_key and settings.azure_chat_deployment:
        llm = _azure_chat_llm()
    elif vertex_provider.vertex_configured():
        return await vertex_provider.chat_invoke(system, user)
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

    if provider == "vertex" and vertex_provider.vertex_configured():
        return vertex_provider.llm_provider_label()
    if provider == "ollama" and settings.ollama_base_url:
        return f"ollama:{settings.ollama_model}"
    if provider == "azure" and settings.azure_chat_deployment:
        return f"azure:{settings.azure_chat_deployment}"
    if vertex_provider.vertex_configured():
        return vertex_provider.llm_provider_label()
    if settings.ollama_base_url:
        return f"ollama:{settings.ollama_model}"
    if settings.azure_chat_deployment:
        return f"azure:{settings.azure_chat_deployment}"
    return grok_provider.llm_provider_label()
