"""Vertex AI Gemini chat via LangChain."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI

from app.core.config import get_settings


def _vertex_chat_llm(model_name: str | None = None) -> ChatVertexAI:
    settings = get_settings()
    project = settings.gcp_project_id
    if not project:
        raise RuntimeError("GCP_PROJECT_ID is required for Vertex AI")
    return ChatVertexAI(
        model_name=model_name or settings.vertex_chat_model,
        project=project,
        location=settings.gcp_region,
        temperature=0.6,
        max_retries=2,
    )


def get_vertex_caregiver_llm() -> ChatVertexAI:
    settings = get_settings()
    return _vertex_chat_llm(settings.vertex_caregiver_chat_model)


def vertex_configured() -> bool:
    settings = get_settings()
    return bool(settings.gcp_project_id.strip())


async def chat_invoke(system: str, user: str) -> str:
    llm = _vertex_chat_llm()
    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = response.content
    return content if isinstance(content, str) else str(content)


async def chat_invoke_messages(messages: list) -> object:
    llm = _vertex_chat_llm()
    return await llm.ainvoke(messages)


def llm_provider_label() -> str:
    settings = get_settings()
    return f"vertex:{settings.vertex_chat_model}"
