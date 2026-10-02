"""Vertex AI Gemini chat via LangChain."""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI

from app.core.config import get_settings
from app.llm.content_text import stringify_ai_content

PRO_MODEL = "gemini-3.1-pro-preview"


def _pro_model(name: str | None) -> str:
    m = (name or "").strip()
    if not m or "flash" in m.lower() or "2.5" in m:
        return PRO_MODEL
    return m


def _vertex_location() -> str:
    # 3.1 Pro is global-only. A regional location returns 404.
    return "global"


def _vertex_chat_llm(model_name: str | None = None, location: str | None = None) -> ChatVertexAI:
    settings = get_settings()
    project = settings.gcp_project_id
    if not project:
        raise RuntimeError("GCP_PROJECT_ID is required for Vertex AI")
    chosen = _pro_model(model_name or settings.vertex_chat_model)
    is_agent_model = model_name == settings.vertex_caregiver_chat_model
    return ChatVertexAI(
        model_name=chosen,
        project=project,
        location=location or _vertex_location(),
        temperature=0.35 if is_agent_model else 0.5,
        max_retries=2,
    )


def get_vertex_caregiver_llm() -> ChatVertexAI:
    settings = get_settings()
    return _vertex_chat_llm(settings.vertex_caregiver_chat_model)


def vertex_configured() -> bool:
    settings = get_settings()
    return bool(settings.gcp_project_id.strip())


def _scrub_surrogates(text: str) -> str:
    # Upstream JS string slicing can split an emoji and leave a lone UTF-16 surrogate,
    # which protobuf refuses to encode as UTF-8.
    return text.encode("utf-8", "replace").decode("utf-8")


def _scrub_messages(messages: list) -> list:
    for m in messages:
        if isinstance(getattr(m, "content", None), str):
            m.content = _scrub_surrogates(m.content)
        elif isinstance(getattr(m, "content", None), list):
            m.content = [
                _scrub_surrogates(p) if isinstance(p, str)
                else {**p, "text": _scrub_surrogates(p["text"])} if isinstance(p, dict) and isinstance(p.get("text"), str)
                else p
                for p in m.content
            ]
    return messages


async def _invoke_with_capacity_fallback(messages: list) -> object:
    """A 429 stays on 3.1 Pro at global. The regional endpoint 404s for this model."""
    messages = _scrub_messages(messages)
    try:
        return await _vertex_chat_llm().ainvoke(messages)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        if not any(k in text for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "Quota")):
            raise
        await asyncio.sleep(2)
        return await _vertex_chat_llm(location="global").ainvoke(messages)


async def chat_invoke(system: str, user: str) -> str:
    response = await _invoke_with_capacity_fallback([SystemMessage(content=system), HumanMessage(content=user)])
    return stringify_ai_content(response.content)


async def chat_invoke_messages(messages: list) -> object:
    return await _invoke_with_capacity_fallback(messages)


def llm_provider_label() -> str:
    settings = get_settings()
    return f"vertex:{_pro_model(settings.vertex_chat_model)}@{_vertex_location()}"
