"""Grok / xAI LLM — API first, Grok CLI fallback."""

from __future__ import annotations

import asyncio
import subprocess

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_xai import ChatXAI

from app.core.config import get_settings


def _valid_api_key(key: str) -> bool:
    k = key.strip()
    if not k:
        return False
    if k in {"xai-...", "sk-...", "sk-your-key"}:
        return False
    if k.endswith("..."):
        return False
    return True


def _has_xai_api_key() -> bool:
    return _valid_api_key(get_settings().xai_api_key)


def get_chat_llm() -> BaseChatModel:
    settings = get_settings()
    if not _has_xai_api_key():
        raise RuntimeError(
            "Set XAI_API_KEY for Grok API, or ensure `grok` CLI is on PATH (grok login)."
        )
    return ChatXAI(
        model=settings.grok_chat_model,
        api_key=settings.xai_api_key,
        temperature=0.6,
        max_retries=2,
    )


async def grok_cli_invoke(system: str, user: str) -> str:
    settings = get_settings()
    args = ["-p", user, "--system-prompt-override", system]

    def run() -> str:
        proc = subprocess.run(
            ["grok", *args],
            capture_output=True,
            text=True,
            timeout=settings.grok_cli_timeout_seconds,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or f"grok exited {proc.returncode}"
            raise RuntimeError(err)
        out = proc.stdout.strip()
        if not out:
            raise RuntimeError("grok returned empty output")
        return out

    return await asyncio.to_thread(run)


async def chat_invoke(system: str, user: str) -> str:
    if _has_xai_api_key():
        llm = get_chat_llm()
        response = await llm.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        content = response.content
        return content if isinstance(content, str) else str(content)

    return await grok_cli_invoke(system, user)


def llm_provider_label() -> str:
    if _has_xai_api_key():
        return f"xai:{get_settings().grok_chat_model}"
    return "grok-cli"
