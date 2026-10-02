"""Multi-model router.

Each role (brain, worker, extract, classify, judge, vision) has an ordered list of models.
A call goes to the first healthy model; a rate limit, outage or missing model trips a
short circuit breaker and the call moves to the next one. Messages and tools use one
neutral shape so a turn can fall over from Claude to Gemini mid-loop.

Neutral messages:
    {"role": "user", "content": [{"type": "text", "text": ...}, {"type": "image", "mime": ..., "data": b64}]}
    {"role": "assistant", "text": ..., "tool_calls": [ToolCall...], "raw": (model, provider blocks)}
    {"role": "tool", "results": [{"id": ..., "name": ..., "content": str, "is_error": bool}]}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMReply:
    text: str
    tool_calls: list[ToolCall]
    model: str
    raw: Any = None
    usage: dict = field(default_factory=dict)
    stop: str = ""

    def as_message(self) -> dict:
        return {"role": "assistant", "text": self.text, "tool_calls": self.tool_calls, "raw": (self.model, self.raw)}


@dataclass(frozen=True)
class Route:
    provider: str  # claude | gemini | fake
    model: str
    location: str = "global"

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}@{self.location}"


DEFAULT_ROUTES: dict[str, list[str]] = {
    "brain": ["claude:claude-opus-5-5", "gemini:gemini-3.1-pro-preview", "gemini:gemini-3.8-flash"],
    "worker": ["claude:claude-sonnet-5-5", "gemini:gemini-3.8-flash", "gemini:gemini-3.1-pro-preview"],
    "extract": ["claude:claude-sonnet-5-5", "gemini:gemini-3.8-flash"],
    "classify": ["claude:claude-haiku-4-5", "gemini:gemini-3.8-flash"],
    "judge": ["claude:claude-opus-5-5", "gemini:gemini-3.1-pro-preview"],
    "vision": ["gemini:gemini-3.8-flash", "claude:claude-sonnet-5-5"],
}


def parse_route(spec: str) -> Route:
    provider, _, rest = spec.partition(":")
    model, _, location = rest.partition("@")
    return Route(provider=provider, model=model, location=location or "global")


def routes_for(role: str) -> list[Route]:
    """MODEL_ROUTES env (JSON {role: [specs]}) overrides the defaults per role."""
    override = os.getenv("MODEL_ROUTES", "").strip()
    table = dict(DEFAULT_ROUTES)
    if override:
        try:
            table.update(json.loads(override))
        except json.JSONDecodeError:
            logger.warning("MODEL_ROUTES is not valid JSON; using defaults")
    return [parse_route(s) for s in table.get(role, table["worker"])]


class ModelUnavailable(Exception):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


class AllModelsFailed(Exception):
    pass


class Provider(Protocol):
    async def complete(
        self,
        route: Route,
        *,
        system_stable: str,
        system_dynamic: str,
        messages: list[dict],
        tools: list[ToolSpec],
        max_tokens: int,
        effort: str,
    ) -> LLMReply: ...


def scrub(text: str) -> str:
    """Drop lone UTF-16 surrogates (half an emoji) that providers refuse to encode."""
    return text.encode("utf-8", "replace").decode("utf-8") if text else text


# ── circuit breaker ────────────────────────────────────────────────────────────

_open_until: dict[str, float] = {}


def _cooldown(status: int | None) -> float:
    if status in (403, 404):
        return 3600.0  # model not enabled for this project, or no quota at all
    if status == 429:
        return 60.0
    return 30.0


def healthy(route: Route, now: float | None = None) -> bool:
    return _open_until.get(route.key, 0.0) <= (now if now is not None else time.monotonic())


def trip(route: Route, status: int | None) -> None:
    _open_until[route.key] = time.monotonic() + _cooldown(status)


def reset_breakers() -> None:
    _open_until.clear()


# ── router ─────────────────────────────────────────────────────────────────────

_providers: dict[str, Provider] = {}


def register_provider(name: str, provider: Provider) -> None:
    _providers[name] = provider


def _provider(name: str) -> Provider:
    if name not in _providers:
        if name == "claude":
            from app.llm.providers.claude import ClaudeProvider

            _providers[name] = ClaudeProvider()
        elif name == "gemini":
            from app.llm.providers.gemini import GeminiProvider

            _providers[name] = GeminiProvider()
        else:
            raise KeyError(f"No provider registered for {name}")
    return _providers[name]


async def complete(
    role: str,
    *,
    system_stable: str,
    system_dynamic: str = "",
    messages: list[dict],
    tools: list[ToolSpec] | None = None,
    max_tokens: int = 8000,
    effort: str = "medium",
    timeout_s: float = 90.0,
    routes: list[Route] | None = None,
) -> LLMReply:
    candidates = routes or routes_for(role)
    live = [r for r in candidates if healthy(r)] or candidates  # all tripped: try anyway
    errors: list[str] = []
    for route in live:
        try:
            reply = await asyncio.wait_for(
                _provider(route.provider).complete(
                    route,
                    system_stable=scrub(system_stable),
                    system_dynamic=scrub(system_dynamic),
                    messages=messages,
                    tools=tools or [],
                    max_tokens=max_tokens,
                    effort=effort,
                ),
                timeout=timeout_s,
            )
            logger.info("llm role=%s model=%s usage=%s", role, route.key, reply.usage)
            return reply
        except ModelUnavailable as exc:
            trip(route, exc.status)
            errors.append(f"{route.key}: {exc.status} {str(exc)[:120]}")
        except asyncio.TimeoutError:
            trip(route, None)
            errors.append(f"{route.key}: timeout")
        logger.warning("llm fallback role=%s from=%s reason=%s", role, route.key, errors[-1])
    raise AllModelsFailed("; ".join(errors))


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"
