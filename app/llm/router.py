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
    provider: str  # claude | gemini | bedrock | fake
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
    "judge_fast": ["gemini:gemini-3.8-flash", "claude:claude-haiku-4-5"],
    "learn": ["gemini:gemini-3.1-pro-preview", "gemini:gemini-3.8-flash"],
    "vision": ["gemini:gemini-3.8-flash", "claude:claude-sonnet-5-5"],
    # the nutrition agent (app/nutrition): food names, allergies and rules stay in India
    "nutrition": ["gemini:gemini-3.5-flash@asia-south1"],
    # voice-note script: two fast models asked at once (app.care.speech), the first good answer wins
    "speech": ["gemini:gemini-3.5-flash@asia-south1", "gemini:gemini-3.5-flash@global", "gemini:gemini-3.8-flash"],
}


def parse_route(spec: str) -> Route:
    provider, _, rest = spec.partition(":")
    model, _, location = rest.partition("@")
    return Route(provider=provider, model=model, location=location or "global")


def configured_roles() -> set[str]:
    """Roles with routes set in MODEL_ROUTES (not just defaults)."""
    override = os.getenv("MODEL_ROUTES", "").strip()
    try:
        return set(json.loads(override)) if override else set()
    except json.JSONDecodeError:
        return set()


GOOGLE_PROVIDERS = {"gemini", "claude"}  # claude here is Claude on Vertex
ROLES_IN_USE = ("brain", "worker", "extract", "classify", "judge", "judge_fast", "learn", "vision", "speech", "nutrition")


def google_bound_roles() -> list[str]:
    """Roles whose every route needs Google (Vertex): on a host without Google (the AWS copy) MODEL_ROUTES must move them."""
    return [r for r in ROLES_IN_USE if all(x.provider in GOOGLE_PROVIDERS for x in routes_for(r))]


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
        system_stable: list[str],
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
        # Gemini on Vertex runs on shared capacity: a 429 passes in seconds. A 60 s trip sent every reply of the next
        # minute to the slow Pro model (live 2026-10-09: 38-75 s replies).
        return 20.0
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
            from app.llm import google_wif
            from app.llm.providers.gemini import GeminiProvider

            # on AWS: federated credentials (no Google key); on GCP: None, the default chain
            _providers[name] = GeminiProvider(credentials=google_wif.credentials())
        elif name == "bedrock":
            from app.llm.providers.bedrock import BedrockProvider

            _providers[name] = BedrockProvider()
        else:
            raise KeyError(f"No provider registered for {name}")
    return _providers[name]


async def complete(
    role: str,
    *,
    system_stable: str | list[str],
    system_dynamic: str = "",
    messages: list[dict],
    tools: list[ToolSpec] | None = None,
    max_tokens: int = 8000,
    effort: str = "medium",
    timeout_s: float = 90.0,
    routes: list[Route] | None = None,
    essential: bool | None = None,
) -> LLMReply:
    """system_stable blocks are cached in order (persona first, then the family's care record);
    system_dynamic is per-turn and never cached."""
    stable = [system_stable] if isinstance(system_stable, str) else list(system_stable)
    candidates = routes or routes_for(role)
    from app.llm import spend

    try:
        mode = await spend.gate(role, essential)
    except spend.SpendCapReached as exc:
        raise AllModelsFailed(str(exc)) from exc
    if mode == "cheap":
        candidates, effort = spend.cheapest(candidates), "low"
    from app.llm import policy_block

    errors: list[str] = [f"{r.key}: provider refused by policy ({policy_block.blocked(r.provider)}); not sent" for r in candidates
                         if policy_block.blocked(r.provider)]
    candidates = [r for r in candidates if not policy_block.blocked(r.provider)]  # a policy refusal is never retried
    if not candidates:
        raise AllModelsFailed("; ".join(errors))
    live = [r for r in candidates if healthy(r)] or candidates  # all tripped: try anyway (never past a policy refusal)
    started = time.monotonic()

    async def call(route: Route) -> LLMReply:
        t0 = time.monotonic()
        try:
            reply = await asyncio.wait_for(
                _provider(route.provider).complete(
                    route,
                    system_stable=[scrub(b) for b in stable if b],
                    system_dynamic=scrub(system_dynamic),
                    messages=messages,
                    tools=tools or [],
                    max_tokens=max_tokens,
                    effort=effort,
                ),
                timeout=timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if route.provider == "gemini":
                from app.obs.emf import note
                status = getattr(exc, "status", None)
                note("Kavach/Gemini", role, errors=0 if status == 429 else 1, throttles=1 if status == 429 else 0,
                     latency_ms=(time.monotonic() - t0) * 1000)
            raise
        if route.provider == "gemini":
            from app.obs.emf import note
            note("Kavach/Gemini", role, latency_ms=(time.monotonic() - t0) * 1000)
        return reply

    async def first_of(route: Route) -> LLMReply:
        """The call, and when it is slow the same request once more on the same model: the first answer wins and the other
        is cancelled. Google's shared capacity has a long tail (live 13 Oct: one brain call is ~4 s at p50 but 12-40 s now
        and then); the same request is no different an answer, only sooner."""
        after = hedge_after(role)
        if not after or after >= timeout_s:
            return await call(route)
        tasks = [asyncio.ensure_future(call(route))]
        try:
            done, _ = await asyncio.wait(tasks, timeout=after)
            if done:
                return tasks[0].result()
            logger.warning("llm_hedge role=%s model=%s after=%.1fs", role, route.key, after)
            tasks.append(asyncio.ensure_future(call(route)))
            pending: set = set(tasks)
            error: BaseException | None = None
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception() is None:
                        return t.result()
                    error = t.exception()
            assert error is not None
            raise error
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    queue = list(live)
    while queue:
        route = queue.pop(0)
        try:
            reply = None
            for attempt, wait in enumerate((0.0, *RETRY_429_S), start=1):
                if wait:
                    await asyncio.sleep(wait)
                try:
                    reply = await first_of(route)
                    break
                except ModelUnavailable as exc:
                    if exc.status != 429:
                        raise
                    # Every 429 is logged, also the ones a retry clears (trust plan C5.1: measure before buying capacity).
                    logger.warning("llm_429 role=%s model=%s attempt=%d", role, route.key, attempt)
                    # Shared capacity usually frees within seconds: back off on the same model (health data stays in
                    # India, founder D5, so there is often no other region to fall to) rather than give a weaker answer.
                    if attempt > len(RETRY_429_S):
                        raise
            assert reply is not None
            inr = await spend.record(role, route.model, reply.usage or {})
            logger.info("llm role=%s model=%s usage=%s inr=%.3f", role, route.key, reply.usage, inr)
            from app.core import trace

            trace.note_timed("model", role, (time.monotonic() - started) * 1000)  # with any 429 wait and fallback
            return reply
        except ModelUnavailable as exc:
            trip(route, exc.status)
            errors.append(f"{route.key}: {exc.status} {str(exc)[:120]}")
            if policy_block.is_policy_refusal(exc.status, str(exc)):
                # the whole provider (project) refuses: no other route of it, no retry, for the block window
                policy_block.note(route.provider, str(exc))
                logger.error("llm policy refusal provider=%s role=%s: %s", route.provider, role, str(exc)[:160])
                queue = [r for r in queue if r.provider != route.provider]
                if not queue:
                    raise AllModelsFailed("; ".join(errors)) from exc
                continue
        except asyncio.TimeoutError:
            trip(route, None)
            errors.append(f"{route.key}: timeout")
        logger.warning("llm fallback role=%s from=%s reason=%s", role, route.key, errors[-1])
    raise AllModelsFailed("; ".join(errors))


RETRY_429_S: tuple[float, ...] = (1.5, 3.0, 6.0)  # about 10 s in all before the next route or AllModelsFailed


def hedge_after(role: str) -> float:
    """Seconds after which a slow call is sent once more (0: never). Only Saheli's own replies (the brain and brain_hard),
    where a person is waiting; BRAIN_HEDGE_AFTER_S sets it (10 s: past the p85 of a brain call, so about one call in seven
    or fewer)."""
    if role not in ("brain", "brain_hard"):
        return 0.0
    try:
        return max(0.0, float(os.getenv("BRAIN_HEDGE_AFTER_S", "10")))
    except ValueError:
        return 10.0


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"
