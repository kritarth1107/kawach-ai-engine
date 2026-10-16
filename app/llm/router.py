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
    global _hedge_off_until
    _open_until.clear()
    _hedge_off_until = 0.0


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
        # wait_for waits until a cancelled call actually stops. A Vertex request that never comes back, and does not
        # notice the cancellation, held the whole caller: recall on 16 Oct sat for the rest of its 120 minutes after
        # two extract calls and wrote nothing. asyncio.wait returns at the timeout either way. The request may still
        # be in flight (and may keep one Gemini slot until it ends); this caller does not.
        work = asyncio.ensure_future(
            _provider(route.provider).complete(
                route,
                system_stable=[scrub(b) for b in stable if b],
                system_dynamic=scrub(system_dynamic),
                messages=messages,
                tools=tools or [],
                max_tokens=max_tokens,
                effort=effort,
            )
        )
        try:
            done, _pending = await asyncio.wait({work}, timeout=timeout_s)
            if work not in done:
                work.cancel()
                raise TimeoutError(f"{route.key} still running after {timeout_s:.0f}s")
            reply = work.result()
        except asyncio.CancelledError:
            work.cancel()
            raise
        except Exception as exc:
            if route.provider == "gemini":
                from app.obs.emf import note
                status = getattr(exc, "status", None)
                note("Kavach/Gemini", role, errors=0 if status == 429 else 1, throttles=1 if status == 429 else 0,
                     latency_ms=(time.monotonic() - t0) * 1000, region=_metric_region(route))
            raise
        if route.provider == "gemini":
            from app.obs.emf import note
            note("Kavach/Gemini", role, latency_ms=(time.monotonic() - t0) * 1000, region=_metric_region(route))
        return reply

    async def first_of(route: Route) -> LLMReply:
        """The call, and when it is slow the same request once more: the first answer wins and the other is cancelled.
        Google's shared capacity has a long tail (live 13 Oct: one brain call is ~4 s at p50 but 12-40 s now and then;
        16 Oct afternoon: Mumbai brain calls sat until the 20 s cap, then Singapore answered in 4-13 s). With the
        Singapore flag on, the copy goes to Singapore (another capacity pool, so it does not crowd Mumbai) and counts as
        this request's one Singapore call; otherwise it is the same route, when BRAIN_HEDGE_AFTER_S asks for it."""
        nonlocal tried_sg
        sg = _sg_route(route) if not tried_sg else None
        after = sg_hedge_after(role) if sg is not None else hedge_after(role)
        if not after or after >= timeout_s:
            return await call(route)
        tasks = [asyncio.ensure_future(call(route))]
        try:
            done, _ = await asyncio.wait(tasks, timeout=after)
            if done:
                return tasks[0].result()
            logger.warning("llm_hedge role=%s model=%s to=%s after=%.1fs", role, route.key, (sg or route).key, after)
            if sg is not None:
                tried_sg = True
            tasks.append(asyncio.ensure_future(call(sg or route)))
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
    tried_sg = False

    async def overflow(failed: Route):
        """One Singapore call for this request. The next request starts in Mumbai again."""
        nonlocal tried_sg
        if tried_sg:
            return None
        tried_sg = True
        return await _singapore(role, failed, call, errors, started)

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
                    note_throttled()
                    # Mumbai first. One overflow call when the founder has allowed it; otherwise back off here.
                    alt = await overflow(route)
                    if alt is not None:
                        return alt
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
            if exc.status == 429:
                alt = await overflow(route)
                if alt is not None:
                    return alt
        except asyncio.TimeoutError:
            trip(route, None)
            errors.append(f"{route.key}: timeout")
            alt = await overflow(route)
            if alt is not None:
                return alt
        logger.warning("llm fallback role=%s from=%s reason=%s", role, route.key, errors[-1])
    raise AllModelsFailed("; ".join(errors))


def sg_enabled() -> bool:
    """One Gemini call in Singapore after this Mumbai call gets a 429 or a timeout. The next call starts in Mumbai.
    Off unless GEMINI_SG_FALLBACK=on (the live task and Terraform set that; founder 15 Oct)."""
    return os.getenv("GEMINI_SG_FALLBACK", "off").strip().lower() in ("on", "1", "true")


def _metric_region(route: Route) -> str | None:
    """Set only when the call left India, so Kavach/Gemini Role+Region counts those and no others."""
    return route.location if route.location == "asia-southeast1" else None


def _sg_route(route: Route) -> Route | None:
    if not sg_enabled() or route.provider != "gemini" or route.location != "asia-south1":
        return None
    return Route(route.provider, route.model, "asia-southeast1")


async def _singapore(role: str, route: Route, call, errors: list[str], started: float):
    """A single call in asia-southeast1 after this Mumbai route failed. None when the flag is off or that call fails too."""
    alt = _sg_route(route)
    if alt is None:
        return None
    logger.warning("llm singapore fallback role=%s from=%s", role, route.key)
    try:
        reply = await call(alt)
    except (ModelUnavailable, asyncio.TimeoutError) as exc:
        errors.append(f"{alt.key}: {type(exc).__name__} {str(exc)[:80]}")
        return None
    from app.llm import spend

    inr = await spend.record(role, alt.model, reply.usage or {})
    logger.info("llm role=%s model=%s usage=%s inr=%.3f", role, alt.key, reply.usage, inr)
    from app.core import trace

    trace.note_timed("model", role, (time.monotonic() - started) * 1000)
    return reply


RETRY_429_S: tuple[float, ...] = (1.5, 3.0, 6.0)  # about 10 s in all before the next route or AllModelsFailed

# A duplicate of a call that is merely slow crowds the same shared capacity and comes back as a 429 (live 15 Oct: five
# hedges in five minutes, then 429s, and the 22 s copy never returned before the person had given up). Off. Set
# BRAIN_HEDGE_AFTER_S to turn a single copy back on. A 429 still pauses it.
HEDGE_DEFAULT_S = 0.0
HEDGE_PAUSE_S = 120.0
_hedge_off_until = 0.0


def note_throttled() -> None:
    """A 429 just happened: do not start a second copy of any brain call until the shared capacity has had a moment."""
    global _hedge_off_until
    _hedge_off_until = time.monotonic() + HEDGE_PAUSE_S


def hedge_after(role: str) -> float:
    """Seconds after which a slow call is sent once more (0: never, the default). Only Saheli's own replies (the brain
    and brain_hard), where a person is waiting. BRAIN_HEDGE_AFTER_S turns it on. A 429 turns it back off for a while."""
    if role not in ("brain", "brain_hard"):
        return 0.0
    if time.monotonic() < _hedge_off_until:
        return 0.0
    try:
        return max(0.0, float(os.getenv("BRAIN_HEDGE_AFTER_S", str(HEDGE_DEFAULT_S))))
    except ValueError:
        return HEDGE_DEFAULT_S


def sg_hedge_after(role: str) -> float:
    """Seconds after which a slow Mumbai reply call also goes to Singapore (0: never). Only Saheli's own replies, and only
    with GEMINI_SG_FALLBACK on. A Mumbai 429 does not pause it: Singapore is a different pool."""
    if role not in ("brain", "brain_hard") or not sg_enabled():
        return 0.0
    try:
        return max(0.0, float(os.getenv("BRAIN_SG_HEDGE_AFTER_S", "7")))
    except ValueError:
        return 7.0


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"
