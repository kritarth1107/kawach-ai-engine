"""Gemini on Vertex AI (google-genai)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time

from google import genai
from google.genai import errors, types

from app.core.config import get_settings
from app.llm.router import LLMReply, ModelUnavailable, Route, ToolCall, ToolSpec, new_call_id, scrub

logger = logging.getLogger(__name__)

# The fixed prefix (persona, or whatever the caller put first, plus the tool schemas) is the same for every family.
# An explicit cache bills those tokens at the cached rate. Anything after the first block stays on the request.
_CACHE_TTL_S = 3600
_CACHE_MIN_CHARS = 8000
_gate: asyncio.Semaphore | None = None
_gate_n = 0

FOREIGN_SIGNATURE = b"skip_thought_signature_validator"

_LEVEL: dict[str, types.ThinkingLevel] = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
    "xhigh": types.ThinkingLevel.HIGH,
    "max": types.ThinkingLevel.HIGH,
}


def gemini_limit() -> asyncio.Semaphore:
    """How many Gemini calls may be in flight. Dream workers and live turns share this."""
    global _gate, _gate_n
    n = max(1, int(os.getenv("GEMINI_PARALLEL", "8")))
    if _gate is None or n != _gate_n:
        _gate = asyncio.Semaphore(n)
        _gate_n = n
    return _gate


def _tool_payload(tools: list[ToolSpec]) -> list[dict]:
    return [{"name": t.name, "description": t.description, "schema": t.schema} for t in tools]


class GeminiProvider:
    def __init__(self, credentials=None) -> None:
        self._credentials = credentials
        self._clients: dict[str, genai.Client] = {}
        self._prefix: dict[str, tuple[str, float]] = {}  # key -> (cache name or "", do not retry before)
        self._making: dict[str, asyncio.Task] = {}  # caches being created in the background

    def _client(self, location: str) -> genai.Client:
        if location not in self._clients:
            self._clients[location] = genai.Client(
                vertexai=True, project=get_settings().gcp_project_id, location=location, credentials=self._credentials
            )
        return self._clients[location]

    def _prefix_cache(self, route: Route, system: str, tools: list[ToolSpec]) -> tuple[str | None, str]:
        """(cache name, key). The name is None while the cache is off, too small, refused, or still being made.

        Never waits: a missing cache starts its creation in the background and this call goes out uncached (live replay
        16 Oct: caches.create sat for over 20 s inside the turn and every reply ran out of time)."""
        if os.getenv("GEMINI_EXPLICIT_CACHE", "off") != "on" or not system or not tools:
            return None, ""
        blob = system + json.dumps(_tool_payload(tools), sort_keys=True, default=str)
        if len(blob) < _CACHE_MIN_CHARS:
            return None, ""
        key = hashlib.sha256(f"{route.location}\n{route.model}\n{blob}".encode()).hexdigest()
        hit = self._prefix.get(key)
        now = time.monotonic()
        if hit and hit[1] > now:
            return (hit[0] or None), key
        if key not in self._making:
            task = asyncio.create_task(self._make_cache(key, route, system, tools))
            self._making[key] = task
            task.add_done_callback(lambda _t, k=key: self._making.pop(k, None))
        return None, key

    async def _make_cache(self, key: str, route: Route, system: str, tools: list[ToolSpec]) -> None:
        try:
            created = await asyncio.wait_for(self._client(route.location).aio.caches.create(
                model=route.model,
                config=types.CreateCachedContentConfig(
                    display_name="saheli-prefix",
                    system_instruction=system,
                    tools=[types.Tool(function_declarations=[
                        types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.schema)
                        for t in tools
                    ])],
                    ttl=f"{_CACHE_TTL_S}s",
                ),
            ), timeout=float(os.getenv("GEMINI_CACHE_CREATE_S", "30")))
        except (Exception, asyncio.CancelledError):  # noqa: BLE001 — a cache miss bills the prefix at full price; replies still go out
            logger.warning("gemini prefix cache was not created model=%s", route.model, exc_info=True)
            self._prefix[key] = ("", time.monotonic() + 600)
            return
        name = getattr(created, "name", None) or ""
        self._prefix[key] = (name, time.monotonic() + _CACHE_TTL_S - 300)

    def _drop_cache(self, key: str) -> None:
        """A cache Vertex no longer serves (expired early, deleted, refused): go uncached for 10 minutes."""
        if key:
            self._prefix[key] = ("", time.monotonic() + 600)

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
    ) -> LLMReply:
        # The first stable block plus the tool schemas are the cached prefix. The rest (this family's record, this
        # turn) changes every call and is not cached.
        prefix = system_stable[0] if system_stable else ""
        rest = "\n\n".join(p for p in (*system_stable[1:], system_dynamic) if p)
        cache_name, cache_key = self._prefix_cache(route, prefix, tools) if tools else (None, "")
        thinking = types.ThinkingConfig(thinking_level=_LEVEL.get(effort, types.ThinkingLevel.MEDIUM))
        no_auto = types.AutomaticFunctionCallingConfig(disable=True)

        def request(cached: str | None) -> tuple[types.GenerateContentConfig, list[types.Content]]:
            contents = to_gemini_contents(messages, route.model)
            if cached:
                config = types.GenerateContentConfig(
                    cached_content=cached, max_output_tokens=max_tokens, thinking_config=thinking,
                    automatic_function_calling=no_auto,
                )
                if rest:
                    lead = types.Part(text=scrub("(Context for this turn, not a message from the person)\n" + rest))
                    if contents and contents[0].role == "user" and contents[0].parts is not None:
                        contents[0].parts.insert(0, lead)
                    else:
                        contents.insert(0, types.Content(role="user", parts=[lead]))
                return config, contents
            config = types.GenerateContentConfig(
                system_instruction="\n\n".join(p for p in (*system_stable, system_dynamic) if p),
                max_output_tokens=max_tokens,
                thinking_config=thinking,
                automatic_function_calling=no_auto,
            )
            if tools:
                config.tools = [
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.schema)
                            for t in tools
                        ]
                    )
                ]
            return config, contents

        config, contents = request(cache_name)
        for attempt in range(2):
            try:
                async with gemini_limit():
                    resp = await self._client(route.location).aio.models.generate_content(
                        model=route.model, contents=contents, config=config
                    )
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if cache_name and code != 429:
                    # the cache, not the model: drop it and send the same call uncached once
                    logger.warning("gemini cached call failed (%s): retrying uncached", code)
                    self._drop_cache(cache_key)
                    cache_name = None
                    config, contents = request(None)
                    try:
                        async with gemini_limit():
                            resp = await self._client(route.location).aio.models.generate_content(
                                model=route.model, contents=contents, config=config
                            )
                    except errors.APIError as exc2:
                        code2 = getattr(exc2, "code", None)
                        if code2 in (403, 404, 429) or (code2 or 500) >= 500:
                            raise ModelUnavailable(code2, str(exc2)) from exc2
                        raise
                elif code in (403, 404, 429) or (code or 500) >= 500:
                    raise ModelUnavailable(code, str(exc)) from exc
                else:
                    raise
            cand = (resp.candidates or [None])[0]
            # A malformed tool call is a one-off generation slip; one retry on the same model usually fixes it.
            if attempt == 0 and "MALFORMED" in str(getattr(cand, "finish_reason", "")):
                continue
            break

        cand = (resp.candidates or [None])[0]
        parts = (cand.content.parts if cand and cand.content else None) or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        calls = [
            ToolCall(id=p.function_call.id or new_call_id(), name=p.function_call.name, args=dict(p.function_call.args or {}))
            for p in parts
            if p.function_call
        ]
        if not text and not calls:
            reason = str(getattr(cand, "finish_reason", "") or "empty")
            raise ModelUnavailable(None, f"empty reply ({reason})")
        meta = resp.usage_metadata
        usage = {
            "in": getattr(meta, "prompt_token_count", 0) or 0,
            "out": getattr(meta, "candidates_token_count", 0) or 0,
            "cache_read": getattr(meta, "cached_content_token_count", 0) or 0,
            "think": getattr(meta, "thoughts_token_count", 0) or 0,  # billed as output
        }
        # Keep the parts (with thought signatures) to replay on the same model; ids must match the calls.
        raw = {"parts": parts, "ids": [c.id for c in calls]}
        return LLMReply(text=text, tool_calls=calls, model=route.model, raw=raw, usage=usage, stop=str(getattr(cand, "finish_reason", "")))


def to_gemini_contents(messages: list[dict], model: str) -> list[types.Content]:
    out: list[types.Content] = []
    names: dict[str, str] = {}

    def push(role: str, parts: list[types.Part]) -> None:
        if out and out[-1].role == role:
            out[-1].parts.extend(parts)
        else:
            out.append(types.Content(role=role, parts=parts))

    for m in messages:
        if m["role"] == "user":
            parts = []
            for part in m["content"]:
                if part["type"] == "text":
                    parts.append(types.Part(text=scrub(part["text"])))
                elif part["type"] == "image":
                    parts.append(types.Part.from_bytes(data=base64.b64decode(part["data"]), mime_type=part["mime"]))
            push("user", parts)
        elif m["role"] == "assistant":
            for c in m.get("tool_calls", []):
                names[c.id] = c.name
            raw_model, raw = m.get("raw") or (None, None)
            if raw_model == model and isinstance(raw, dict) and raw.get("parts"):
                push("model", list(raw["parts"]))
                continue
            parts = [types.Part(text=scrub(m["text"]))] if m.get("text") else []
            # Calls made by another model carry no thought signature; Gemini accepts this documented
            # placeholder for history it did not produce.
            parts += [
                types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.args), thought_signature=FOREIGN_SIGNATURE)
                for c in m.get("tool_calls", [])
            ]
            push("model", parts or [types.Part(text="…")])
        elif m["role"] == "tool":
            parts = []
            for r in m["results"]:
                try:
                    body = json.loads(r["content"])
                    if not isinstance(body, dict):
                        body = {"result": body}
                except (json.JSONDecodeError, TypeError):
                    body = {"result": scrub(r["content"])}
                if r.get("is_error"):
                    body = {"error": body}
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(id=r["id"], name=r.get("name") or names.get(r["id"], ""), response=body)
                    )
                )
            push("user", parts)
    return out
