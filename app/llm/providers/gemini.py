"""Gemini on Vertex AI (google-genai)."""

from __future__ import annotations

import base64
import json

from google import genai
from google.genai import errors, types

from app.core.config import get_settings
from app.llm.router import LLMReply, ModelUnavailable, Route, ToolCall, ToolSpec, new_call_id, scrub

FOREIGN_SIGNATURE = b"skip_thought_signature_validator"

_LEVEL = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH", "max": "HIGH"}


class GeminiProvider:
    def __init__(self, credentials=None) -> None:
        self._credentials = credentials
        self._clients: dict[str, genai.Client] = {}

    def _client(self, location: str) -> genai.Client:
        if location not in self._clients:
            self._clients[location] = genai.Client(
                vertexai=True, project=get_settings().gcp_project_id, location=location, credentials=self._credentials
            )
        return self._clients[location]

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
        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(p for p in (*system_stable, system_dynamic) if p),
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=_LEVEL.get(effort, "MEDIUM")),
            # We run tools ourselves; never let the SDK try to call them.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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
        contents = to_gemini_contents(messages, route.model)
        for attempt in range(2):
            try:
                resp = await self._client(route.location).aio.models.generate_content(
                    model=route.model, contents=contents, config=config
                )
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code in (403, 404, 429) or (code or 500) >= 500:
                    raise ModelUnavailable(code, str(exc)) from exc
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
