"""Claude on Vertex AI Model Garden."""

from __future__ import annotations

import anthropic
from anthropic import AsyncAnthropicVertex

from app.core.config import get_settings
from app.llm.router import LLMReply, ModelUnavailable, Route, ToolCall, ToolSpec, scrub

# Haiku 4.5 rejects `effort`; the 5.x models take it (Opus 5.5 defaults to medium).
_NO_EFFORT = ("claude-haiku-4-5",)


class ClaudeProvider:
    def __init__(self, access_token: str | None = None) -> None:
        # access_token is for local runs without ADC; Cloud Run uses its service account.
        self._token = access_token
        self._clients: dict[str, AsyncAnthropicVertex] = {}

    def _client(self, location: str) -> AsyncAnthropicVertex:
        if location not in self._clients:
            # One SDK retry only: the router falls over to the next model instead of waiting.
            self._clients[location] = AsyncAnthropicVertex(
                project_id=get_settings().gcp_project_id, region=location, max_retries=1, access_token=self._token
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
        # Each stable block ends a cache breakpoint (max 4 per request, tools ride on the first).
        system = [{"type": "text", "text": b, "cache_control": {"type": "ephemeral"}} for b in system_stable[:3]]
        if system_stable[3:]:
            system.append({"type": "text", "text": "\n\n".join(system_stable[3:])})
        if system_dynamic:
            system.append({"type": "text", "text": system_dynamic})
        kwargs: dict = {
            "model": route.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": to_claude_messages(messages, route.model),
        }
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in tools]
        if route.model not in _NO_EFFORT:
            kwargs["output_config"] = {"effort": effort}
        try:
            resp = await self._client(route.location).messages.create(**kwargs)
        except anthropic.NotFoundError as exc:
            raise ModelUnavailable(404, str(exc)) from exc
        except anthropic.PermissionDeniedError as exc:
            raise ModelUnavailable(403, str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise ModelUnavailable(429, str(exc)) from exc
        except anthropic.InternalServerError as exc:
            raise ModelUnavailable(exc.status_code, str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise ModelUnavailable(None, str(exc)) from exc

        if resp.stop_reason == "refusal":
            # Let another model try; the brain must still answer the family.
            raise ModelUnavailable(None, "refusal")
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, args=dict(b.input or {})) for b in resp.content if b.type == "tool_use"]
        usage = {
            "in": resp.usage.input_tokens,
            "out": resp.usage.output_tokens,
            "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        }
        raw = [b.model_dump(exclude_none=True) for b in resp.content]
        return LLMReply(text=text, tool_calls=calls, model=route.model, raw=raw, usage=usage, stop=resp.stop_reason or "")


def to_claude_messages(messages: list[dict], model: str) -> list[dict]:
    out: list[dict] = []

    def push(role: str, blocks: list[dict]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        if m["role"] == "user":
            blocks = []
            for part in m["content"]:
                if part["type"] == "text":
                    blocks.append({"type": "text", "text": scrub(part["text"])})
                elif part["type"] == "image" and part["mime"].startswith("image/"):
                    blocks.append(
                        {"type": "image", "source": {"type": "base64", "media_type": part["mime"], "data": part["data"]}}
                    )
                elif part["type"] == "image" and part["mime"] == "application/pdf":
                    blocks.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": part["data"]}})
                elif part["type"] == "image":  # video or audio: this model cannot watch or hear it
                    blocks.append({"type": "text", "text": f"[a {part['mime'].split('/')[0]} was attached that this model cannot open; say so and ask what it shows]"})
            push("user", blocks)
        elif m["role"] == "assistant":
            raw_model, raw = m.get("raw") or (None, None)
            if raw_model == model and raw:
                # Same model: replay its blocks unchanged so thinking stays valid.
                push("assistant", list(raw))
                continue
            blocks = [{"type": "text", "text": scrub(m["text"])}] if m.get("text") else []
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in m.get("tool_calls", [])]
            push("assistant", blocks or [{"type": "text", "text": "…"}])
        elif m["role"] == "tool":
            push(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": r["id"],
                        "content": scrub(r["content"]),
                        "is_error": bool(r.get("is_error")),
                    }
                    for r in m["results"]
                ],
            )
    return out
