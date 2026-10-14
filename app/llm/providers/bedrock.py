"""Amazon Bedrock (Converse API): one shape for every Bedrock model (OpenAI, Kimi, Qwen, …).

Route spec: "bedrock:<model or inference-profile id>@<region>", e.g. "bedrock:in.moonshotai.kimi-k3@ap-south-1".
Only India-resident ids (in.* profiles or plain model ids) are allowed by the AWS account's IAM; global.* and apac.*
profiles are refused there. Credentials come from the default AWS chain (task role on ECS, SSO profile locally).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from app.llm.router import LLMReply, ModelUnavailable, Route, ToolCall, ToolSpec, new_call_id, scrub

logger = logging.getLogger(__name__)

# Models whose Converse request takes a reasoning effort; the others reject unknown fields.
_EFFORT_PREFIXES = ("openai.", "in.openai.")
# Prompt caching (cachePoint blocks) is only accepted by some Bedrock models; elsewhere it is a validation error.
_CACHE_PREFIXES: tuple[str, ...] = ()  # none of the tested models accepts cachePoint yet (Kimi K3: ValidationException, 14 Oct)
_IMAGE_FORMATS = {"image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpeg", "image/gif": "gif", "image/webp": "webp"}


def _supports(model: str, prefixes: tuple[str, ...]) -> bool:
    return model.startswith(prefixes)


class BedrockProvider:
    def __init__(self, client_factory=None) -> None:
        self._factory = client_factory
        self._clients: dict[str, Any] = {}

    def _client(self, region: str):
        if region not in self._clients:
            if self._factory:
                self._clients[region] = self._factory(region)
            else:
                import boto3
                from botocore.config import Config

                # One SDK retry only: the router falls over to the next model instead of waiting.
                self._clients[region] = boto3.client(
                    "bedrock-runtime", region_name=region, config=Config(retries={"max_attempts": 2, "mode": "standard"}, read_timeout=120)
                )
        return self._clients[region]

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
        region = "ap-south-1" if route.location in ("", "global") else route.location
        cache = _supports(route.model, _CACHE_PREFIXES)
        system: list[dict] = [{"text": b} for b in system_stable if b]
        if cache and system:
            system.append({"cachePoint": {"type": "default"}})
        if system_dynamic:
            system.append({"text": system_dynamic})
        req: dict[str, Any] = {
            "modelId": route.model,
            "messages": to_converse_messages(messages, route.model),
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            req["system"] = system
        if tools:
            specs: list[dict] = [
                {"toolSpec": {"name": t.name, "description": t.description or t.name, "inputSchema": {"json": t.schema}}} for t in tools
            ]
            if cache:
                specs.append({"cachePoint": {"type": "default"}})
            req["toolConfig"] = {"tools": specs}
        if _supports(route.model, _EFFORT_PREFIXES):
            req["additionalModelRequestFields"] = {"reasoning_effort": effort if effort in ("low", "medium", "high") else "low"}

        client = self._client(region)
        try:
            resp = await asyncio.to_thread(client.converse, **req)
        except Exception as exc:  # botocore ClientError and connection errors
            raise _unavailable(exc) from exc

        content = (resp.get("output") or {}).get("message", {}).get("content", []) or []
        text = "".join(b["text"] for b in content if "text" in b)
        calls = [
            ToolCall(id=b["toolUse"].get("toolUseId") or new_call_id(), name=b["toolUse"]["name"], args=dict(b["toolUse"].get("input") or {}))
            for b in content
            if "toolUse" in b
        ]
        u = resp.get("usage") or {}
        usage = {
            "in": int(u.get("inputTokens", 0)) + int(u.get("cacheReadInputTokens", 0) or 0) + int(u.get("cacheWriteInputTokens", 0) or 0),
            "out": int(u.get("outputTokens", 0)),
            "cache_read": int(u.get("cacheReadInputTokens", 0) or 0),
        }
        stop = resp.get("stopReason", "") or ""
        if stop in ("guardrail_intervened", "content_filtered") and not calls and not text:
            # Let another model try; the brain must still answer the family.
            raise ModelUnavailable(None, f"bedrock stop {stop}")
        raw = [b for b in content if "reasoningContent" not in b]
        return LLMReply(text=text, tool_calls=calls, model=route.model, raw=raw, usage=usage, stop=stop)


def _unavailable(exc: Exception) -> ModelUnavailable:
    resp = getattr(exc, "response", None) or {}
    code = (resp.get("Error") or {}).get("Code", "")
    status = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    if code in ("ThrottlingException", "ServiceQuotaExceededException", "TooManyRequestsException"):
        status = 429
    elif code in ("AccessDeniedException",):
        status = 403
    elif code in ("ResourceNotFoundException",):
        status = 404
    elif code in ("ModelNotReadyException", "ServiceUnavailableException", "InternalServerException", "ModelTimeoutException"):
        status = 503
    return ModelUnavailable(status, f"{code or type(exc).__name__}: {str(exc)[:300]}")


def to_converse_messages(messages: list[dict], model: str) -> list[dict]:
    out: list[dict] = []

    def push(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        if m["role"] == "user":
            blocks: list[dict] = []
            for part in m["content"]:
                if part["type"] == "text":
                    if part["text"]:
                        blocks.append({"text": scrub(part["text"])})
                elif part["type"] == "image" and part["mime"] in _IMAGE_FORMATS:
                    blocks.append({"image": {"format": _IMAGE_FORMATS[part["mime"]], "source": {"bytes": base64.b64decode(part["data"])}}})
                elif part["type"] == "image" and part["mime"] == "application/pdf":
                    blocks.append({"document": {"format": "pdf", "name": "document", "source": {"bytes": base64.b64decode(part["data"])}}})
                elif part["type"] == "image":  # video or audio: this model cannot watch or hear it
                    blocks.append({"text": f"[a {part['mime'].split('/')[0]} was attached that this model cannot open; say so and ask what it shows]"})
            push("user", blocks or [{"text": "…"}])
        elif m["role"] == "assistant":
            raw_model, raw = m.get("raw") or (None, None)
            if raw_model == model and raw:
                push("assistant", [dict(b) for b in raw])
                continue
            blocks = [{"text": scrub(m["text"])}] if m.get("text") else []
            blocks += [{"toolUse": {"toolUseId": c.id, "name": c.name, "input": c.args}} for c in m.get("tool_calls", [])]
            push("assistant", blocks or [{"text": "…"}])
        elif m["role"] == "tool":
            push(
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": r["id"],
                            "content": [{"text": scrub(r["content"]) or "(empty)"}],
                            "status": "error" if r.get("is_error") else "success",
                        }
                    }
                    for r in m["results"]
                ],
            )
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"text": "…"}]})
    return out

