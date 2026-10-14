"""Bedrock provider: neutral messages ↔ Converse shape, tool calls, errors → router fallbacks."""

import asyncio
import base64

import pytest

from app.llm.providers.bedrock import BedrockProvider, _unavailable, to_converse_messages
from app.llm.router import ModelUnavailable, ToolCall, ToolSpec, parse_route


class FakeClient:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def converse(self, **req):
        self.calls.append(req)
        if self.exc:
            raise self.exc
        return self.resp


class ClientError(Exception):
    def __init__(self, code, status):
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


def run(provider, route="bedrock:openai.gpt-oss-120b-1:0@ap-south-1", **kw):
    args = dict(system_stable=["persona"], system_dynamic="today", messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                tools=[], max_tokens=100, effort="low")
    args.update(kw)
    return asyncio.run(provider.complete(parse_route(route), **args))


def test_tool_call_and_usage_come_back_in_router_shape():
    resp = {"output": {"message": {"content": [{"text": "ok"}, {"toolUse": {"toolUseId": "t1", "name": "log_dose", "input": {"medicine": "X"}}}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 0}, "stopReason": "tool_use"}
    fake = FakeClient(resp)
    r = run(BedrockProvider(lambda region: fake), tools=[ToolSpec("log_dose", "d", {"type": "object", "properties": {}})])
    assert r.text == "ok" and r.tool_calls[0].name == "log_dose" and r.tool_calls[0].args == {"medicine": "X"}
    assert r.usage == {"in": 10, "out": 5, "cache_read": 0}
    req = fake.calls[0]
    assert req["system"] == [{"text": "persona"}, {"text": "today"}]
    assert req["toolConfig"]["tools"][0]["toolSpec"]["name"] == "log_dose"
    assert req["additionalModelRequestFields"] == {"reasoning_effort": "low"}  # OpenAI models take an effort


def test_non_openai_models_get_no_effort_field():
    fake = FakeClient({"output": {"message": {"content": [{"text": "x"}]}}, "usage": {}})
    run(BedrockProvider(lambda region: fake), route="bedrock:in.moonshotai.kimi-k3@ap-south-1")
    assert "additionalModelRequestFields" not in fake.calls[0]


@pytest.mark.parametrize("code,status", [("ThrottlingException", 429), ("AccessDeniedException", 403), ("ResourceNotFoundException", 404),
                                         ("ServiceUnavailableException", 503)])
def test_errors_map_to_router_statuses(code, status):
    fake = FakeClient(exc=ClientError(code, 400))
    with pytest.raises(ModelUnavailable) as e:
        run(BedrockProvider(lambda region: fake))
    assert e.value.status == status


def test_blocked_reply_without_content_falls_over():
    fake = FakeClient({"output": {"message": {"content": []}}, "usage": {}, "stopReason": "content_filtered"})
    with pytest.raises(ModelUnavailable):
        run(BedrockProvider(lambda region: fake))


def test_history_conversion_tool_round_trip_and_images():
    img = base64.b64encode(b"\x89PNG").decode()
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "photo"}, {"type": "image", "mime": "image/png", "data": img},
                                     {"type": "image", "mime": "audio/ogg", "data": img}]},
        {"role": "assistant", "text": "", "tool_calls": [ToolCall("c1", "log_dose", {"m": 1})], "raw": ("other-model", None)},
        {"role": "tool", "results": [{"id": "c1", "name": "log_dose", "content": "saved", "is_error": False}]},
    ]
    out = to_converse_messages(msgs, "openai.gpt-oss-120b-1:0")
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[0]["content"][1]["image"]["format"] == "png" and out[0]["content"][1]["image"]["source"]["bytes"] == b"\x89PNG"
    assert "cannot open" in out[0]["content"][2]["text"]
    assert out[1]["content"][0]["toolUse"] == {"toolUseId": "c1", "name": "log_dose", "input": {"m": 1}}
    assert out[2]["content"][0]["toolResult"]["status"] == "success"


def test_history_starting_with_assistant_gets_a_user_turn_first():
    out = to_converse_messages([{"role": "assistant", "text": "hello", "tool_calls": []}], "m")
    assert out[0]["role"] == "user"


def test_unavailable_for_plain_connection_errors_has_no_status():
    assert _unavailable(ConnectionError("down")).status is None
