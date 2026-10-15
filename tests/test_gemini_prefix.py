"""The fixed prefix is an explicit Gemini cache, and a cold chat does not send tools it cannot use yet."""

import asyncio

import pytest

from app.brain import tools
from app.llm import spend
from app.llm.providers import gemini
from app.llm.router import LLMReply, Route, ToolSpec


def test_a_cold_chat_keeps_care_and_orders_and_leaves_a_running_task(monkeypatch):
    monkeypatch.setenv("LLM_PRICES_INR", '{"gemini-3.5-flash":[64,383]}')
    cold = {t.name for t in tools.specs_for()}
    assert "remember" in cold and "log_food" in cold and "start_task" in cold and "note_trip" in cold
    assert "daily_brief" in cold and "past_orders" in cold and "nutrition_report" in cold
    assert "task_input" not in cold and "cancel_task" not in cold
    assert "task_input" in {t.name for t in tools.specs_for(open_tasks=True)}
    assert "weekly_report" in {t.name for t in tools.specs_for(system=True)}
    before = spend.cost_inr("gemini-3.5-flash", {"in": 40000})
    # 15 Oct was about 40k input tokens a call at full price. The prefix is cached at a quarter; the family's own
    # record stays outside it. One call, about 1,000 fresh tokens.
    from app.brain.persona import PERSONA, REPLY_FORMAT
    import json
    prefix = (len(PERSONA) + len(REPLY_FORMAT) + len(json.dumps([
        {"n": t.name, "d": t.description, "s": t.schema} for t in tools.specs_for()
    ]))) // 4
    after = spend.cost_inr("gemini-3.5-flash", {"in": prefix + 1000, "cache_read": prefix})
    assert before > 2.4
    assert after < before / 3


class _Part:
    def __init__(self, text):
        self.text = text
        self.thought = False
        self.function_call = None


class _Resp:
    class _Cand:
        finish_reason = "STOP"

        class content:
            parts = [_Part("ok")]

    candidates = [_Cand()]

    class usage_metadata:
        prompt_token_count = 120
        candidates_token_count = 2
        cached_content_token_count = 100
        thoughts_token_count = 0


@pytest.mark.asyncio
async def test_a_cached_prefix_is_not_sent_again_as_system_and_tools(monkeypatch):
    monkeypatch.setenv("GEMINI_PARALLEL", "4")
    gemini._gate = None
    provider = gemini.GeminiProvider()
    seen = {}

    async def cached(route, system, tools):
        return "cachedContents/prefix"

    class _Models:
        async def generate_content(self, *, model, contents, config):
            seen["config"] = config
            seen["contents"] = contents
            return _Resp()

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    monkeypatch.setattr(provider, "_prefix_cache", cached)
    monkeypatch.setattr(provider, "_client", lambda location: _Client())
    route = Route(provider="gemini", model="gemini-3.5-flash", location="asia-south1")
    reply = await provider.complete(
        route, system_stable=["PERSONA", "FAMILY"], system_dynamic="NOW",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tools=[ToolSpec("remember", "save", {"type": "object"})], max_tokens=20, effort="low",
    )
    assert isinstance(reply, LLMReply)
    assert seen["config"].cached_content == "cachedContents/prefix"
    assert seen["config"].system_instruction is None
    assert seen["config"].tools is None
    lead = seen["contents"][0].parts[0].text
    assert "FAMILY" in lead and "NOW" in lead and "PERSONA" not in lead
    assert reply.usage["cache_read"] == 100


@pytest.mark.asyncio
async def test_gemini_calls_share_one_concurrency_gate(monkeypatch):
    monkeypatch.setenv("GEMINI_PARALLEL", "1")
    gemini._gate = None
    provider = gemini.GeminiProvider()
    peak = 0
    held = 0

    async def cached(route, system, tools):
        return None

    class _Models:
        async def generate_content(self, *, model, contents, config):
            nonlocal peak, held
            held += 1
            peak = max(peak, held)
            await asyncio.sleep(0.05)
            held -= 1
            return _Resp()

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    monkeypatch.setattr(provider, "_prefix_cache", cached)
    monkeypatch.setattr(provider, "_client", lambda location: _Client())
    route = Route(provider="gemini", model="gemini-3.5-flash", location="asia-south1")

    async def one():
        return await provider.complete(
            route, system_stable=["p"], system_dynamic="", messages=[], tools=[], max_tokens=8, effort="low",
        )

    await asyncio.gather(one(), one())
    assert peak == 1
