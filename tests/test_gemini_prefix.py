"""The fixed prefix is an explicit Gemini cache, and a cold chat does not send tools it cannot use yet."""

import asyncio

import pytest

from app.brain import tools
from app.llm import spend
from app.llm.providers import gemini
from app.llm.router import LLMReply, Route, ToolSpec


def test_every_call_starts_with_the_same_tool_prefix():
    """A situation only adds tools after the ones every call already sends."""
    cold = [t.name for t in tools.specs_for()]
    with_task = [t.name for t in tools.specs_for(open_tasks=True)]
    with_system = [t.name for t in tools.specs_for(system=True)]
    assert with_task[: len(cold)] == cold
    assert with_system[: len(cold)] == cold
    assert "task_input" in with_task[len(cold):]
    assert "start_task" in with_task[len(cold):]


def test_a_normal_turn_sends_the_core_and_stays_under_9k(monkeypatch):
    """Persona plus tools on a chit-chat, counted the same way as the old 37k (characters / 4)."""
    monkeypatch.setenv("LLM_PRICES_INR", '{"gemini-3.5-flash":[64,383]}')
    import json

    from app.brain.persona import REPLY_FORMAT, for_groups
    cold = {t.name for t in tools.specs_for()}
    assert {"remember", "log_dose", "log_vital", "set_reminder", "send_message", "alert_caregiver", "need_tools"} <= cold
    assert "log_food" not in cold and "start_task" not in cold and "nutrition_report" not in cold
    assert "daily_brief" not in cold and "change_reminder" not in cold
    assert "start_task" in {t.name for t in tools.specs_for(situation="order")}
    assert "note_trip" in {t.name for t in tools.specs_for(situation="order")}
    assert "log_food" in {t.name for t in tools.specs_for(images=True)}
    assert "daily_brief" in {t.name for t in tools.specs_for(onboarding=True)}
    assert "change_reminder" in {t.name for t in tools.specs_for(situation="med_question")}
    assert "task_input" not in {t.name for t in tools.specs_for(situation="order")}
    assert "task_input" in {t.name for t in tools.specs_for(open_tasks=True)}
    assert "weekly_report" in {t.name for t in tools.specs_for(system=True)}
    groups = tools.groups_for()
    text = for_groups(groups) + "\n\n" + REPLY_FORMAT
    tool_tok = len(json.dumps([
        {"n": t.name, "d": t.description, "s": t.schema} for t in tools.specs_for()
    ])) // 4
    normal = len(text) // 4 + tool_tok
    assert normal < 9000
    # The family's record stays in the system instruction, after this prefix.
    assert "CARE RECORD" in text
    before = spend.cost_inr("gemini-3.5-flash", {"in": 40000})
    after = spend.cost_inr("gemini-3.5-flash", {"in": normal + 1000, "cache_read": normal})
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

    def cached(route, system, tools):
        return "cachedContents/prefix", "k"

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

    def cached(route, system, tools):
        return None, ""

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


def _client_with(models, caches=None):
    class _Aio:
        pass

    _Aio.models = models
    _Aio.caches = caches

    class _Client:
        aio = _Aio()

    return _Client()


@pytest.mark.asyncio
async def test_a_slow_cache_never_holds_up_a_reply(monkeypatch):
    """Live replay 16 Oct: caches.create sat past the 20 s call cap and every reply ran out of time."""
    monkeypatch.setenv("GEMINI_EXPLICIT_CACHE", "on")
    monkeypatch.setenv("GEMINI_CACHE_CREATE_S", "0.2")
    gemini._gate = None
    provider = gemini.GeminiProvider()
    seen = []

    class _Caches:
        async def create(self, *, model, config):
            await asyncio.sleep(5)

    class _Models:
        async def generate_content(self, *, model, contents, config):
            seen.append(config.cached_content)
            return _Resp()

    monkeypatch.setattr(provider, "_client", lambda location: _client_with(_Models(), _Caches()))
    route = Route(provider="gemini", model="gemini-3.5-flash", location="asia-south1")
    big = [ToolSpec("remember", "save " * 3000, {"type": "object"})]
    reply = await asyncio.wait_for(provider.complete(
        route, system_stable=["PERSONA"], system_dynamic="NOW", messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tools=big, max_tokens=20, effort="low"), timeout=1)
    assert isinstance(reply, LLMReply) and seen == [None]
    await asyncio.sleep(0.4)  # the background create times out and is remembered as refused
    assert not provider._making and all(v[0] == "" for v in provider._prefix.values())


@pytest.mark.asyncio
async def test_a_cache_vertex_no_longer_serves_is_dropped_and_the_call_goes_uncached(monkeypatch):
    from google.genai import errors

    gemini._gate = None
    provider = gemini.GeminiProvider()
    seen = []

    class _Models:
        async def generate_content(self, *, model, contents, config):
            seen.append(config.cached_content)
            if config.cached_content:
                raise errors.ClientError(404, {"error": {"code": 404, "message": "cached content not found", "status": "NOT_FOUND"}})
            return _Resp()

    monkeypatch.setattr(provider, "_prefix_cache", lambda route, system, tools: ("cachedContents/gone", "k"))
    monkeypatch.setattr(provider, "_client", lambda location: _client_with(_Models()))
    route = Route(provider="gemini", model="gemini-3.5-flash", location="asia-south1")
    reply = await provider.complete(
        route, system_stable=["PERSONA"], system_dynamic="NOW", messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tools=[ToolSpec("remember", "save", {"type": "object"})], max_tokens=20, effort="low")
    assert isinstance(reply, LLMReply)
    assert seen == ["cachedContents/gone", None]
    assert provider._prefix["k"][0] == ""


@pytest.mark.asyncio
async def test_an_uncached_call_keeps_the_family_in_the_system_instruction(monkeypatch):
    """Moving the family record into the user part dropped replay cases. It stays after the persona."""
    monkeypatch.delenv("GEMINI_EXPLICIT_CACHE", raising=False)
    gemini._gate = None
    provider = gemini.GeminiProvider()
    seen = {}

    class _Models:
        async def generate_content(self, *, model, contents, config):
            seen["config"] = config
            seen["contents"] = contents
            return _Resp()

    monkeypatch.setattr(provider, "_client", lambda location: _client_with(_Models()))
    route = Route(provider="gemini", model="gemini-3.5-flash", location="asia-south1")
    await provider.complete(
        route, system_stable=["PERSONA", "FAMILY"], system_dynamic="NOW",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tools=[ToolSpec("remember", "save", {"type": "object"})], max_tokens=20, effort="low",
    )
    text = seen["config"].system_instruction
    assert text.startswith("PERSONA") and "FAMILY" in text and "NOW" in text
    assert seen["config"].cached_content is None
    assert "FAMILY" not in (seen["contents"][0].parts[0].text or "")


def test_the_prefix_cache_is_off_unless_turned_on(monkeypatch):
    monkeypatch.delenv("GEMINI_EXPLICIT_CACHE", raising=False)
    provider = gemini.GeminiProvider()
    route = Route(provider="gemini", model="gemini-3.5-flash", location="asia-south1")
    assert provider._prefix_cache(route, "P" * 20000, [ToolSpec("remember", "save", {"type": "object"})]) == (None, "")
