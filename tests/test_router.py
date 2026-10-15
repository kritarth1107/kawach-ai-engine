import pytest

from app.llm import router
from app.llm.providers.claude import to_claude_messages
from app.llm.providers.gemini import to_gemini_contents
from app.llm.router import LLMReply, ModelUnavailable, Route, ToolCall


class Scripted:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def complete(self, route, **kw):
        self.calls.append(route.model)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return LLMReply(text=out, tool_calls=[], model=route.model)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(router, "RETRY_429_S", (0, 0, 0))
    router.reset_breakers()
    yield
    router.reset_breakers()


ROUTES = [Route("a", "m1"), Route("b", "m2")]
MSG = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


async def test_falls_back_on_rate_limit_and_skips_tripped_model():
    a, b = Scripted([ModelUnavailable(429, "quota")] * 4), Scripted(["from b", "again b"])
    router.register_provider("a", a)
    router.register_provider("b", b)
    r = await router.complete("brain", system_stable="s", messages=MSG, routes=ROUTES)
    assert r.text == "from b" and r.model == "m2"
    r = await router.complete("brain", system_stable="s", messages=MSG, routes=ROUTES)
    assert r.text == "again b"
    assert a.calls == ["m1"] * 4  # backed off three times, then the breaker is open: not tried on the second call


async def test_a_longer_rate_limit_is_waited_out_on_the_same_model():
    """Trust plan C5 (founder D5: health data stays in India, so Mumbai Flash is often the only route): back off on the
    same model before falling to another route or a holding line."""
    a, b = Scripted([ModelUnavailable(429, "q"), ModelUnavailable(429, "q"), ModelUnavailable(429, "q"), "from a"]), Scripted(["from b"])
    router.register_provider("a", a)
    router.register_provider("b", b)
    r = await router.complete("brain", system_stable="s", messages=MSG, routes=ROUTES)
    assert r.text == "from a" and b.calls == [] and a.calls == ["m1"] * 4


async def test_a_passing_rate_limit_is_retried_on_the_same_fast_model():
    """Live 2026-10-09: one 429 on Flash sent replies to the slow Pro model (38-75 s). A short retry usually works."""
    a, b = Scripted([ModelUnavailable(429, "quota"), "from a"]), Scripted(["from b"])
    router.register_provider("a", a)
    router.register_provider("b", b)
    r = await router.complete("brain", system_stable="s", messages=MSG, routes=ROUTES)
    assert r.text == "from a" and b.calls == [] and router.healthy(ROUTES[0])


async def test_all_failed_raises():
    router.register_provider("a", Scripted([ModelUnavailable(500, "x")]))
    router.register_provider("b", Scripted([ModelUnavailable(404, "y")]))
    with pytest.raises(router.AllModelsFailed):
        await router.complete("brain", system_stable="s", messages=MSG, routes=ROUTES)


async def test_when_every_model_is_tripped_it_still_tries():
    for r in ROUTES:
        router.trip(r, 429)
    router.register_provider("a", Scripted(["ok"]))
    r = await router.complete("brain", system_stable="s", messages=MSG, routes=ROUTES)
    assert r.text == "ok"


def test_routes_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["gemini:gemini-3.8-flash@global"]}')
    assert router.routes_for("brain") == [Route("gemini", "gemini-3.8-flash", "global")]


def _history(raw_model, raw):
    call = ToolCall(id="call_1", name="get_schedule", args={"day": "today"})
    return [
        {"role": "user", "content": [{"type": "text", "text": "aaj kya hai \ud83d"}]},
        {"role": "assistant", "text": "", "tool_calls": [call], "raw": (raw_model, raw)},
        {"role": "tool", "results": [{"id": "call_1", "name": "get_schedule", "content": '{"items": []}'}]},
    ]


def test_claude_replays_own_blocks_and_rebuilds_foreign_ones():
    own = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "tool_use", "id": "call_1", "name": "get_schedule", "input": {}}]
    msgs = to_claude_messages(_history("claude-opus-5-5", own), "claude-opus-5-5")
    assert msgs[1]["content"] == own
    assert "\ud83d" not in msgs[0]["content"][0]["text"]
    msgs = to_claude_messages(_history("gemini-3.1-pro-preview", {"parts": []}), "claude-opus-5-5")
    assert msgs[1]["content"] == [{"type": "tool_use", "id": "call_1", "name": "get_schedule", "input": {"day": "today"}}]
    assert msgs[2]["content"][0]["type"] == "tool_result"


def test_gemini_rebuilds_tool_call_and_response():
    contents = to_gemini_contents(_history("claude-opus-5-5", []), "gemini-3.1-pro-preview")
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "get_schedule"
    resp = contents[2].parts[0].function_response
    assert resp.name == "get_schedule" and resp.response == {"items": []}


class Timed:
    """A provider whose calls take the given seconds (a value, or an exception raised after the wait)."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = 0
        self.cancelled = 0

    async def complete(self, route, **kw):
        import asyncio

        delay, out = self.plan.pop(0)
        self.calls += 1
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if isinstance(out, Exception):
            raise out
        return LLMReply(text=out, tool_calls=[], model=route.model)


async def test_a_slow_brain_call_is_sent_once_more_and_the_first_answer_wins(monkeypatch):
    """Reply speed: Google's shared capacity has a long tail (a brain call ~4 s at p50, 12-40 s now and then). Past
    BRAIN_HEDGE_AFTER_S the same request goes once more on the same model; the first answer wins, the other is cancelled."""
    import asyncio

    from app.core import trace

    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0.05")
    a = Timed([(0.5, "slow"), (0.01, "fast")])
    router.register_provider("a", a)
    with trace.collect_timing() as timing:
        r = await router.complete("brain", system_stable="s", messages=MSG, routes=[Route("a", "m1")])
    await asyncio.sleep(0.01)  # the losing copy's cancellation lands
    assert r.text == "fast" and a.calls == 2 and a.cancelled == 1
    assert timing["model"][0][0] == "brain" and timing["model"][0][1] < 400


async def test_a_quick_call_is_never_sent_twice_and_other_roles_never_hedge(monkeypatch):
    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0.2")
    a = Timed([(0.01, "quick")])
    router.register_provider("a", a)
    assert (await router.complete("brain", system_stable="s", messages=MSG, routes=[Route("a", "m1")])).text == "quick"
    assert a.calls == 1
    b = Timed([(0.4, "judge slow")])
    router.register_provider("b", b)
    assert (await router.complete("judge_fast", system_stable="s", messages=MSG, routes=[Route("b", "m2")])).text == "judge slow"
    assert b.calls == 1
    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0")  # off
    c = Timed([(0.3, "only one")])
    router.register_provider("c", c)
    assert (await router.complete("brain", system_stable="s", messages=MSG, routes=[Route("c", "m3")])).text == "only one"
    assert c.calls == 1


async def test_when_the_slow_call_fails_the_second_one_still_answers(monkeypatch):
    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0.05")
    a = Timed([(0.1, ModelUnavailable(500, "boom")), (0.2, "second")])
    router.register_provider("a", a)
    r = await router.complete("brain", system_stable="s", messages=MSG, routes=[Route("a", "m1")])
    assert r.text == "second" and a.calls == 2


async def test_both_copies_rate_limited_is_one_rate_limit_backed_off_as_before(monkeypatch):
    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0.05")
    a = Timed([(0.1, ModelUnavailable(429, "q")), (0.1, ModelUnavailable(429, "q")), (0.01, "after backoff")])
    router.register_provider("a", a)
    r = await router.complete("brain", system_stable="s", messages=MSG, routes=[Route("a", "m1")])
    assert r.text == "after backoff" and a.calls == 3


def test_a_rate_limit_pauses_the_hedge_and_the_default_waits_until_a_call_is_stuck(monkeypatch):
    monkeypatch.delenv("BRAIN_HEDGE_AFTER_S", raising=False)
    assert router.hedge_after("brain") == router.HEDGE_DEFAULT_S
    assert router.hedge_after("judge_fast") == 0
    router.note_throttled()
    assert router.hedge_after("brain") == 0


async def test_a_cancelled_turn_cancels_both_copies(monkeypatch):
    import asyncio

    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0.05")
    a = Timed([(5, "never"), (5, "never")])
    router.register_provider("a", a)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(router.complete("brain", system_stable="s", messages=MSG, routes=[Route("a", "m1")]), timeout=0.2)
    await asyncio.sleep(0.01)
    assert a.calls == 2 and a.cancelled == 2
