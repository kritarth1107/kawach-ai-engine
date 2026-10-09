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
    monkeypatch.setattr(router, "RETRY_429_S", 0)
    router.reset_breakers()
    yield
    router.reset_breakers()


ROUTES = [Route("a", "m1"), Route("b", "m2")]
MSG = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


async def test_falls_back_on_rate_limit_and_skips_tripped_model():
    a, b = Scripted([ModelUnavailable(429, "quota"), ModelUnavailable(429, "quota")]), Scripted(["from b", "again b"])
    router.register_provider("a", a)
    router.register_provider("b", b)
    r = await router.complete("brain", system_stable="s", messages=MSG, routes=ROUTES)
    assert r.text == "from b" and r.model == "m2"
    r = await router.complete("brain", system_stable="s", messages=MSG, routes=ROUTES)
    assert r.text == "again b"
    assert a.calls == ["m1", "m1"]  # one short retry, then the breaker is open: not tried on the second call


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
