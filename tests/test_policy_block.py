"""A provider's policy refusal is never retried (KP 14 Oct): one 403 "rejected in accordance with our terms of use" blocks
every route of that provider in the process for the window; nothing is sent meanwhile; other providers still work."""

import pytest

from app.llm import policy_block, router, spend
from app.llm.router import LLMReply, ModelUnavailable, parse_route


class Refuses:
    def __init__(self):
        self.calls = 0

    async def complete(self, route, **kw):
        self.calls += 1
        raise ModelUnavailable(403, "403 PERMISSION_DENIED. Request rejected in accordance with our terms of use and policies.")


class Answers:
    def __init__(self):
        self.calls = 0

    async def complete(self, route, **kw):
        self.calls += 1
        return LLMReply(text="ok", tool_calls=[], model=route.model)


@pytest.fixture(autouse=True)
def clean():
    policy_block.reset()
    router.reset_breakers()
    spend.reset()
    yield
    policy_block.reset()
    router.reset_breakers()
    for name in ("fakeg", "fakeb"):
        router._providers.pop(name, None)


def _routes(*specs):
    return [parse_route(s) for s in specs]


async def test_one_refusal_blocks_every_route_of_that_provider_and_sends_nothing_after():
    g = Refuses()
    router.register_provider("fakeg", g)
    routes = _routes("fakeg:m1@asia-south1", "fakeg:m1@global", "fakeg:m2")
    with pytest.raises(router.AllModelsFailed):
        await router.complete("brain", system_stable="x", messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}], routes=routes)
    assert g.calls == 1  # not the other two routes of the same project
    with pytest.raises(router.AllModelsFailed) as e:
        await router.complete("brain", system_stable="x", messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}], routes=routes)
    assert g.calls == 1 and "not sent" in str(e.value)  # no "try anyway"


async def test_another_provider_still_answers():
    g, b = Refuses(), Answers()
    router.register_provider("fakeg", g)
    router.register_provider("fakeb", b)
    rep = await router.complete("brain", system_stable="x", messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                                routes=_routes("fakeg:m1", "fakeg:m2", "fakeb:other"))
    assert rep.text == "ok" and g.calls == 1 and b.calls == 1


def test_only_a_policy_refusal_counts():
    assert policy_block.is_policy_refusal(403, "Request rejected in accordance with our terms of use and policies.")
    assert not policy_block.is_policy_refusal(403, "Permission denied on resource project (model not enabled)")
    assert not policy_block.is_policy_refusal(429, "Resource exhausted")


async def test_web_search_sends_nothing_while_blocked(monkeypatch):
    from app.brain import websearch

    calls = []

    class Models:
        async def generate_content(self, **kw):
            calls.append(kw["contents"])
            raise RuntimeError("403 PERMISSION_DENIED. Request rejected in accordance with our terms of use and policies.")

    client = type("C", (), {"aio": type("A", (), {"models": Models()})()})()
    monkeypatch.setattr(websearch, "_gemini", lambda location=None: client)
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "gemini")  # the Google path (off by default where no Google project is set)
    with pytest.raises(RuntimeError):
        await websearch.search("a")
    with pytest.raises(RuntimeError) as e:
        await websearch.grounded("b")  # the nutrition lookup's call too
    assert len(calls) == 1 and "not sent" in str(e.value)
