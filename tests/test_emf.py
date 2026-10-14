"""Gemini calls publish a CloudWatch line. A 429 is a throttle, not an error."""
import json

import pytest

from app.llm import router
from app.llm.router import AllModelsFailed, ModelUnavailable


class Once:
    def __init__(self, exc):
        self.exc = exc

    async def complete(self, route, **kw):
        raise self.exc


async def test_a_gemini_429_is_a_throttle(monkeypatch, capsys):
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["gemini:gemini-3.5-flash"]}')
    monkeypatch.setattr(router, "RETRY_429_S", ())
    previous = router._providers.get("gemini")
    router.register_provider("gemini", Once(ModelUnavailable(429, "slow")))
    router.reset_breakers()
    try:
        with pytest.raises(AllModelsFailed):
            await router.complete(role="brain", system_stable="s", messages=[{"role": "user", "content": "hi"}])
    finally:
        if previous is None:
            router._providers.pop("gemini", None)
        else:
            router._providers["gemini"] = previous
        router.reset_breakers()
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{") and "Kavach/Gemini" in line]
    assert lines and lines[-1]["Throttles"] == 1 and lines[-1]["Errors"] == 0 and lines[-1]["Calls"] == 1
