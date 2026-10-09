"""A phone call turn runs through Saheli's own brain (same record as WhatsApp); off unless CALLS_ENABLED=on."""

import httpx
import pytest

from app.brain import loop
from app.brain.loop import TurnRequest
from app.core.security import verify_api_secret
from app.db.session import get_db
from app.main import app


def test_call_mode_tells_the_brain_it_is_a_live_call():
    req = TurnRequest(family_id="f", elder={"id": "e"}, speaker={"id": "e"}, members=[], text="hello", channel="call", modality="voice")
    assert "LIVE PHONE CALL" in loop.voice_block(req)


@pytest.mark.parametrize("flag,code", [("off", 503)])
async def test_calls_are_off_by_default(monkeypatch, flag, code):
    monkeypatch.setenv("CALLS_ENABLED", flag)
    app.dependency_overrides[verify_api_secret] = lambda: None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/v2/calls/turn", json={"family_id": "f", "person_id": "p", "call_id": "c1", "text": "hi"})
            assert r.status_code == code
    finally:
        app.dependency_overrides.clear()
