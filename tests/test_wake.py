from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import wake
from app.care import store
from app.care.models import OpenLoop
from app.core import clock
from app.llm import router
from app.llm.router import LLMReply, ToolCall
from app.sim.world import SimHost

FAM = "fam-wake"
ELDER = {"id": "elder-w", "name": "Kamla", "role": "elder"}
DAUGHTER = {"id": "cg-w", "name": "Asha", "role": "primary caregiver"}


class Brain:
    """Scripted brain: on the wake-up turn it sends a follow-up and closes nothing."""

    def __init__(self):
        self.seen = []

    async def complete(self, route, *, messages, **kw):
        last = messages[-1]
        if last["role"] == "user":
            text = last["content"][0]["text"]
            self.seen.append(text)
            if "Scheduled wake-up" in text:
                return LLMReply(text="", tool_calls=[ToolCall("c1", "send_message", {"to": ELDER["id"], "text": "Kamla ji, BP check kiya?"})], model="fake")
        return LLMReply(text="none", tool_calls=[], model="fake")


@pytest.fixture
def fake_brain(monkeypatch):
    b = Brain()
    router.register_provider("fake", b)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    yield b
    router._providers.pop("fake", None)


async def test_due_loop_wakes_brain_and_sends_followup(db, at, fake_brain):
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    now = at("2026-10-02 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, DAUGHTER])
    loop = await store.open_loop(db, family_id=FAM, subject_id=ELDER["id"], kind="question", title="Ask Kamla ji for her BP",
                                 wake_at=now + timedelta(minutes=30), alert_rule="ask_again")
    await db.commit()
    host = SimHost()
    at("2026-10-02 10:31")
    stats = await wake.wake_due(sessions, lambda fid: host)
    assert stats["ran"] == 1
    assert host.world.sent == [{"to": ELDER["id"], "text": "Kamla ji, BP check kiya?", "at": host.world.sent[0]["at"]}]
    assert "Scheduled wake-up" in fake_brain.seen[0]
    await db.refresh(loop)
    assert loop.detail["wakes"] == 1 and loop.status == "open"
    turns = await store.recent_turns(db, FAM, ELDER["id"])
    assert turns[-1].text == "Kamla ji, BP check kiya?" and turns[-1].meta["proactive"]


async def test_quiet_hours_defer_and_expiry(db, at, fake_brain):
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    now = at("2026-10-02 23:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, DAUGHTER])
    loop = await store.open_loop(db, family_id=FAM, subject_id=ELDER["id"], kind="followup", title="Ask about sleep",
                                 wake_at=now - timedelta(minutes=1), alert_rule="ask_again")
    await db.commit()
    stats = await wake.wake_due(sessions, lambda fid: SimHost())
    assert stats["deferred"] == 1
    await db.refresh(loop)
    assert clock.ist(loop.wake_at).strftime("%d %H:%M") == "03 07:30"
    loop.detail = {"wakes": wake.MAX_WAKES}
    loop.wake_at = now
    await db.commit()
    at("2026-10-03 08:00")
    stats = await wake.wake_due(sessions, lambda fid: SimHost())
    assert stats["expired"] == 1
    await db.refresh(loop)
    assert loop.status == "expired"
