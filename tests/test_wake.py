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


class TaskBrain:
    """Scripted brain: every system turn (task update or wake-up) tries to message the elder."""

    async def complete(self, route, *, messages, **kw):
        last = messages[-1]
        if last["role"] == "user":
            text = last["content"][0]["text"]
            if "[Task update]" in text:
                return LLMReply(text="", tool_calls=[ToolCall("t1", "send_message", {"to": ELDER["id"], "text": "Kamla ji, Blinkit ne login code bheja hai, bata dijiye."})], model="fake")
            if "Scheduled wake-up" in text:
                return LLMReply(text="", tool_calls=[ToolCall("w1", "send_message", {"to": ELDER["id"], "text": "Kamla ji, BP check kiya?"})], model="fake")
        return LLMReply(text="none", tool_calls=[], model="fake")


async def test_task_update_reaches_elder_before_she_answers(db, at, monkeypatch):
    """Live 2026-10-08: 'Order diet coke' → 'I'll tell you when the cart is ready' → the login-code and timeout updates were
    both blocked by the unanswered-message cooldown, so she never heard back. Task updates must go out; nudges still wait."""
    router.register_provider("fake", TaskBrain())
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    try:
        sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
        now = at("2026-10-08 14:43")
        await store.save_roster(db, FAM, ELDER, [ELDER, DAUGHTER])
        await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="user", text="Order diet coke", speaker_id=ELDER["id"], message_ref="wamid.1")
        await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="assistant", text="Ji, order laga rahi hoon. Cart taiyar hote hi bataungi.")
        loop = await store.open_loop(db, family_id=FAM, subject_id=ELDER["id"], kind="question", title="Ask Kamla ji for her BP",
                                     wake_at=now + timedelta(minutes=10), alert_rule="ask_again")
        await db.commit()
        host = SimHost()
        at("2026-10-08 14:47")
        await wake.system_turn(sessions, host, FAM, "[Task update] [t-1] Blinkit order: Diet Coke, waiting for otp. The service sent a login code to phone; ask the person who has that phone for it.", "task:abc123")
        assert [m["text"] for m in host.world.sent] == ["Kamla ji, Blinkit ne login code bheja hai, bata dijiye."]
        at("2026-10-08 14:54")
        await wake.wake_due(sessions, lambda fid: host)
        assert len(host.world.sent) == 1, "a nudge still waits for her answer"
        await db.refresh(loop)
        assert loop.detail["wakes"] == 1
    finally:
        router._providers.pop("fake", None)


class PlainTextBrain:
    """Scripted brain that answers a task update as a plain reply (no send_message), as Gemini did live."""

    async def complete(self, route, *, messages, **kw):
        text = messages[-1]["content"][0]["text"] if messages[-1]["role"] == "user" else ""
        if "[Task update]" in text:
            return LLMReply(text="Kamla ji, Blinkit par Diet Coke ₹50 mein mil rahi hai. Order karun?", tool_calls=[], model="fake")
        return LLMReply(text="none", tool_calls=[], model="fake")


async def test_task_update_written_as_plain_reply_still_reaches_the_asker(db, at, monkeypatch):
    """Live 2026-10-08 21:41: the brain wrote the price and go-ahead question as its final reply, which on a system turn
    goes to no one. The update is now delivered to the person who asked."""
    router.register_provider("fake", PlainTextBrain())
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    try:
        sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
        at("2026-10-08 21:41")
        await store.save_roster(db, FAM, ELDER, [ELDER, DAUGHTER])
        await db.commit()
        host = SimHost()
        await wake.system_turn(sessions, host, FAM, "[Task update] [t-2] Blinkit order: Diet Coke, waiting for go.", "task:p1", deliver_to=ELDER["id"])
        assert [m["text"] for m in host.world.sent] == ["Kamla ji, Blinkit par Diet Coke ₹50 mein mil rahi hai. Order karun?"]
        assert (await store.recent_turns(db, FAM, ELDER["id"]))[-1].meta["fallback_delivery"] == "task:p1"
        # a wake-up (no deliver_to) still sends nothing on its own
        await wake.system_turn(sessions, host, FAM, "[Task update] [t-3] Blinkit order: Diet Coke, waiting for go.", "task:p2")
        assert len(host.world.sent) == 1
    finally:
        router._providers.pop("fake", None)
