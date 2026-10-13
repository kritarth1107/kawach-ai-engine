from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import wake
from app.care import store
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
    sent = host.world.sent
    assert sent == [{"to": ELDER["id"], "text": "Kamla ji, BP check kiya?", "at": sent[0]["at"], "kind": "message", "ref": sent[0]["ref"]}]
    assert sent[0]["ref"].startswith("wake:")  # the backend ledger keeps what it was (a wake-up's follow-up is a message)
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
    assert clock.ist(loop.wake_at).strftime("%d %H:%M") == "03 08:30"  # nothing at night (21:00-08:00)
    loop.detail = {"wakes": wake.MAX_WAKES}
    loop.wake_at = now
    await db.commit()
    at("2026-10-03 08:40")
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
    both blocked by the unanswered-message cooldown, so she never heard back. Task updates must go out; a scheduled question
    waits while the order waits for her answer (without using up its wake), then goes."""
    from app.tasks.models import Task

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
        db.add(Task(family_id=FAM, subject_id=ELDER["id"], requested_by=ELDER["id"], service="blinkit", kind="order", goal="diet coke",
                    status="needs_input", phase="otp", input_needed="otp", created_at=now, updated_at=clock.now(), deadline_at=now + timedelta(minutes=30)))
        await db.commit()
        await wake.system_turn(sessions, host, FAM, "[Task update] [t-1] Blinkit order: Diet Coke, waiting for otp. The service sent a login code to phone; ask the person who has that phone for it.", "task:abc123")
        # mid-conversation (Saheli wrote 4 minutes ago): straight to the point, no name opener
        assert [m["text"] for m in host.world.sent] == ["Blinkit ne login code bheja hai, bata dijiye."]
        at("2026-10-08 14:54")
        stats = await wake.wake_due(sessions, lambda fid: host)
        assert len(host.world.sent) == 1 and stats["deferred"] == 1, "the BP question waits while the order waits for her login code"
        await db.refresh(loop)
        assert (loop.detail or {}).get("wakes", 0) == 0 and clock.ist(loop.wake_at).strftime("%H:%M") == "15:24"
        at("2026-10-08 15:25")  # the order has waited 38 minutes for her: it no longer holds her questions back
        await wake.wake_due(sessions, lambda fid: host)
        assert len(host.world.sent) == 2 and "BP check kiya?" in host.world.sent[-1]["text"]
        await db.refresh(loop)
        assert loop.detail["wakes"] == 1
    finally:
        router._providers.pop("fake", None)


async def test_urgent_follow_up_is_not_held_back_by_an_order(db, at, fake_brain):
    """A follow-up whose rule alerts the caregiver (a health worry) wakes on time even while an order is being placed."""
    from app.tasks.models import Task

    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    now = at("2026-10-02 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, DAUGHTER])
    db.add(Task(family_id=FAM, subject_id=ELDER["id"], requested_by=DAUGHTER["id"], service="blinkit", kind="order", goal="atta",
                status="running", phase="place", created_at=now, updated_at=now, deadline_at=now + timedelta(minutes=30)))
    await store.open_loop(db, family_id=FAM, subject_id=ELDER["id"], kind="question", title="Ask Kamla ji if the chest pain has eased",
                          wake_at=now + timedelta(minutes=10), alert_rule="alert_caregiver")
    await db.commit()
    host = SimHost()
    at("2026-10-02 10:11")
    stats = await wake.wake_due(sessions, lambda fid: host)
    assert stats["ran"] == 1 and stats["deferred"] == 0 and len(host.world.sent) == 1


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
        # live 2026-10-09: the backend refuses an actor who is not in the family ("saheli"), so nothing went out
        assert [c["actor"] for c in host.world.calls if c["tool"] == "send_whatsapp"] == [ELDER["id"]]
        # a wake-up (no deliver_to) still sends nothing on its own
        await wake.system_turn(sessions, host, FAM, "[Task update] [t-3] Blinkit order: Diet Coke, waiting for go.", "task:p2")
        assert len(host.world.sent) == 1
    finally:
        router._providers.pop("fake", None)


class SilentFirstBrain:
    """Answers a task update with 'none' the first time (as live 2026-10-10 20:29 and 2026-10-12 02:03), with the message
    when asked again."""

    def __init__(self):
        self.calls = 0

    async def complete(self, route, *, messages, **kw):
        text = messages[-1]["content"][0]["text"] if messages[-1]["role"] == "user" else ""
        if "[Task update]" in text:
            self.calls += 1
            if "sent NOTHING" in text:
                return LLMReply(text="Kamla ji, cart ₹175 ready hai. Order karun?", tool_calls=[], model="fake")
        return LLMReply(text="none", tool_calls=[], model="fake")


async def test_an_update_that_sent_nothing_is_asked_again_even_right_after_saheli_wrote(db, at, monkeypatch):
    """Live 2026-10-12 02:03: the login-code ask got "none" and was not asked again because Saheli had written 26 s before.
    A time window never decides; what the person was told does (tests/test_updates_reach.py has the task-data cases)."""
    brain = SilentFirstBrain()
    router.register_provider("fake", brain)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    try:
        sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
        at("2026-10-10 20:29")
        await store.save_roster(db, FAM, ELDER, [ELDER, DAUGHTER])
        await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="assistant", text="Ji, dekh rahi hoon.")
        await db.commit()
        host = SimHost()
        await wake.system_turn(sessions, host, FAM, "[Task update] [t-9] Instamart order: khakhra, waiting for confirm, total ₹175.", "task:a1",
                               deliver_to=ELDER["id"])
        # mid-conversation: no name opener
        assert [m["text"] for m in host.world.sent] == ["Cart ₹175 ready hai. Order karun?"] and brain.calls == 2
    finally:
        router._providers.pop("fake", None)
