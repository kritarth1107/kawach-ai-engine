"""Every update a person must hear reaches them (founder 2026-10-12: the login-code ask and the booked ride never reached
him: the brain answered "none" and the retry was skipped because Saheli had written to him in the last two minutes).
What decides is whether they were told this state, not a time window; when the brain sends nothing twice, or no model
answers, the news written from the task data goes, in their words when a model answers."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import wake
from app.care import store
from app.llm import router
from app.llm.router import LLMReply
from app.sim.world import SimHost
from app.tasks import runtime
from app.tasks.models import TaskTold

FAM = "fam-reach"
ELDER = {"id": "elder-reach", "name": "Kamla", "role": "elder"}
SON = {"id": "son-reach", "name": "Ravi", "role": "primary caregiver"}


class Brain:
    """The brain as it behaved live: "none" to a task update; asked again, it writes the message (unless lazy=True);
    asked to put plain news in their words, it gives it back marked."""

    def __init__(self, lazy: bool = False, down: bool = False):
        self.lazy, self.down, self.calls, self.rewrites = lazy, down, 0, 0

    async def complete(self, route, *, messages, **kw):
        if self.down:
            raise RuntimeError("every model is down")
        text = messages[-1]["content"][0]["text"] if messages[-1]["role"] == "user" else ""
        if any("Rewrite this WhatsApp message" in b for b in kw.get("system_stable") or []):
            self.rewrites += 1
            return LLMReply(text="[hi] " + text.split("MESSAGE:\n", 1)[1], tool_calls=[], model="fake")
        if "[Task update]" in text:
            self.calls += 1
            if "has NOT heard this" in text and not self.lazy:
                return LLMReply(text="Ravi, Ola ne login code bheja hai, bata dijiye.", tool_calls=[], model="fake")
        return LLMReply(text="none", tool_calls=[], model="fake")


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


@pytest.fixture
def brain(monkeypatch):
    def use(b: Brain) -> Brain:
        router.register_provider("fake", b)
        monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "worker": ["fake:m"]}')
        router.reset_breakers()
        return b

    yield use
    router._providers.pop("fake", None)


async def _ride(db, **state) -> runtime.Task:
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER["id"], requested_by=SON["id"], service="ola", kind="ride",
                             goal="Cab to the airport", details={"pickup": "Home", "drop": "Airport"})
    for k, v in state.items():
        setattr(t, k, v)
    await db.commit()
    return t


def _update(t) -> runtime.TaskUpdate:
    token = runtime._UPDATE_FOR.set({str(t.id): runtime._stage(t)})
    try:
        return runtime.delivery_check()
    finally:
        runtime._UPDATE_FOR.reset(token)


async def _told(db, t) -> list[tuple[str, str]]:
    return [(r.person, r.how) for r in (await db.execute(select(TaskTold).where(TaskTold.task_id == t.id))).scalars()]


async def test_a_code_ask_right_after_saheli_wrote_still_reaches_him_and_is_not_sent_twice(db, at, sessions, brain):
    b = brain(Brain())
    at("2026-10-12 02:02")
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="user", text="Look in Ola first", speaker_id=SON["id"], message_ref="wamid.r1")
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="assistant", text="Going ahead with Ola.", meta={"reply_to": "wamid.r1"})
    t = await _ride(db, status="needs_input", phase="prepare", input_needed="otp")
    t.details = {**t.details, "login": "2001", "codes_sent": 1}
    await db.commit()
    at("2026-10-12 02:03")  # 26 s after Saheli's last message, as live
    host = SimHost()
    prompt = f"[Task update] [{t.id}] Ola ride: waiting for otp. The service sent a login code to the phone ending 2001; ask for it."
    await wake.system_turn(sessions, host, FAM, prompt, "task:r1", deliver_to=SON["id"], still_true=_update(t))
    assert [m["text"] for m in host.world.sent] == ["Ravi, Ola ne login code bheja hai, bata dijiye."] and b.calls == 2
    assert await _told(db, t) == [(SON["id"], "reply")]
    # the same state again (an overlapping loop): not sent twice, the brain not even asked
    await wake.system_turn(sessions, host, FAM, prompt, "task:r2", deliver_to=SON["id"], still_true=_update(t))
    assert len(host.world.sent) == 1 and b.calls == 2
    # a NEW code is new news
    t.details = {**t.details, "codes_sent": 2}
    await db.commit()
    await wake.system_turn(sessions, host, FAM, prompt, "task:r3", deliver_to=SON["id"], still_true=_update(t))
    assert len(host.world.sent) == 2


async def test_a_booked_ride_the_brain_will_not_announce_goes_out_from_the_task_data_with_every_detail(db, at, sessions, brain):
    b = brain(Brain(lazy=True))
    at("2026-10-12 02:06")
    t = await _ride(db, status="done", phase="place")
    t.details = {**t.details, "choice": "Mini ₹548"}
    t.result = {"booked": True, "ride_id": "CRN-1", "driver": "Ajay", "vehicle": "White Dzire", "plate": "CG04 AB 0001",
                "pickup_eta": "4 min", "ride_pin": "4321", "pickup_used": "12 Civil Lines, Raipur"}
    await db.commit()
    host = SimHost()
    await wake.system_turn(sessions, host, FAM, f"[Task update] [{t.id}] Ola ride: Booked.", "task:b1", deliver_to=SON["id"], still_true=_update(t))
    assert b.calls == 2 and b.rewrites == 1
    sent = host.world.sent[-1]["text"]
    assert sent.startswith("[hi] Booked on Ola: Mini ₹548, from 12 Civil Lines, Raipur to Airport")
    for bit in ("ride id CRN-1", "driver Ajay", "car White Dzire CG04 AB 0001", "pickup in about 4 min", "ride PIN 4321"):
        assert bit in sent, bit
    assert await _told(db, t) == [(SON["id"], "composed")]


async def test_when_no_model_answers_the_news_still_goes_in_plain_words(db, at, sessions, brain):
    brain(Brain(down=True))
    at("2026-10-12 02:10")
    t = await _ride(db, status="awaiting_confirm", phase="prepare", input_needed="go")
    t.result = {"options": [{"type": "Auto", "fare": "₹142", "eta": "4 min"}, {"type": "Mini", "fare": "₹210", "eta": "6 min"}]}
    runtime._ride_options_message(t)  # numbers the fares on the task, as the update did
    await db.commit()
    host = SimHost()
    await wake.system_turn(sessions, host, FAM, f"[Task update] Fares on Ola ...", "task:f1", deliver_to=SON["id"], still_true=_update(t))
    sent = host.world.sent[-1]["text"]
    assert sent.startswith("Fares on Ola, from Home to Airport (nothing is booked yet):\n1. Auto ₹142 (pickup in 4 min) - my suggestion\n2. Mini ₹210")
    assert sent.endswith("Which one shall I book? Reply with its number.")


async def test_a_cart_confirm_from_the_task_data_keeps_its_buttons(db, at, sessions, brain):
    brain(Brain(lazy=True))
    at("2026-10-12 02:12")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER["id"], requested_by=ELDER["id"], service="blinkit", kind="order",
                             goal="milk", details={"items": [{"name": "milk", "qty": 2}]})
    t.status, t.phase, t.input_needed = "awaiting_confirm", "prepare", "confirm"
    t.result = {"items": [{"name": "Amul Taaza Toned Milk", "pack": "500 ml", "qty": 2, "price": "₹56"}], "total": "₹71", "eta": "12 min"}
    t.details = {**t.details, "cart_fp": "abcdef1234", "limits": {"place": {"nickname": "Ghar"}}}
    await db.commit()
    host = SimHost()
    await wake.system_turn(sessions, host, FAM, f"[Task update] [{t.id}] cart ready", "task:c1", deliver_to=ELDER["id"], still_true=_update(t))
    msg = host.world.sent[-1]
    assert "Blinkit cart: 2 x Amul Taaza Toned Milk (500 ml). Total ₹71 with fees, to Ghar, cash on delivery" in msg["text"]
    assert [x["id"] for x in msg["buttons"]][0] == f"v2:od:yes:{t.id}:abcdef12"


async def test_when_they_wrote_after_the_update_the_brain_may_judge_they_already_heard_it(db, at, sessions, brain):
    """Live 2026-10-10 20:45: Maa asked, Saheli told her it was placed, then the "placed" update went out again."""
    b = brain(Brain(lazy=True))
    at("2026-10-12 02:20")
    t = await _ride(db, status="done", phase="place")
    t.result = {"booked": True, "ride_id": "CRN-2"}
    await db.commit()
    upd = _update(t)
    at("2026-10-12 02:20:30")
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="user", text="booked?", speaker_id=SON["id"], message_ref="wamid.r9")
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="assistant", text="Yes, booked: CRN-2.", meta={"reply_to": "wamid.r9"})
    await db.commit()
    host = SimHost()
    await wake.system_turn(sessions, host, FAM, f"[Task update] [{t.id}] Ola ride: Booked.", "task:s1", deliver_to=SON["id"], still_true=upd)
    assert not host.world.sent and b.calls == 2 and await _told(db, t) == []


async def test_an_update_whose_task_moved_on_is_never_sent_from_the_task_data(db, at, sessions, brain):
    brain(Brain(lazy=True))
    at("2026-10-12 02:25")
    t = await _ride(db, status="needs_input", phase="prepare", input_needed="otp")
    upd = _update(t)
    t.status, t.input_needed = "queued", None  # the code came in meanwhile
    await db.commit()
    host = SimHost()
    await wake.system_turn(sessions, host, FAM, f"[Task update] [{t.id}] waiting for otp", "task:m1", deliver_to=SON["id"], still_true=upd)
    assert not host.world.sent
