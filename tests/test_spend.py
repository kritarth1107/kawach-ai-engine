"""Spend cap and flood limits (after the 3 Oct 2026 suspension: journal/2026-10-03_2250)."""

from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain.loop import PERSON_BURST, TurnRequest, run_turn
from app.care import store
from app.core import clock
from app.llm import router, spend
from app.llm.router import AllModelsFailed, LLMReply
from app.sim.world import SimHost


class Paid:
    """A provider that reports big usage, so spend climbs fast."""

    def __init__(self):
        self.calls = []

    async def complete(self, route, **kw):
        self.calls.append((route.model, kw.get("effort")))
        return LLMReply(text="ok", tool_calls=[], model=route.model, usage={"in": 200_000, "out": 20_000, "think": 30_000})


@pytest.fixture
def paid(monkeypatch):
    p = Paid()
    router.register_provider("paid", p)
    router.reset_breakers()
    spend.reset()
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["paid:gemini-3.1-pro", "paid:gemini-3.8-flash"], "judge": ["paid:gemini-3.1-pro"], '
                                       '"extract": ["paid:gemini-3.8-flash"]}')
    yield p
    router._providers.pop("paid", None)
    spend.reset()


def test_price_counts_thinking_and_cache():
    base = spend.cost_inr("gemini-3.8-flash", {"in": 1_000_000, "out": 0})
    assert spend.cost_inr("gemini-3.8-flash", {"in": 0, "out": 0, "think": 1_000_000}) > base  # thinking billed as output
    assert spend.cost_inr("gemini-3.8-flash", {"in": 1_000_000, "cache_read": 1_000_000}) < base  # cache discount
    assert spend.cost_inr("gemini-3.1-pro-preview", {"out": 1000}) > spend.cost_inr("gemini-3.8-flash", {"out": 1000})


def test_price_override(monkeypatch):
    monkeypatch.setenv("LLM_PRICES_INR", '{"gemini-3.8-flash": [10, 100]}')
    assert spend.cost_inr("gemini-3.8-flash", {"in": 1_000_000, "out": 1_000_000}) == pytest.approx(110)


async def test_caps_step_down_then_refuse(paid, monkeypatch):
    monkeypatch.setenv("LLM_SOFT_CAP_INR", "50")
    monkeypatch.setenv("LLM_HARD_CAP_INR", "100")
    while spend.process_spent() < 50:
        await router.complete("judge", system_stable="x", messages=[])
    # Over the soft cap: non-essential roles refused, the brain uses only its cheapest model at low effort.
    with pytest.raises(AllModelsFailed, match="soft cap"):
        await router.complete("judge", system_stable="x", messages=[])
    paid.calls.clear()
    await router.complete("brain", system_stable="x", messages=[], effort="medium")
    assert paid.calls == [("gemini-3.8-flash", "low")]
    while spend.process_spent() < 100:
        await router.complete("brain", system_stable="x", messages=[])
    with pytest.raises(AllModelsFailed, match="hard cap"):
        await router.complete("extract", system_stable="x", messages=[])
    reply = await router.complete("brain", system_stable="x", messages=[])  # a person writing still gets an answer
    assert reply.text == "ok"


async def test_ledger_persists_and_is_shared(db, paid):
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    spend.configure(sessions)
    # (The test shares one connection; production writes use their own pooled connection.)
    await router.complete("brain", system_stable="x", messages=[])
    await spend.flush()
    await router.complete("brain", system_stable="x", messages=[])
    await spend.flush()
    rows = await spend.summary(sessions, days=1)
    assert rows and rows[0]["calls"] == 2 and rows[0]["costInr"] > 0
    # Another instance (empty memory) reads the same total from the database.
    spend._mem.clear()
    spend._cache.clear()
    assert await spend.spent_today() == pytest.approx(rows[0]["costInr"], rel=0.01)


# ── flood limits ──

FAM = "fam-flood"
ELDER = {"id": "elder-f", "name": "Kamla", "role": "elder"}


class Echo:
    def __init__(self):
        self.calls = 0

    async def complete(self, route, **kw):
        self.calls += 1
        return LLMReply(text="Theek hai ji.", tool_calls=[], model="fake")


@pytest.fixture
def echo(monkeypatch):
    e = Echo()
    router.register_provider("fake", e)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    yield e
    router._providers.pop("fake", None)


async def _flood(db, n, text="hello hello"):
    for i in range(n):
        await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="user", text=text, speaker_id=ELDER["id"], message_ref=f"f{i}")
    await db.commit()


async def test_flood_gets_canned_reply_without_model(db, at, echo):
    at("2026-10-02 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER])
    await _flood(db, PERSON_BURST)
    req = TurnRequest(family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER], text="Kya haal hai aapka beta", message_ref="x1")
    out = await run_turn(db, SimHost(), req)
    assert echo.calls == 0 and out.model == "none" and "रुक रही" in out.reply
    out2 = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER], text="hello?", message_ref="x2"))
    assert out2.reply == "🙏" and echo.calls == 0


async def test_flood_never_blocks_an_emergency(db, at, echo):
    at("2026-10-02 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER])
    await _flood(db, PERSON_BURST + 5)
    out = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER],
                                                   text="I fell in the bathroom", message_ref="x3"))
    assert echo.calls >= 1 and out.model == "fake"


async def test_flood_window_expires(db, at, echo):
    now = at("2026-10-02 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER])
    await _flood(db, PERSON_BURST + 5)
    clock.set_now(now + timedelta(minutes=11))
    out = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER], text="hi", message_ref="x4"))
    assert echo.calls >= 1 and out.model == "fake"


async def test_capped_brain_keeps_every_fallback(paid, monkeypatch):
    monkeypatch.setenv("LLM_SOFT_CAP_INR", "1")
    while spend.process_spent() < 1:
        await router.complete("judge", system_stable="x", messages=[])

    class Down:
        async def complete(self, route, **kw):
            from app.llm.router import ModelUnavailable

            raise ModelUnavailable(503, "vertex down")

    router.register_provider("down", Down())
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["down:gemini-3.8-flash", "paid:gemini-3.1-pro"]}')
    reply = await router.complete("brain", system_stable="x", messages=[])  # cheapest is down; the next one answers
    assert reply.model == "gemini-3.1-pro"
    router._providers.pop("down", None)


async def test_scheduler_turns_stop_at_hard_cap_people_do_not(paid, monkeypatch):
    monkeypatch.setenv("LLM_SOFT_CAP_INR", "1")
    monkeypatch.setenv("LLM_HARD_CAP_INR", "2")
    while spend.process_spent() < 2:
        await router.complete("judge", system_stable="x", messages=[])
    with pytest.raises(AllModelsFailed, match="hard cap"):
        await router.complete("brain", system_stable="x", messages=[], essential=False)
    assert (await router.complete("brain", system_stable="x", messages=[], essential=True)).text == "ok"


async def test_task_updates_are_never_throttled(db, at, echo):
    at("2026-10-02 10:00")
    from app.brain.loop import SYSTEM_BURST

    system = {"id": "saheli-scheduler", "name": "Scheduler", "role": "system"}
    await store.save_roster(db, FAM, ELDER, [ELDER])
    for i in range(SYSTEM_BURST + 2):
        await store.add_turn(db, family_id=FAM, thread_id=system["id"], role="user", text="wake", speaker_id=system["id"], message_ref=f"w{i}")
    await db.commit()
    out = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ELDER, speaker=system, members=[ELDER],
                                                   text="[Task update] unclear, do not order again", message_ref="task:abc"))
    assert out.model == "fake" and echo.calls >= 1
    out2 = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ELDER, speaker=system, members=[ELDER],
                                                    text="[Scheduled wake-up] loop", message_ref="wake:zz"))
    assert out2.model == "none"
