"""Reply speed (overnight 14 Oct): where each turn's time goes is kept on the why-trail, a turn never waits behind a
stuck one for minutes, and a turn never runs past the time the person is still waiting.

Live 12 Oct 20:33 UTC: one turn hung, Cloud Run cut the request at 300 s but the turn kept the family locked until its
instance shut down 10 minutes later; five messages waited behind it (125-248 s each). The backend stops waiting at 120 s."""

import asyncio

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.brain import loop
from app.care import store, why
from app.llm import router, spend
from app.llm.router import LLMReply
from app.sim.world import SimHost
from tests.conftest import TEST_DB

FAM = "fam-speed"
ME = {"id": "self-speed", "name": "Asha Rao", "role": "primary caregiver"}


class Slow:
    def __init__(self, delay: float, text: str = "Okay."):
        self.delay, self.text, self.calls = delay, text, 0

    async def complete(self, route, **kw):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return LLMReply(text=self.text, tool_calls=[], model="fake")


@pytest.fixture
def model(monkeypatch):
    def make(delay: float, reply: str = "Okay."):
        p = Slow(delay, reply)
        router.register_provider("fake", p)
        monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"], "classify": ["fake:m"]}')
        monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0")
        router.reset_breakers()
        spend.reset()
        return p
    yield make
    router._providers.pop("fake", None)
    router.reset_breakers()


def req(text_: str, ref: str) -> loop.TurnRequest:
    return loop.TurnRequest(family_id=FAM, elder=ME, speaker=ME, members=[ME], text=text_, message_ref=ref)


async def test_the_why_trail_keeps_where_the_time_went(db, model):
    await store.save_roster(db, FAM, ME, [ME])
    model(0.01, "Good morning Asha!")
    await loop.run_turn(db, SimHost(), req("good morning", "speed-1"))
    row = (await db.execute(select(why.TraceTurn).where(why.TraceTurn.message_ref == "speed-1"))).scalar_one()
    t = row.timing
    assert {"lockMs", "contextMs", "afterMs"} <= set(t) and all(isinstance(t[k], int) for k in ("lockMs", "contextMs", "afterMs"))
    assert [r for r, _ in t["model"]][0] == "brain" and all(ms >= 0 for _, ms in t["model"])
    assert "Asha" not in str(t)  # step names and milliseconds only


async def test_a_turn_never_waits_minutes_behind_a_stuck_one(db, monkeypatch):
    other = create_async_engine(TEST_DB)
    try:
        async with other.connect() as conn:
            async with conn.begin():
                await conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:f))"), {"f": "fam-busy"})  # a stuck turn
                with pytest.raises(loop.FamilyBusy):
                    await loop._lock_family(db, "fam-busy", wait_s=0.3)
    finally:
        await other.dispose()
    assert issubclass(loop.FamilyBusy, router.AllModelsFailed)  # answered like an outage: 503, the backend's holding line


async def test_the_wait_bound_is_only_for_taking_the_turn(db):
    before = (await db.execute(text("SHOW lock_timeout"))).scalar_one()
    await loop._lock_family(db, "fam-free", wait_s=5)
    assert (await db.execute(text("SHOW lock_timeout"))).scalar_one() == before  # the rest of the turn is as before


async def test_a_turn_never_runs_past_the_time_the_person_waits(db, model, monkeypatch):
    """Nothing done yet when the budget runs out: the same as no model answering (503, the backend's holding line)."""
    await store.save_roster(db, FAM, ME, [ME])
    monkeypatch.setitem(loop.TURN_BUDGET_S, "message", 0.4)
    monkeypatch.setattr(loop, "MIN_CALL_S", 0.05)
    p = model(2.0)
    started = asyncio.get_running_loop().time()
    with pytest.raises(router.AllModelsFailed):
        await loop.run_turn(db, SimHost(), req("are you there?", "speed-2"))
    assert asyncio.get_running_loop().time() - started < 1.5 and p.calls == 1


async def test_hedged_brain_calls_answer_a_slow_turn_sooner(db, monkeypatch):
    """The whole turn with a brain call that is slow once: the copy sent after BRAIN_HEDGE_AFTER_S answers."""
    await store.save_roster(db, FAM, ME, [ME])

    class OnceSlow:
        def __init__(self):
            self.calls = 0

        async def complete(self, route, **kw):
            self.calls += 1
            await asyncio.sleep(3.0 if self.calls == 1 else 0.01)
            return LLMReply(text="Here I am, Asha.", tool_calls=[], model="fake")

    p = OnceSlow()
    router.register_provider("fake", p)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"]}')
    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0.1")
    monkeypatch.setenv("GROUNDING_CHECK", "off")
    router.reset_breakers()
    spend.reset()
    try:
        out = await loop.run_turn(db, SimHost(), req("hello?", "speed-3"))
    finally:
        router._providers.pop("fake", None)
    assert out.reply.startswith("Here I am") and out.ms < 2000
