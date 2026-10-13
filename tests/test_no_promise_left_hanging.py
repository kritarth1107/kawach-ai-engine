"""No promise is ever left hanging (founder 2026-10-13, 17:40 IST): mints and chewing gum were not found on three stores;
"try again on Blinkit" got "I'm retrying on Blinkit, I'll let you know", and nothing came for 19 minutes. start_task had
answered "already running": the Blinkit look-up that found nothing was still open, waiting for a pick with nothing to pick.
Now a finished look-up with nothing to pick ends when the options go out, a store asked again is looked at again, and a
store that is really busy says whether an update will come."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import tools
from app.care import store
from app.sim.agent import CART, FakeAgent
from app.sim.world import SimHost
from app.tasks import runtime

FAM, ME = "fam-np", "self-np"
M = {"id": ME, "name": "Vikram", "role": "elder"}
GUM = {"kind": "order", "goal": "Mints and chewing gum", "items": [{"name": "mints"}, {"name": "chewing gum"}], "for_whom": ME}
# a look-up that found products, none of them what was asked (the AI picked none)
NOTHING_EXACT = {"logged_in": False, "needs_otp": False, "blocked": False, "problem": "", "deliverable": True,
                 "items": [{"name": "Toothpaste Mint 100 g", "price": "₹95", "available": True}]}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


@pytest.fixture(autouse=True)
def picks_none(monkeypatch):
    from app.tasks import matcher

    async def choose(items, listings, **kw):
        return [{"exact": [], "closest": None, "why": "not this"} for _ in items]

    monkeypatch.setattr(matcher, "choose", choose)


class H:
    def __init__(self, sessions, agent):
        self.sessions, self.agent, self.told = sessions, agent, []

    async def profile_for(self, task):
        return {"profileId": f"prof-{task.service}", "loginPhone": "9000044444"}

    async def notify(self, family_id, requested_by, prompt):
        self.told.append(prompt)

    async def tick(self):
        return await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


def _ctx(db, ref):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=M, speaker=M, members=[M], message_ref=ref)


async def _looked_and_found_nothing(db, sessions, monkeypatch) -> tuple[H, list]:
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    await store.save_roster(db, FAM, M, [M])
    await store.record_event(db, family_id=FAM, subject_id=ME, kind="order_address_asked", summary="asked",
                             payload={"speaker": ME, "ref": "m0", "where": "Home"})
    await db.commit()
    h = H(sessions, FakeAgent(script={"browse": [NOTHING_EXACT], "prepare": [CART]}))
    out = await tools.start_task(_ctx(db, "m1"), {**GUM, "category": "grocery"})
    assert out["update_coming"] is True and len(out["task_ids"]) == 3
    for _ in range(6):
        await h.tick()
    return h, await runtime.live_tasks(db, FAM)


async def test_stores_that_found_nothing_are_closed_when_the_options_go_out(db, at, sessions, monkeypatch):
    at("2026-10-13 17:39")
    h, live = await _looked_and_found_nothing(db, sessions, monkeypatch)
    assert any("It is not available on any of them" in x for x in h.told)
    assert live == [], "nothing is left open waiting for a pick when there is nothing to pick"


async def test_try_again_on_blinkit_really_looks_again_and_the_update_comes(db, at, sessions, monkeypatch):
    at("2026-10-13 17:40")
    h, _ = await _looked_and_found_nothing(db, sessions, monkeypatch)
    at("2026-10-13 17:41")
    out = await tools.start_task(_ctx(db, "m2"), {**GUM, "service": "blinkit"})
    assert out["status"] == "started" and out["update_coming"] is True
    t = (await runtime.live_tasks(db, FAM))[0]
    assert t.service == "blinkit" and t.status == "queued"
    before = len(h.told)
    for _ in range(4):
        await h.tick()
    await db.refresh(t)
    assert len(h.told) > before, "the promised update came"


async def test_an_open_look_up_with_nothing_to_pick_is_replaced_not_reported_running(db, at, sessions):
    """Also for an older look-up still open with nothing to pick (as his was): it is looked at again."""
    at("2026-10-13 17:42")
    await store.save_roster(db, FAM, M, [M])
    old = await runtime.create(db, family_id=FAM, subject_id=ME, requested_by=ME, service="blinkit", kind="order", goal="gum",
                               details={"items": [{"name": "chewing gum"}], "for_set": True, "browsed": True})
    old.phase, old.status, old.input_needed = "browse", "awaiting_confirm", "go"
    old.result = {"items": NOTHING_EXACT["items"]}
    old.details = {**old.details, "chosen": None, "compare": "g-old", "compare_announced": True,
                   "matched": {"chewing gum": {"exact": [], "closest": None, "why": "not this"}}}  # a store of a comparison, as his was
    await store.record_event(db, family_id=FAM, subject_id=ME, kind="order_address_asked", summary="asked",
                             payload={"speaker": ME, "ref": "m0", "where": "Home"})
    await db.commit()
    out = await tools.start_task(_ctx(db, "m3"), {**GUM, "service": "blinkit"})
    await db.refresh(old)
    assert out["status"] == "started" and old.status == "cancelled" and old.details["replaced"]


@pytest.mark.parametrize("status,input_needed,coming", [("running", None, True), ("awaiting_confirm", "confirm", False),
                                                         ("needs_input", "otp", False)])
async def test_a_store_really_busy_says_whether_an_update_will_come(db, at, status, input_needed, coming):
    at("2026-10-13 17:43")
    await store.save_roster(db, FAM, M, [M])
    t = await runtime.create(db, family_id=FAM, subject_id=ME, requested_by=ME, service="blinkit", kind="order", goal="gum",
                             details={"items": [{"name": "chewing gum"}], "for_set": True, "cart_fp": "fp"})
    t.phase, t.status, t.input_needed = "prepare", status, input_needed
    await store.record_event(db, family_id=FAM, subject_id=ME, kind="order_address_asked", summary="asked",
                             payload={"speaker": ME, "ref": "m0", "where": "Home"})
    await db.commit()
    out = await tools.start_task(_ctx(db, "m4"), {**GUM, "service": "blinkit"})
    assert out["update_coming"] is coming and "already_running" in out
    if not coming:
        assert "NO update will come" in out["next"] and "Never say you are retrying" in out["next"]
    assert len(await runtime.live_tasks(db, FAM)) == 1
