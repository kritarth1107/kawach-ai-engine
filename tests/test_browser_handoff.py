"""A session belongs to one Browser Use API. These cover every store a chat can order from."""

import dataclasses

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.sim.agent import CART, FakeAgent
from app.specialists import guard
from app.tasks import runtime
from app.tasks.models import Task

STORES = ("blinkit", "instamart", "zepto", "swiggy", "zomato", "apollo", "1mg", "pharmeasy", "uber", "ola", "rapido")
RIDES = {"uber", "ola", "rapido"}
ME = "me-h"


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class H:
    def __init__(self, sessions, agent):
        self.sessions, self.agent = sessions, agent

    async def profile_for(self, task):
        return f"prof-{task.family_id}-{task.service}"

    async def notify(self, family_id, requested_by, prompt):
        return None

    async def tick(self):
        return await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


class Gone(FakeAgent):
    """The session id belongs to the other API: Browser Use answers 404."""

    async def run(self, *, session_id, **kw):
        if session_id:
            raise RuntimeError('browser-use POST v4 404: {"detail":"session not found"}')
        return await super().run(session_id=session_id, **kw)


class BadJson(FakeAgent):
    async def poll(self, task_id, *, api=None):
        run = await super().poll(task_id, api=api)
        return dataclasses.replace(run, error="bad json", output=None, path=[])


async def _task(db, service: str, case: str) -> Task:
    kind = "ride" if service in RIDES else "order"
    details = {"pickup": "Home", "drop": "Clinic"} if kind == "ride" else {"items": [{"name": "Milk 1L", "qty": 1}]}
    t = await runtime.create(
        db, family_id=f"fam-{service}-{case}", subject_id=ME, requested_by=ME, service=service, kind=kind,
        goal="milk" if kind == "order" else "a ride home", details=details,
    )
    await db.commit()
    t.details = {**(t.details or {}), "channel": "browser", "browsed": True, "session_api": "v4", "browser_api": "v4"}
    await db.commit()
    return t


def _confirm(t: Task) -> None:
    fp = "fp-1"
    by = ME
    if t.kind == "ride":
        choice = "auto"
        used = runtime._ride_fp(fp, choice)
        extra = {"choice": choice, "cart_fp": fp}
    else:
        used = fp
        extra = {"cart_fp": fp}
    t.phase = "place"
    t.details = {**(t.details or {}), **extra, "confirmed_fp": used, "confirmed_by": by, "confirmed_total": "₹100",
                 "confirm_token": guard.confirm_token(str(t.id), used, by)}


async def test_saved_steps_leave_a_v4_browser_alone():
    """Saved steps need a v2 CDP browser. They must not open a second one over a V4 session."""

    class Opener:
        def __init__(self):
            self.opened: list = []

        async def open_session(self, profile):
            self.opened.append(profile)
            return "v2-new"

        async def cdp_url(self, sid):
            raise AssertionError(sid)

    opener = Opener()
    place = {"lat": 12.97, "lng": 77.59, "pincode": "560001"}
    for service in ("blinkit", "instamart", "zepto", "swiggy", "zomato", "apollo", "1mg", "pharmeasy", "rapido"):
        kind = "ride" if service == "rapido" else "order"
        t = Task(service=service, kind=kind, phase="prepare", family_id="f", subject_id="s", requested_by="s",
                 details={"session_api": "v4", "limits": {"place": place}})
        assert await runtime._fast_lookup(None, opener, t, None) is None
    cart = Task(service="blinkit", kind="order", phase="prepare", family_id="f", subject_id="s", requested_by="s",
                details={"session_api": "v4", "chosen": [{"name": "Milk 1L", "store_id": "1"}]})
    assert await runtime._fast_cart(None, opener, cart, None) is None
    place_task = Task(service="instamart", kind="order", phase="place", family_id="f", subject_id="s", requested_by="s",
                      details={"session_api": "v4", "fast_place_check": {"total": 10}})
    assert await runtime._fast_place(None, opener, place_task, None) is None
    assert opener.opened == []


async def test_each_store_handles_a_missing_session_a_stuck_run_and_bad_json(db, at, sessions, monkeypatch):
    monkeypatch.setenv("BROWSER_API", "v4")
    at("2026-10-16 22:00")
    for service in STORES:
        gone = Gone(script={"prepare": [CART], "place": [CART]})
        h = H(sessions, gone)
        t = await _task(db, service, "prepare")
        t.agent_session = "dead-v4"
        await db.commit()
        await h.tick()
        assert len(gone.runs) == 1 and gone.runs[0]["session"] is None and gone.runs[0]["api"] == "v4", service
        await db.refresh(t)
        assert t.details["session_api"] == "v4"
        t.status = "failed"
        await db.commit()

        gone = Gone(script={"place": [CART]})
        h = H(sessions, gone)
        t = await _task(db, service, "place")
        t.agent_session = "dead-v4"
        _confirm(t)
        await db.commit()
        await h.tick()
        assert gone.runs == [], service  # a place is not started again on a fresh browser
        await db.refresh(t)
        t.status = "failed"
        await db.commit()

        stuck = FakeAgent(script={"prepare": [CART]}, finish_after_polls=50, path=[])
        h = H(sessions, stuck)
        t = await _task(db, service, "stuck")
        await h.tick()
        at("2026-10-16 22:13")
        await h.tick()
        await db.refresh(t)
        assert t.status == "failed" and stuck.stops and stuck.stops[-1]["api"] == "v4", service
        at("2026-10-16 22:00")
        await db.refresh(t)
        t.status = "failed"
        await db.commit()

        bad = BadJson(script={"prepare": [CART]}, path=[])
        h = H(sessions, bad)
        t = await _task(db, service, "json")
        await h.tick()
        await h.tick()
        await h.tick()
        assert bad.runs[-1]["api"] == "v4" and bad.runs[-1]["session"] and "JSON" in bad.runs[-1]["goal"], service
        assert "dead" not in (bad.runs[-1]["session"] or "")
        await db.refresh(t)
        t.status = "failed"
        await db.commit()
