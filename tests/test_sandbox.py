"""Browser task sandbox: profiles never cross families, every browser is stopped, login state, audit, sweeper."""

import json
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.dash import task_json
from app.brain import tools
from app.core import clock
from app.sim.agent import CART, OTP, FakeAgent
from app.sim.world import SimHost
from app.tasks import runtime, sandbox
from app.tasks.models import Task


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class H:
    def __init__(self, sessions, agent, profile=None):
        self.sessions, self.agent, self.told = sessions, agent, []
        self.profile = profile or (lambda t: f"prof-{t.family_id}-{t.service}")

    async def profile_for(self, task):
        return self.profile(task)

    async def notify(self, fid, by, prompt):
        self.told.append(prompt)

    async def tick(self, n=1):
        for _ in range(n):
            await runtime.tick(self.sessions, self.agent, profile_for=self.profile_for, notify=self.notify)


async def order(db, fam="fam-sb", service="zepto"):
    t = await runtime.create(db, family_id=fam, subject_id="e", requested_by="e", service=service, kind="order", goal="Atta",
                             details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]})
    await db.commit()
    return t


async def open_rows(db):
    return (await db.execute(select(sandbox.BrowserSession).where(sandbox.BrowserSession.stopped_at.is_(None)))).scalars().all()


async def test_a_profile_never_crosses_families(db, at, sessions):
    at("2026-10-04 10:00")
    h = H(sessions, FakeAgent(script={"prepare": [CART]}), profile=lambda t: "prof-shared")
    await order(db, "fam-a")
    await h.tick()
    assert h.agent.runs[-1]["profile"] == "prof-shared"
    t2 = await order(db, "fam-b")
    await h.tick(4)
    await db.refresh(t2)
    assert all(r["profile"] == "prof-shared" for r in h.agent.runs) and len(h.agent.runs) == 1  # fam-b never got a browser
    assert t2.status == "failed" and any("stopped; nothing was placed" in m for m in h.told)


async def test_same_profile_for_another_store_is_refused(db):
    await sandbox.bind_profile(db, "fam-a", "zepto", "p1")
    with pytest.raises(sandbox.ProfileConflict):
        await sandbox.bind_profile(db, "fam-a", "uber", "p1")
    await sandbox.bind_profile(db, "shadow:fam-a", "zepto", "p1")  # shadow memory is the same family


@pytest.mark.parametrize("path", ["done", "store_failed", "cancelled", "timeout", "crash"])
async def test_every_exit_path_stops_the_browser(db, at, sessions, path, monkeypatch):
    now = at("2026-10-04 10:00")
    script = {"prepare": [{"logged_in": True, "needs_otp": False, "blocked": False, "problem": "store closed"}] if path == "store_failed" else [CART]}
    agent = FakeAgent(script=script, finish_after_polls=99 if path == "timeout" else 1)
    h = H(sessions, agent)
    t = await order(db)
    await h.tick()  # run started
    assert len(await open_rows(db)) == 1
    if path == "done":
        await db.refresh(t)
        t.status = "done"
        await db.commit()
    elif path == "cancelled":
        await runtime.request_cancel(db, agent, t, by="e", reason="changed mind")
        await db.commit()
    elif path == "timeout":
        clock.set_now(now + runtime.RUN_TIMEOUT + timedelta(minutes=1))
    elif path == "crash":
        async def boom(task_id):
            raise ValueError("unreadable report")
        monkeypatch.setattr(agent, "poll", boom)
    await h.tick(4)
    if path in ("store_failed",):
        clock.set_now(clock.now() + runtime.IDLE_CLOSE + timedelta(minutes=1))
        await h.tick()
    await db.refresh(t)
    assert t.status in ("done", "failed", "cancelled", "awaiting_confirm"), t.status
    if t.status == "awaiting_confirm":  # store_failed may still offer a cart; waiting → browser released after idle
        clock.set_now(clock.now() + runtime.IDLE_CLOSE + timedelta(minutes=1))
        await h.tick()
    assert agent.sessions_stopped == ["s-t1"], (path, agent.sessions_stopped)
    assert await open_rows(db) == []


async def test_sweeper_stops_orphans_only(db, at, sessions):
    now = at("2026-10-04 10:00")
    agent = FakeAgent()
    for sid, mins in (("old-orphan", 30), ("young", 5), ("old-in-use", 30)):
        await sandbox.started(db, session_id=sid, task_id="x", family_id="fam-s", service="zepto", profile_id=None)
        (await db.get(sandbox.BrowserSession, sid)).started_at = now - timedelta(minutes=mins)
    t = await order(db, "fam-s")
    t.status, t.agent_session = "running", "old-in-use"
    await db.commit()
    out = await sandbox.sweep(sessions, agent)
    assert agent.sessions_stopped == ["old-orphan"] and out["stopped"] == 1
    assert {r.session_id for r in await open_rows(db)} == {"young", "old-in-use"}


async def test_sweeper_retries_when_stop_fails(db, at, sessions):
    now = at("2026-10-04 10:00")
    await sandbox.started(db, session_id="s9", task_id="x", family_id="f", service="ola", profile_id=None)
    (await db.get(sandbox.BrowserSession, "s9")).started_at = now - timedelta(hours=1)
    await db.commit()

    class Down(FakeAgent):
        async def stop_session(self, session_id):
            raise RuntimeError("browser-use PATCH sessions 503: down")

    out = await sandbox.sweep(sessions, Down())
    assert out == {"checked": 1, "stopped": 0, "failed": 1} and len(await open_rows(db)) == 1


async def test_login_state_audit_and_service_status(db, at, sessions):
    at("2026-10-04 10:00")
    h = H(sessions, FakeAgent(script={"prepare": [OTP]}))
    t = await order(db, "fam-l", "uber")
    await h.tick(2)
    rows = {r["service"]: r for r in await sandbox.logins(db, "fam-l")}
    assert rows["uber"]["state"] == "expired" and rows["uber"]["problem"] == "asked for a login code"
    await sandbox.note_login(db, "fam-l", "uber", CART)
    assert (await sandbox.logins(db, "fam-l"))[0]["state"] == "ok"
    await db.refresh(t)
    audit = task_json(t)["audit"]
    assert audit and audit[0].startswith("Prepare run finished after 12 steps on shop.test")
    c = tools.TurnCtx(session=db, host=SimHost(), family_id="fam-l", elder={"id": "e", "name": "Kamla"}, speaker={"id": "e", "role": "elder"}, members=[])
    out, err = await tools.run(c, "service_status", {"service": "uber"})
    apps = json.loads(out)["apps"]
    assert not err and apps == [{"service": "Uber", "login": "ok", "lastWorked": apps[0]["lastWorked"], "note": "logged in last time"}]
    out, _ = await tools.run(c, "service_status", {"service": "rapido"})
    assert json.loads(out)["apps"][0]["login"] == "never used"


async def test_task_rows_unchanged_shape(db):
    t = Task  # the audit lives in details; no new task columns
    assert "audit" not in t.__table__.columns
