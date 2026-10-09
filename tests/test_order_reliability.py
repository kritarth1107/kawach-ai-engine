"""Orders do not silently break: the founder hears before the browser credit runs out, and a placed order is followed
up until it arrives."""

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care import store, work
from app.care.models import OpenLoop
from app.tasks import credits, runtime
from tests.test_tasks import CART, PLACED, FakeAgent, Harness, _order

import pytest


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class Agent:
    def __init__(self, usd):
        self.usd = usd

    async def _req(self, method, path, body=None):
        assert path == "/billing/account"
        return {"totalCreditsBalanceUsd": self.usd}


class Host:
    def __init__(self):
        self.calls = []

    async def call(self, tool, args, **kw):
        self.calls.append((tool, args))
        return {"sent": True}


async def test_low_credit_emails_the_founder_once_per_half_day(db, at, sessions):
    at("2026-10-09 10:00")
    h = Host()
    assert (await credits.check(sessions, Agent(12.0), h))["alerted"] is False and not h.calls
    out = await credits.check(sessions, Agent(2.4), h)
    assert out["alerted"] and h.calls[0][0] == "ops_alert" and "$2.40" in h.calls[0][1]["subject"]
    at("2026-10-09 11:30")
    assert (await credits.check(sessions, Agent(2.1), h))["alerted"] is False, "once per half day"
    at("2026-10-09 14:00")
    assert (await credits.check(sessions, Agent(1.9), h))["alerted"] is True


async def test_placed_order_gets_one_arrival_follow_up(db, at, sessions):
    at("2026-10-09 10:00")
    h = Harness(sessions, FakeAgent(script={"prepare": [CART], "place": [{**PLACED, "eta": "12 mins"}]}))
    t = await _order(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    await runtime.provide_input(db, t, kind="confirm", value="yes", by="cg-t", by_is_elder=False)
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "done"
    loops = list((await db.execute(select(OpenLoop).where(OpenLoop.family_id == t.family_id, OpenLoop.kind == "delivery"))).scalars())
    assert len(loops) == 1 and loops[0].owner_id == t.requested_by
    assert timedelta(minutes=30) <= loops[0].wake_at - t.updated_at <= timedelta(minutes=34), "ETA 12 min + 20 min margin"
    item = next(r for r in await work.items(db, t.family_id) if r["kind"] == "delivery")
    assert item["next_action"] == "confirm the order arrived" and item["owner"] == t.requested_by
