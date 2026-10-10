"""The family's boundaries are enforced in code (the delegate model's 4th ring): limits, categories, who may order,
monthly cap, and only an approver's yes places something outside them."""

import json

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import tools
from app.care import boundaries, store
from app.sim.world import SimHost
from app.tasks import runtime
from app.tasks.models import Task
from tests.test_tasks import CART, PLACED, FakeAgent, Harness

FAM, ELDER, SON, NIECE = "fam-bd", "elder-bd", "son-bd", "niece-bd"
ROSTER = [{"id": ELDER, "name": "Kamla", "role": "elder"}, {"id": SON, "name": "Ravi", "role": "primary caregiver"},
          {"id": NIECE, "name": "Asha", "role": "caregiver"}]


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


async def _family(db, **policy):
    await store.save_roster(db, FAM, ROSTER[0], ROSTER)
    if policy:
        await boundaries.save(db, FAM, policy, by=SON)
    await db.commit()


async def _order(db, by=ELDER, service="instamart", kind="order") -> Task:
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=by, service=service, kind=kind, goal="Atta for Amma",
                             details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]})
    await db.commit()
    return t


async def _ready(db, h, t):
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm"


async def test_defaults_keep_todays_behaviour(db):
    await _family(db)
    p = await boundaries.get(db, FAM)
    assert p["elder_order_limit"] == 1500 and p["elder_ride_limit"] == 800 and p["monthly_cap"] is None
    roster = await store.roster(db, FAM)
    assert boundaries.approvers(p, roster) == [SON], "the primary caregiver approves by default"
    assert await boundaries.reasons(db, family_id=FAM, service="blinkit", kind="order", amount=900, requested_by=ELDER, elder_id=ELDER) == []


async def test_approval_round_trip_only_the_approver_counts(db, at, sessions):
    at("2026-10-09 10:00")
    await _family(db, elder_order_limit=300)
    h = Harness(sessions, FakeAgent(script={"prepare": [CART], "place": [PLACED]}))  # CART total ₹318
    t = await _order(db)
    await _ready(db, h, t)
    msg = await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    assert "needs approval first" in msg and "above the ₹300 limit" in msg and "Ravi" in msg
    await db.commit()
    # the request goes to the approver once (task tick), with what to call when they answer
    await h.tick(); await h.tick()
    asks = [m for m in h.told if m.startswith("[Approval needed]")]
    assert len(asks) == 1 and f"task_id {t.id}, kind approve" in asks[0]
    await db.refresh(t)
    # someone who is not the approver cannot approve; the elder's second "yes" does not place it either
    assert "only the family's approver" in await runtime.provide_input(db, t, kind="approve", value="yes", by=NIECE, by_is_elder=False)
    assert "waiting for the family approver" in await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    out = await runtime.provide_input(db, t, kind="approve", value="yes", by=SON, by_is_elder=False)
    assert out.startswith("approved. confirmed; placing it now") and t.phase == "place" and t.details["approved_by"] == SON
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "done" and t.result.get("placed")


async def test_approver_can_decline(db, at, sessions):
    at("2026-10-09 10:00")
    await _family(db, elder_order_limit=300)
    h = Harness(sessions, FakeAgent(script={"prepare": [CART]}))
    t = await _order(db)
    await _ready(db, h, t)
    await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True)
    out = await runtime.provide_input(db, t, kind="approve", value="no", by=SON, by_is_elder=False)
    assert t.status == "cancelled" and f"Tell person {ELDER}" in out
    assert not [r for r in h.agent.runs if r["phase"] == "place"]


async def test_categories_members_and_monthly_cap(db, at, sessions):
    at("2026-10-09 10:00")
    await _family(db, approval_categories=["food"], members={NIECE: {"can_order": False}}, monthly_cap=1000, anyone_over=2000)
    r = lambda **kw: boundaries.reasons(db, family_id=FAM, elder_id=ELDER, **kw)
    assert "approve every food order" in (await r(service="zomato", kind="order", amount=200, requested_by=SON))[0]
    assert "not allowed to order" in (await r(service="blinkit", kind="order", amount=100, requested_by=NIECE))[0]
    assert await r(service="rapido", kind="ride", amount=100, requested_by=NIECE) == [], "she may still book rides"
    assert "₹2000 approval limit" in " ".join(await r(service="blinkit", kind="order", amount=2500, requested_by=SON))
    # ₹900 already placed this month → a ₹200 order goes past the ₹1000 cap
    done = Task(family_id=FAM, subject_id=ELDER, requested_by=SON, service="blinkit", kind="order", goal="x", details={"confirmed_total": "₹900"},
                status="done", phase="place", result={"placed": True, "total": "₹900"}, history=[], created_at=at("2026-10-09 09:00"),
                updated_at=at("2026-10-09 09:00"), deadline_at=at("2026-10-09 11:00"))
    db.add(done); await db.commit()
    at("2026-10-09 10:00")
    assert "over the ₹1000 monthly limit" in " ".join(await r(service="blinkit", kind="order", amount=200, requested_by=SON))
    # the approver ordering for themselves goes through (they are the approver)
    h = Harness(sessions, FakeAgent(script={"prepare": [CART]}))
    t = await _order(db, by=SON, service="zomato")
    await _ready(db, h, t)
    assert (await runtime.provide_input(db, t, kind="confirm", value="yes", by=SON, by_is_elder=False)).startswith("confirmed; placing")


async def test_only_an_approver_changes_the_limits_on_whatsapp(db, at):
    at("2026-10-09 10:00")
    await _family(db)

    def ctx(who, role):
        return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ROSTER[0], speaker={"id": who, "name": who, "role": role}, members=ROSTER)

    out, err = await tools.run(ctx(ELDER, "elder"), "set_boundaries", {"elder_order_limit": 99999})
    assert err and "Only the family's approver" in out
    out, err = await tools.run(ctx(NIECE, "caregiver"), "set_boundaries", {"elder_order_limit": 500})
    assert err
    out, err = await tools.run(ctx(SON, "primary caregiver"), "set_boundaries", {"elder_order_limit": 500, "approval_categories": ["food"]})
    assert not err and "over ₹500" in json.loads(out)["saved"]
    out, err = await tools.run(ctx(SON, "primary caregiver"), "set_boundaries", {"approvers": [ELDER]})
    assert err and "cannot be the approver" in out
    read = json.loads((await tools.run(ctx(NIECE, "caregiver"), "boundaries", {}))[0])
    assert read["policy"]["elder_order_limit"] == 500 and read["approvers"] == ["Ravi"] and read["can_change"] is False
    # every change is a version with who changed it
    rows = await store.events(db, "fam-bd", "family", kinds=["boundary_changed"])
    assert rows and rows[-1].actor_id == SON
