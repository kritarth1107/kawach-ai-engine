"""Every dashboard page is about one person: the orders, rides, spending, jobs and approvals shown for someone are the
ones placed FOR them (care_tasks.subject_id, start_task's for_whom), never the rest of the family's (founder
2026-10-13: on his mother's profile, Orders & rides showed his own orders and rides)."""

import uuid
from datetime import timedelta

import httpx
import pytest

from app.api import dash
from app.care import features, store, work
from app.core.security import verify_api_secret
from app.db.session import get_db
from app.main import app
from app.sim.world import SimHost, World
from app.tasks.models import Task

FAM, MAA, SON = "fam-scope", "maa-scope", "son-scope"
ROSTER = [{"id": MAA, "name": "Kamla Devi", "role": "elder"}, {"id": SON, "name": "Ravi", "role": "primary caregiver"}]
ACTOR = {"id": SON, "name": "Ravi"}


@pytest.fixture
async def client(db, monkeypatch):
    host = SimHost(World(backend_record={"schedules": []}))
    monkeypatch.setattr(dash, "make_host", lambda: host)

    async def _db():
        yield db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[verify_api_secret] = lambda: None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c
    app.dependency_overrides.clear()


def _task(when, *, subject, by, service, kind, goal, status="done", total=None, input_needed=None) -> Task:
    placed = status == "done"
    result = ({"placed": True} if kind == "order" else {"booked": True}) | {"total": total} if placed else {}
    return Task(id=uuid.uuid4(), family_id=FAM, subject_id=subject, requested_by=by, service=service, kind=kind, goal=goal,
                details={"for_set": True}, status=status, phase="place" if placed else "prepare", input_needed=input_needed,
                result=result, history=[], created_at=when, updated_at=when, deadline_at=when + timedelta(hours=2))


async def _family(db, at):
    """Ravi cares for his mother and keeps his own self care: his own orders and a ride, and orders he placed for her."""
    now = at("2026-10-13 10:00")
    await store.save_roster(db, FAM, ROSTER[0], ROSTER)
    rows = {
        "son_order": _task(now - timedelta(hours=5), subject=SON, by=SON, service="blinkit", kind="order", goal="Protein bars for Ravi", total="₹300"),
        "son_ride": _task(now - timedelta(hours=4), subject=SON, by=SON, service="uber", kind="ride", goal="Cab to the office", total="₹250"),
        "son_live": _task(now - timedelta(hours=1), subject=SON, by=SON, service="zepto", kind="order", goal="Mints for Ravi",
                          status="awaiting_confirm", input_needed="confirm"),
        "son_approve": _task(now - timedelta(minutes=50), subject=SON, by=SON, service="swiggy", kind="order", goal="Biryani for Ravi",
                             status="awaiting_confirm", input_needed="approve"),
        "maa_order": _task(now - timedelta(hours=3), subject=MAA, by=SON, service="apollo", kind="order", goal="BP tablets for Maa", total="₹500"),
        "maa_live": _task(now - timedelta(minutes=30), subject=MAA, by=SON, service="instamart", kind="order", goal="Sugar for Maa",
                          status="awaiting_confirm", input_needed="confirm"),
        "maa_approve": _task(now - timedelta(minutes=20), subject=MAA, by=MAA, service="zomato", kind="order", goal="Thali for Maa",
                             status="awaiting_confirm", input_needed="approve"),
    }
    db.add_all(rows.values())
    await store.open_loop(db, family_id=FAM, subject_id=SON, kind="family_task", title="Renew Ravi's gym pass", owner_id=SON,
                          wake_at=now - timedelta(hours=6))
    await store.open_loop(db, family_id=FAM, subject_id=MAA, kind="family_task", title="Take Maa for the eye test", owner_id=SON,
                          wake_at=now + timedelta(hours=6))
    await db.commit()
    return {k: str(t.id) for k, t in rows.items()}


async def test_orders_and_rides_are_the_selected_persons_only(client, db, at):
    ids = await _family(db, at)
    maa = (await client.get(f"/v2/dash/{FAM}/{MAA}/overview")).json()["tasks"]
    assert {t["id"] for t in maa} == {ids["maa_order"], ids["maa_live"], ids["maa_approve"]}
    assert all(t["subjectId"] == MAA for t in maa)
    assert not [t for t in maa if "Ravi" in t["goal"] or t["kind"] == "ride"], "his orders and his ride are not on her page"
    son = (await client.get(f"/v2/dash/{FAM}/{SON}/overview")).json()["tasks"]
    assert {t["id"] for t in son} == {ids["son_order"], ids["son_ride"], ids["son_live"], ids["son_approve"]}
    assert all(t["subjectId"] == SON for t in son)


async def test_home_lists_and_asks_only_about_the_selected_person(client, db, at):
    ids = await _family(db, at)
    home = (await client.get(f"/v2/dash/{FAM}/{MAA}/home")).json()
    assert {t["id"] for t in home["tasks"]} == {ids["maa_order"], ids["maa_live"], ids["maa_approve"]}
    asks = {n["taskId"] for n in home["needsYou"] if n["kind"] == "task"}
    assert asks == {ids["maa_live"], ids["maa_approve"]}
    stuck = [n["title"] for n in home["needsYou"] if n["kind"] == "stuck"]
    assert "Renew Ravi's gym pass" not in stuck, "his overdue chore is not on her Home"
    mine = (await client.get(f"/v2/dash/{FAM}/{SON}/home")).json()
    assert {n["taskId"] for n in mine["needsYou"] if n["kind"] == "task"} == {ids["son_live"], ids["son_approve"]}
    assert "Renew Ravi's gym pass" in [n["title"] for n in mine["needsYou"] if n["kind"] == "stuck"]


async def test_spending_counts_only_what_was_for_the_selected_person(client, db, at):
    await _family(db, at)
    s = (await client.get(f"/v2/dash/{FAM}/{MAA}/spending", params={"month": "2026-10"})).json()
    assert s["total"] == 500 and s["count"] == 1 and set(s["byPerson"]) == {MAA}
    assert [i["goal"] for i in s["items"]] == ["BP tablets for Maa"]
    mine = (await client.get(f"/v2/dash/{FAM}/{SON}/spending", params={"month": "2026-10"})).json()
    assert mine["total"] == 550 and set(mine["byPerson"]) == {SON} and mine["byKind"] == {"order": 300, "ride": 250}
    # the brain's spending tool still answers for the whole family ("how much did we spend?")
    assert (await features.spending(db, FAM, "2026-10"))["total"] == 1050


async def test_handed_over_jobs_and_approvals_are_the_selected_persons(client, db, at):
    ids = await _family(db, at)
    rows = (await client.get(f"/v2/dash/{FAM}/{MAA}/work", params={"all": True})).json()["items"]
    assert rows and {r["subject"] for r in rows} == {MAA}
    assert "Renew Ravi's gym pass" not in [r["title"] for r in rows]
    assert not [r for r in rows if "Ravi" in r["title"]]
    limits = (await client.get(f"/v2/dash/{FAM}/{MAA}/boundaries")).json()
    assert [w["taskId"] for w in limits["waiting"]] == [ids["maa_approve"]] and limits["waiting"][0]["subjectId"] == MAA
    assert limits["monthSpent"] == 1050, "the monthly cap is the family's: its spend stays the whole family's"
    assert [w["taskId"] for w in (await client.get(f"/v2/dash/{FAM}/{SON}/boundaries")).json()["waiting"]] == [ids["son_approve"]]
    # the brain and WhatsApp still see every job of the family
    assert {r["subject"] for r in await work.items(db, FAM)} == {MAA, SON}


async def test_a_task_for_someone_else_cannot_be_answered_from_this_persons_page(client, db, at):
    ids = await _family(db, at)
    for path, body in ((f"tasks/{ids['son_live']}/input", {"actor": ACTOR, "kind": "confirm", "value": "yes"}),
                       (f"tasks/{ids['son_live']}/cancel", {"actor": ACTOR})):
        r = await client.post(f"/v2/dash/{FAM}/{MAA}/{path}", json=body)
        assert r.status_code == 404, path
    assert (await client.get(f"/v2/dash/{FAM}/{MAA}/tasks/{ids['son_live']}/live")).status_code == 404
    t = await db.get(Task, uuid.UUID(ids["son_live"]), populate_existing=True)
    assert t is not None and t.status == "awaiting_confirm" and not t.cancel_requested, "nothing changed on his order"
    # from his own page it is his to answer
    assert (await client.get(f"/v2/dash/{FAM}/{SON}/tasks/{ids['son_live']}/live")).json() == {"url": None}
