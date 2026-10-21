"""The admin's insight routes (founder 21 Oct): every query runs, test families are kept apart, no message text leaves."""

from datetime import timedelta

import pytest

from app.api import admin
from app.care import store
from app.care.models import NudgeLedger
from app.care.why import TraceTurn
from app.core import clock
from app.quality.models import ReplyVerdict
from app.tasks.models import Task

F = "fam-insights-1"


@pytest.mark.asyncio
async def test_every_insight_route_answers_with_counts_only(db):
    now = clock.now()
    await store.add_turn(db, family_id=F, thread_id="p1", role="user", text="hello", speaker_id="p1")
    await store.add_turn(db, family_id=F, thread_id="p1", role="user", text="hello", speaker_id="p1")
    await store.add_turn(db, family_id=F, thread_id="p1", role="assistant", text="nudge", meta={"proactive": True})
    await store.add_turn(db, family_id=F, thread_id="p1", role="assistant", text="your 8 pm tablet", meta={"proactive": True, "reminder_id": "r1"})
    await store.add_turn(db, family_id="sim-1", thread_id="p9", role="user", text="test", speaker_id="p9")
    db.add(TraceTurn(trace_id="t1", family_id=F, thread_id="p1", kind="message", at=now, ms=900, cost_inr=1.5, models=[]))
    db.add(TraceTurn(trace_id="t2", family_id="sim-2", thread_id="p9", kind="message", at=now, ms=900, cost_inr=4.0, models=[]))
    db.add(NudgeLedger(family_id=F, person_id="p1", kind="know", slot_key="p1:x", status="skipped", reason='guard:this question was already asked today: "Secret words"',
                       ref="nudge:1", text="Secret words", planned_at=now))
    db.add(NudgeLedger(family_id=F, person_id="p1", kind="know", slot_key="p1:y", status="sent", reason="", ref="nudge:2", text="hi", planned_at=now, sent_at=now, replied_at=now))
    await store.record_event(db, family_id=F, subject_id="p1", kind="reminder_failed", summary="Thyronorm 08:00 — window_closed", payload={}, actor_id="saheli", at=now, ref="r1")
    db.add(Task(family_id=F, subject_id="p1", requested_by="p1", service="blinkit", kind="order", goal="milk", status="failed", details={"fail_code": "not_deliverable"},
                created_at=now - timedelta(minutes=5), updated_at=now, deadline_at=now + timedelta(hours=1)))
    db.add(ReplyVerdict(reply_id=1, family_id=F, thread_id="p1", at=now, day=clock.ist_day(), kind="reply", passed=False, issues=["ignored_ask"], verdict={},
                        judged_at=now))
    await db.flush()

    costs = await admin.insights_family_costs(db, days=7)
    names = {f["familyId"]: f for f in costs["families"]}
    assert names[F]["costInr"] == 1.5 and names["tests and simulations"]["costInr"] == 4.0

    people = await admin.insights_people(db, days=30)
    me = next(p for p in people["people"] if p["familyId"] == F)
    assert me["in7"] == 2 and all(p["familyId"] != "sim-1" for p in people["people"])
    assert me["unansweredSinceLastReply"] == 1  # the reminder is not counted

    out = await admin.insights_outreach(db, days=7)
    assert out["nudges"]["sent"] == 1 and out["nudges"]["answered"] == 1
    assert "Secret" not in str(out) and out["nudges"]["skipReasons"][0][0].startswith("guard:this question was already asked today")

    q = await admin.insights_quality(db, days=7)
    assert q["issues"][0] == ("ignored_ask", 1) and q["daily"][0]["judged"] == 1

    o = await admin.insights_orders(db, days=7)
    assert o["stores"][0]["service"] == "blinkit" and o["failCodes"][0] == ("not_deliverable", 1)


@pytest.mark.asyncio
async def test_a_reported_reply_becomes_one_anonymised_open_case(db):
    from app.learn.models import ReplyLog
    from app.quality.models import ReplayCase
    from sqlalchemy import select

    await store.save_roster(db, "fam-report", {"id": "e1", "name": "Savitri"}, [{"id": "c1", "name": "Arjun", "role": "primary caregiver"}])
    db.add(ReplyLog(family_id="fam-report", thread_id="c1", at=clock.now(), kind="reply", situation="chat", speaker_role="caregiver",
                    user_text="aaj aur kal ka food diary", text="Kal: chawal, dal", tools=[]))
    await db.flush()
    body = admin.ReportReplyIn(family_id="fam-report", contains="chawal", note="Asked for today and yesterday; gave only yesterday.")
    out = await admin.admin_case_from_reply(body, db)
    assert out["created"] is True
    again = await admin.admin_case_from_reply(body, db)
    assert again["created"] is False
    case = (await db.execute(select(ReplayCase).where(ReplayCase.source == "founder"))).scalars().one()
    assert case.status == "open" and case.anonymised and "Savitri" not in str(case.context)
