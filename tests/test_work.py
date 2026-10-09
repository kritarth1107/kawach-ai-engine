"""Every job the family handed over has a state, an owner and a next action; stuck jobs are flagged (founder's slides)."""

import json
from datetime import timedelta

from app.brain import tools
from app.care import store, work
from app.core import clock
from app.sim.world import SimHost
from app.tasks import runtime

FAM, MAA, SON = "fam-work", "maa-work", "son-work"
ROSTER = [{"id": MAA, "name": "Kamla", "role": "elder"}, {"id": SON, "name": "Ravi", "role": "primary caregiver"}]


def ctx(db, who=SON, role="primary caregiver", system=False):
    speaker = {"id": "saheli-scheduler", "name": "Scheduler", "role": "system"} if system else {"id": who, "name": who, "role": role}
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ROSTER[0], speaker=speaker, members=ROSTER)


async def test_orders_and_loops_in_one_list_with_owner_and_next_action(db, at):
    now = at("2026-10-09 10:00")
    await store.save_roster(db, FAM, ROSTER[0], ROSTER)
    t = await runtime.create(db, family_id=FAM, subject_id=MAA, requested_by=MAA, service="blinkit", kind="order", goal="Atta",
                             details={"items": [{"name": "Atta", "qty": 1}]})
    t.status, t.input_needed = "needs_input", "otp"
    await store.open_loop(db, family_id=FAM, subject_id=MAA, kind="family_task", title="Fix the geyser", owner_id=SON,
                          wake_at=now + timedelta(hours=3))
    # a check-in question opened on a scheduled turn belongs to the person asked, not to the scheduler
    out, err = await tools.run(ctx(db, system=True), "open_loop", {"kind": "question", "title": "Weight check", "check_back_in_minutes": 1440})
    assert not err
    await db.commit()
    rows = await work.items(db, FAM)
    by_title = {r["title"].split(":")[-1].strip(): r for r in rows}
    order = next(r for r in rows if r["source"] == "task")
    assert order["state"] == "waiting" and order["owner"] == MAA and order["next_action"].startswith("send the login code")
    assert by_title["Fix the geyser"]["owner"] == SON and by_title["Fix the geyser"]["next_action"] == "do it: Fix the geyser"
    assert by_title["Weight check"]["owner"] == MAA, "owned by the person asked"
    assert all(r["next_action"] for r in rows)
    # WhatsApp: "what do I need to do?"
    mine = json.loads((await tools.run(ctx(db), "whats_pending", {"mine": True}))[0])
    assert [x["title"] for x in mine["open"]] == ["Fix the geyser"]
    # the brain sees who acts next
    text = work.brief(rows, {MAA: "Kamla", SON: "Ravi"})
    assert "→ Kamla: send the login code" in text and "→ Ravi: do it: Fix the geyser" in text


async def test_a_job_past_its_deadline_is_flagged_stuck(db, at):
    now = at("2026-10-09 10:00")
    await store.save_roster(db, FAM, ROSTER[0], ROSTER)
    await store.open_loop(db, family_id=FAM, subject_id=MAA, kind="family_task", title="Buy a new walker", owner_id=SON, wake_at=now)
    await db.commit()
    at("2026-10-09 15:00")
    rows = await work.items(db, FAM)
    assert rows[0]["stuck"] == "past its deadline"
    assert "[STUCK: past its deadline]" in work.brief(rows, {SON: "Ravi"})
