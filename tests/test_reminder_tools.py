"""Saheli's reminder tools (founder 2026-10-12: reminders show in the calendar and can be skipped, stopped or changed
there, and the same on WhatsApp). The brain reads the words and passes the schedule as data; the reminder service in
the backend keeps them (its own tests: kavach-backend scripts/test-reminders.ts)."""

import json

from app.brain import persona, tools
from app.brain.host import WRITE_TOOLS

FAM = "fam-rem"
ELDER = {"id": "e-rem", "name": "Shanti", "role": "elder"}
SON = {"id": "c-rem", "name": "Mohan", "role": "primary caregiver"}
DAUGHTER = {"id": "c2-rem", "name": "Asha", "role": "co-caregiver"}


class Host:
    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or {}

    async def call(self, tool, args, **kw):
        self.calls.append((tool, args, kw))
        return self.answers.get(tool, {"ok": True, "status": "created", "reminder": {"reminderId": "r1", "text": args.get("text"),
                                                                                   "schedule": "Every day at 19:30", "status": "active"}})


def ctx(db, speaker, host):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ELDER, speaker=speaker, members=[ELDER, SON, DAUGHTER])


async def run(db, speaker, host, name, args):
    out, err = await tools.run(ctx(db, speaker, host), name, args)
    return json.loads(out), err


async def test_set_reminder_sends_the_schedule_as_data(db):
    h = Host()
    _, err = await run(db, SON, h, "set_reminder", {"text": "Call the bank", "times": ["17:00"], "repeat": "once", "date": "2026-10-15", "about": SON["id"]})
    assert not err
    tool, args, kw = h.calls[-1]
    assert tool == "create_reminder" and args == {"text": "Call the bank", "schedule": {"repeat": "once", "times": ["17:00"], "date": "2026-10-15"}}
    assert kw["subject_id"] == SON["id"] and kw["actor_id"] == SON["id"]
    # weekly days: the care record counts from Monday (0), the reminder service from Sunday (0)
    await run(db, SON, h, "set_reminder", {"text": "Yoga", "times": ["07:00"], "repeat": "weekly", "days": [0, 2, 4], "until": "2026-11-30"})
    assert h.calls[-1][1]["schedule"] == {"repeat": "weekly", "times": ["07:00"], "days": [1, 3, 5], "endDate": "2026-11-30"}
    assert h.calls[-1][2]["subject_id"] == ELDER["id"], "no about: the care recipient"
    await run(db, ELDER, h, "set_reminder", {"text": "Drink water", "every_hour": {"from": "09:00", "to": "21:00"}})
    assert h.calls[-1][1]["schedule"] == {"repeat": "daily", "windowStart": "09:00", "windowEnd": "21:00"}


async def test_set_reminder_asks_instead_of_guessing(db):
    h = Host({"create_reminder": {"status": "not_set", "error": "a once reminder needs its date (YYYY-MM-DD)"}})
    out, err = await run(db, SON, h, "set_reminder", {"text": "Pay rent", "times": ["10:00"], "repeat": "once"})
    assert err and "date" in out["refused"]
    out, err = await run(db, SON, Host(), "set_reminder", {"text": "Pay rent"})
    assert err and "times" in out["refused"]


async def test_reminders_lists_with_ids_and_next_time(db):
    h = Host({"list_reminders": {"reminders": [{"reminderId": "r1", "text": "Get ready for the gym", "schedule": "Every day at 19:30",
                                                 "status": "active", "next": {"day": "2026-10-12", "time": "19:30"}, "skips": []}]}})
    out, err = await run(db, SON, h, "reminders", {"about": SON["id"]})
    assert not err and out["reminders"] == [{"id": "r1", "text": "Get ready for the gym", "schedule": "Every day at 19:30", "status": "active",
                                              "next": "2026-10-12 19:30", "skipped": None}]
    assert h.calls[-1][1] == {"status": "active"}
    await run(db, SON, h, "reminders", {"about": SON["id"], "include_stopped": True})
    assert h.calls[-1][1] == {"status": "all"}


async def test_who_may_see_and_change_whose_reminders(db):
    h = Host()
    # a caregiver: their own and the care recipient's; never another caregiver's own (self care)
    for speaker, about, allowed in [(SON, SON["id"], True), (SON, ELDER["id"], True), (DAUGHTER, SON["id"], False),
                                    (ELDER, ELDER["id"], True), (ELDER, SON["id"], None)]:
        out, err = await run(db, speaker, h, "change_reminder", {"id": "r1", "action": "stop", "about": about})
        if allowed is None:  # the elder's "about" for someone else falls back to herself (ctx.subject)
            continue
        assert bool(err) == (not allowed), (speaker["name"], about, out)
    system = {"id": "saheli", "role": "system"}
    _, err = await run(db, system, h, "change_reminder", {"id": "r1", "action": "stop"})
    assert err


async def test_change_reminder_skips_stops_and_changes_by_id(db):
    h = Host({"cancel_reminder": {"ok": True, "reminder": {"reminderId": "r1", "status": "cancelled", "schedule": "Every day at 19:30"}},
              "skip_reminder": {"ok": True, "skipped": "2026-10-13@19:30", "reminder": {"reminderId": "r1", "status": "active"}},
              "update_reminder": {"ok": True, "reminder": {"reminderId": "r1", "status": "active", "schedule": "Every day at 19:00"}}})
    out, err = await run(db, SON, h, "change_reminder", {"id": "r1", "action": "skip", "date": "2026-10-13", "about": SON["id"]})
    assert not err and out["result"] == "skipped" and out["skipped"] == "2026-10-13@19:30"
    assert h.calls[-1][:2] == ("skip_reminder", {"reminderId": "r1", "date": "2026-10-13", "time": None, "undo": False})
    await run(db, SON, h, "change_reminder", {"id": "r1", "action": "unskip", "date": "2026-10-13", "about": SON["id"]})
    assert h.calls[-1][1]["undo"] is True
    out, _ = await run(db, SON, h, "change_reminder", {"id": "r1", "action": "change", "times": ["19:00"], "about": SON["id"]})
    assert h.calls[-1][:2] == ("update_reminder", {"reminderId": "r1", "schedule": {"times": ["19:00"]}}), "what is not given stays (no repeat sent)"
    assert out["reminder"]["schedule"] == "Every day at 19:00"
    _, err = await run(db, SON, h, "change_reminder", {"id": "r1", "action": "change", "about": SON["id"]})
    assert err, "nothing to change"
    out, _ = await run(db, SON, h, "change_reminder", {"id": "r1", "action": "stop", "about": SON["id"]})
    assert out["result"] == "stopped" and h.calls[-1][:2] == ("cancel_reminder", {"reminderId": "r1"})
    bad = Host({"skip_reminder": {"ok": False, "error": "It does not go out on that day."}})
    out, err = await run(db, SON, bad, "change_reminder", {"id": "r1", "action": "skip", "date": "2026-10-14", "about": SON["id"]})
    assert err and "does not go out" in out["refused"]


def test_reminder_writes_are_not_run_in_shadow_and_saheli_knows_the_tools():
    assert {"create_reminder", "cancel_reminder", "skip_reminder", "update_reminder"} <= WRITE_TOOLS
    assert "change_reminder" in persona.PERSONA and "reminders tool" in persona.PERSONA
    names = {s.name for s in tools.specs()}
    assert {"set_reminder", "reminders", "change_reminder"} <= names
