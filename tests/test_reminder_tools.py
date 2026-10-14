"""Saheli's reminder tools (founder 2026-10-12: reminders show in the calendar and can be skipped, stopped or changed
there, and the same on WhatsApp). The brain reads the words and passes the schedule as data; the reminder service in
the backend keeps them (its own tests: kavach-backend scripts/test-reminders.ts)."""

import json

from app.brain import loop, persona, tools
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
    out, err = await run(db, SON, h, "set_reminder", {"text": "Yoga", "times": ["07:00"], "repeat": "weekly", "days": [0, 2, 4], "until": "2026-11-30"})
    assert err and "whose" in out["refused"], "no about, and nothing just delivered: ask whose, do not use the care recipient"
    await run(db, SON, h, "set_reminder", {"text": "Yoga", "times": ["07:00"], "repeat": "weekly", "days": [0, 2, 4], "until": "2026-11-30", "about": ELDER["id"]})
    assert h.calls[-1][1]["schedule"] == {"repeat": "weekly", "times": ["07:00"], "days": [1, 3, 5], "endDate": "2026-11-30"}
    assert h.calls[-1][2]["subject_id"] == ELDER["id"]
    await run(db, ELDER, h, "set_reminder", {"text": "Drink water", "repeat": "daily", "every_hour": {"from": "09:00", "to": "21:00"}})
    assert h.calls[-1][1]["schedule"] == {"repeat": "daily", "windowStart": "09:00", "windowEnd": "21:00"}


async def test_a_one_off_stays_a_one_off(db, at):
    """Live 2026-10-12: "Remind me in next 1 hour" at 23:16 became a reminder at 00:15 every night. Saheli must say
    whether it repeats, and "in an hour" is worked out by code (the date rolls over past midnight)."""
    h = Host()
    at("2026-10-11 23:16")
    out, err = await run(db, SON, h, "set_reminder", {"text": "Take your magnesium tablet", "times": ["00:15"], "about": SON["id"]})
    assert err and "repeats" in out["refused"], "no silent every-day default"
    assert not h.calls
    _, err = await run(db, SON, h, "set_reminder", {"text": "Take your magnesium tablet", "in_minutes": 60, "about": SON["id"],
                                                     "item": "medicine:pure_nutrition_magnesium_glycinate_with_zinc"})
    assert not err
    tool, args, _ = h.calls[-1]
    assert tool == "create_reminder"
    assert args == {"text": "Take your magnesium tablet", "schedule": {"repeat": "once", "times": ["00:16"], "date": "2026-10-12"},
                    "item": "medicine:pure_nutrition_magnesium_glycinate_with_zinc"}
    out, err = await run(db, SON, h, "set_reminder", {"text": "x", "in_minutes": 60, "times": ["10:00"], "about": SON["id"]})
    assert err and "not both" in out["refused"]
    out, err = await run(db, SON, h, "set_reminder", {"text": "x", "in_minutes": 0, "about": SON["id"]})
    assert err
    # every day only when they asked for it
    _, err = await run(db, SON, h, "set_reminder", {"text": "Gym", "times": ["19:30"], "repeat": "daily", "about": SON["id"]})
    assert not err and h.calls[-1][1]["schedule"] == {"repeat": "daily", "times": ["19:30"]} and "item" not in h.calls[-1][1]


async def test_a_reminders_words_can_be_changed(db):
    """Live 2026-10-12: "add a cigarette emoji and a red heart in the reminder message" was saved as a note, and the
    reminder kept going out with its old words. The words are changed on the reminder itself."""
    h = Host({"update_reminder": {"ok": True, "reminder": {"reminderId": "r1", "text": "🚭 Skip the cigarette ❤️", "status": "active"}}})
    out, err = await run(db, SON, h, "change_reminder", {"id": "r1", "action": "change", "text": "🚭 Skip the cigarette ❤️", "about": SON["id"]})
    assert not err and h.calls[-1][:2] == ("update_reminder", {"reminderId": "r1", "text": "🚭 Skip the cigarette ❤️"})
    await run(db, SON, h, "change_reminder", {"id": "r1", "action": "change", "item": "", "about": SON["id"]})
    assert h.calls[-1][1] == {"reminderId": "r1", "item": None}, "an empty item removes the link"
    spec = next(x for x in tools.specs() if x.name == "change_reminder")
    assert "words" in spec.description and "note or skill never changes them" in spec.description
    assert "saved words" in persona.PERSONA and "REMINDERS DUE SOON" in persona.PERSONA
    assert "do not ask what it is for" in persona.PERSONA
    assert "log_activity it in that same turn" in persona.PERSONA
    assert "one short message per fact" in persona.REPLY_FORMAT
    spec = next(x for x in tools.specs() if x.name == "set_reminder")
    assert "do not ask what it is for" in spec.description
    walk = next(x for x in tools.specs() if x.name == "log_activity")
    assert "before you praise them" in walk.description


async def test_a_dose_just_sent_is_named_for_the_reply(db):
    """A reply to "Time for Metformin" is that dose: a new time is change_reminder on that medicine key, and finishing
    it is log_dose. The person's words are not matched."""
    from app.care import store

    await store.write_fact(db, family_id=FAM, subject_id=SON["id"], domain="medicine", key="medicine:metformin",
                           value={"name": "Metformin", "dose": "500 mg", "times": ["08:00"]},
                           text="Metformin 500 mg at 08:00", source_kind="caregiver_said")
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="assistant", text="Time for Metformin 500 mg.")
    await db.commit()
    req = loop.TurnRequest(family_id=FAM, elder=ELDER, speaker=SON, members=[ELDER, SON], text="later")
    dynamic, _ = await loop.turn_context(db, req)
    assert "THE LAST MESSAGE YOU SENT was a dose reminder for Metformin 500 mg at 08:00" in dynamic
    assert "change_reminder id medicine:metformin" in dynamic and "log_dose" in dynamic
    assert "a line that is only ---" in dynamic
    other = loop.dose_answer_line("The maid comes at 11.", [("Metformin", "500 mg", ["08:00"])])
    assert other == ""


async def test_changing_a_medicine_dose_time_updates_the_record(db):
    from app.care import store

    await store.write_fact(db, family_id=FAM, subject_id=SON["id"], domain="medicine", key="medicine:metformin",
                           value={"name": "Metformin", "dose": "500 mg", "times": ["08:00"]},
                           text="Metformin 500 mg at 08:00", source_kind="caregiver_said")
    await db.commit()
    h = Host()
    out, err = await run(db, SON, h, "change_reminder",
                         {"id": "medicine:metformin", "action": "change", "times": ["2:00"], "about": SON["id"]})
    assert not err and out["result"] == "changed" and out["times"] == ["02:00"]
    assert all(c[0] != "update_reminder" for c in h.calls)
    assert any(c[0] == "sync_medicine_schedule" and c[1]["times"] == ["02:00"] for c in h.calls)
    fact = await store.active_fact(db, FAM, SON["id"], "medicine:metformin")
    assert fact is not None and fact.value["times"] == ["02:00"]


async def test_saheli_sees_the_reminders_due_soon(db, at):
    """Live 2026-10-12: at 00:12 "i had the medicine", at 00:15 the reminder for it came anyway. Saheli now sees what is
    due soon (sent by the backend with the message) and is told to skip one that is already done."""
    at("2026-10-12 00:12")
    soon = [{"id": "rem-mag", "person": SON["id"], "text": "Take your Magnesium Glycinate tablet", "day": "2026-10-12", "time": "00:15",
             "repeat": "once", "item": None},
            {"id": "rem-walk", "person": ELDER["id"], "text": "Evening walk", "day": "2026-10-12", "time": "07:00", "repeat": "daily",
             "item": None}]
    req = loop.TurnRequest(family_id=FAM, elder=ELDER, speaker=SON, members=[ELDER, SON, DAUGHTER], text="i had the medicine",
                           reminders_soon=soon)
    dynamic, _ = await loop.turn_context(db, req)
    block = dynamic.split("REMINDERS DUE SOON", 1)[1]
    assert "[rem-mag] 2026-10-12 00:15 for the speaker" in block and "[rem-walk] 2026-10-12 07:00 for Shanti" in block
    assert "skip that time with change_reminder" in block
    plain = loop.TurnRequest(family_id=FAM, elder=ELDER, speaker=SON, members=[ELDER, SON, DAUGHTER], text="hi")
    assert "REMINDERS DUE SOON" not in (await loop.turn_context(db, plain))[0]


def test_the_backend_can_send_the_reminders_due_soon():
    from app.api.brain import TurnIn

    body = TurnIn.model_validate({"family_id": FAM, "elder": ELDER, "speaker": SON, "text": "hi",
                                  "reminders_soon": [{"id": "r1", "person": SON["id"], "text": "Gym", "day": "2026-10-12", "time": "19:30",
                                                      "repeat": "daily", "item": None}]})
    assert body.reminders_soon[0].time == "19:30"
    assert TurnIn.model_validate({"family_id": FAM, "elder": ELDER, "speaker": SON, "text": "hi"}).reminders_soon == []


async def test_set_reminder_asks_instead_of_guessing(db):
    h = Host({"create_reminder": {"status": "not_set", "error": "a once reminder needs its date (YYYY-MM-DD)"}})
    out, err = await run(db, SON, h, "set_reminder", {"text": "Pay rent", "times": ["10:00"], "repeat": "once", "about": SON["id"]})
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


async def test_a_caregivers_own_reminder_is_not_saved_on_the_care_recipient(db):
    """Live 2026-10-14: a caregiver answered the reminder just sent to him; about was left out and a new reminder was
    created for the care recipient. The item just delivered, or a care-record key only they have, decides the subject."""
    from app.care import store

    h = Host()
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="assistant", text="Time for your tablet",
                         meta={"proactive": True, "via": "reminder", "reminder_id": "r-own"})
    await db.commit()
    _, err = await run(db, SON, h, "set_reminder", {"text": "Take it", "times": ["02:00"], "repeat": "daily"})
    assert not err and h.calls[-1][2]["subject_id"] == SON["id"]
    _, err = await run(db, SON, h, "change_reminder", {"id": "r-own", "action": "change", "times": ["02:00"], "repeat": "daily"})
    assert not err and h.calls[-1][2]["subject_id"] == SON["id"]
    # a nudge on his thread about the care recipient stays theirs
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="assistant", text="Her walk",
                         meta={"proactive": True, "nudge": {"kind": "upcoming", "about_ref": ELDER["id"]}})
    await db.commit()
    _, err = await run(db, SON, h, "set_reminder", {"text": "Walk", "times": ["07:00"], "repeat": "daily"})
    assert not err and h.calls[-1][2]["subject_id"] == ELDER["id"]
    # his own medicine key, with no message to go on
    await store.write_fact(db, family_id=FAM, subject_id=SON["id"], domain="medicine", key="medicine:tablet",
                           value={"name": "Tablet"}, text="Tablet", source_kind="caregiver_said", replace=True)
    await db.commit()
    fresh = Host()
    # the last assistant turn is still the nudge about the elder, so the key must win when it is given
    _, err = await run(db, SON, fresh, "set_reminder", {"text": "Take it", "times": ["02:00"], "repeat": "daily", "item": "medicine:tablet"})
    assert not err and fresh.calls[-1][2]["subject_id"] == SON["id"]
    await store.write_fact(db, family_id=FAM, subject_id=ELDER["id"], domain="medicine", key="medicine:tablet",
                           value={"name": "Tablet"}, text="Tablet", source_kind="caregiver_said", replace=True)
    await db.commit()
    out, err = await run(db, SON, Host(), "set_reminder", {"text": "Take it", "times": ["02:00"], "repeat": "daily", "item": "medicine:tablet"})
    assert err and "whose" in out["refused"], "both have that key: ask whose"


async def test_a_failed_change_does_not_create_a_second_reminder(db):
    h = Host({"update_reminder": {"ok": False, "error": "Reminder not found."}})
    c = ctx(db, SON, h)
    out, err = await tools.run(c, "change_reminder", {"id": "nope", "action": "change", "times": ["02:00"], "repeat": "daily", "about": SON["id"]})
    refused = json.loads(out)
    assert err and "Do not call set_reminder" in refused["refused"]
    out, err = await tools.run(c, "set_reminder", {"text": "Take it", "times": ["02:00"], "repeat": "daily", "about": SON["id"]})
    assert err and "could not change it" in json.loads(out)["refused"]
    assert not any(call[0] == "create_reminder" for call in h.calls)


def test_reminder_writes_are_not_run_in_shadow_and_saheli_knows_the_tools():
    assert {"create_reminder", "cancel_reminder", "skip_reminder", "update_reminder"} <= WRITE_TOOLS
    assert "change_reminder" in persona.PERSONA and "reminders tool" in persona.PERSONA
    names = {s.name for s in tools.specs()}
    assert {"set_reminder", "reminders", "change_reminder"} <= names
