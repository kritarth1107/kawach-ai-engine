"""Nothing reaches a person in their night (found by the world message-budget sim, 2026-10-13).

- A scheduled turn about one person once messaged another family member at 22:00: send_message on a system turn now
  checks the recipient's own night (or the household's), except a no-answer alert loop and an order or ride update.
- The nightly brief's default ("after the elder is asleep") read only the old routine:sleep key, so every brief went at
  22:00; bedtimes are activities now. A default that lands in the caregiver's own known night moves out of it.
"""

import pytest

from app.brain import tools
from app.care import brief, store
from app.sim.world import SimHost

FAM = "fam-night"
MAA = {"id": "maa-n", "name": "Kamla", "role": "elder"}
SON = {"id": "son-n", "name": "Ravi", "role": "primary caregiver"}
SYSTEM = {"id": "saheli-scheduler", "role": "system"}


def ctx(db, host, ref):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=MAA, speaker=SYSTEM, members=[MAA, SON], message_ref=ref)


async def routine(db, who: str, type_: str, start: str):
    await store.write_fact(db, family_id=FAM, subject_id=who, domain="activity", key=f"activity:{type_}",
                           value={"type": type_, "kind": "routine", "start": start}, text=f"{type_} {start}", source_kind="caregiver_said")


async def test_a_scheduled_message_to_someone_in_the_household_night_is_refused(db, at):
    at("2026-10-12 22:30")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    host = SimHost()
    with pytest.raises(tools.ToolRefused, match="night for them"):
        await tools.send_message(ctx(db, host, "checkin:feelings:son-n:2026-10-12"), {"to": SON["id"], "text": "Kaisa raha din?"})
    assert not host.world.sent


async def test_their_own_night_counts_not_the_household_one(db, at):
    at("2026-10-12 10:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await routine(db, SON["id"], "sleep", "03:00")
    await routine(db, SON["id"], "wake_up", "11:00")
    at("2026-10-12 22:30")  # a late sleeper is awake at 22:30
    host = SimHost()
    await tools.send_message(ctx(db, host, "checkin:feelings:son-n:2026-10-12"), {"to": SON["id"], "text": "Kaisa raha din?"})
    assert host.world.sent and host.world.sent[-1]["to"] == SON["id"]
    at("2026-10-13 09:00")  # and asleep at 9 in the morning
    with pytest.raises(tools.ToolRefused, match="night for them"):
        await tools.send_message(ctx(db, host, "checkin:feelings:son-n:2026-10-13"), {"to": SON["id"], "text": "Good morning!"})


async def test_a_no_answer_alert_and_an_order_update_still_go_at_night(db, at):
    at("2026-10-12 23:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    loop = await store.open_loop(db, family_id=FAM, subject_id=MAA["id"], kind="question", title="Did Maa take the night tablet?",
                                 owner_id=MAA["id"], alert_rule="alert_caregiver")
    host = SimHost()
    await tools.send_message(ctx(db, host, f"wake:{loop.id}:3"), {"to": SON["id"], "text": "Kamla ji ne raat ki dawa ke baare mein jawab nahi diya."})
    await tools.send_message(ctx(db, host, "task:abc:placed"), {"to": SON["id"], "text": "Order placed: milk and bread, arriving in 20 min."})
    assert len(host.world.sent) == 2


async def test_the_brief_follows_the_elders_bedtime_saved_as_an_activity(db, at):
    at("2026-10-12 10:00")
    await routine(db, MAA["id"], "sleep", "21:00")
    await routine(db, MAA["id"], "wake_up", "05:30")
    t = await brief.default_time(db, FAM, MAA["id"])
    assert t.strftime("%H:%M") == "21:30"


async def test_a_default_brief_moves_out_of_the_caregivers_own_night_but_a_chosen_time_stays(db, at):
    at("2026-10-12 10:00")
    await routine(db, MAA["id"], "sleep", "22:00")
    await routine(db, MAA["id"], "wake_up", "06:00")
    await routine(db, SON["id"], "sleep", "21:30")
    await routine(db, SON["id"], "wake_up", "05:00")
    t = await brief.brief_time(db, FAM, SON["id"], MAA["id"])
    assert t.strftime("%H:%M") == "21:00"  # 30 minutes before his own bedtime, not 22:30 in his sleep
    await store.write_fact(db, family_id=FAM, subject_id=SON["id"], domain="routine", key=brief.SETTING_KEY, value={"time": "23:00"},
                           text="brief at 23:00", source_kind="caregiver_said")
    t = await brief.brief_time(db, FAM, SON["id"], MAA["id"])
    assert t.strftime("%H:%M") == "23:00"  # he chose it
