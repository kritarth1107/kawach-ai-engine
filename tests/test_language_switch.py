"""Switching language replaces the whole setting (live 2026-10-09: 'Talk to me in Gujarati' kept dialect Marwari)."""

import json

from app.brain import tools
from app.care import store
from app.sim.world import SimHost

FAM, MAA = "fam-lang", "maa-lang"


async def test_switching_language_drops_the_old_dialect(db, at):
    at("2026-10-09 23:58")
    elder = {"id": MAA, "name": "Sarla", "role": "elder"}
    await store.save_roster(db, FAM, elder, [elder])
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=elder, speaker=elder, members=[elder])

    async def set_(**a):
        out, err = await tools.run(c, "language_preference", {"about": MAA, **a})
        assert not err, out
        return json.loads(out)

    assert (await set_(dialect="Marwari"))["dialect"] == "mwr"
    got = await set_(language="Gujarati")
    assert got["language"] == "gu" and not got.get("dialect")
    row = await store.active_fact(db, FAM, MAA, "language:preferred")
    assert row.value.get("language") == "gu" and "dialect" not in row.value, row.value
    assert (await set_(dialect="Chhattisgarhi"))["dialect"] == "hne"
    row = await store.active_fact(db, FAM, MAA, "language:preferred")
    assert row.value == {"language": "hi", "dialect": "hne", "script": "native"} or row.value.get("dialect") == "hne"


async def test_a_switch_applies_to_this_turns_checks(db, at):
    from app.brain import guards

    at("2026-10-10 00:40")
    elder = {"id": MAA, "name": "Sarla", "role": "elder"}
    await store.save_roster(db, FAM, elder, [elder])
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=elder, speaker=elder, members=[elder],
                      profiles={MAA: {"script": "latin", "saved": {"language": "gu"}, "now": "latin"}})
    assert guards.target(c.profiles[MAA]) == "gujarati"
    out, err = await tools.run(c, "language_preference", {"dialect": "Chhattisgarhi", "about": MAA})
    assert not err, out
    assert guards.target(c.profiles[MAA]) == "devanagari" and guards.dialect_of(c.profiles[MAA]) == "hne"


async def test_whose_language_is_always_said_and_a_caregiver_can_ask_for_a_light_touch(db, at):
    """Live 2026-10-12, caregiver: 'shift that back to Hindi then slightly Marwadi, not proper Marwadi'. The setting is for
    the person named (about is required), and a light touch is kept as such."""
    at("2026-10-12 02:20")
    elder = {"id": MAA, "name": "Babli", "role": "elder"}
    son = {"id": "son-lang", "name": "Ravi", "role": "primary caregiver"}
    await store.save_roster(db, FAM, elder, [elder, son])
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=elder, speaker=son, members=[elder, son])
    out, err = await tools.run(c, "language_preference", {"language": "hi", "dialect": "Marwari"})
    assert err, "whose setting must be said"
    out, err = await tools.run(c, "language_preference", {"about": MAA, "language": "hi", "dialect": "Marwari", "mix": "light"})
    assert not err, out
    assert json.loads(out)["label"] == "Hindi with a light Marwari touch (मारवाड़ी)"
    row = await store.active_fact(db, FAM, MAA, "language:preferred")
    assert row.value.get("mix") == "light" and row.value.get("dialect") == "mwr"
    # "talk to me in Hindi" from the caregiver is his own setting, not Maa's
    out, err = await tools.run(c, "language_preference", {"about": son["id"], "language": "hi"})
    assert not err and (await store.active_fact(db, FAM, son["id"], "language:preferred")).value == {"language": "hi", "script": "native"} \
        or (await store.active_fact(db, FAM, son["id"], "language:preferred")).value.get("language") == "hi"
    assert (await store.active_fact(db, FAM, MAA, "language:preferred")).value.get("mix") == "light", "Maa's setting untouched"


async def test_a_message_saheli_starts_follows_the_saved_setting(db, at):
    """Live 2026-10-12: Maa's saved setting was set at 02:20 by the caregiver, yet the 08:30 follow-up came in the
    Chhattisgarhi of her 01:49 chat. A scheduled message for one person now says which setting to write in."""
    from app.brain import wake

    at("2026-10-12 08:30")
    elder = {"id": MAA, "name": "Babli", "role": "elder"}
    await store.save_roster(db, FAM, elder, [elder])
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="language", key="language:preferred",
                           value={"language": "hi", "dialect": "mwr", "mix": "light"}, text="Speaks Hindi with a light Marwari touch",
                           source_kind="caregiver_said", replace=True)
    await db.commit()
    line = await wake.saved_language_line(db, FAM, MAA)
    assert "Write it in Hindi with a light Marwari touch" in line and "not the language of an older chat" in line
    assert await wake.saved_language_line(db, FAM, "nobody") == ""
    assert await wake.saved_language_line(db, FAM, None) == ""
