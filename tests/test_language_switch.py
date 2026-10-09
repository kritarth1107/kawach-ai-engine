"""Switching language replaces the whole setting (live 2026-10-09: 'Talk to me in Gujarati' kept dialect Marwari)."""

import json

from app.brain import tools
from app.care import store
from app.sim.world import SimHost

FAM, MAA = "fam-lang", "maa-lang"


async def test_switching_language_drops_the_old_dialect(db, at):
    at("2026-10-09 23:58")
    elder = {"id": MAA, "name": "Vasundara", "role": "elder"}
    await store.save_roster(db, FAM, elder, [elder])
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=elder, speaker=elder, members=[elder])

    async def set_(**a):
        out, err = await tools.run(c, "language_preference", a)
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
    elder = {"id": MAA, "name": "Vasundara", "role": "elder"}
    await store.save_roster(db, FAM, elder, [elder])
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=elder, speaker=elder, members=[elder],
                      profiles={MAA: {"script": "latin", "saved": {"language": "gu"}, "now": "latin"}})
    assert guards.target(c.profiles[MAA]) == "gujarati"
    out, err = await tools.run(c, "language_preference", {"dialect": "Chhattisgarhi"})
    assert not err, out
    assert guards.target(c.profiles[MAA]) == "devanagari" and guards.dialect_of(c.profiles[MAA]) == "hne"
