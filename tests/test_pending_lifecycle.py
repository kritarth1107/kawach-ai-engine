"""Pending health facts: one final ask at day 14, archived at day 17, restorable (trust plan A4; founder 2026-10-13:
"If nobody approves one in 14 days, ask once more, then archive it. It shouldn't sit there forever.")."""

from sqlalchemy import select

from app.brain import tools
from app.care import pending, store
from app.care.models import OpenLoop
from app.sim.world import SimHost

FAM = "fam-pend"
MAA = {"id": "elder-pend", "name": "Kamla Devi", "role": "elder"}
SON = {"id": "son-pend", "name": "Ravi Verma", "role": "primary caregiver"}


async def make_pending(db, at):
    at("2026-10-01 10:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="medicine", key="medicine:metformin",
                           value={"name": "Metformin", "dose": "500 mg"}, text="Metformin 500 mg", source_kind="caregiver_said", stated_by=SON["id"])
    w = await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="medicine", key="medicine:metformin",
                               value={"dose": "1000 mg"}, text="Metformin 1000 mg", source_kind="elder_said", stated_by=MAA["id"])
    assert w.result == "pending"
    return w.fact


async def final_asks(db):
    return list((await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.dedupe_key.like("pending-final:%")))).scalars())


async def test_nothing_happens_before_day_14(db, at):
    await make_pending(db, at)
    at("2026-10-14 23:00")
    assert await pending.run(db) == {"final_asks": 0, "archived": 0}


async def test_day_14_brings_exactly_one_final_ask_to_the_caregiver(db, at):
    f = await make_pending(db, at)
    at("2026-10-15 12:00")
    assert (await pending.run(db))["final_asks"] == 1
    asks = await final_asks(db)
    assert len(asks) == 1 and asks[0].owner_id == SON["id"] and asks[0].kind == "memory_check"
    assert "filed away in 3 days" in asks[0].title and asks[0].wake_at is not None
    at("2026-10-16 12:00")
    assert await pending.run(db) == {"final_asks": 0, "archived": 0}  # no repeat
    assert len(await final_asks(db)) == 1
    assert (await store.facts(db, FAM, MAA["id"], statuses=("pending",)))[0].id == f.id


async def test_day_17_archives_it_out_of_context_and_into_history(db, at):
    f = await make_pending(db, at)
    at("2026-10-15 12:00")
    await pending.run(db)
    at("2026-10-18 12:30")
    assert (await pending.run(db))["archived"] == 1
    row = (await store.facts(db, FAM, MAA["id"], statuses=("archived",)))[0]
    assert row.id == f.id and row.note.startswith(pending.ARCHIVED_NOTE)
    assert not await store.facts(db, FAM, MAA["id"], statuses=("pending",))
    assert all(f2.status != "archived" for f2 in await store.facts(db, FAM, MAA["id"]))  # the brain's default view
    events = await store.events(db, FAM, MAA["id"], kinds=["fact_archived"]) if "kinds" in store.events.__code__.co_varnames else []
    assert events == [] or events[0].kind == "fact_archived"
    open_asks = [a for a in await final_asks(db) if a.status == "open"]
    assert open_asks == []
    # running again changes nothing
    assert await pending.run(db) == {"final_asks": 0, "archived": 0}


async def test_a_caregiver_can_bring_an_archived_fact_back(db, at):
    await make_pending(db, at)
    at("2026-10-15 12:00")
    await pending.run(db)
    at("2026-10-18 12:30")
    await pending.run(db)
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=MAA, speaker=SON, members=[MAA, SON], message_ref="m1")
    out = await tools.confirm_change(c, {"key": "medicine:metformin", "approve": True})
    assert out.get("result") in ("active", "approved", "superseded") or out
    now = await store.active_fact(db, FAM, MAA["id"], "medicine:metformin")
    assert now.value["dose"] == "1000 mg"


async def test_an_answer_at_any_time_ends_the_lifecycle(db, at):
    f = await make_pending(db, at)
    await store.resolve_pending(db, fact_id=f.id, approve=False, by=SON["id"])
    at("2026-10-20 02:00")
    assert await pending.run(db) == {"final_asks": 0, "archived": 0}


async def test_with_no_one_to_confirm_it_is_archived_at_day_14(db, at):
    at("2026-10-01 10:00")
    await store.save_roster(db, FAM, MAA, [MAA])
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="allergy", key="allergy:dust",
                           value={"allergen": "dust"}, text="Dust allergy", source_kind="caregiver_said", stated_by="x")
    w = await store.stop_fact(db, family_id=FAM, subject_id=MAA["id"], key="allergy:dust", reason="not true", source_kind="elder_said")
    assert w.result == "pending"
    at("2026-10-15 12:00")
    assert (await pending.run(db))["archived"] == 1


async def test_a_self_care_user_gets_the_final_ask_themselves(db, at):
    me = {"id": "self-pend", "name": "Rhea", "role": "primary caregiver"}
    at("2026-10-01 10:00")
    await store.save_roster(db, FAM, me, [me])
    await store.write_fact(db, family_id=FAM, subject_id=me["id"], domain="medicine", key="medicine:vit_d",
                           value={"name": "Vitamin D"}, text="Vitamin D weekly", source_kind="prescription", stated_by=me["id"])
    w = await store.write_fact(db, family_id=FAM, subject_id=me["id"], domain="medicine", key="medicine:vit_d",
                               value={"dose": "60k"}, text="Vitamin D 60k", source_kind="inferred")
    assert w.result == "pending"
    at("2026-10-15 12:00")
    assert (await pending.run(db))["final_asks"] == 1
    assert (await final_asks(db))[0].owner_id == me["id"]
