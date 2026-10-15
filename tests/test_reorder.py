"""A stock count that runs out within 3 days is offered once, the usual way, and not while that order is moving."""

import json
import uuid
from datetime import timedelta

from sqlalchemy import select

from app.brain import tools, wake
from app.care import reorder, store
from app.care.models import OpenLoop
from app.core import clock
from app.nutrition.models import NutritionEntry
from app.nutrition import pantry
from app.sim.world import SimHost
from app.tasks import runtime

FAM, ELDER, CG = "fam-ro", "elder-ro", "cg-ro"
E = {"id": ELDER, "name": "Kamla", "role": "elder"}
C = {"id": CG, "name": "Asha", "role": "primary caregiver"}


def ctx(db, speaker=C):
    return tools.TurnCtx(
        session=db, host=SimHost(), family_id=FAM, elder=E, speaker=speaker, members=[E, C],
    )


async def call(db, name, args, speaker=C, members=None):
    c = ctx(db, speaker)
    if members is not None:
        c.members = members
    out, err = await tools.run(c, name, args)
    assert not err, out
    return json.loads(out)


async def _medicine(db, count, *, about=None):
    await store.save_roster(db, FAM, E, [E, C])
    who = about or ELDER
    await call(db, "remember", {"domain": "medicine", "name": "Amlodipine", "about": who,
                                "details": {"dose": "5 mg", "times": ["08:00", "20:00"]}, "sentence": "Amlodipine 5 mg twice"})
    await call(db, "set_stock", {"medicine": "Amlodipine", "count": count, "about": who})


async def _loops(db):
    return list((await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "refill"))).scalars())


async def test_a_medicine_within_3_days_is_offered_once_per_run_out(db, at):
    at("2026-10-02 10:00")
    await _medicine(db, 10)  # 2 a day: 5 days left, still on the dashboard, not offered yet
    assert await reorder.open_offers(db) == 0
    at("2026-10-02 10:01")
    await call(db, "set_stock", {"medicine": "Amlodipine", "count": 6})  # 3 days
    assert await reorder.open_offers(db) == 1
    assert await reorder.open_offers(db) == 0
    loop = (await _loops(db))[0]
    assert loop.owner_id == CG and loop.detail["pay"] == "cod" and loop.detail["category"] == "medicine"
    assert loop.detail["days_left"] == 3 and loop.detail["episode"]
    prompt = await wake.wake_prompt(db, loop, ELDER)
    assert "Cash on delivery only" in prompt and "One short question" in prompt and "Only after a yes" in prompt
    assert "caregiver's approval" in prompt and f"Ask person {CG}" in prompt
    # A no closes it. The same count is not offered again.
    loop.status = "done"
    await db.commit()
    assert await reorder.open_offers(db) == 0
    # A new count that is still within 3 days is a new run-out.
    at("2026-10-03 10:00")
    await call(db, "set_stock", {"medicine": "Amlodipine", "count": 4})
    assert await reorder.open_offers(db) == 1


async def test_no_offer_while_an_order_for_it_is_moving(db, at):
    at("2026-10-02 10:00")
    await _medicine(db, 4)  # 2 days
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=CG, service="apollo", kind="order",
                             goal="Amlodipine", details={"items": [{"name": "Amlodipine", "qty": 1}]})
    t.status = "queued"
    await db.commit()
    assert await reorder.open_offers(db) == 0
    t.status = "cancelled"
    await db.commit()
    assert await reorder.open_offers(db) == 1
    loop = (await _loops(db))[0]
    assert loop.detail["service"] == "" and loop.detail["where"] == "no usual store yet"


async def test_the_usual_store_and_amount_come_from_the_last_placed_order(db, at):
    at("2026-10-02 10:00")
    await _medicine(db, 6)
    placed = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=CG, service="apollo", kind="order",
                                  goal="Amlodipine", details={"items": [{"name": "Amlodipine", "qty": 10}]})
    placed.status = "done"
    placed.result = {"placed": True, "items": [{"name": "Amlodipine 5 mg", "qty": 10}]}
    await db.commit()
    assert await reorder.open_offers(db) == 1
    loop = (await _loops(db))[0]
    assert loop.detail["service"] == "apollo" and loop.detail["qty"] == 10
    assert loop.detail["where"] == "Apollo 24|7 Pharmacy" and loop.detail["item"] == "Amlodipine 5 mg"
    prompt = await wake.wake_prompt(db, loop, ELDER)
    assert "Amlodipine 5 mg" in prompt and "Apollo 24|7 Pharmacy" in prompt and "10" in prompt


async def test_a_regular_grocery_is_offered_from_the_pantry_count(db, at):
    at("2026-10-10 10:00")
    await store.save_roster(db, FAM, E, [E, C])
    name = "Amul Taaza Toned Milk"
    for _ in range(2):
        t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order",
                                 goal=name, details={"items": [{"name": name, "qty": 6}]})
        t.status, t.result = "done", {"placed": True, "items": [{"name": name, "qty": 6}]}
    await pantry.add(db, family_id=FAM, subject_id=ELDER, name=name, count=2, unit="piece")
    now = clock.now()
    meal = uuid.uuid4()
    for eaten, qty in ((now - timedelta(days=6), 2), (now, 2)):
        db.add(NutritionEntry(
            id=uuid.uuid4(), family_id=FAM, subject_id=ELDER, logged_by=ELDER, meal_id=meal, meal="breakfast",
            kind="food", eaten_at=eaten, day=clock.ist_day(eaten), name=name, name_said=name, quantity=qty, unit="piece",
            nutrients={}, confidence="said", source="chat", consumed="yes", status="active", version=1,
            created_at=now, updated_at=now,
        ))
    await db.commit()
    # 4 pieces over 6 days, 2 left: 3 days. Offered once for this pantry row.
    assert await reorder.open_offers(db) == 1
    assert await reorder.open_offers(db) == 0
    loop = (await _loops(db))[0]
    assert loop.detail["category"] == "grocery" and loop.detail["service"] == "blinkit" and loop.detail["qty"] == 6
    assert loop.owner_id == CG  # it is for the elder
    loop.status = "done"
    await db.commit()
    assert await reorder.open_offers(db) == 0


async def test_self_care_is_asked_of_the_person_themselves(db, at):
    at("2026-10-02 10:00")
    await _medicine(db, 4, about=CG)
    assert await reorder.open_offers(db) == 1
    assert (await _loops(db))[0].owner_id == CG


async def test_one_person_hears_the_sooner_item_and_the_other_the_next_day(db, at):
    at("2026-10-02 10:00")
    await _medicine(db, 4)  # Amlodipine, twice a day: 2 days, asked of the caregiver
    await call(db, "remember", {"domain": "medicine", "name": "Metformin", "about": ELDER,
                                "details": {"dose": "500 mg", "times": ["08:00"]}, "sentence": "Metformin once a day"})
    await call(db, "set_stock", {"medicine": "Metformin", "count": 3, "about": ELDER})  # 3 days
    assert await reorder.open_offers(db) == 1
    titles = [loop.title for loop in await _loops(db)]
    assert titles == ["Reorder Amlodipine: about 2 days left (4 left) for person elder-ro"]
    assert await reorder.open_offers(db) == 0
    at("2026-10-03 10:00")
    assert await reorder.open_offers(db) == 1
    assert any(loop.title.startswith("Reorder Metformin") for loop in await _loops(db))


async def test_two_people_can_each_hear_one_offer_the_same_morning(db, at):
    at("2026-10-02 10:00")
    other = {"id": "other-ro", "name": "Mohan", "role": "caregiver"}
    people = [E, C, other]
    await store.save_roster(db, FAM, E, people)
    await call(db, "remember", {"domain": "medicine", "name": "Amlodipine", "about": ELDER,
                                "details": {"dose": "5 mg", "times": ["08:00", "20:00"]}, "sentence": "Amlodipine twice"}, members=people)
    await call(db, "set_stock", {"medicine": "Amlodipine", "count": 4, "about": ELDER}, members=people)
    await call(db, "remember", {"domain": "medicine", "name": "Vitamin D", "about": other["id"],
                                "details": {"dose": "1", "times": ["08:00"]}, "sentence": "Vitamin D daily"}, members=people)
    await call(db, "set_stock", {"medicine": "Vitamin D", "count": 2, "about": other["id"]}, members=people)
    assert await reorder.open_offers(db) == 2
    assert {loop.owner_id for loop in await _loops(db)} == {CG, other["id"]}
