"""What reads the food log: the dashboard view, counters and streaks, the nightly brief's FOOD block, the doctor report and
its share link, the nudge hooks, the meal pattern, the backfill, groceries into the pantry, and account deletion."""

import json
import uuid
from datetime import timedelta

from app.brain import tools
from app.care import brief, patterns, store
from app.core import clock
from app.nutrition import backfill, entries, nudge_hooks, pantry, report
from app.nutrition.models import NutritionEntry, NutritionFood, NutritionJob, NutritionPantry
from app.sim.world import SimHost, World
from tests import nutrition_fakes

nfake = nutrition_fakes.nfake  # the fixture

FAM = "fam-rep"
NANI = {"id": "e-rep", "name": "Savitri", "role": "elder"}
SON = {"id": "c-rep", "name": "Arjun", "role": "primary caregiver"}
OTHER = {"id": "c2-rep", "name": "Kiran", "role": "family member"}


def ctx(db, speaker=NANI):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=NANI, speaker=speaker, members=[NANI, SON, OTHER],
                         message_ref=f"m-{uuid.uuid4().hex[:6]}")


async def call(db, name, args, speaker=NANI):
    out, err = await tools.run(ctx(db, speaker), name, args)
    return json.loads(out), err


async def log(db, at, when, items, **kw):
    at(when)
    r, err = await call(db, "log_food", {"items": items, **kw})
    assert not err, r
    return r


EGGS = [{"name": "boiled egg", "quantity": 2, "unit": "piece"}]
LUNCH = [{"name": "chapati", "quantity": 2, "unit": "piece"}, {"name": "dal", "quantity": 1, "unit": "katori"}]
DINNER = [{"name": "chicken biryani", "quantity": 1, "unit": "plate"}]


async def a_week(db, at):
    for d in (8, 9, 10, 11, 12, 13):
        await log(db, at, f"2026-10-{d:02d} 08:30", EGGS + [{"name": "tea", "quantity": 1, "unit": "cup"}])
        await log(db, at, f"2026-10-{d:02d} 13:30", LUNCH)
        await log(db, at, f"2026-10-{d:02d} 20:30", DINNER + [{"name": "water", "quantity": 2, "unit": "glass"}])


async def test_the_page_view_has_the_day_the_week_the_month_and_counters(db, at, nfake):
    await a_week(db, at)
    await call(db, "nutrition_target", {"for_whom": NANI["id"], "set": {"protein_g": 60, "water_ml": 2000}, "set_by": "doctor", "how": "guide"}, speaker=SON)
    await log(db, at, "2026-10-14 08:40", EGGS)
    at("2026-10-14 16:00")
    v = await report.view(db, FAM, NANI["id"], days=7)
    meals = {m["meal"]: m for m in v["today"]["meals"]}
    assert meals["breakfast"]["entries"][0]["qty"] == "2" and meals["breakfast"]["entries"][0]["confidence"] == "said"
    assert meals["lunch"]["notLogged"] and not meals["dinner"]["notLogged"]  # dinner is not due yet
    assert v["today"]["progress"][0]["label"] in ("Protein", "Water")
    eggs = next(c for c in v["counters"] if c["key"] == "egg")
    assert eggs["today"] == 2 and eggs["usual"] == 2 and len(eggs["week"]) == 7
    water = next(c for c in v["counters"] if c["key"] == "water")
    assert water["week"][-2] == 2  # two glasses yesterday
    assert len(v["week"]) == 7 and len(v["month"]["calendar"]) == 30 and v["week"][-2]["complete"] == 1.0
    assert v["streaks"]["allMeals"] == 6
    assert any(n["key"] == "target:protein_g" for n in v["noticed"])
    assert v["units"] and v["mealSlots"][0] == "breakfast"


async def test_food_log_answers_totals_counts_and_one_item(db, at, nfake):
    await a_week(db, at)
    at("2026-10-14 10:00")
    r, _ = await call(db, "food_log", {"from": "2026-10-08", "to": "2026-10-13", "what": "counts"})
    assert r["counts"]["egg"] == 12 and r["counts"]["tea_coffee"] == 6
    r, _ = await call(db, "food_log", {"from": "2026-10-08", "to": "2026-10-13", "what": "item", "item": "egg"})
    assert r["count"] == 12
    r, _ = await call(db, "food_log", {"day": "yesterday", "what": "totals"})
    assert r["days"][0]["main_meals_logged"] == 3 and r["days"][0]["kcal"] > 1000
    r, _ = await call(db, "food_log", {"what": "totals"})
    assert "Nothing logged" in r["note"]


async def test_someone_else_cannot_read_her_food_log(db, at, nfake):
    at("2026-10-14 10:00")
    r, err = await call(db, "food_log", {"for_whom": NANI["id"]}, speaker=OTHER)
    assert err and "caregiver" in r["refused"]


async def test_the_nightly_brief_has_a_food_block_and_section(db, at, nfake):
    await log(db, at, "2026-10-14 08:30", EGGS)
    await log(db, at, "2026-10-14 13:30", LUNCH)
    await call(db, "log_food", {"skipped": True, "meal": "dinner"})
    text = await brief.digest(db, SimHost(World()), FAM, NANI, "2026-10-14")
    assert "FOOD (from the food log" in text and "breakfast (08:30): 2 boiled egg [said]" in text
    assert "dinner: skipped (they said)" in text and "totals:" in text
    assert "food and nutrition" in brief.prompt_for(SON, NANI, "2026-10-14", text)


async def test_the_brief_says_not_logged_and_waiting_order_meals(db, at, nfake):
    from tests.test_nutrition_log import _order

    at("2026-10-14 13:00")
    t = _order(subject=NANI["id"])
    t.family_id = FAM
    db.add(t)
    await db.flush()
    await log(db, at, "2026-10-14 08:30", EGGS)
    block = await report.brief_block(db, FAM, NANI["id"], "2026-10-14")
    assert "not logged: lunch, dinner" in block and "delivered, not yet confirmed eaten: Chicken Dum Biryani from Paradise" in block


async def test_doctor_report_and_its_share_link_only_for_a_caregiver(db, at, nfake):
    await a_week(db, at)
    at("2026-10-14 10:00")
    r = await report.doctor(db, FAM, NANI["id"], 7)
    assert r["daysLogged"] == 6 and r["mealPattern"] == {"breakfast": 6, "lunch": 6, "dinner": 6} and r["topFoods"]
    assert r["confidence"]["said"] > 0
    out, err = await call(db, "nutrition_report", {"for_whom": NANI["id"], "days": 7, "share_with_doctor": True}, speaker=SON)
    assert not err and out["share_link"].startswith("https://kavach.test/n/") and "logged on 6 of 7 days" in out["summary"]
    out, err = await call(db, "nutrition_report", {"for_whom": NANI["id"], "share_with_doctor": True}, speaker=OTHER)
    assert err


async def test_the_person_cared_for_never_shares_their_report_a_self_care_adult_does(db, at, nfake):
    """Founder Q8: shared only when the caregiver chooses (or someone managing their own care); the elder asking about their
    own report gets the report, never a doctor's link."""
    await a_week(db, at)
    at("2026-10-14 10:00")
    host = SimHost()
    elder_ctx = tools.TurnCtx(session=db, host=host, family_id=FAM, elder=NANI, speaker=NANI, members=[NANI, SON, OTHER], message_ref="m-share-1")
    out, err = await tools.run(elder_ctx, "nutrition_report", {"share_with_doctor": True})
    assert err and "share_link" not in json.loads(out)
    out, err = await tools.run(elder_ctx, "nutrition_report", {})
    assert not err and "summary" in json.loads(out)
    me = {"id": "s-rep", "name": "Meera", "role": "primary caregiver"}
    self_ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=me, speaker=me, members=[me], message_ref="m-share-2")
    out, err = await tools.run(self_ctx, "nutrition_report", {"share_with_doctor": True})
    assert not err and json.loads(out)["share_link"]


async def test_a_caregiver_cannot_read_or_share_another_caregivers_own_food_log(db, at, nfake):
    co = {"id": "c3-rep", "name": "Ravi", "role": "caregiver"}
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=NANI, speaker=SON, members=[NANI, SON, co], message_ref="m-share-3")
    at("2026-10-14 10:00")
    out, err = await tools.run(c, "nutrition_report", {"for_whom": co["id"], "share_with_doctor": True})
    assert err and "another caregiver" in json.loads(out).get("error", out)


async def test_nudge_hooks(db, at, nfake):
    await a_week(db, at)
    at("2026-10-14 10:00")
    assert await nudge_hooks.meal_logged(db, FAM, NANI["id"], "lunch", "2026-10-13")
    assert not await nudge_hooks.meal_logged(db, FAM, NANI["id"], "lunch", "2026-10-14")
    times = await nudge_hooks.usual_meal_times(db, FAM, NANI["id"])
    assert times["breakfast"] == "08:30" and times["evidence"]["lunch"] == 6 and not times["stable"]["lunch"]  # 6 days < 7
    await log(db, at, "2026-10-14 08:30", [{"name": "boiled egg"}])  # count unclear: a question waits
    pend = await nudge_hooks.pending_confirmations(db, FAM, NANI["id"])
    assert pend and pend[0]["kind"] == "question" and pend[0]["reason"] == "quantity"
    assert await nudge_hooks.meal_logged(db, FAM, NANI["id"], "breakfast", "2026-10-14")  # the conversation is going
    p = await nudge_hooks.progress(db, FAM, NANI["id"], "2026-10-13")
    assert p["meals_logged"] == 3 and "Chicken dum biryani" in p["own_foods"]


async def test_habit_learnings_are_visible_facts(db, at, nfake):
    for d in range(1, 9):
        await log(db, at, f"2026-10-{d:02d} 13:30", LUNCH)
    at("2026-10-09 22:00")
    got = await nudge_hooks.learn_meal_habits(db, FAM, NANI["id"])
    assert got.get("times") == "created" and got.get("portions") == "created"
    f = await store.active_fact(db, FAM, NANI["id"], "learned:nudges_meals_times")
    assert f.value["lunch"] == "13:30" and "1:30 pm" in f.text


async def test_skipped_meals_make_a_pattern_and_plain_words_do_not(db, at, nfake):
    for d in (10, 12, 13):
        at(f"2026-10-{d} 15:00")
        await call(db, "log_food", {"skipped": True, "meal": "lunch"})
    at("2026-10-13 18:00")
    await store.record_event(db, family_id=FAM, subject_id=NANI["id"], kind="meal", summary="skipped dinner, no appetite")  # words only
    at("2026-10-14 10:00")
    found = {p.kind: p for p in await patterns.find(db, FAM, NANI["id"])}
    assert "meals_skipped" in found and "3 days" in found["meals_skipped"].title


async def test_backfill_turns_old_meal_notes_into_entries_dry_run_first(db, at, nfake):
    at("2026-10-13 23:20")
    eid = await store.record_event(db, family_id=FAM, subject_id=NANI["id"], kind="meal", summary="Ate four boiled eggs despite stomach upset",
                                   actor_id=NANI["id"])
    nfake.notes[eid] = {"skipped": False, "items": [{"name": "boiled egg", "quantity": 4, "unit": "piece"}]}
    at("2026-10-14 10:00")
    plan = await backfill.family(db, FAM, days=7, apply=False)
    assert plan[0]["items"] == [{"name": "boiled egg", "quantity": 4, "unit": "piece"}] and "written" not in plan[0]
    assert await entries.on_day(db, FAM, NANI["id"], "2026-10-13") == []
    plan = await backfill.family(db, FAM, days=7, apply=True)
    rows = await entries.on_day(db, FAM, NANI["id"], "2026-10-13")
    assert len(rows) == 1 and rows[0].source == "backfill" and rows[0].confidence == "said" and round(rows[0].nutrients["energy_kcal"]) == 310
    assert await backfill.family(db, FAM, days=7, apply=True) == []  # done once


async def test_a_delivered_grocery_order_goes_to_the_pantry_once(db, at, nfake):
    from app.tasks.models import Task

    at("2026-10-14 12:00")
    now = clock.now()
    t = Task(id=uuid.uuid4(), family_id=FAM, subject_id=NANI["id"], requested_by=SON["id"], service="blinkit", kind="order", goal="protein bars",
             details={"chosen": [{"name": "Choco Protein Bar", "pack": "pack of 6", "for_item": "protein bar"}], "items": [{"name": "protein bar", "qty": 1}]},
             status="done", phase="place", result={"placed": True}, history=[], runs=1, created_at=now, updated_at=now, deadline_at=now)
    db.add(t)
    await db.flush()
    out = await pantry.add_from_order(db, t)
    assert out["added"] == [{"id": out["added"][0]["id"], "name": "Choco Protein Bar", "count": 6.0, "unit": "piece"}]
    again = await pantry.add_from_order(db, t)
    assert again["added"] == [] and again["kept"]
    t.service = "swiggy"
    assert (await pantry.add_from_order(db, t))["skipped"] == "not a grocery order"
    assert await entries.on_day(db, FAM, NANI["id"], "2026-10-14") == []  # nothing is eaten by being bought


async def test_account_deletion_removes_her_food_data(db, at, nfake):
    await log(db, at, "2026-10-14 08:30", EGGS)
    await log(db, at, "2026-10-14 09:00", [{"name": "boiled egg"}])  # a question waiting
    await pantry.add(db, family_id=FAM, subject_id=NANI["id"], name="Dates", count=10)
    out = await entries.erase(db, FAM, NANI["id"])
    assert out["nutrition_entries"] == 1 and out["nutrition_jobs"] == 2 and out["nutrition_pantry"] == 1 and out["memory_versions_food"] >= 1
    for model in (NutritionEntry, NutritionJob, NutritionPantry):
        assert (await db.execute(model.__table__.select().where(model.family_id == FAM))).first() is None
    assert (await db.execute(NutritionFood.__table__.select().where(NutritionFood.family_id.is_(None)))).first() is not None  # generic stays
    ev = await store.events(db, FAM, NANI["id"], kinds=["meal"])
    assert ev == []


async def test_the_erase_route_includes_the_food_log(db, at, nfake):
    from app.rag.memory_lifecycle import erase_elder_memory

    fid, eid = uuid.uuid4(), uuid.uuid4()
    at("2026-10-14 08:30")
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=str(fid), elder={"id": str(eid), "name": "x"}, speaker={"id": str(eid), "name": "x", "role": "elder"},
                      members=[{"id": str(eid)}], message_ref="m-erase")
    await tools.run(c, "log_food", {"items": EGGS})
    out = await erase_elder_memory(db, family_id=fid, elder_id=eid)
    assert out["nutrition"]["nutrition_entries"] == 1


def test_counters_read_tags_not_names():
    e = NutritionEntry(kind="food", status="active", quantity=3, unit="piece", nutrients={"protein_g": 18.9}, flags={"tags": ["egg"]}, name="Anything")
    w = NutritionEntry(kind="food", status="active", quantity=2, unit="glass", nutrients={"water_ml": 500}, flags={"tags": ["water"]}, name="Paani")
    c = entries.counters([e, w])
    assert c["egg"] == 3 and c["water"] == 2 and c["protein"] == 18.9


def test_days_helper():
    assert entries.day_days("2026-10-14", 3) == ["2026-10-12", "2026-10-13", "2026-10-14"]
    assert (clock.now() - timedelta(days=1)) < clock.now()


async def test_each_day_says_what_each_meal_had_and_unknown_nutrients_stay_unknown(db, at, nfake):
    """The week cards and the calendar dots read each day's meals; a nutrient no food carried is unknown, not 0 (a doctor's
    report must not say 'saturated fat 0 g'); water is in the averages, not in 'most eaten'."""
    await a_week(db, at)
    at("2026-10-12 09:00")
    await call(db, "log_food", {"meal": "breakfast", "skipped": True})
    at("2026-10-14 10:00")
    v = await report.view(db, FAM, NANI["id"], days=7)
    day = next(r for r in v["week"] if r["day"] == "2026-10-13")
    assert [m["meal"] for m in day["meals"]] == ["breakfast", "lunch", "dinner"]
    assert day["meals"][0]["names"] == ["Egg, whole, boiled", "Tea with milk and sugar"] and day["meals"][0]["kcal"] > 0
    assert day["logged"] == ["breakfast", "lunch", "dinner"] and day["skipped"] == []
    cal = {c["day"]: c for c in v["month"]["calendar"]}
    assert cal["2026-10-13"]["logged"] == ["breakfast", "lunch", "dinner"] and cal["2026-10-01"]["logged"] == []
    nut = {n["key"]: n for n in v["nutrients"]}
    assert nut["sat_fat_g"]["avg7"] is None and nut["energy_kcal"]["avg7"] > 0
    r = await report.doctor(db, FAM, NANI["id"], 7)
    assert r["averages"]["sat_fat_g"] is None and r["averages"]["water_ml"] > 0
    assert "Water" not in [f["name"] for f in r["topFoods"]]
