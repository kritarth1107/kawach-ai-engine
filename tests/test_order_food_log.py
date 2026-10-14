"""Something ordered is not something eaten (founder 2026-10-14, nutrition plan §N6): a restaurant meal goes in the food
log only after the person's own answer that it was eaten (all, most, half, a little), asked in the order's follow-up as one
of its questions; groceries go to the pantry at home when they arrive, never logged as eaten; a food_delivered event marks
what came. The nutrition agent's own functions (app.nutrition) are faked here: their contract is
OpenBot/Shared/kavach-nutrition/INTERFACES.md §1."""

from datetime import timedelta

import pytest

from app.care import store
from app.core import clock
from app.tasks import aftercare, runtime

FAM, ME, SON = "fam-fl", "elder-fl", "son-fl"
E = {"id": ME, "name": "Kamla", "role": "elder"}
S = {"id": SON, "name": "Ravi", "role": "primary caregiver"}


class FakeNutrition:
    def __init__(self):
        self.meals, self.pantry = [], []

    async def log_order_meal(self, session, task, *, consumed, portion_eaten=None, eaten_by=None, by, channel="whatsapp"):
        self.meals.append({"task": str(task.id), "consumed": consumed, "portion": portion_eaten, "eaten_by": eaten_by, "by": by})
        return {"result": "saved" if consumed in ("yes", "part") else "nothing", "say": f"Saved: {portion_eaten or 'nothing'}"}

    async def add_from_order(self, session, task, *, items=None):
        self.pantry.append(str(task.id))
        return {"added": [], "kept": [], "skipped": None}

    def entries(self):
        return [m for m in self.meals if m["consumed"] in ("yes", "part")]


@pytest.fixture
def nutrition(monkeypatch):
    fake = FakeNutrition()
    monkeypatch.setattr(aftercare, "_nutrition", lambda: (fake, fake))
    return fake


async def _placed(db, service, items, *, restaurant=None):
    await store.save_roster(db, FAM, E, [E, S])
    t = await runtime.create(db, family_id=FAM, subject_id=ME, requested_by=ME, service=service, kind="order", goal="food",
                             details={"items": [{"name": i} for i in items], "for_set": True})
    t.status, t.phase = "done", "place"
    t.result = {"placed": True, "order_id": "X1", "total": "₹249", "eta": "30 min",
                "items": [{"name": i, "qty": 1, **({"restaurant": restaurant} if restaurant else {})} for i in items]}
    await runtime.delivery_followup(db, t)
    await db.commit()
    return t


async def test_a_restaurant_meal_asks_whether_it_was_eaten_first_and_logs_only_that_answer(db, at, nutrition):
    at("2026-10-14 13:00")
    t = await _placed(db, "swiggy", ["Paneer Butter Masala"], restaurant="Punjabi Tadka")
    out = await aftercare.record(db, t, by=ME, source_kind="elder_said", confidence=0.9, arrived="yes")
    assert nutrition.meals == [], "it came: nothing is logged as eaten"
    loop = await aftercare.loop_for(db, t)
    loop.detail = {**loop.detail, "stage": "feedback", "wakes": 0}
    t.result = {**t.result, "feedback": {"arrived": "yes"}}
    await db.commit()
    prompt = await aftercare.wake_prompt(db, loop)
    assert "whether it was eaten: all of it, most, half, a little, or someone else ate it" in prompt
    assert "never log_food for this order" in prompt
    out = await aftercare.record(db, t, by=ME, source_kind="elder_said", confidence=0.9, eaten="half")
    assert nutrition.meals == [{"task": str(t.id), "consumed": "part", "portion": "half", "eaten_by": ME, "by": ME}]
    assert t.result["feedback"]["eaten"] == "half" and "Saved: half" in out["saved"][0]
    # asked once: the next question is about the dish itself
    nq = aftercare.next_question(t, t.result["feedback"], {})
    assert nq and "eaten" not in nq


@pytest.mark.parametrize("eaten", ["none", "someone_else"])
async def test_not_eaten_logs_nothing_for_them(db, at, nutrition, eaten):
    at("2026-10-14 13:10")
    t = await _placed(db, "zomato", ["Veg Biryani"], restaurant="Shree Annapoorna")
    await aftercare.record(db, t, by=ME, source_kind="elder_said", confidence=0.9, arrived="yes")
    await aftercare.record(db, t, by=ME, source_kind="elder_said", confidence=0.9, eaten=eaten)
    assert nutrition.entries() == [] and [m["consumed"] for m in nutrition.meals] == ["no"]


async def test_no_answer_no_entry_and_never_from_a_scheduled_turn(db, at, nutrition):
    at("2026-10-14 13:20")
    t = await _placed(db, "swiggy", ["Dal Khichdi"], restaurant="Home Kitchen")
    await aftercare.store_delivered(db, t, clock.now())
    at("2026-10-14 15:00")
    loop = await aftercare.loop_for(db, t)
    if loop is not None:
        await aftercare.wake_prompt(db, loop)  # the scheduled follow-up turn: a question, never an entry
    assert nutrition.meals == [], "delivered and silence: nothing is logged"


async def test_groceries_go_to_the_pantry_once_never_as_eaten(db, at, nutrition):
    at("2026-10-14 13:30")
    t = await _placed(db, "blinkit", ["RiteBite Choco Berry bar", "Amul Taaza milk"])
    await aftercare.store_delivered(db, t, clock.now())
    await aftercare.record(db, t, by=ME, source_kind="elder_said", confidence=0.9, arrived="yes")
    assert nutrition.pantry == [str(t.id)] and nutrition.meals == []
    ev = await store.events(db, FAM, ME, since=clock.now() - timedelta(hours=1), kinds=["food_delivered"])
    assert len(ev) == 1 and [i["name"] for i in ev[0].payload["items"]] == ["RiteBite Choco Berry bar", "Amul Taaza milk"]


async def test_medicines_and_rides_are_never_food(db, at, nutrition):
    at("2026-10-14 13:40")
    t = await _placed(db, "1mg", ["Dolo 650"])
    await aftercare.store_delivered(db, t, clock.now())
    await aftercare.record(db, t, by=ME, source_kind="elder_said", confidence=0.9, arrived="yes", eaten="all")
    assert nutrition.meals == [] and nutrition.pantry == []
    assert not await store.events(db, FAM, ME, since=clock.now() - timedelta(hours=1), kinds=["food_delivered"])


async def test_without_the_nutrition_agent_nothing_breaks(db, at, monkeypatch):
    monkeypatch.setattr(aftercare, "_nutrition", lambda: (None, None))
    at("2026-10-14 13:50")
    t = await _placed(db, "swiggy", ["Paneer Butter Masala"], restaurant="Punjabi Tadka")
    await aftercare.record(db, t, by=ME, source_kind="elder_said", confidence=0.9, arrived="yes")
    out = await aftercare.record(db, t, by=ME, source_kind="elder_said", confidence=0.9, eaten="all")
    assert t.result["feedback"]["eaten"] == "all" and out["saved"] == ["eaten: all"]


async def test_with_the_real_nutrition_agent_not_eaten_saves_nothing(db, at):
    """The real app.nutrition (no fake): "someone else ate it" reaches log_order_meal as consumed=no, and the food log has
    no entry for her (the nutrition agent records the answer only)."""
    from sqlalchemy import func, select

    from app.nutrition.models import NutritionEntry

    at("2026-10-14 14:30")
    t = await _placed(db, "swiggy", ["Paneer Butter Masala"], restaurant="Punjabi Tadka")
    t.result = {**t.result, "delivered": True}
    await aftercare.record(db, t, by=ME, source_kind="elder_said", confidence=0.9, arrived="yes")
    await aftercare.record(db, t, by=ME, source_kind="elder_said", confidence=0.9, eaten="someone_else")
    await db.flush()
    n = await db.scalar(select(func.count()).select_from(NutritionEntry).where(NutritionEntry.family_id == FAM))
    assert n == 0 and t.result["feedback"]["eaten"] == "someone_else"
