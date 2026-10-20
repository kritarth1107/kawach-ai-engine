"""Live 17-20 Oct: food saves failed, "girls" raised an emergency, reminders stopped, "Saved" was said after a failed
save, and "When should I stay quiet?" replaced unrelated answers. Synthetic data only."""

import uuid
from types import SimpleNamespace

from app.api.brain import ComposeIn
from app.brain import guards, loop
from app.core import clock
from app.learn import situations
from app.llm import router
from app.nutrition import catalogue
from app.nutrition.models import NutritionFood
from sqlalchemy import func, select


async def test_catalogue_loads_around_a_food_already_there(db, at):
    at("2026-10-17 12:00")
    now = clock.now()
    db.add(NutritionFood(id=uuid.uuid4(), family_id=None, name="Chilli paneer", aliases=["chilli paneer"], kind="generic",
                         per_100g={"energy_kcal": 250}, source="web", confidence=0.8, checked={}, uses=0, created_at=now, updated_at=now))
    await db.flush()
    await catalogue.ensure(db)  # used to raise the unique-name error on every food save
    n = await db.scalar(select(func.count()).select_from(NutritionFood).where(NutritionFood.source == "indb"))
    assert n and n >= len(catalogue.dishes()) - 5
    same = await db.scalar(select(func.count()).select_from(NutritionFood).where(func.lower(NutritionFood.name) == "chilli paneer"))
    assert same == 1
    await catalogue.ensure(db)  # a second call does nothing


def test_girls_is_not_a_fall():
    for t in ("Do you know if have fucked any girls", "my girlfriend called", "girl power"):
        assert situations.tag(text=t, role="caregiver", tools=[]) != "emergency", t
    for t in ("Maa bathroom mein gir gayi", "papa gire the subah", "girne se chot lagi"):
        assert guards.RED_FLAG_WORDS.search(t), t


async def test_a_word_alone_alerts_only_if_the_fast_check_says_emergency(monkeypatch):
    calls = []

    async def run(ctx, name, args):
        calls.append(name)
        return "{}", False

    async def answer(word):
        async def complete(role, **kw):
            return SimpleNamespace(text=word)
        return complete

    monkeypatch.setenv("EMERGENCY_CHECK", "on")
    monkeypatch.setattr(loop.tools, "run", run)
    ctx = SimpleNamespace(actions=[], user_text="I could finally breathe with that")
    monkeypatch.setattr(router, "complete", await answer("NO"))
    await loop.cover_red_flag(ctx, "emergency")
    assert calls == []
    monkeypatch.setattr(router, "complete", await answer("YES"))
    await loop.cover_red_flag(ctx, "emergency")
    assert calls == ["log_event", "alert_caregiver"]


def test_saved_is_never_said_when_the_save_failed():
    en = {"saved": {"language": "en"}}
    out = guards.not_saved("Saved: 4 parathas and patta gobhi sabji for your lunch. How was it?", en)
    assert "Saved:" not in out and "did not save" in out and "How was it?" in out
    assert guards.not_saved("Glad it helped.", en) == "Glad it helped."


def test_an_invented_time_is_dropped_not_replaced_by_the_quiet_question():
    reply = "Your dinner is noted at 10:00 PM. It had about 950 kcal."
    assert guards.drop_invented_clocks(reply, "4 parathas") == "It had about 950 kcal."


def test_compose_takes_the_backends_24_lines():
    body = ComposeIn(family_id="f", subject_id="s", purpose="water", recent=[f"line {i}" for i in range(24)])
    assert len(body.recent) == 24
