"""Saheli's night: learn from the day, once per family, step by step, within a budget."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care import baselines, dream, store
from app.care.models import CareEvent, FamilyRoster
from app.core import clock
from app.llm import router, spend
from app.llm.router import LLMReply
from app.tasks.models import SkillNote

FAM = "fam-dream"
ELDER = {"id": "elder-d", "name": "Kamla", "role": "elder"}
CG = {"id": "cg-d", "name": "Asha", "role": "primary caregiver"}


class Model:
    def __init__(self):
        self.calls = 0

    async def complete(self, route, **kw):
        self.calls += 1
        return LLMReply(text="Took her tablets, BP 130/80, knee pain in the evening.", tool_calls=[], model="fake")


@pytest.fixture
def model(monkeypatch):
    m = Model()
    router.register_provider("fake", m)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    yield m
    router._providers.pop("fake", None)


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


def ist(d, hhmm):
    h, m = map(int, hhmm.split(":"))
    return datetime(2026, 10, d, h, m, tzinfo=clock.IST)


async def seed(db):
    from sqlalchemy import delete

    await db.execute(delete(FamilyRoster).where(FamilyRoster.family_id != FAM))  # only this family (rolled back after)
    await store.save_roster(db, FAM, ELDER, [ELDER, CG])
    for d in range(1, 15):
        clock.set_now(ist(d, "09:00"))
        await store.record_event(db, family_id=FAM, subject_id=ELDER["id"], kind="vital", summary="bp 130/80", payload={"kind": "bp", "value": "130/80"})
        if d % 3 == 0:
            await store.record_event(db, family_id=FAM, subject_id=ELDER["id"], kind="symptom", summary="knee pain", payload={"severity": "watch"})
        await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="user", text="dawai le li", speaker_id=ELDER["id"])
    await db.commit()


async def test_night_learns_once_per_family(db, sessions, model):
    await seed(db)
    clock.set_now(ist(15, "02:00"))
    stats = await dream.dream_all(sessions)
    assert stats["day"] == "2026-10-14" and stats["done_now"] == 1 and stats["errors"] == 0
    assert stats["patterns"] >= 1  # knee pain keeps coming back
    assert (await baselines.get(db, FAM, ELDER["id"]))["vitals"]["bp"]["systolic"]["median"] == 130
    diary = [n for n in await store.notes(db, FAM, [ELDER["id"]]) if n.slug == "diary"]
    assert diary and diary[0].body_md.startswith("- 2026-10-14:")
    again = await dream.dream_all(sessions)
    assert again["done_now"] == 0  # already done tonight
    marks = (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "dream"))).scalars().all()
    assert len(marks) == 1


async def test_no_model_calls_without_models_and_a_failing_step_does_not_stop_the_rest(db, sessions, model, monkeypatch):
    await seed(db)
    clock.set_now(ist(15, "02:00"))

    async def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(dream.patterns, "record_new", broken)
    roster = await db.get(FamilyRoster, FAM)
    rep = await dream.dream_family(sessions, roster, "2026-10-14", with_models=False)
    assert model.calls == 0 and rep["baselines"] == 2 and any(e.startswith("patterns:") for e in rep["errors"])


async def test_diary_stays_small(db, sessions, model):
    await seed(db)
    for d in range(1, 80):
        day = (datetime(2026, 7, 1) + timedelta(days=d)).strftime("%Y-%m-%d")
        clock.set_now(datetime.fromisoformat(day).replace(hour=10, tzinfo=clock.IST))
        await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="user", text="hello", speaker_id=ELDER["id"])
        async with sessions() as s:
            await dream.diary(s, FAM, ELDER, day)
            await s.commit()
    note = [n for n in await store.notes(db, FAM, [ELDER["id"]]) if n.slug == "diary"][0]
    assert len(note.body_md.splitlines()) == dream.DIARY_KEEP_LINES


async def test_upkeep_keeps_recent_skill_notes(db, sessions):
    clock.set_now(ist(15, "02:00"))
    for i in range(dream.SKILL_NOTES_KEEP + 15):
        db.add(SkillNote(service="zepto", note=f"note {i}", created_at=clock.now()))
    await db.commit()
    out = await dream.upkeep(sessions)
    assert out["skill_notes_removed"] >= 15
    left = (await db.execute(select(SkillNote).where(SkillNote.service == "zepto"))).scalars().all()
    assert len(left) == dream.SKILL_NOTES_KEEP and max(n.note for n in left).startswith("note")


def test_night_of():
    assert dream.night_of(ist(4, "02:00")) == "2026-10-03"
    assert dream.night_of(ist(4, "23:00")) == "2026-10-04"
