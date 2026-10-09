"""With few families, every correction is the lesson: collected nightly, anonymised, exported as eval fixtures."""

import json
from datetime import timedelta

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care import skillbook, store
from app.learn import corrections
from app.learn.models import ReplyLog
from app.tasks.models import Task

import pytest

FAM, MAA, SON = "fam-corr", "maa-corr", "son-corr"


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


async def test_guard_rewrites_chat_and_memory_corrections_become_cases(db, at, sessions):
    t0 = at("2026-10-09 19:00")
    await store.save_roster(db, FAM, {"id": MAA, "name": "Vasundara", "role": "elder"}, [{"id": SON, "name": "Kritarth", "role": "primary caregiver"}])
    await corrections.note_guard(db, family_id=FAM, thread_id=MAA, draft="राम राम सा वसुंधरा जी, मैं कीरतन जी ने खबर कर दी है",
                                 problems=["repeats what you already said"], user_text="chakkar aave se")
    db.add(ReplyLog(family_id=FAM, thread_id=MAA, turn_id=1, at=t0, kind="reply", situation="dose", lang="hi", user_text="dawai?",
                    text="आपने सारी दवाई ले ली।", tools=[], corrected=True, speaker_role="elder"))
    await store.add_turn(db, family_id=FAM, thread_id=MAA, role="user", text="नहीं, गलत, मैंने कोई दवाई नहीं ली", speaker_id=MAA)
    (await store.recent_turns(db, FAM, MAA))[-1].at = t0 + timedelta(minutes=1)
    await store.record_event(db, family_id=FAM, subject_id=MAA, kind="fact_created", summary="Shelcal 500 at 2 pm",
                             payload={"key": "medicine:shelcal"}, actor_id=MAA)
    at("2026-10-09 20:00")
    await store.record_event(db, family_id=FAM, subject_id=MAA, kind="dashboard_edit", summary="remember: Shelcal 250 at 2 pm",
                             payload={"key": "medicine:shelcal"}, actor_id=SON)
    await db.commit()
    at("2026-10-09 23:00")
    got = await corrections.collect(sessions)
    assert got == {"chat": 1, "guard": 1, "memory": 1}
    rows = await corrections.cases(db)
    kinds = {c["kind"] for c in rows}
    assert kinds == {"chat_correction", "guard_rewrite", "memory_correction"}
    assert not any("वसुंधरा" in json.dumps(c, ensure_ascii=False) or "Vasundara" in json.dumps(c) for c in rows), "anonymised"
    line = json.loads(corrections.as_fixture(next(c for c in rows if c["kind"] == "chat_correction")))
    assert line["bad_reply"] == "आपने सारी दवाई ले ली।" and "गलत" in line["problem"]
    # nightly again: nothing twice
    assert await corrections.collect(sessions) == {}


async def test_usual_orders_become_a_skill_to_approve(db, at):
    t0 = at("2026-10-01 10:00")
    await store.save_roster(db, FAM, {"id": MAA, "name": "Vasundara", "role": "elder"}, [{"id": SON, "name": "Kritarth", "role": "primary caregiver"}])
    for d in range(3):
        db.add(Task(family_id=FAM, subject_id=MAA, requested_by=MAA, service="blinkit", kind="order", goal="milk", status="done", phase="place",
                    details={}, result={"placed": True, "items": [{"name": "Amul Taaza Milk 500 ml", "qty": 2}]}, history=[],
                    created_at=t0 + timedelta(days=d), updated_at=t0 + timedelta(days=d), deadline_at=t0 + timedelta(days=d, hours=1)))
    await db.commit()
    at("2026-10-09 23:00")
    out = await corrections.propose_usuals(db, FAM, MAA)
    assert out and out.get("saved")
    rows = [skillbook.view(s) for s in await skillbook.family_skills(db, FAM, [MAA], statuses=("proposed", "active"))] \
        if "statuses" in skillbook.family_skills.__code__.co_varnames else [skillbook.view(s) for s in await skillbook.family_skills(db, FAM, [MAA])]
    assert any("Amul Taaza Milk from Blinkit" in json.dumps(r, ensure_ascii=False) for r in rows), rows
