"""Every scheduled message is checked against what the person said or logged today (live 20 Oct: "Skipping gym" at
19:15, then a gym reminder at 19:30 and "are you back from the gym?" at 21:15). Synthetic data only."""

from types import SimpleNamespace

from app.care import presend, store
from app.llm import router

FAM, ME = "fam-ps", "p-ps"


async def test_nothing_today_sends_without_asking_a_model(db, at, monkeypatch):
    at("2026-10-20 21:15")
    monkeypatch.setenv("PRESEND_CHECK", "on")

    async def boom(*a, **k):
        raise AssertionError("no model call when nothing was said today")

    monkeypatch.setattr(router, "complete", boom)
    assert await presend.still_right(db, FAM, ME, "Gym check-in: are you back?") == (True, "")


async def test_a_change_said_today_holds_it_and_the_model_decides(db, at, monkeypatch):
    at("2026-10-20 19:15")
    monkeypatch.setenv("PRESEND_CHECK", "on")
    await store.add_turn(db, family_id=FAM, thread_id=ME, role="user", text="Skipping gym")
    await db.flush()
    at("2026-10-20 21:15")
    seen = []

    async def model(role, **kw):
        seen.append(kw["messages"][0]["content"][0]["text"])
        return SimpleNamespace(text='{"send": false, "why": "skipping gym today"}')

    monkeypatch.setattr(router, "complete", model)
    assert await presend.still_right(db, FAM, ME, "Gym check-in: are you back?") == (False, "skipping gym today")
    assert "they wrote: Skipping gym" in seen[0] and "PLANNED: Gym" in seen[0]


async def test_a_failed_check_still_sends(db, at, monkeypatch):
    at("2026-10-20 19:15")
    monkeypatch.setenv("PRESEND_CHECK", "on")
    await store.add_turn(db, family_id=FAM, thread_id=ME, role="user", text="doctor stopped the evening tablet")
    await db.flush()

    async def down(*a, **k):
        raise TimeoutError

    monkeypatch.setattr(router, "complete", down)
    assert (await presend.still_right(db, FAM, ME, "Evening tablet"))[0] is True
