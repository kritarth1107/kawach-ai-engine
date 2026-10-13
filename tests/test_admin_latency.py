"""Saheli's reply time for the hourly live check (quality plan F8): message turns of the last hours, nearest-rank p50/p90,
sims, shadow families, system and dashboard turns left out."""

from datetime import timedelta

from app.api import admin
from app.care import why
from app.core import clock


async def test_reply_time_is_the_message_turns_of_the_window(db, at):
    at("2026-10-14 01:00")
    old = clock.now() - timedelta(hours=3)
    await why.record_turn(db, trace_id="t-old", family_id="fam-lat", thread_id="p1", kind="message", at=old, ms=99_000)
    for i, ms in enumerate([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 40_000]):
        await why.record_turn(db, trace_id=f"t{i}", family_id="fam-lat", thread_id="p1", kind="message", at=clock.now() - timedelta(minutes=i), ms=ms)
    await why.record_turn(db, trace_id="t-sys", family_id="fam-lat", thread_id="saheli", kind="system", ms=200_000)
    await why.record_turn(db, trace_id="t-sim", family_id="sim-x", thread_id="p", kind="message", ms=300_000)
    await why.record_turn(db, trace_id="t-shadow", family_id="shadow:canary", thread_id="p", kind="message", ms=300_000)
    await db.commit()
    out = await admin.admin_latency(db, hours=1)
    assert out == {"hours": 1, "turns": 10, "p50Ms": 5000, "p90Ms": 9000, "maxMs": 40_000}
    wide = await admin.admin_latency(db, hours=4)
    assert wide["turns"] == 11 and wide["maxMs"] == 99_000


async def test_no_turns_is_no_numbers(db, at):
    at("2026-10-14 01:00")
    assert await admin.admin_latency(db, hours=1) == {"hours": 1, "turns": 0, "p50Ms": None, "p90Ms": None, "maxMs": None}
