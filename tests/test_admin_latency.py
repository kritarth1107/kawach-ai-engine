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
    assert out == {"hours": 1, "turns": 10, "p50Ms": 5000, "p90Ms": 9000, "maxMs": 40_000, "timedTurns": 0, "steps": {}}
    wide = await admin.admin_latency(db, hours=4)
    assert wide["turns"] == 11 and wide["maxMs"] == 99_000


async def test_no_turns_is_no_numbers(db, at):
    at("2026-10-14 01:00")
    assert await admin.admin_latency(db, hours=1) == {"hours": 1, "turns": 0, "p50Ms": None, "p90Ms": None, "maxMs": None, "timedTurns": 0,
                                                    "steps": {}}


async def test_where_the_time_went(db, at):
    """Each turn's timing (lock wait, context, model calls by role, tools, checks, after the reply) as p50/p90 per step."""
    at("2026-10-14 01:00")
    for i in range(10):
        timing = {"lockMs": 100 * i, "contextMs": 300, "checkMs": 1200, "afterMs": 150,
                  "model": [["brain", 4000 + 1000 * i], ["brain", 3000], ["judge_fast", 900]], "tool": [["recall", 200], ["set_reminder", 400]]}
        await why.record_turn(db, trace_id=f"tt{i}", family_id="fam-lat2", thread_id="p1", kind="message",
                              at=clock.now() - timedelta(minutes=i), ms=10_000 + 1000 * i, timing=timing)
    await why.record_turn(db, trace_id="tt-old-row", family_id="fam-lat2", thread_id="p1", kind="message", ms=5000)  # before timing
    await db.commit()
    out = await admin.admin_latency(db, hours=1)
    assert out["turns"] == 11 and out["timedTurns"] == 10
    s = out["steps"]
    assert s["brain"] == {"p50Ms": 11_000, "p90Ms": 15_000}
    assert s["otherModels"] == {"p50Ms": 900, "p90Ms": 900}
    assert s["tools"] == {"p50Ms": 600, "p90Ms": 600}
    assert s["lock"] == {"p50Ms": 400, "p90Ms": 800}
    assert s["own"] == {"p50Ms": 13_600, "p90Ms": 17_200}  # the turn's own time: its total less the wait for the family
    assert s["check"]["p50Ms"] == 1200 and s["after"]["p50Ms"] == 150 and s["context"]["p50Ms"] == 300
