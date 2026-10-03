"""Efficiency and no-leak guarantees (audit 2026-10-04)."""

from datetime import date

import httpx
import pytest
from sqlalchemy import event

from app.agents import tool_client
from app.brain import guards
from app.brain.loop import TurnRequest, run_turn
from app.care import store
from app.llm import router, spend
from app.llm.router import LLMReply
from app.sim.world import SimHost


def test_spend_counters_keep_only_today(monkeypatch):
    spend.reset()
    days = iter(["2026-10-01", "2026-10-02", "2026-10-03"])
    for d in days:
        monkeypatch.setattr(spend, "today", lambda d=d: d)
        import asyncio

        asyncio.run(spend.record("brain", "gemini-3.8-flash", {"in": 10, "out": 10}))
    assert {k[0] for k in spend._mem} == {"2026-10-03"}
    spend.reset()


def test_name_patterns_are_cached():
    guards._rx.cache_clear()
    for _ in range(50):
        guards.false_claims("Maine Asha ko bata diya hai.", others={"asha": "a"}, messaged=set(), ordering_ok=False)
    info = guards._rx.cache_info()
    assert info.hits > info.misses


async def test_ledger_writes_are_batched(db, monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    spend.reset()
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    spend.configure(sessions)
    for _ in range(5):
        await spend.record("brain", "gemini-3.8-flash", {"in": 100, "out": 50})
    assert len(spend._unsaved) == 1  # five calls, one pending row
    await spend.flush()
    rows = await spend.summary(sessions, days=1)
    assert rows[0]["calls"] == 5
    spend.reset()


class Fast:
    async def complete(self, route, **kw):
        return LLMReply(text="Theek hai ji.", tool_calls=[], model="fake")


async def test_turn_reads_each_thing_once(db, at, monkeypatch):
    at("2026-10-02 10:00")
    router.register_provider("fake", Fast())
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    elder, cg = {"id": "e-q", "name": "Kamla", "role": "elder"}, {"id": "c-q", "name": "Asha", "role": "primary caregiver"}
    await store.save_roster(db, "fam-q", elder, [elder, cg])
    for i in range(30):
        await store.add_turn(db, family_id="fam-q", thread_id=cg["id"], role="user", text=f"msg {i}", speaker_id=cg["id"])
    await db.commit()
    stmts = []

    def count(conn, cursor, statement, *a):
        stmts.append(statement.split()[0:6])

    sync_engine = db.bind.sync_engine if hasattr(db.bind, "sync_engine") else db.bind.engine.sync_engine
    event.listen(sync_engine, "before_cursor_execute", count)
    try:
        await run_turn(db, SimHost(), TurnRequest(family_id="fam-q", elder=elder, speaker=cg, members=[elder, cg], text="Mummy kaisi hain?", message_ref="q1"))
    finally:
        event.remove(sync_engine, "before_cursor_execute", count)
    turns_reads = [s for s in stmts if "turns" in " ".join(s).lower() and s[0].upper() == "SELECT"]
    assert len(stmts) <= 30, f"{len(stmts)} statements"
    assert len(turns_reads) <= 4  # was 6+ (history, profiles per person, guard, elder thread, compaction)
    router._providers.pop("fake", None)


async def test_backend_write_is_not_retried_after_timeout(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout("slow", request=req)

    import asyncio

    loop = asyncio.get_running_loop()
    tool_client._clients[id(loop)] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.TimeoutException):
        await tool_client.execute_backend_tool(tool="send_whatsapp", args={}, family_id="f", elder_id="e", actor_user_id="a")
    assert len(calls) == 1  # the message may already have gone: never send twice
    calls.clear()
    with pytest.raises(httpx.TimeoutException):
        await tool_client.execute_backend_tool(tool="get_today_schedule", args={}, family_id="f", elder_id="e", actor_user_id="a")
    assert len(calls) == 2  # a read is safe to retry
    await tool_client.close_clients()
