"""A burst from one person is one follow-up turn. Another person in the family does not wait on it."""

import asyncio
import json

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.brain import batch, loop
from app.care import store
from app.care.models import CareFact, TurnInbox
from app.llm import router, spend
from app.llm.router import LLMReply
from app.sim.world import SimHost
from tests.conftest import TEST_DB

FAM = "fam-batch"
ELDER = {"id": "elder-batch", "name": "Mina Das", "role": "elder"}
A = {"id": "person-a", "name": "Ravi Das", "role": "primary caregiver"}
B = {"id": "person-b", "name": "Leela Das", "role": "caregiver"}
ROUTES = (
    '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], '
    '"worker": ["fake:m"], "classify": ["fake:m"], "judge": ["fake:m"]}'
)


class Fake:
    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.hold_first = False
        self.pause = 0.0
        self.inflight = 0
        self.most_at_once = 0

    async def complete(self, route, **kw):
        self.calls += 1
        self.inflight += 1
        self.most_at_once = max(self.most_at_once, self.inflight)
        self.seen.append(json.dumps(kw.get("messages"), default=str))
        try:
            if self.hold_first and self.calls == 1:
                self.started.set()
                await self.release.wait()
            elif self.pause:
                await asyncio.sleep(self.pause)
        finally:
            self.inflight -= 1
        return LLMReply(text="Noted.", tool_calls=[], model="fake")


@pytest.fixture
def model(monkeypatch):
    fake = Fake()
    router.register_provider("fake", fake)
    monkeypatch.setenv("MODEL_ROUTES", ROUTES)
    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0")
    monkeypatch.setenv("GROUNDING_CHECK", "off")
    router.reset_breakers()
    spend.reset()
    yield fake
    router._providers.pop("fake", None)
    router.reset_breakers()


def _req(who: dict, text_: str, ref: str, *, role: str | None = None) -> loop.TurnRequest:
    speaker = {**who, "role": role} if role else who
    return loop.TurnRequest(
        family_id=FAM, elder=ELDER, speaker=speaker, members=[ELDER, A, B], text=text_, message_ref=ref,
    )


@pytest.fixture
async def sessions(db):
    """Real commits, so two connections can see each other's inbox rows. Cleaned up after the test."""
    engine = create_async_engine(TEST_DB)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield maker
    finally:
        async with maker() as session:
            for table in ("turn_inbox", "turns", "trace_turns", "family_rosters", "care_facts", "open_loops", "reply_log", "capability_gaps"):
                await session.execute(text(f"DELETE FROM {table} WHERE family_id = :f"), {"f": FAM})
            await session.commit()
        await engine.dispose()


async def _run(maker, who, text_, ref, *, role: str | None = None):
    async with maker() as session:
        return await loop.run_turn(session, SimHost(), _req(who, text_, ref, role=role))


async def test_a_burst_from_one_person_is_one_follow_up(sessions, model, monkeypatch):
    entered = {"n": 0}
    waiting_for = [10**9]
    both_waiting = asyncio.Event()
    real_lock = batch._lock_speaker

    async def count_lock(conn, key, wait_s):
        entered["n"] += 1
        if entered["n"] >= waiting_for[0]:
            both_waiting.set()
        await real_lock(conn, key, wait_s)

    monkeypatch.setattr(batch, "_lock_speaker", count_lock)
    model.hold_first = False
    await _run(sessions, A, "solo hello", "batch-solo")
    one = model.calls
    assert one >= 1
    model.calls = 0
    model.seen.clear()
    model.hold_first = True
    waiting_for[0] = entered["n"] + 3  # the reply in progress, then the two that join it

    async def later():
        await model.started.wait()
        pending = asyncio.gather(
            _run(sessions, A, "beta two", "batch-b"),
            _run(sessions, A, "gamma three", "batch-c"),
        )
        await both_waiting.wait()
        model.release.set()
        return await pending

    first, (second, third) = await asyncio.gather(_run(sessions, A, "alpha one", "batch-a"), later())
    assert first.reply == "Noted." and not first.duplicate
    replies = [second, third]
    assert sum(1 for r in replies if r.reply == "Noted." and not r.duplicate) == 1
    assert sum(1 for r in replies if r.duplicate and not r.reply) == 1
    assert any("beta two" in s and "gamma three" in s for s in model.seen)
    assert not any(("beta two" in s) ^ ("gamma three" in s) for s in model.seen)
    assert model.calls <= one * 2 + 2
    async with sessions() as session:
        rows = (await session.execute(select(TurnInbox).where(TurnInbox.family_id == FAM))).scalars().all()
    assert {r.message_ref for r in rows} >= {"batch-a", "batch-b", "batch-c"}
    assert {r.status for r in rows if r.message_ref in {"batch-a", "batch-b", "batch-c"}} == {"answered"}


async def test_two_people_do_not_wait_for_each_other(sessions, model):
    model.pause = 0.35
    await _run(sessions, A, "ping from one", "batch-p1")
    model.most_at_once = 0
    one, two = await asyncio.gather(
        _run(sessions, A, "ping from one again", "batch-p2"),
        _run(sessions, B, "ping from two", "batch-p3"),
    )
    assert one.reply and two.reply and not one.duplicate and not two.duplicate
    # their model calls ran at the same time (a wall-clock ratio was flaky on a loaded runner)
    assert model.most_at_once >= 2


async def test_a_message_does_not_wait_on_the_family_lock(sessions, model, monkeypatch):
    other = create_async_engine(TEST_DB)
    try:
        async with other.connect() as conn:
            async with conn.begin():
                await conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:f))"), {"f": FAM})
                out = await _run(sessions, A, "hello there", "batch-free")
                assert out.reply == "Noted." and not out.duplicate
                monkeypatch.setitem(loop.LOCK_WAIT_S, "system", 0.3)
                with pytest.raises(loop.FamilyBusy):
                    await _run(sessions, A, "scheduled", "batch-sys", role="system")
    finally:
        await other.dispose()


async def test_a_retried_message_is_not_answered_twice(sessions, model):
    first = await _run(sessions, A, "the only line", "batch-once")
    calls = model.calls
    again = await _run(sessions, A, "the only line", "batch-once")
    assert first.reply == "Noted." and not first.duplicate
    assert again.duplicate and again.reply == "Noted."
    assert model.calls == calls


async def test_an_unfinished_turn_is_still_resumed(db, model):
    await store.add_turn(
        db, family_id=FAM, thread_id=A["id"], role="user", text="the only line", speaker_id=A["id"], message_ref="batch-resume",
    )
    out = await loop.run_turn(db, SimHost(), _req(A, "the only line", "batch-resume"))
    assert out.reply == "Noted." and not out.duplicate and model.calls >= 1
    calls = model.calls
    again = await loop.run_turn(db, SimHost(), _req(A, "the only line", "batch-resume"))
    assert again.duplicate and model.calls == calls


async def test_two_writers_of_one_record_leave_one_active_fact(sessions):
    async def put(dose: str) -> None:
        async with sessions() as session:
            await store.write_fact(
                session, family_id=FAM, subject_id=ELDER["id"], domain="medicine", key="medicine:metformin",
                value={"name": "Metformin", "dose": dose}, text=f"Metformin {dose}",
                source_kind="caregiver_said", stated_by=A["id"],
            )

    await asyncio.gather(put("500 mg"), put("850 mg"))
    async with sessions() as session:
        active = (
            await session.execute(
                select(func.count()).select_from(CareFact).where(
                    CareFact.family_id == FAM, CareFact.key == "medicine:metformin", CareFact.status == "active",
                )
            )
        ).scalar_one()
    assert active == 1
