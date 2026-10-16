"""Reply speed (overnight 14 Oct): where each turn's time goes is kept on the why-trail, a turn never waits behind a
stuck one for minutes, and a turn never runs past the time the person is still waiting.

Live 12 Oct 20:33 UTC: one turn hung, Cloud Run cut the request at 300 s but the turn kept the family locked until its
instance shut down 10 minutes later; five messages waited behind it (125-248 s each). The backend stops waiting at 120 s."""

import asyncio
import json

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.brain import loop
from app.care import store, why
from app.llm import router, spend
from app.care.models import OpenLoop
from app.llm.router import LLMReply, ToolCall
from app.sim.world import SimHost
from tests.conftest import TEST_DB

FAM = "fam-speed"
ME = {"id": "self-speed", "name": "Asha Rao", "role": "primary caregiver"}


class Slow:
    def __init__(self, delay: float, text: str = "Okay."):
        self.delay, self.text, self.calls = delay, text, 0

    async def complete(self, route, **kw):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return LLMReply(text=self.text, tool_calls=[], model="fake")


@pytest.fixture
def model(monkeypatch):
    def make(delay: float, reply: str = "Okay."):
        p = Slow(delay, reply)
        router.register_provider("fake", p)
        monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"], "classify": ["fake:m"]}')
        monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0")
        router.reset_breakers()
        spend.reset()
        return p
    yield make
    router._providers.pop("fake", None)
    router.reset_breakers()


def req(text_: str, ref: str) -> loop.TurnRequest:
    return loop.TurnRequest(family_id=FAM, elder=ME, speaker=ME, members=[ME], text=text_, message_ref=ref)


async def test_a_tool_line_is_sent_without_another_model_call(db, monkeypatch):
    """The tool already said what to tell them. A second brain call and a judge would only repeat it."""
    await store.save_roster(db, FAM, ME, [ME])
    seen: list[str] = []

    class Rec:
        async def complete(self, route, **kw):
            seen.append("call")
            if len(seen) == 1:
                return LLMReply(text="", tool_calls=[ToolCall("c1", "remember", {"domain": "preference"})], model="fake")
            return LLMReply(text="This second draft must not be sent.", tool_calls=[], model="fake")

    async def fake_run(ctx, name, args):
        ctx.actions.append({"tool": name, "args": args, "ok": True})
        return json.dumps({"result": "saved", "say": "Saved: two eggs, 140 kcal."}), False

    router.register_provider("fake", Rec())
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"]}')
    monkeypatch.setenv("JUDGE_PER_FAMILY", "3")
    monkeypatch.setattr(loop.tools, "run", fake_run)
    router.reset_breakers()
    spend.reset()
    result = await loop.run_turn(db, SimHost(), req("two eggs", "speed-say"))
    assert seen == ["call"]
    assert "140 kcal" in result.reply and "second draft" not in result.reply


async def _schedule_run(ctx, name, args):
    meds = [{"name": "Zinc tablet", "dose": "1 tablet", "time": "23:30"}]
    ctx.actions.append({"tool": name, "args": args, "ok": True, "medicines": meds,
                        "result": json.dumps({"medicines": meds})})
    return json.dumps({"medicines": meds}), False


async def test_a_question_mark_on_a_quote_is_not_answered_with_the_schedule(db, monkeypatch):
    """They quoted a dinner line and sent ?. Looking up the day does not replace the reply with the medicine list."""
    person = {"id": "self-quote", "name": "Asha Rao", "role": "primary caregiver"}
    await store.save_roster(db, "fam-quote", person, [person])

    class Rec:
        async def complete(self, route, **kw):
            calls = getattr(self, "n", 0)
            self.n = calls + 1
            if calls == 0:
                return LLMReply(text="", tool_calls=[ToolCall("c1", "today_schedule", {})], model="fake")
            return LLMReply(text="Hope dinner went well. What did you have?", tool_calls=[], model="fake")

    router.register_provider("fake", Rec())
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"]}')
    monkeypatch.setenv("JUDGE_PER_FAMILY", "0")
    monkeypatch.setattr(loop.tools, "run", _schedule_run)
    router.reset_breakers()
    spend.reset()
    result = await loop.run_turn(
        db, SimHost(),
        loop.TurnRequest(family_id="fam-quote", elder=person, speaker=person, members=[person], text="?",
                         message_ref="quote-dinner",
                         media_note="They are replying to your message: Enjoy your dinner. I will ask in an hour what you had."))
    assert "Zinc tablet" not in result.reply
    assert "dinner" in result.reply.lower()


async def test_a_medicine_question_still_gets_the_full_schedule(db, monkeypatch):
    """Asked which medicines are due: the tool's list is the reply when the draft left one out."""
    person = {"id": "self-due", "name": "Asha Rao", "role": "primary caregiver"}
    await store.save_roster(db, "fam-due", person, [person])

    class Rec:
        async def complete(self, route, **kw):
            calls = getattr(self, "n", 0)
            self.n = calls + 1
            if calls == 0:
                return LLMReply(text="", tool_calls=[ToolCall("c1", "today_schedule", {})], model="fake")
            return LLMReply(text="You seem fine.", tool_calls=[], model="fake")

    router.register_provider("fake", Rec())
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"]}')
    monkeypatch.setenv("JUDGE_PER_FAMILY", "0")
    monkeypatch.setattr(loop.tools, "run", _schedule_run)
    router.reset_breakers()
    spend.reset()
    result = await loop.run_turn(
        db, SimHost(),
        loop.TurnRequest(family_id="fam-due", elder=person, speaker=person, members=[person],
                         text="which medicines are due?", message_ref="due-ask"))
    assert "Zinc tablet" in result.reply and "23:30" in result.reply


async def test_need_tools_brings_the_group_on_the_next_call(db, monkeypatch):
    """A missing group is one extra call, then those tools are on the next step."""
    await store.save_roster(db, FAM, ME, [ME])
    seen: list[list[str]] = []

    class Rec:
        async def complete(self, route, **kw):
            seen.append([t.name for t in (kw.get("tools") or [])])
            if len(seen) == 1:
                return LLMReply(text="", tool_calls=[ToolCall("c1", "need_tools", {"group": "order"})], model="fake")
            return LLMReply(text="Which store?", tool_calls=[], model="fake")

    router.register_provider("fake", Rec())
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"]}')
    monkeypatch.setenv("JUDGE_PER_FAMILY", "3")
    router.reset_breakers()
    spend.reset()
    result = await loop.run_turn(db, SimHost(), req("buy milk", "speed-need"))
    assert "start_task" not in seen[0] and "start_task" in seen[1]
    assert "Which store?" in result.reply


async def test_the_why_trail_keeps_where_the_time_went(db, model):
    await store.save_roster(db, FAM, ME, [ME])
    model(0.01, "Good morning Asha!")
    await loop.run_turn(db, SimHost(), req("good morning", "speed-1"))
    row = (await db.execute(select(why.TraceTurn).where(why.TraceTurn.message_ref == "speed-1"))).scalar_one()
    t = row.timing
    assert {"lockMs", "contextMs", "afterMs"} <= set(t) and all(isinstance(t[k], int) for k in ("lockMs", "contextMs", "afterMs"))
    assert [r for r, _ in t["model"]][0] == "brain" and all(ms >= 0 for _, ms in t["model"])
    assert "Asha" not in str(t)  # step names and milliseconds only


async def test_a_turn_never_waits_minutes_behind_a_stuck_one(db, monkeypatch):
    other = create_async_engine(TEST_DB)
    try:
        async with other.connect() as conn:
            async with conn.begin():
                await conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:f))"), {"f": "fam-busy"})  # a stuck turn
                with pytest.raises(loop.FamilyBusy):
                    await loop._lock_family(db, "fam-busy", wait_s=0.3)
    finally:
        await other.dispose()
    assert issubclass(loop.FamilyBusy, router.AllModelsFailed)  # answered like an outage: 503, the backend's holding line


async def test_the_wait_bound_is_only_for_taking_the_turn(db):
    before = (await db.execute(text("SHOW lock_timeout"))).scalar_one()
    await loop._lock_family(db, "fam-free", wait_s=5)
    assert (await db.execute(text("SHOW lock_timeout"))).scalar_one() == before  # the rest of the turn is as before


async def test_a_draft_already_written_is_sent_when_the_budget_runs_out(db, monkeypatch):
    """Five night probes died with no reply after the budget. The draft from the first call is the reply."""
    await store.save_roster(db, FAM, ME, [ME])

    class Two:
        def __init__(self):
            self.calls = 0

        async def complete(self, route, **kw):
            self.calls += 1
            if self.calls == 1:
                return LLMReply(text="How is the heat? Are you alright?", tool_calls=[], model="fake")
            await asyncio.sleep(2)
            return LLMReply(text="this rewrite never arrives", tool_calls=[], model="fake")

    p = Two()
    router.register_provider("fake", p)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"], "classify": ["fake:m"]}')
    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0")
    monkeypatch.setitem(loop.TURN_BUDGET_S, "message", 0.6)
    monkeypatch.setattr(loop, "MIN_CALL_S", 0.05)
    router.reset_breakers()
    spend.reset()
    result = await loop.run_turn(db, SimHost(), req("the heat today", "speed-draft"))
    assert "heat" in result.reply and "never arrives" not in result.reply


async def test_a_turn_never_runs_past_the_time_the_person_waits(db, model, monkeypatch):
    """Nothing done yet when the budget runs out: the same as no model answering (503, the backend's holding line)."""
    await store.save_roster(db, FAM, ME, [ME])
    monkeypatch.setitem(loop.TURN_BUDGET_S, "message", 0.4)
    monkeypatch.setattr(loop, "MIN_CALL_S", 0.05)
    p = model(2.0)
    started = asyncio.get_running_loop().time()
    with pytest.raises(router.AllModelsFailed):
        await loop.run_turn(db, SimHost(), req("are you there?", "speed-2"))
    assert asyncio.get_running_loop().time() - started < 1.5 and p.calls == 1


async def test_hedged_brain_calls_answer_a_slow_turn_sooner(db, monkeypatch):
    """The whole turn with a brain call that is slow once: the copy sent after BRAIN_HEDGE_AFTER_S answers."""
    await store.save_roster(db, FAM, ME, [ME])

    class OnceSlow:
        def __init__(self):
            self.calls = 0

        async def complete(self, route, **kw):
            self.calls += 1
            await asyncio.sleep(3.0 if self.calls == 1 else 0.01)
            return LLMReply(text="Here I am, Asha.", tool_calls=[], model="fake")

    p = OnceSlow()
    router.register_provider("fake", p)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"]}')
    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0.1")
    monkeypatch.setenv("GROUNDING_CHECK", "off")
    router.reset_breakers()
    spend.reset()
    try:
        out = await loop.run_turn(db, SimHost(), req("hello?", "speed-3"))
    finally:
        router._providers.pop("fake", None)
    assert out.reply.startswith("Here I am") and out.ms < 2000


async def test_a_long_reply_is_cut_in_code_on_any_turn(db, model, monkeypatch):
    await store.save_roster(db, FAM, ME, [ME])
    monkeypatch.setenv("GROUNDING_CHECK", "off")
    draft = "One thing here. Two things here. Three things here. Four things here. Five things here."
    p = model(0.01, draft)
    out = await loop.run_turn(db, SimHost(), req("please order milk", "speed-long"))
    assert p.calls == 1 and "Four things" not in out.reply and "One thing here." in out.reply


async def test_a_stall_after_a_save_promises_a_reply_and_opens_a_loop(db, monkeypatch):
    await store.save_roster(db, FAM, ME, [ME])
    monkeypatch.setattr(loop, "CALL_CAP_S", 0.3)
    monkeypatch.setenv("GROUNDING_CHECK", "off")
    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0")
    monkeypatch.delenv("GEMINI_SG_FALLBACK", raising=False)

    class ThenStuck:
        def __init__(self):
            self.calls = 0

        async def complete(self, route, **kw):
            self.calls += 1
            if self.calls == 1:
                return LLMReply(text="", tool_calls=[ToolCall(
                    id="c1", name="remember",
                    args={"domain": "dish", "name": "khichdi", "details": {"food": "khichdi"}, "sentence": "Likes khichdi"},
                )], model="fake")
            await asyncio.sleep(2)
            return LLMReply(text="Saved.", tool_calls=[], model="fake")

    p = ThenStuck()
    router.register_provider("fake", p)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    try:
        out = await loop.run_turn(db, SimHost(), req("ok", "speed-hold"))
    finally:
        router._providers.pop("fake", None)
    assert "reply shortly" in out.reply
    row = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "finish"))).scalar_one()
    assert row.status == "open" and row.owner_id == ME["id"]
