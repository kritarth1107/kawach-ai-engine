"""The read-only live score and memory score of 13 Oct, every failure's root cause (trust plan C1, A2, A3, C4):

- messages Saheli starts (check-ins, order updates) called a caregiver by his roster name, not the name he asked for: the
  household block now says how to address everyone, and every reply and every send_message is checked against it;
- "hold off on that tablet tonight" passed the advice check as general information: a dose change is never hers to say;
- two questions at once to the person cared for: one question per message, checked on her own text;
- the judge graded an order's options as "made up" because a message she started was logged without the scheduler's
  prompt or the turn's tool results; tool results in the trace were cut at 1,500 characters;
- checks 2 and 3 errored on long conversations: the thinking model's output limit cut its JSON (OUT_TOKENS);
- check 2 graded the scheduler's own thread as a person's conversation;
- check 3 failed a correction for every message after a later correction of the same setting.
Scripted models; invented families.
"""

import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import grounding, guards, loop, tools
from app.care import digest, naming, store, versions
from app.care.models import FamilyRoster
from app.core import clock
from app.learn.models import ReplyLog
from app.llm import router, spend
from app.llm.router import LLMReply
from app.quality import daily, judge
from app.sim.world import SimHost

FAM = "fam-lsf"
MAA = {"id": "e-lsf", "name": "Kamla Devi", "role": "elder"}
SON = {"id": "c-lsf", "name": "Rohit Verma", "role": "primary caregiver"}
SCHED = {"id": loop.SYSTEM_ID, "name": "Saheli scheduler", "role": "system"}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


@pytest.fixture(autouse=True)
def clean():
    spend.reset()
    router.reset_breakers()
    yield
    spend.reset()


class Fake:
    """answer(system_text, user_text, kw) -> text; records every call."""

    def __init__(self, answer):
        self.answer, self.seen = answer, []

    async def complete(self, route, **kw):
        sys = kw.get("system_stable") or ""
        sys = "\n".join(sys) if isinstance(sys, list) else sys
        user = "".join(c.get("text", "") for c in kw["messages"][-1]["content"]) if kw.get("messages") else ""
        self.seen.append({"route": route, "system": sys, "user": user, "max_tokens": kw.get("max_tokens")})
        return LLMReply(text=self.answer(sys, user, kw), tool_calls=[], model="fake", usage={"in": 100, "out": 50})


@pytest.fixture
def fake(monkeypatch):
    def use(answer):
        f = Fake(answer)
        router.register_provider("fake", f)
        return f

    monkeypatch.setenv("MODEL_ROUTES", json.dumps({r: ["fake:m"] for r in ("judge", "judge_fast", "brain", "extract", "worker", "classify")}))
    monkeypatch.setenv("GROUNDING_CHECK", "on")
    yield use
    router._providers.pop("fake", None)
    router.reset_breakers()


async def name_fact(db, subject, value, text, by):
    with versions.attribution(actor_id=by, source="whatsapp"):
        await store.write_fact(db, family_id=FAM, subject_id=subject, domain="naming", key="naming:address_as", value=value, text=text,
                               source_kind="caregiver_said", stated_by=by)


# ── names in every message ────────────────────────────────────────────────────


def test_address_problems_roster_name_where_another_name_was_asked_for():
    who = {"name": "Rinku", "avoid": []}
    assert naming.address_problems("Rohit, your order is ready.", roster_name="Rohit Verma", who=who) == [
        "call them Rinku, the name they asked for, not Rohit"]
    assert naming.address_problems("Rinku, your order is ready.", roster_name="Rohit Verma", who=who) == []
    assert naming.address_problems("Your order is ready.", roster_name="Rohit Verma", who=who) == []
    # the name to use contains the roster name ("Kamla ji" for Kamla Devi): nothing to check
    assert naming.address_problems("Kamla, dawai ka time", roster_name="Kamla Devi", who={"name": "Kamla ji", "avoid": []}) == []
    assert naming.address_problems("Maa, dawai ka time", roster_name="Kamla Devi", who={"name": "Kamla ji", "avoid": ["Maa"]}) == [
        "never call them 'Maa'; call them Kamla ji"]
    assert naming.address_problems("Rohit, hi", roster_name="Rohit Verma", who=None) == []


def test_household_says_how_to_address_each_person():
    block = digest.household(MAA, [MAA, SON], SON["id"], {SON["id"]: {"name": "Rinku", "avoid": []}, MAA["id"]: {"name": "Amma", "avoid": ["Maa"]}})
    assert "Care recipient: Kamla Devi (id e-lsf) — call them Amma; never: Maa" in block
    assert "Rohit Verma (id c-lsf) — call them Rinku  ← speaking now" in block


async def test_the_brain_sees_everyones_name_in_a_scheduled_turn(db, fake):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await name_fact(db, SON["id"], {"name": "Rinku"}, "Call Rohit 'Rinku'", SON["id"])
    req = loop.TurnRequest(family_id=FAM, elder=MAA, speaker=SCHED, members=[MAA, SON], text="[Wake] check on Rohit's order", message_ref="task:x")
    block, _known, _avoid = await loop.family_block(db, req)
    assert "Rohit Verma (id c-lsf) — call them Rinku" in block


async def test_send_message_refuses_the_roster_name_and_two_questions_to_the_elder(db, fake):
    fake(lambda s, u, kw: '{"claims": []}')
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await name_fact(db, SON["id"], {"name": "Rinku"}, "Call Rohit 'Rinku'", SON["id"])
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=MAA, speaker=SCHED, members=[MAA, SON], message_ref="loop:1")
    got = await tools.send_problems(ctx, SON["id"], "Rohit, did the trainer send your plan?")
    assert "call them Rinku, the name they asked for, not Rohit" in got
    assert not any("Rinku" in p for p in await tools.send_problems(ctx, SON["id"], "Rinku, did the trainer send your plan?"))
    two = await tools.send_problems(ctx, MAA["id"], "आज नींद अच्छी आई? कैसी है तबियत?")
    assert any("2 questions at once" in p for p in two)
    assert not any("questions at once" in p for p in await tools.send_problems(ctx, MAA["id"], "आज तबियत कैसी है?"))


def test_one_question_per_message_counts_every_script():
    assert guards.question_problems("Sleep well? How is your health?")
    assert guards.question_problems("نیند اچھی آئی؟ طبیعت کیسی ہے؟")
    assert guards.question_problems("How are you today?") == []
    assert guards.length_problems("One. Two. Three. Four.")
    assert guards.length_problems("One. Two. Three.", cap=2)
    assert guards.length_problems("One. Two. Three.") == []
    assert guards.opener_problems("I have noted this. She did not eat.")
    assert guards.opener_problems("According to health sources, rest helps.")
    assert guards.opener_problems("She did not eat. I have noted this.") == []


async def test_a_reply_to_the_elder_with_two_questions_is_rewritten(db, fake):
    replies = iter(["जानकर अच्छा लगा। रसोई हो गई? भोजन कर लिया?", "जानकर अच्छा लगा। भोजन कर लिया आपने?"])

    def answer(sys, user, kw):
        return '{"claims": []}' if sys.startswith(grounding.PROMPT[:40]) else next(replies, "ठीक है।")

    fake(answer)
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    res = await loop.run_turn(db, SimHost(), loop.TurnRequest(family_id=FAM, elder=MAA, speaker=MAA, members=[MAA, SON], text="theek hu",
                                                               message_ref="lsf-q1"))
    assert res.reply == "जानकर अच्छा लगा। भोजन कर लिया आपने?"


# ── a dose change is never hers ───────────────────────────────────────────────


def test_a_dose_change_is_a_problem_whatever_the_source_except_the_doctor():
    raw = json.dumps({"claims": [
        {"kind": "advice", "quote": "Hold off on that Magnesium tablet tonight", "supported": True, "prescriptive": True, "source": "tool", "dose_change": True},
        {"kind": "advice", "quote": "skip the missed dose", "supported": False, "prescriptive": True, "source": "safety", "dose_change": True},
        {"kind": "advice", "quote": "the doctor said to stop Ecosprin before the surgery", "supported": True, "prescriptive": True, "source": "doctor",
         "dose_change": True},
        {"kind": "advice", "quote": "if you have not taken it yet, please take it now", "supported": True, "prescriptive": True, "source": "safety",
         "dose_change": False},
    ]})
    cs = grounding._parse(raw)
    assert [c.quote for c in cs if c.problem] == ["Hold off on that Magnesium tablet tonight", "skip the missed dose"]
    assert "only their doctor decides that" in grounding.problems([cs[0]])[0]


# ── what the judge sees ───────────────────────────────────────────────────────


async def test_a_message_she_starts_is_logged_with_its_prompt_and_tool_results(db, fake):
    fake(lambda s, u, kw: '{"claims": []}')
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    prompt = "[Task update] Blinkit: 1. Choco Almond bar ₹80 ← suggested; Instamart did not answer in time. Show them the options."
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=MAA, speaker=SCHED, members=[MAA, SON], message_ref="task:t1",
                        prompt=prompt, tool_texts=["tasks: {\"t1\": \"options ready\"}"])
    out, err = await tools.run(ctx, "send_message", {"to": SON["id"], "text": "Instamart did not answer in time. Blinkit: 1. Choco Almond bar ₹80. Which one?"})
    assert not err, out
    row = (await db.execute(select(ReplyLog).where(ReplyLog.family_id == FAM))).scalars().one()
    assert row.kind == "proactive" and row.user_text == prompt
    assert row.trace[0] == {"role": "user", "text": prompt} and row.trace[1]["results"][0]["name"] == "tasks"


async def test_the_judge_reads_a_proactive_prompt_as_a_source(db, at):
    at("2026-10-13 10:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    roster = await db.get(FamilyRoster, FAM)
    r = ReplyLog(family_id=FAM, thread_id=SON["id"], at=clock.now(), kind="proactive", situation="order", speaker_role="caregiver", lang="latin",
                 user_text="[Task update] Blinkit: Choco Almond ₹80", text="Blinkit has Choco Almond for ₹80.", text_len=10, tools=[])
    db.add(r)
    await db.flush()
    inp = await judge.build_input(db, r, roster)
    payload = json.loads(inp.user)
    assert payload["the scheduled prompt she was answering (its data is a source)"] == "[Task update] Blinkit: Choco Almond ₹80"
    assert "the scheduled prompt she was answering" in judge.JUDGE_PROMPT
    old = ReplyLog(family_id=FAM, thread_id=SON["id"], at=clock.now(), kind="proactive", situation="order", speaker_role="caregiver", lang="latin",
                   user_text="", text="x", text_len=1, tools=[])
    db.add(old)
    await db.flush()
    assert json.loads((await judge.build_input(db, old, roster)).user)["the scheduled prompt she was answering (its data is a source)"] == "(not recorded)"


def test_trace_keeps_long_tool_results():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "list her medicines"}]},
            {"role": "tool", "results": [{"name": "search_records", "content": "x" * 3500, "is_error": False}]}]
    tr = loop.turn_trace(msgs, 0, "done")
    assert len(tr[1]["results"][0]["content"]) == 3500


# ── checks 2 and 3 ────────────────────────────────────────────────────────────


async def test_judge_calls_leave_room_for_thinking(db, at, sessions, fake):
    at("2026-10-05 09:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="user", text="Mera beta Ravi har Sunday aata hai", speaker_id=MAA["id"])
    await store.add_turn(db, family_id=FAM, thread_id=loop.SYSTEM_ID, role="user", text="[Wake] check-in loop due", speaker_id=loop.SYSTEM_ID)
    await db.commit()
    f = fake(lambda s, u, kw: json.dumps({"items": [], "invented": []}))
    out = await judge.remembered(sessions, FAM, "2026-10-05")
    assert [c["thread"] for c in out] == [MAA["id"]]  # the scheduler's thread is not a person's conversation
    assert f.seen and all(c["max_tokens"] >= 16000 for c in f.seen)


async def test_check3_stops_at_a_later_correction_of_the_same_setting(db, at, sessions, fake):
    at("2026-10-05 09:00")
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    with versions.attribution(actor_id=SON["id"], source="whatsapp"):
        await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="language", key="language:preferred", value={"language": "hi"},
                               text="Speaks Hindi", source_kind="caregiver_said", stated_by=SON["id"])
    for when, value, text in (("2026-10-05 10:00", {"language": "hi", "dialect": "hne"}, "Speaks Chhattisgarhi"),
                              ("2026-10-05 12:00", {"language": "en"}, "Speaks English")):
        at(when)
        with versions.attribution(actor_id=SON["id"], source="whatsapp"):
            await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="language", key="language:preferred", value=value, text=text,
                                   source_kind="caregiver_said", stated_by=SON["id"])
    at("2026-10-05 11:00")
    await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="assistant", text="का हाल हे दाई?")
    at("2026-10-05 13:00")
    await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="assistant", text="How are you feeling today?")
    await db.commit()
    seen: list[dict] = []

    def answer(sys, user, kw):
        data = json.loads(user)
        seen.append(data)
        return json.dumps({"messages": [{"n": m["n"], "relevant": True, "held": True} for m in data["MESSAGES"]]})

    fake(answer)
    got = await daily.corrections_held(sessions, FAM, "2026-10-05")
    by_new = {d["CORRECTION"]["new"]: [m["text"] for m in d["MESSAGES"]] for d in seen}
    assert by_new["Speaks Chhattisgarhi"] == ["का हाल हे दाई?"]  # the English message after the next change is not held against it
    assert by_new["Speaks English"] == ["How are you feeling today?"]
    assert (got["num"], got["den"]) == (2, 2)


# ── no invention in messages she starts (C4.5) ───────────────────────────────


def _claims(*quotes):
    return json.dumps({"claims": [{"kind": "event", "quote": q, "supported": False} for q in quotes]})


async def test_send_message_is_checked_for_inventions_then_stripped_on_the_second_try(db, fake, at):
    at("2026-10-13 18:00")
    inv = "She went to the temple with Mala after lunch."
    seen = []

    def answer(sys, user, kw):
        seen.append(user)
        return _claims(inv) if inv in user.split("REPLY")[-1] else '{"claims": []}'

    fake(answer)
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    prompt = "[Daily brief] … WHAT KAMLA SAID: 21:15 bhindi"
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=MAA, speaker=SCHED, members=[MAA, SON], message_ref="task:x",
                        prompt=prompt, known=["HOUSEHOLD …", "", "", "", prompt])
    text = f"Kamla ji had bhindi for dinner. {inv}"
    out, err = await tools.run(ctx, "send_message", {"to": SON["id"], "text": text})
    assert err and "is not in memory or this chat" in out  # first time: back to the brain with the exact words
    assert "21:15 bhindi" in seen[0].split("REPLY")[0]  # the scheduled prompt's data is a source
    out, err = await tools.run(ctx, "send_message", {"to": SON["id"], "text": text})
    assert not err
    row = (await db.execute(select(ReplyLog).where(ReplyLog.family_id == FAM))).scalars().one()
    assert row.text == "Kamla ji had bhindi for dinner."  # second time: sent without the invented sentence


def test_the_recall_question_names_the_item_the_family_changed():
    from datetime import datetime, timezone

    from app.quality import memory_score
    from app.quality.corrections import Correction

    t = memory_score.Truths(subject_id="e1", name="Kamla Devi")
    assert '"what": "the most recent thing the family changed"' in memory_score.recall_question(t)
    t.correction = Correction("f", "e1", "activity:sleep", "activity", "change", datetime(2026, 10, 13, tzinfo=timezone.utc), "c1", "whatsapp",
                              "sleeps 00:00-01:00", "sleeps 01:00-02:00")
    q = memory_score.recall_question(t)
    assert '"what": "activity: sleep"' in q and "Kamla Devi (id e1)" in q


async def test_a_medicine_dose_that_is_a_time_of_day_is_sent_back(db):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=MAA, speaker=SON, members=[MAA, SON], message_ref="m1", user_text="her meds")
    for dose in ("10 am", "T am"):
        out, err = await tools.run(ctx, "remember", {"domain": "medicine", "name": "Empsi L", "details": {"dose": dose, "times": ["08:00"]},
                                                    "sentence": f"Empsi L {dose}", "about": MAA["id"]})
        assert err and "is a time of day" in out
    out, err = await tools.run(ctx, "remember", {"domain": "medicine", "name": "Empsi L", "details": {"dose": "10 mg", "times": ["08:00"]},
                                                "sentence": "Empsi L 10 mg at 08:00", "about": MAA["id"]})
    assert not err, out
