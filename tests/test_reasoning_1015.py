"""Reasoning checks from the 15 Oct chats: a timed promise is scheduled, a pending change is asked first,
a short reply is not a new instruction, and a question already asked today is not asked again."""

from datetime import datetime

import pytest

from app.brain import grounding, guards, tools
from app.brain.loop import TurnRequest, run_turn
from app.brain.persona import PERSONA, REPLY_FORMAT
from app.care import store
from app.core import clock
from app.llm import router
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM = "fam-reason"
ELDER = {"id": "elder-r", "name": "Meera Iyer", "role": "elder"}
CG = {"id": "cg-r", "name": "Arjun Iyer", "role": "primary caregiver"}


def test_a_timed_promise_with_no_reminder_is_caught():
    said = "Theek hai, 8:30 baje message bhej kar utha dungi."
    assert guards.timed_promise(said, [{"tool": "log_event", "ok": True}])
    assert not guards.timed_promise(said, [{"tool": "set_reminder", "ok": True}])
    assert not guards.timed_promise(said, [{"tool": "open_loop", "ok": True}])
    assert not guards.timed_promise("Your reminder is at 08:00.", [])
    assert not guards.timed_promise("Shall I message you at 8:30?", [])


def test_a_later_promise_is_unsupported_unless_this_turn_scheduled_it():
    raw = '{"claims": [{"kind": "action", "quote": "I will message you at 8:30", "supported": true, "later": true}]}'
    claim = grounding._parse(raw)[0]
    assert claim.later and claim.supported
    claim.supported = False
    assert "set_reminder" in grounding.problems([claim])[0]
    assert grounding.scheduled(["set_reminder: {\"at\": \"08:30\"}"])
    assert not grounding.scheduled(["log_event: noted", "remember: saved"])


def test_a_greeting_does_not_answer_a_question():
    t0 = datetime(2026, 10, 15, 10, 35, tzinfo=clock.IST)
    t1 = datetime(2026, 10, 15, 13, 40, tzinfo=clock.IST)
    asked = [(t0, "How is your stomach after the ice cream?")]
    assert guards.unanswered(asked, [(t1, "Hi")]) == ["How is your stomach after the ice cream?"]
    assert guards.unanswered(asked, [(t1, "Stomach is better now")]) == []
    again = guards.question_again("How is your stomach after the ice cream?", guards.unanswered(asked, [(t1, "Hi")]))
    assert again


def test_a_brief_does_not_invent_a_mood_from_hi():
    assert guards.brief_mood_claims("She is doing completely fine, stable and peaceful.", "hi")
    assert not guards.brief_mood_claims("Lata wrote hi. No mood or health update today.", "hi")


def test_persona_covers_the_reference_the_frame_and_the_follow_up():
    assert "Name it only when the frame itself shows" in PERSONA
    assert "ask one short question and call no tool" in PERSONA
    assert "ask one specific question about that thing" in PERSONA
    assert "set_reminder or open_loop" in REPLY_FORMAT
    assert "ALREADY ASKED TODAY" in REPLY_FORMAT


class Script:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    async def complete(self, route, *, messages, **kw):
        parts = []
        for key in ("system_stable", "system_dynamic"):
            val = kw.get(key) or ""
            parts.append(val if isinstance(val, str) else "\n".join(val))
        self.seen.append({"route": route, "text": "\n".join(parts)})
        return self.replies.pop(0) if self.replies else LLMReply(text="Hi.", tool_calls=[], model="fake")


@pytest.fixture
def brain(monkeypatch):
    holder = {}

    def use(replies):
        holder["s"] = Script(replies)
        router.register_provider("fake", holder["s"])
        return holder["s"]

    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"]}')
    router.reset_breakers()
    yield use
    router._providers.pop("fake", None)


async def _roster(db):
    await store.save_roster(db, FAM, ELDER, [ELDER, CG])


def _ctx(db, said="", *, channel="whatsapp"):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ELDER, speaker=CG, members=[ELDER, CG],
                         user_text=said, channel=channel)


async def test_anything_else_does_not_approve_a_pending_medicine(db, at):
    at("2026-10-15 02:01")
    await _roster(db)
    await store.write_fact(db, family_id=FAM, subject_id=ELDER["id"], domain="medicine", key="medicine:magnesium",
                           value={"name": "Magnesium", "dose": "200 mg"}, text="Magnesium 200 mg", source_kind="caregiver_said", stated_by=CG["id"])
    w = await store.write_fact(db, family_id=FAM, subject_id=ELDER["id"], domain="medicine", key="medicine:magnesium",
                               value={"name": "Magnesium", "dose": "400 mg"}, text="Magnesium 400 mg", source_kind="elder_said", stated_by=ELDER["id"])
    assert w.result == "pending"
    with pytest.raises(tools.ToolRefused, match="have not answered"):
        await tools.confirm_change(_ctx(db, "Anything else left for today?"), {"key": "medicine:magnesium", "approve": True})
    dash = await tools.confirm_change(_ctx(db, "Anything else left for today?", channel="dashboard"),
                                     {"key": "medicine:magnesium", "approve": False})
    assert dash["result"] in ("rejected", "stopped", "retracted", "active")
    w2 = await store.write_fact(db, family_id=FAM, subject_id=ELDER["id"], domain="medicine", key="medicine:magnesium",
                                value={"name": "Magnesium", "dose": "400 mg"}, text="Magnesium 400 mg", source_kind="elder_said", stated_by=ELDER["id"])
    assert w2.result == "pending"
    await store.add_turn(db, family_id=FAM, thread_id=CG["id"], role="assistant", text="Magnesium is still waiting for your yes.")
    out = await tools.confirm_change(_ctx(db, "Yes"), {"key": "medicine:magnesium", "approve": True})
    assert out["result"] in ("active", "superseded")


async def test_a_short_reply_does_not_save_a_skill_or_ask_the_wrong_thing(db, at):
    at("2026-10-15 14:00")
    await _roster(db)
    await store.add_turn(db, family_id=FAM, thread_id=CG["id"], role="assistant", text="Take your time.")
    with pytest.raises(tools.ToolRefused, match="last messages"):
        await tools.save_family_skill(_ctx(db, "As you wish"), {"text": "keep chat short and plain"})
    await store.add_turn(db, family_id=FAM, thread_id=CG["id"], role="user", text="When is she landing in Bangalore?", speaker_id=CG["id"])
    with pytest.raises(tools.ToolRefused, match="last messages"):
        await tools.send_message(_ctx(db, "Ask her please"), {"to": ELDER["id"], "text": "What time do you wake, eat lunch and go to bed?"})
    # the landing words are in the chat, so that question may be sent
    sent = await tools.send_message(_ctx(db, "Ask her please"), {"to": ELDER["id"], "text": "When are you landing in Bangalore?"})
    assert "refused" not in sent


async def test_a_promised_wake_up_is_rewritten_when_nothing_was_scheduled(db, at, brain):
    at("2026-10-15 03:37")
    await _roster(db)
    await db.commit()
    brain([
        LLMReply(text="Theek hai, 8:30 baje message bhej kar utha dungi.", tool_calls=[], model="fake"),
        LLMReply(text="Theek hai.", tool_calls=[], model="fake"),
    ])
    out = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ELDER, speaker=CG, members=[ELDER, CG],
                                                    text="8:30 baje message bhej kar utha dena", message_ref="m-promise"))
    assert "8:30" not in (out.reply or "")


async def test_hi_does_not_get_yesterdays_question_again(db, at, brain):
    at("2026-10-15 13:43")
    await _roster(db)
    await store.add_turn(db, family_id=FAM, thread_id=CG["id"], role="assistant",
                         text="How is your stomach after the ice cream?")
    await db.commit()
    s = brain([
        LLMReply(text="How is your stomach after the ice cream?", tool_calls=[], model="fake"),
        LLMReply(text="Hi.", tool_calls=[], model="fake"),
    ])
    out = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=ELDER, speaker=CG, members=[ELDER, CG],
                                                    text="Hi", message_ref="m-hi"))
    assert "stomach" not in (out.reply or "").lower()
    assert any("ALREADY ASKED TODAY" in x["text"] for x in s.seen)
