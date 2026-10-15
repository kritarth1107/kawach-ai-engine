"""A greeting or an ack thinks as little as the model allows, and only the recent thread is sent.
A too-long draft of that kind is shortened in code instead of another brain call. The reply-path judge still runs.
A medicine question is unchanged."""

from app.brain import loop
from app.care import store
from app.llm import router, spend
from app.llm.router import LLMReply, ToolCall
from app.sim.world import SimHost

FAM = "fam-light"
ME = {"id": "self-light", "name": "Kittu Sharma", "role": "primary caregiver"}


class Recorder:
    def __init__(self, script):
        self.script = list(script)
        self.efforts: list[str] = []
        self.messages: list = []
        self.max_tokens: list = []

    async def complete(self, route, **kw):
        self.efforts.append(kw.get("effort"))
        self.messages.append(list(kw.get("messages") or []))
        self.max_tokens.append(kw.get("max_tokens"))
        out = self.script.pop(0) if self.script else LLMReply(text="Okay.", tool_calls=[], model="fake")
        return out


def brain(monkeypatch, script):
    r = Recorder(script)
    router.register_provider("fake", r)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"], "classify": ["fake:m"]}')
    monkeypatch.setattr(loop, "BRAIN_EFFORT", "low")
    monkeypatch.setattr(loop, "BRAIN_EFFORT_HARD", "medium")
    router.reset_breakers()
    spend.reset()
    return r


def req(text, ref):
    return loop.TurnRequest(family_id=FAM, elder=ME, speaker=ME, members=[ME], text=text, message_ref=ref)


async def _thread(db, n, fam):
    await store.save_roster(db, fam, ME, [ME])
    for i in range(n):
        await store.add_turn(db, family_id=fam, thread_id=ME["id"], role="user", text=f"earlier note {i} about the day")
        await store.add_turn(db, family_id=fam, thread_id=ME["id"], role="assistant", text=f"noted that {i}")


async def test_an_ack_thinks_little(db, monkeypatch):
    await store.save_roster(db, FAM, ME, [ME])
    r = brain(monkeypatch, [LLMReply(text="I am right here with you.", tool_calls=[], model="fake")])
    await loop.run_turn(db, SimHost(), req("ok", "L1"))
    assert r.efforts[0] == "minimal"
    assert r.max_tokens[0] == loop.LIGHT_MAX_TOKENS


async def test_a_greeting_sends_only_the_recent_thread(db, monkeypatch):
    fam = "fam-light-hi"
    await _thread(db, 12, fam)
    r = brain(monkeypatch, [LLMReply(text="Hello.", tool_calls=[], model="fake")])
    await loop.run_turn(db, SimHost(), loop.TurnRequest(
        family_id=fam, elder=ME, speaker=ME, members=[ME], text="Hi", message_ref="L2"))
    assert r.efforts == ["minimal"]
    sent = " ".join(
        (m.get("content") or [{}])[0].get("text", "") if m.get("role") == "user" else m.get("text", "")
        for m in r.messages[0]
    )
    assert "earlier note 0 " not in sent
    assert len(r.messages[0]) <= loop.LIGHT_HISTORY + 1


async def test_an_order_keeps_the_thread_and_the_normal_depth(db, monkeypatch):
    fam = "fam-light-order"
    await _thread(db, 12, fam)
    r = brain(monkeypatch, [LLMReply(text="I can help with that.", tool_calls=[], model="fake")])
    await loop.run_turn(db, SimHost(), loop.TurnRequest(
        family_id=fam, elder=ME, speaker=ME, members=[ME], text="please order milk", message_ref="L3"))
    assert r.efforts == ["low"]
    assert len(r.messages[0]) > loop.LIGHT_HISTORY + 1


async def test_a_long_ack_is_shortened_without_a_second_call(db, monkeypatch):
    await store.save_roster(db, FAM, ME, [ME])
    draft = "One thing here. Two things here. Three things here. Four things here. Five things here."
    r = brain(monkeypatch, [LLMReply(text=draft, tool_calls=[], model="fake")])
    result = await loop.run_turn(db, SimHost(), req("Yes", "L4"))
    assert r.efforts[0] == "minimal"
    # The reply-path judge still runs (its own max tokens). A second brain call does not.
    assert r.max_tokens.count(loop.LIGHT_MAX_TOKENS) == 1
    assert "Three things" not in result.reply
    assert "One thing here." in result.reply


async def test_a_medicine_question_still_reaches_the_judge(db, monkeypatch):
    await store.save_roster(db, FAM, ME, [ME])
    checks = []

    async def fake_check(draft, **kw):
        checks.append(draft)
        return []

    monkeypatch.setattr(loop.grounding, "check", fake_check)
    r = brain(monkeypatch, [LLMReply(text="Please check that with the doctor first.", tool_calls=[], model="fake")])
    await loop.run_turn(db, SimHost(), req("can I take the dolo tablet?", "L5"))
    assert r.efforts[0] == "low"
    assert checks


async def test_a_tool_on_an_ack_thinks_at_the_normal_depth_afterwards(db, monkeypatch):
    await store.save_roster(db, FAM, ME, [ME])
    call = ToolCall(id="c1", name="remember", args={"domain": "dish", "name": "khichdi", "details": {"food": "khichdi"},
                                                     "sentence": "Likes khichdi"})
    r = brain(monkeypatch, [
        LLMReply(text="", tool_calls=[call], model="fake"),
        LLMReply(text="Saved, you like khichdi.", tool_calls=[], model="fake"),
    ])
    await loop.run_turn(db, SimHost(), req("ok", "L6"))
    assert r.efforts[:2] == ["minimal", "low"]


def test_clip_reply_keeps_the_sentence_cap_and_drops_a_note():
    assert loop.guards.clip_reply("One. Two. Three. Four.", cap=2) == "One. Two."
    assert loop.guards.clip_reply("I have noted this. She ate.", cap=3) == "She ate."
