"""Hard turns think harder (trust plan C6): health, a correction or an upset person, judged from data and the brain's own
tool choices, never from the person's words. Off until the A/B shows it helps (founder decision D3)."""

import pytest

from app.brain import loop
from app.care import store
from app.llm import router, spend
from app.llm.router import LLMReply, ToolCall
from app.sim.world import SimHost

FAM = "fam-hard"
ME = {"id": "self-hard", "name": "Kittu Sharma", "role": "primary caregiver"}


class Recorder:
    def __init__(self, script):
        self.script = list(script)
        self.efforts: list[str] = []

    async def complete(self, route, **kw):
        self.efforts.append(kw.get("effort"))
        out = self.script.pop(0) if self.script else LLMReply(text="Okay.", tool_calls=[], model="fake")
        return out


@pytest.fixture
def brain(monkeypatch):
    def make(script):
        r = Recorder(script)
        router.register_provider("fake", r)
        monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"], "classify": ["fake:m"]}')
        monkeypatch.setattr(loop, "BRAIN_EFFORT", "low")
        monkeypatch.setattr(loop, "BRAIN_EFFORT_HARD", "medium")
        router.reset_breakers()
        spend.reset()
        return r
    yield make
    router._providers.pop("fake", None)
    router.reset_breakers()


def req(text, ref):
    return loop.TurnRequest(family_id=FAM, elder=ME, speaker=ME, members=[ME], text=text, message_ref=ref)


async def test_an_ordinary_turn_stays_at_low_effort(db, brain):
    await store.save_roster(db, FAM, ME, [ME])
    r = brain([LLMReply(text="I can help with that.", tool_calls=[], model="fake")])
    await loop.run_turn(db, SimHost(), req("please order milk", "h1"))
    assert r.efforts[0] == "low"


async def test_a_symptom_logged_today_makes_the_next_turn_hard(db, brain):
    await store.save_roster(db, FAM, ME, [ME])
    await store.record_event(db, family_id=FAM, subject_id=ME["id"], kind="symptom", summary="Loose motions since morning")
    r = brain([LLMReply(text="How are you feeling now?", tool_calls=[], model="fake")])
    await loop.run_turn(db, SimHost(), req("hmm", "h2"))
    assert r.efforts[0] == "medium"


async def test_a_health_change_mid_turn_makes_the_reply_think_harder(db, brain):
    await store.save_roster(db, FAM, ME, [ME])
    call = ToolCall(id="c1", name="remember", args={"domain": "allergy", "name": "peanut", "details": {"allergen": "peanut"},
                                                     "sentence": "Allergic to peanuts"})
    r = brain([LLMReply(text="", tool_calls=[call], model="fake"), LLMReply(text="Noted, no peanuts.", tool_calls=[], model="fake")])
    await loop.run_turn(db, SimHost(), req("I am allergic to peanuts", "h3"))
    assert r.efforts[:2] == ["minimal", "medium"]


async def test_off_until_switched_on(db, brain, monkeypatch):
    await store.save_roster(db, FAM, ME, [ME])
    await store.record_event(db, family_id=FAM, subject_id=ME["id"], kind="symptom", summary="Headache")
    r = brain([LLMReply(text="Okay", tool_calls=[], model="fake")])
    monkeypatch.setattr(loop, "BRAIN_EFFORT_HARD", "low")
    await loop.run_turn(db, SimHost(), req("hmm", "h4"))
    assert r.efforts[0] == "low"


def test_what_counts_as_a_hard_call():
    assert loop.hard_call(ToolCall("1", "remember", {"domain": "medicine"}))
    assert loop.hard_call(ToolCall("1", "remember", {"domain": "preference", "correction": True}))
    assert loop.hard_call(ToolCall("1", "log_event", {"kind": "symptom"}))
    assert loop.hard_call(ToolCall("1", "alert_caregiver", {}))
    assert not loop.hard_call(ToolCall("1", "remember", {"domain": "dish"}))
    assert not loop.hard_call(ToolCall("1", "log_event", {"kind": "meal"}))
