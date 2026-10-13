"""The founder's seven points of 2026-10-13, the parts in the brain and care memory:
restricted items (never ordered, never discussed; harmless alternatives), mood read from the whole conversation, repeated
problems grouped for the doctor (three falls with each one's cause), Saheli's scope (no code; a general question gets one
sentence), and a caregiver's request reaching a closed chat (tests/test_whatsapp_window.py)."""

import json

import pytest

from app.brain import guards, loop, tools
from app.brain.persona import PERSONA, REPLY_FORMAT
from app.care import extract, mood, restricted, store, symptom_patterns
from app.llm import router, spend
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM = "fam-points"
ME = {"id": "self-p", "name": "Kittu Sharma", "role": "primary caregiver"}


class Fake:
    """A model that answers by which job is asking (the first line of its system prompt)."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.answers: dict[str, list[str]] = {}

    async def complete(self, route, **kw):
        system = kw.get("system_stable") or ""
        system = " ".join(system) if isinstance(system, list) else system
        text = kw["messages"][-1]["content"][0]["text"]
        key = next((k for k in self.answers if system.startswith(k)), None)
        self.calls.append((key or system[:40], text))
        out = self.answers[key].pop(0) if key and self.answers[key] else "{}"
        return LLMReply(text=out, tool_calls=[], model="fake")


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    router.register_provider("fake", f)
    monkeypatch.setenv("MODEL_ROUTES", '{"worker": ["fake:m"], "extract": ["fake:m"], "brain": ["fake:m"], "judge_fast": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    yield f
    router._providers.pop("fake", None)
    router.reset_breakers()


# ── 2. restricted items ──────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,cat", [
    ("Classic Ice Burst", "tobacco"), ("Classic Ice Burst cigarettes", "tobacco"), ("Gold Flake Kings", "tobacco"),
    ("Rajnigandha pan masala", "gutka"), ("Kingfisher Strong Beer 650ml", "alcohol"), ("Old Monk rum", "alcohol"),
    ("vape pod", "vape"), ("ganja", "illegal:cannabis"), ("heroin", "illegal:opioids"), ("LSD", "illegal:psychedelics"),
])
def test_restricted_items_are_recognised(name, cat):
    assert restricted.category(name) == cat


@pytest.mark.parametrize("name", ["Polo mints", "Center Fresh chewing gum", "Nicotex nicotine gum", "Ginger beer", "root beer",
                                  "Diet Coke", "Joint care tablets", "Hand sanitizer 70% alcohol", "Zerodol MR", "Iced tea",
                                  "Speed Post envelope", "Folic acid 5 mg", "Classic salted crackers"])
def test_harmless_things_are_not_restricted(name):
    assert restricted.category(name) is None


def test_the_refusal_offers_harmless_alternatives_and_the_brain_knows_the_rules():
    r = restricted.refusal("tobacco")
    assert r["ok"] is False and r["order"] == "none" and "sugar-free mints" in r["alternatives"]
    assert "say" in restricted.refusal("illegal:opioids") and "alternatives" not in restricted.refusal("illegal:opioids")
    assert "never order, search" in PERSONA.lower() or "you never order, search" in PERSONA.lower()
    assert "classic ice burst" in PERSONA and "sugar-free mints" in PERSONA and "never answer anything" in PERSONA
    assert "harmful or illegal substances" in REPLY_FORMAT


# ── 6. scope ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_brain_is_told_its_scope():
    assert "What is Python?\" is the snake" in PERSONA and "ONE short, simple sentence" in PERSONA
    assert "search engine or a general AI assistant" in PERSONA


@pytest.mark.parametrize("text,code", [("```python\nprint('hi')\n```", True), ("def sort(xs):\n    return sorted(xs)", True),
                                       ("Python is a large non-venomous snake.", False), ("Take it after food at 9 PM.", False)])
def test_code_in_saheli_reply_is_caught(text, code):
    assert guards.code_in_reply(text) is code


async def test_a_reply_that_still_holds_code_after_the_rewrite_becomes_the_scope_line(db, at, fake):
    at("2026-10-13 18:00")
    await store.save_roster(db, FAM, ME, [ME])
    await db.commit()
    code = "Sure!\n```python\nxs.sort()\n```"
    fake.answers["You are Saheli"] = [code, code, code]  # the rewrite is checked again (trust plan C4)
    out = await loop.run_turn(db, SimHost(), loop.TurnRequest(family_id=FAM, elder=ME, speaker=ME, members=[ME],
                                                              text="write python code to sort a list", message_ref="scope1"))
    assert "```" not in out.reply and "outside what I do" in out.reply


# ── 1. mood from the whole conversation ──────────────────────────────────────────────────────────────────────────

async def _stomach_day(db, at):
    at("2026-10-13 13:00")
    await store.save_roster(db, FAM, ME, [ME])
    for text in ["pet kharab hai yaar, shawarma kha liya tha", "baar baar potty ja raha hoon"]:
        await store.add_turn(db, family_id=FAM, thread_id=ME["id"], role="user", text=text, speaker_id=ME["id"])
    await store.record_event(db, family_id=FAM, subject_id=ME["id"], kind="symptom", summary="Frequent loose stools after shawarma")
    at("2026-10-13 13:30")
    await store.add_turn(db, family_id=FAM, thread_id=ME["id"], role="user", text="fever nahi hai", speaker_id=ME["id"])
    await db.commit()


async def test_a_mood_is_read_again_from_the_whole_day_and_the_reported_problems(db, at, fake):
    await _stomach_day(db, at)
    fake.answers["You decide how a person is feeling NOW"] = [json.dumps(
        {"score": 2, "word": "uncomfortable", "summary": "Still has loose motions after shawarma; no fever.", "confident": True})]
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ME, speaker=ME, members=[ME], message_ref="m1")
    out = await tools.log_event(c, {"kind": "mood", "summary": "He is not feverish or unwell.", "mood_score": 3, "mood_word": "relieved"})
    assert out["word"] == "uncomfortable"
    seen = fake.calls[-1][1]
    assert "baar baar potty" in seen and "Frequent loose stools" in seen and "fever nahi hai" in seen  # the whole picture
    assert (await mood.now_for(db, FAM, ME["id"]))["word"] == "uncomfortable"


async def test_when_the_reading_fails_the_brains_own_mood_is_kept(db, at, fake):
    await _stomach_day(db, at)
    fake.answers["You decide how a person is feeling NOW"] = ["not json"]
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ME, speaker=ME, members=[ME], message_ref="m2")
    out = await tools.log_event(c, {"kind": "mood", "summary": "Unwell with loose motions.", "mood_score": 2, "mood_word": "unwell"})
    assert out["word"] == "unwell"


def test_the_assessor_is_told_a_denied_symptom_is_not_relief():
    assert "\"No fever\" while the loose motions go on = still unwell" in mood.ASSESS
    assert "Better is not over" in mood.ASSESS


def test_the_reread_uses_the_whole_picture_too():
    import inspect

    assert "mood.assess(" in inspect.getsource(extract)


# ── 5. repeated problems for the doctor ──────────────────────────────────────────────────────────────────────────

async def _three_falls(db, at):
    for day, what in (("2026-08-20 10:00", "Fell while walking, did not see the dog"),
                      ("2026-09-12 16:00", "Fell, felt dizzy and dehydrated in the heat"),
                      ("2026-10-08 08:00", "Fell coming out of the shower, wet floor")):
        at(day)
        await store.record_event(db, family_id=FAM, subject_id=ME["id"], kind="symptom", summary=what, payload={"severity": "watch"})
    at("2026-10-13 18:00")
    await db.commit()


async def test_three_falls_weeks_apart_reach_the_doctor_as_one_pattern_with_each_cause(db, at, fake):
    await _three_falls(db, at)
    fake.answers["You prepare the \"repeated problems\""] = [json.dumps({"patterns": [{"theme": "Falls", "count": 3, "events": [
        {"date": "2026-08-20", "what": "Fell while walking, did not see the dog"},
        {"date": "2026-09-12", "what": "Fell, felt dizzy and dehydrated in the heat"},
        {"date": "2026-10-08", "what": "Fell coming out of the shower, wet floor"}],
        "ask_doctor": "Three falls in seven weeks: should balance, blood pressure on standing or hydration be checked?"}]})]
    found = await symptom_patterns.patterns(db, FAM, ME["id"])
    assert found[0]["theme"] == "Falls" and found[0]["count"] == 3
    sent = fake.calls[-1][1]
    assert "2026-08-20" in sent and "dog" in sent and "shower" in sent  # all three, beyond a 7-day window
    again = await symptom_patterns.patterns(db, FAM, ME["id"])  # cached: no second model call
    assert again == found and len(fake.calls) == 1
    lines = symptom_patterns.text(found)
    assert "Falls (3 times)" in lines and "did not see the dog" in lines and "shower" in lines and "Ask:" in lines


async def test_the_doctor_report_and_the_tool_carry_the_patterns(db, at, fake):
    from app.care import features

    await _three_falls(db, at)
    fake.answers["You prepare the \"repeated problems\""] = [json.dumps({"patterns": [{"theme": "Falls", "count": 2, "events": [
        {"date": "2026-09-12", "what": "dehydrated fall"}, {"date": "2026-10-08", "what": "shower fall"}], "ask_doctor": "x"}]})]
    c = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ME, speaker=ME, members=[ME], message_ref="r1")
    await store.save_roster(db, FAM, ME, [ME])
    out = await tools.weekly_report(c, {"days": 7})
    assert "Repeated problems (last 3 months)" in out["summary"] and "shower fall" in out["summary"]
    r = await features.report(db, FAM, ME["id"], 7)
    assert "patterns" not in r or r["patterns"] is not None


async def test_a_single_problem_is_not_a_pattern_and_costs_nothing(db, at, fake):
    at("2026-10-10 10:00")
    await store.record_event(db, family_id=FAM, subject_id=ME["id"], kind="symptom", summary="Headache")
    await db.commit()
    assert await symptom_patterns.patterns(db, FAM, ME["id"]) == [] and not fake.calls


def test_falls_are_logged_with_their_circumstances():
    assert "fell coming out of the shower" in PERSONA and "repeated problem with each time's cause" in PERSONA
