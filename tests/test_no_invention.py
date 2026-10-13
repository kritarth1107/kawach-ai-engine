"""No invention (trust plan C4): every name, food, event or time in Saheli's reply comes from memory, this chat or a tool,
or she rewrites it; a rewrite is checked again and, if it still invents, those sentences are removed. Her own medical
advice is caught the same way. Round 2 cases are the fixtures."""

import json

import pytest

from app.brain import grounding, loop
from app.care import store
from app.llm import router, spend
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM = "fam-ground"
MAA = {"id": "elder-g", "name": "Kamla Devi", "role": "elder"}
SON = {"id": "son-g", "name": "Rohit Verma", "role": "primary caregiver"}


def claims(*items):
    return json.dumps({"claims": [dict(zip(("kind", "quote", "supported", "prescriptive", "source"), i)) for i in items]})


# ── the decision in code ─────────────────────────────────────────────────────

def test_unsupported_facts_and_own_advice_are_problems_sourced_ones_are_not():
    cs = grounding._parse(claims(
        ("person", "Mala temple gayi hain", False, False, "none"),
        ("food", "one rasgulla is fine", False, True, "none"),
        ("advice", "keep potassium under 3.5", False, True, "none"),
        ("advice", "the search says ankle swelling is common; check with your doctor", True, False, "tool"),
        ("advice", "call 112 now", True, True, "safety"),
        ("medicine", "Metformin 500 mg at 8", True, False, "none"),
    ))
    bad = [c.quote for c in cs if c.problem]
    assert bad == ["Mala temple gayi hain", "one rasgulla is fine", "keep potassium under 3.5"]


def test_food_advice_counts_as_unsupported_food_claim():
    c = grounding._parse(claims(("food", "one rasgulla is fine", False, True, "none")))[0]
    assert c.problem


def test_strip_removes_only_the_sentences_with_the_invention():
    text = "Good morning Kamla ji! Mala went to the temple today. Your Metformin is at 8."
    cs = grounding._parse(claims(("person", "Mala went to the temple today", False, False, "none")))
    assert grounding.strip(text, cs) == "Good morning Kamla ji! Your Metformin is at 8."


def test_a_bad_judge_answer_never_blocks_a_reply():
    assert grounding._parse("not json") == [] and grounding._parse('{"claims": [{"kind": "alien", "quote": "x"}]}') == []


# ── in the reply loop ────────────────────────────────────────────────────────

class Fake:
    """The brain and the grounding judge, answered by which job asks."""

    def __init__(self, brain_replies, judge_replies):
        self.brain = list(brain_replies)
        self.judge = list(judge_replies)
        self.judge_inputs: list[str] = []

    async def complete(self, route, **kw):
        system = kw.get("system_stable") or ""
        system = " ".join(system) if isinstance(system, list) else system
        if system.startswith(grounding.PROMPT[:40]):
            self.judge_inputs.append(kw["messages"][-1]["content"][0]["text"])
            return LLMReply(text=self.judge.pop(0) if self.judge else '{"claims": []}', tool_calls=[], model="fake")
        return LLMReply(text=self.brain.pop(0) if self.brain else "Okay.", tool_calls=[], model="fake")


@pytest.fixture
def fake(monkeypatch):
    def make(brain, judge):
        f = Fake(brain, judge)
        router.register_provider("fake", f)
        monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "judge_fast": ["fake:m"], "extract": ["fake:m"], "worker": ["fake:m"], "classify": ["fake:m"]}')
        monkeypatch.setenv("GROUNDING_CHECK", "on")
        router.reset_breakers()
        spend.reset()
        return f
    yield make
    router._providers.pop("fake", None)
    router.reset_breakers()


async def turn(db, text, ref, speaker=SON):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    return await loop.run_turn(db, SimHost(), loop.TurnRequest(family_id=FAM, elder=MAA, speaker=speaker, members=[MAA, SON],
                                                               text=text, message_ref=ref))


async def test_an_invented_life_detail_is_rewritten(db, fake):
    fake(["Kamla ji is fine, Mala took her to the temple this morning.", "Kamla ji is fine today. Shall I ask her how she slept?"],
             [claims(("person", "Mala took her to the temple this morning", False, False, "none")), '{"claims": []}'])
    res = await turn(db, "how is mummy today?", "g1")
    assert res.reply == "Kamla ji is fine today. Shall I ask her how she slept?"
    assert "Mala" not in res.reply


async def test_her_own_earlier_reply_is_not_a_source(db, fake):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="assistant", text="Gopal's number is 98765 43210.")
    f = fake(["Kamla ji is resting after lunch, all calm today."], ['{"claims": []}'])
    await turn(db, "ok, anything else?", "g2")
    sources = f.judge_inputs[0]
    assert "98765 43210" not in sources.split("REPLY")[0]  # not offered as a source


async def test_a_rewrite_that_still_invents_loses_those_sentences(db, fake):
    inv = claims(("person", "Mala visited", False, False, "none"))
    fake(["Mala visited. She ate well.", "Mala visited. She ate well.", "Mala visited. She ate well."], [inv, inv, inv])
    res = await turn(db, "how was her day?", "g3")
    assert res.reply == "She ate well."


async def test_own_medical_advice_is_caught(db, fake):
    adv = claims(("advice", "one rasgulla is fine for her sugar", False, True, "none"))
    fake(["One rasgulla is fine for her sugar.", "I'd check with her doctor before sweets; shall I add it to the doctor questions?"],
         [adv, '{"claims": []}'])
    res = await turn(db, "can mummy have a rasgulla?", "g4")
    assert "rasgulla is fine" not in res.reply and "doctor" in res.reply


async def test_memory_and_their_messages_are_the_sources(db, fake):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="medicine", key="medicine:metformin",
                           value={"name": "Metformin", "dose": "500 mg", "times": ["08:00"]}, text="Metformin 500 mg at 08:00", source_kind="caregiver_said")
    f = fake(["Her Metformin 500 mg is at 8 in the morning."], ['{"claims": []}'])
    await turn(db, "when is her metformin?", "g5")
    src = f.judge_inputs[0].split("REPLY")[0]
    assert "Metformin 500 mg at 08:00" in src and "when is her metformin?" in src
