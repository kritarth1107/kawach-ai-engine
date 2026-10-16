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


# ── what she says she did or will do (13 Oct: "I will track your proteins and calories from now on") ──

async def test_a_second_bookkeeping_opener_is_cut(db, fake):
    fake(["I have noted this. She has not eaten since morning.",
          "According to health sources, rest is enough. She has not eaten since morning."],
         ['{"claims": []}', '{"claims": []}'])
    res = await turn(db, "mummy has not eaten since morning", "g10")
    assert "noted" not in res.reply.lower() and "health sources" not in res.reply.lower()
    assert "eaten" in res.reply


async def test_a_promise_no_tool_can_keep_is_rewritten(db, fake):
    promise = claims(("action", "I will track your proteins and calories from now on", False, False, "none"))
    f = fake(["Noted the 4 boiled eggs. I will track your proteins and calories from now on.",
              "Noted the 4 boiled eggs in today's log. I can't count calories or protein yet."],
             [promise, '{"claims": []}'])
    res = await turn(db, "I ate four boiled eggs, track my protein and calories from now on", "g6")
    assert "from now on" not in res.reply and "can't count" in res.reply
    assert len(f.judge_inputs) == 2


async def test_the_reply_check_knows_her_tools_and_this_turns_results(db, fake, monkeypatch):
    from app.llm.router import ToolCall

    f = fake([], [])
    seen: list[str] = []
    brain = [LLMReply(text="", tool_calls=[ToolCall("c1", "log_event", {"kind": "meal", "summary": "Two rotis and dal", "about": SON["id"]})], model="fake"),
             LLMReply(text="Saved your lunch: two rotis and dal.", tool_calls=[], model="fake")]

    async def answer(route, **kw):  # a tool call needs the whole reply, so this one answers directly
        system = kw.get("system_stable") or ""
        system = " ".join(system) if isinstance(system, list) else system
        if system.startswith(grounding.PROMPT[:40]):
            seen.append(system)
            f.judge_inputs.append(kw["messages"][-1]["content"][0]["text"])
            return LLMReply(text='{"claims": []}', tool_calls=[], model="fake")
        return brain.pop(0) if brain else LLMReply(text="Okay.", tool_calls=[], model="fake")

    monkeypatch.setattr(f, "complete", answer)
    res = await turn(db, "had two rotis and dal for lunch", "g7")
    assert res.reply == "Saved your lunch: two rotis and dal."
    assert seen and "SAHELI'S ABILITIES" in seen[0] and "- log_event:" in seen[0] and "- set_reminder:" in seen[0]
    assert "log_event: " in f.judge_inputs[0].split("REPLY")[0]  # this turn's tool results carry the tool's name


async def test_action_claims_count_only_in_the_reply_loop(fake):
    f = fake([], [claims(("action", "I'll check on you tomorrow", False, False, "none"))])
    assert await grounding.check("Rest well tonight, I'll check on you tomorrow.", memory="", their_messages=[], tool_results=[]) == []
    f.judge = [claims(("action", "I'll check on you tomorrow", False, False, "none"))]
    found = await grounding.check("Rest well tonight, I'll check on you tomorrow.", memory="", their_messages=[], tool_results=[],
                                  actions=True)
    assert [c.kind for c in found] == ["action"] and "no tool did" in grounding.problems(found)[0]


def test_health_information_without_a_doctor_note_is_a_problem_but_nothing_is_stripped():
    cs = grounding._parse(json.dumps({"claims": [], "doctor_note_missing": True}))
    assert [c.problem for c in cs] == [True]
    assert grounding._parse(json.dumps({"claims": [], "doctor_note_missing": True}), doctor_note=False) == [], "only on look-up turns"
    assert "when to see a doctor" in grounding.problems(cs)[0]
    text = "Zerodol MR can cause drowsiness. Did you hurt yourself?"
    assert grounding.strip(text, cs) == text, "the rewrite adds the line; strip never removes the information"
    assert grounding._parse(json.dumps({"claims": [], "doctor_note_missing": False})) == []


def test_a_tease_and_a_missed_followup_are_rewritten_not_stripped():
    raw = json.dumps({"claims": [], "unwanted_apology": True, "followup_missing": True})
    cs = grounding._parse(raw)
    assert {c.source for c in cs} == {"unwanted_apology", "followup_missing"}
    assert all(c.problem for c in cs)
    text = "Sorry, please forgive me. Enjoy the film."
    assert grounding.strip(text, cs) == text
    notes = " ".join(grounding.problems(cs))
    assert "do not apologise" in notes and "one question" in notes
    both = grounding._parse(json.dumps({"claims": [], "framing_missing": True, "question_unanswered": True}))
    assert {c.source for c in both} == {"framing_missing", "question_unanswered"}
    assert grounding.strip("Metformin helps. Okay, noted.", both) == "Metformin helps. Okay, noted."
    more = " ".join(grounding.problems(both))
    assert "general information" in more and "Do not only say noted" in more


def test_a_medicine_explanation_is_framed_by_code():
    from app.brain import guards

    explained = ("Metformin helps keep blood sugar steady throughout the day. It reduces the sugar the liver releases. "
                 "It also slows sugar from meals. It helps the body use insulin.")
    line = guards.with_framing({"saved": {"language": "en"}}, explained)
    assert guards.has_framing(line)
    assert len(guards.sentences(line)) <= 3
    assert "general information" in line and "doctor" in line
    hindi = guards.with_framing({"saved": {"language": "hi"}, "now": "devanagari"}, "मेटफॉर्मिन दिन भर शुगर संभालती है।")
    assert guards.script_of(hindi) == "devanagari"
    assert "डॉक्टर" in hindi
    already = "Metformin helps. This is general information; confirm it with your doctor or pharmacist."
    assert guards.with_framing(None, already) == already
    schedule = "Aaj Maa ki Telma 40 mg subah 8 baje hai."
    kept = guards.settle_reply(
        schedule, {"saved": {"language": "hi"}, "script": "latin", "roman": "indic"},
        situation="med_question", needs_framing=False, force_light=False, held_topic_text="", actions=[],
        prior_assistant=True, naming=False, user_text="What medicines are due today?")
    assert "Telma" in kept and "general information" not in kept and "सामान्य जानकारी" not in kept


def test_an_apology_with_no_earlier_message_of_hers_is_rewritten():
    from app.brain import guards

    sorry = "अरे नहीं अर्जुन जी, मेरा वो मतलब बिल्कुल नहीं था। मुझसे बड़ी गलती हो गई, मुझे माफ़ कीजिएगा।"
    hi = {"saved": {"language": "hi"}, "now": "devanagari"}
    out = guards.settle_reply(sorry, hi, situation="chit_chat", needs_framing=False, force_light=False, held_topic_text="",
                              actions=[], prior_assistant=False, naming=False)
    assert not guards.apology_in(out)
    assert guards.script_of(out) == "devanagari"
    kept = guards.settle_reply("Sorry, I had your name wrong.", {"saved": {"language": "en"}}, situation="chit_chat",
                               needs_framing=False, force_light=False, held_topic_text="", actions=[], prior_assistant=True, naming=False)
    assert kept.startswith("Sorry")
    cannot = "माफ़ कीजिए, मैं ये नहीं कर सकती, बिजली का बिल भरना अभी मेरे पास नहीं है।"
    assert guards.settle_reply(cannot, {"saved": {"language": "hi"}, "now": "devanagari"}, situation="order",
                               needs_framing=False, force_light=False, held_topic_text="", actions=[], prior_assistant=False,
                               naming=False) == cannot


def test_a_short_answer_stays_on_her_question():
    from app.brain import guards

    assert guards.held_topic("Trouble sleeping?", "Yes") == "Trouble sleeping"
    assert guards.held_topic("How is your stomach after the ice cream?", "Hi") == ""
    assert guards.held_topic("Trouble sleeping?", "What time is dinner?") == ""
    assert guards.held_topic("Trouble sleeping?", "I went out for a long walk today") == ""
    assert guards.blocks_held_thread("setup_progress", {}, "Trouble sleeping")
    assert not guards.blocks_held_thread("log_food", {}, "Trouble sleeping")
    setup = "To get started, what does the family call Meera ji, and how should I address her?"
    drifted = "Yes. What is on your mind today?"
    for reply in (setup, drifted):
        out = guards.settle_reply(reply, {"saved": {"language": "en"}}, situation="chit_chat", needs_framing=False, force_light=False,
                                  held_topic_text="Trouble sleeping", actions=[], prior_assistant=True, naming=False)
        assert out == "Trouble sleeping. What's keeping you up?"
    on_topic = "Trouble sleeping, what woke you?"
    assert guards.settle_reply(on_topic, {"saved": {"language": "en"}}, situation="chit_chat", needs_framing=False,
                               force_light=False, held_topic_text="Trouble sleeping", actions=[], prior_assistant=True,
                               naming=False) == on_topic


def test_todays_slips_are_open_replay_cases_and_protein_stays_open():
    from eval.replay_cases import LIVE, SEEDS, load_cases

    live = [json.loads(line) for line in LIVE.read_text().splitlines() if line.strip()]
    curious = [c for c in live if c["kind"].startswith("curious_")]
    assert len(curious) == 10
    still_open = {c["id"] for c in live if c["status"] == "open"}
    assert "live-tease-hi" not in still_open and "live-advice-medicine-en" not in still_open
    assert all(c["status"] == "fixed" for c in live if c["id"] in {
        "live-advice-sleep-en", "live-friends-en", "live-trip-en", "live-tease-en", "live-tease-hinglish",
        "live-tease-hi", "live-advice-medicine-en", "live-advice-medicine-hi",
        "live-question-quiet-en", "live-question-quiet-hi"})
    from eval.replay_cases import REASON
    reason = [json.loads(line) for line in REASON.read_text().splitlines() if line.strip()]
    assert next(c["status"] for c in reason if c["id"] == "reason-consent-pending") == "fixed"
    assert next(c["status"] for c in reason if c["id"] == "reason-yes-sleep") == "fixed"
    seeds = [json.loads(line) for line in SEEDS.read_text().splitlines() if line.strip()]
    protein = next(c for c in seeds if c["id"] == "seed-promise-protein-tracking")
    assert protein["status"] == "open"
    loaded = load_cases([])
    assert any(c["id"] == "live-advice-sleep-en" for c in loaded)


async def test_a_side_effect_reply_gets_its_doctor_line(db, fake, monkeypatch):
    from app.llm.router import ToolCall

    f = fake([], [])
    seen: list[str] = []
    judge = [json.dumps({"claims": [], "doctor_note_missing": True}), '{"claims": [], "doctor_note_missing": false}']
    brain = [LLMReply(text="", tool_calls=[ToolCall("w1", "web_search", {"query": "metformin side effects"})], model="fake"),
             LLMReply(text="Metformin can upset the stomach at first.", tool_calls=[], model="fake"),
             LLMReply(text="Metformin can upset the stomach at first; check with your doctor if it lasts.", tool_calls=[], model="fake")]

    async def answer(route, **kw):
        system = kw.get("system_stable") or ""
        system = " ".join(system) if isinstance(system, list) else system
        if system.startswith(grounding.PROMPT[:40]):
            seen.append(system)
            return LLMReply(text=judge.pop(0) if judge else '{"claims": []}', tool_calls=[], model="fake")
        return brain.pop(0) if brain else LLMReply(text="Okay.", tool_calls=[], model="fake")

    async def search(ctx, a):
        return {"results": [{"snippet": "Metformin: stomach upset, nausea."}]}

    monkeypatch.setattr(f, "complete", answer)
    from app.brain import tools as brain_tools

    spec, _ = brain_tools._TOOLS["web_search"]
    monkeypatch.setitem(brain_tools._TOOLS, "web_search", (spec, search))
    res = await turn(db, "what are metformin side effects?", "g8")
    assert "doctor" in res.reply
    assert "doctor_note_missing" in seen[0], "asked only because this turn looked something up"


async def test_no_doctor_line_asked_when_nothing_was_looked_up(db, fake):
    f = fake(["Your Metformin is at 8 in the morning."], ['{"claims": []}'])
    await turn(db, "when is my metformin?", "g9")
    assert f.judge_inputs  # the check ran without the doctor-note rule (see _system)


def test_an_answer_a_tool_used_is_not_pulled_back_to_her_question():
    """Live 16 Oct 2026: "Plain maggie" answered her food question, log_food used it, and the reply was replaced with
    the question again plus "What's the one thing behind it?". A used answer keeps the reply; a drifted question is
    asked again as she asked it, with no canned tail."""
    from app.brain import guards

    en = {"saved": {"language": "en"}}
    saved = "Saved: 1 plate Maggi, 310 kcal, 9 g fat."
    used = [{"tool": "log_food", "ok": True, "args": {"items": [{"name": "Maggi"}]}}]
    assert guards.settle_reply(saved, en, situation="food", needs_framing=False, force_light=False,
                               held_topic_text="What's in the cooked instant noodles, roughly", actions=used,
                               prior_assistant=True, naming=False) == saved
    drifted = guards.settle_reply("Yes. What is on your mind today?", en, situation="chit_chat", needs_framing=False,
                                  force_light=False, held_topic_text="Did you go for the walk", actions=[],
                                  prior_assistant=True, naming=False)
    assert drifted == "Did you go for the walk?" and "one thing behind" not in drifted
