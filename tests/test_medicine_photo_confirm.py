"""Medicine photos and confirm-before-save (live 17 Oct 2026: a photo of four medicine strips was read, three names were
cut as "not in memory", then "which are they?" got "the photo isn't attached, please send it again"; a quoted "Here" got
"Okay, noted"; another caregiver was asked for a medicine count the dashboard already showed; saves happened without a yes).

Replays of those turns with a scripted model (the model's words are fixed; what is tested is what the engine does with
them: what the model is shown, what is kept, what is held, what is saved).

These scripted-model tests prove the plumbing only: that the code never reads people's words to decide a yes, that a save
runs only through answer_pending on a later turn with the exact read-back arguments, and what happens when the model does
or does not call it. Whether the real model reads "acha" as a yes and "Is it correct?" as not one is measured by the
real-model replay (eval/), not here."""

import json

import pytest
from sqlalchemy import select

from app.brain import confirm, guards, loop, photos

from app.brain.tools.memory import _course
from app.care import features, store
from app.care.models import Turn
from app.llm import router, spend
from app.llm.router import LLMReply, ToolCall
from app.sim.world import SimHost

FAM = "fam-medphoto"
SON = {"id": "son-mp", "name": "Ravi", "role": "primary caregiver"}
MAA = {"id": "maa-mp", "name": "Kamla", "role": "elder"}
IMG = [{"mime": "image/jpeg", "data": "aGVsbG8="}]

FOUR = {"kind": "medicines", "summary": "Four medicine strips on a table", "unclear": "", "items": [
    {"name": "Sompraz 40", "salt": "Esomeprazole", "strength": "40 mg", "form": "tablet", "usual_use": "acidity"},
    {"name": "Dart", "salt": "Paracetamol, Caffeine", "form": "tablet", "usual_use": "headache and pain"},
    {"name": "Montair LC", "salt": "Montelukast, Levocetirizine", "form": "tablet", "usual_use": "allergy"},
    {"name": "Dolo 650", "salt": "Paracetamol", "strength": "650 mg", "form": "tablet", "usual_use": "fever and pain"}]}


class Fake:
    """The vision read answers with `reading`; Saheli's turns pop `brain` (an LLMReply each). Every prompt is kept."""

    def __init__(self):
        self.reading: dict | None = FOUR
        self.brain: list[LLMReply] = []
        self.prompts: list[str] = []
        self.reads = 0

    async def complete(self, route, **kw):
        system = kw.get("system_stable") or ""
        system = " ".join(system) if isinstance(system, list) else system
        if system.startswith(photos.PROMPT[:30]):
            self.reads += 1
            return LLMReply(text=json.dumps(self.reading) if self.reading else "not json", tool_calls=[], model="fake")
        if "Saheli" in system[:400]:
            self.prompts.append(json.dumps(kw, default=str, ensure_ascii=False))
            return self.brain.pop(0) if self.brain else LLMReply(text="Okay.", tool_calls=[], model="fake")
        return LLMReply(text="{}", tool_calls=[], model="fake")


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    router.register_provider("fake", f)
    monkeypatch.setenv("MODEL_ROUTES", json.dumps({r: ["fake:m"] for r in ("brain", "vision", "worker", "extract", "judge_fast", "judge")}))
    monkeypatch.delenv("SAVE_CONFIRM", raising=False)
    router.reset_breakers()
    spend.reset()
    yield f
    router._providers.pop("fake", None)
    router.reset_breakers()


def pid(tool: str, args: dict) -> str:
    return confirm.pending_id(confirm.signature(tool, args))


def answer(ans: str, *saves: tuple[str, dict]) -> LLMReply:
    """The model's structured answer to held saves (the code checks only the id and the turn, never their words)."""
    return LLMReply(text="", tool_calls=[ToolCall(id=f"a{i}", name="answer_pending", args={"id": pid(t, a), "answer": ans})
                                         for i, (t, a) in enumerate(saves)], model="fake")


def yes(*saves: tuple[str, dict]) -> LLMReply:
    return answer("yes", *saves)


def say(text: str) -> LLMReply:
    return LLMReply(text=text, tool_calls=[], model="fake")


def call(name: str, args: dict, n: int = 1) -> LLMReply:
    return LLMReply(text="", tool_calls=[ToolCall(id=f"c{n}", name=name, args=args)], model="fake")


async def turn(db, text, ref, *, images=None, reply_to=None, speaker=SON):
    out = await loop.run_turn(db, SimHost(), loop.TurnRequest(family_id=FAM, elder=MAA, speaker=speaker, members=[MAA, SON],
                                                              text=text, message_ref=ref, images=images or [], reply_to=reply_to))
    await db.commit()
    return out


async def setup(db, at, when="2026-10-17 12:50"):
    at(when)
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await db.commit()


async def user_turn(db, ref) -> Turn:
    return (await db.execute(select(Turn).where(Turn.family_id == FAM, Turn.message_ref == ref, Turn.role == "user"))).scalar_one()


async def med(db, who, name):
    return await store.active_fact(db, FAM, who, f"medicine:{name}")


LIST_REPLY = ("I can see four medicines:\n1. Sompraz 40 (esomeprazole)\n2. Dart\n"
              "3. Montair LC\n4. Dolo 650\nAre these for you or for Kamla?")


# ── the photo stays in the conversation ──────────────────────────────────────────────────────────────────────────

async def test_a_photo_with_no_words_is_read_listed_whole_kept_and_asks_whom(db, at, fake):
    await setup(db, at)
    fake.brain = [say(LIST_REPLY)]
    out = await turn(db, "", "wamid.p1", images=IMG)
    assert fake.reads == 1, "one look beside the turn"
    for name in ("Sompraz 40", "Dart", "Montair LC", "Dolo 650"):
        assert name in out.reply, f"{name} was cut from the list"
    assert "for you or for Kamla" in out.reply
    kept = (await user_turn(db, "wamid.p1")).meta["photo"]
    assert kept["ref"] == "wamid.p1" and len(kept["read"]["items"]) == 4
    assert not await store.facts(db, FAM, SON["id"]) or not [f for f in await store.facts(db, FAM, SON["id"]) if f.domain == "medicine"]


async def test_which_are_they_later_is_answered_from_the_kept_reading_not_asked_again(db, at, fake):
    await setup(db, at)
    fake.brain = [say(LIST_REPLY), say("They are Sompraz 40, Dart, Montair LC and Dolo 650.")]
    await turn(db, "", "wamid.p1", images=IMG)
    at("2026-10-17 12:54")
    await turn(db, "Which are they", "wamid.t2")
    prompt = fake.prompts[-1]
    assert "PHOTOS IN THIS CHAT" in prompt and "Never ask them to send a photo listed here again" in prompt
    assert "Montair LC" in prompt and "they sent a photo here; it showed" in prompt
    assert fake.reads == 1, "the photo is not read twice"


async def test_a_reply_quoting_the_photo_brings_back_what_it_showed(db, at, fake):
    await setup(db, at)
    fake.brain = [say(LIST_REPLY), say("Those four: Sompraz 40, Dart, Montair LC and Dolo 650.")]
    await turn(db, "", "wamid.p1", images=IMG)
    at("2026-10-17 12:56")
    await turn(db, "Here", "wamid.t3", reply_to="wamid.p1")
    assert "they are replying to the photo they sent" in fake.prompts[-1] and "Dolo 650" in fake.prompts[-1]


async def test_a_blurry_photo_says_what_was_read_and_marks_the_rest(db, at, fake):
    await setup(db, at)
    fake.reading = {"kind": "medicines", "summary": "Two strips, one blurred", "unclear": "the second strip's name is blurred",
                    "items": [{"name": "Telma 40", "strength": "40 mg", "usual_use": "blood pressure"},
                              {"name": "Glyc", "legible": False}]}
    fake.brain = [say("I can read Telma 40. The second strip is blurred; could you send a clearer photo of it or type its name?")]
    await turn(db, "these are papa's", "wamid.p2", images=IMG)
    kept = (await user_turn(db, "wamid.p2")).meta["photo"]
    text = photos.reading_text(kept["read"])
    assert "Glyc [only partly readable]" in text and "could not read: the second strip's name is blurred" in text
    assert kept["caption"] == "these are papa's"


async def test_a_failed_read_still_answers_and_the_thread_says_it_could_not_be_read(db, at, fake):
    await setup(db, at)
    fake.reading = None
    fake.brain = [say("I couldn't make out the names. Could you type them?")]
    out = await turn(db, "", "wamid.p3", images=IMG)
    assert out.reply
    assert photos.history_note((await user_turn(db, "wamid.p3")).meta) == "(they sent a photo here; it could not be read)"


# ── nothing is saved until they say yes ──────────────────────────────────────────────────────────────────────────

SOMPRAZ = {"domain": "medicine", "name": "Sompraz 40", "about": SON["id"], "sentence": "Takes Sompraz 40 before breakfast.",
           "details": {"name": "Sompraz 40", "salt": "Esomeprazole", "strength": "40 mg", "dose": "1 tablet", "form": "tablet",
                       "purpose": "acidity", "times": ["08:00"], "food_timing": "before_food", "duration_days": 14, "stock": 10}}


async def test_a_medicine_is_read_back_first_and_saved_only_after_yes_with_everything_known(db, at, fake):
    await setup(db, at)
    fake.brain = [say(LIST_REPLY), call("remember", SOMPRAZ),
                  say("I'll add Sompraz 40 for you at 8:00 before breakfast for 14 days. Shall I save it?")]
    await turn(db, "", "wamid.p1", images=IMG)
    at("2026-10-17 12:57")
    held = await turn(db, "Sompraz is mine, 8 am before breakfast for 2 weeks, 10 left", "wamid.t4")
    assert await med(db, SON["id"], "Sompraz 40") is None, "saved before a yes"
    assert held.actions[0].get("held") is True
    mine = (await db.execute(select(Turn).where(Turn.id == held.turn_id))).scalar_one()
    assert [p["tool"] for p in mine.meta["pending_saves"]] == ["remember"]

    at("2026-10-17 12:58")
    fake.brain = [yes(("remember", SOMPRAZ)), say("Saved Sompraz 40 for you at 8:00.")]
    await turn(db, "haan", "wamid.t5")
    f = await med(db, SON["id"], "Sompraz 40")
    assert f is not None and f.status == "active"
    v = f.value
    assert (v["starts_on"], v["ends_on"], v["duration_days"]) == ("2026-10-17", "2026-10-30", 14)
    assert (v["salt"], v["purpose"], v["photo_ref"]) == ("Esomeprazole", "acidity", "wamid.p1")
    rows = {r["key"]: r for r in await features.stock(db, FAM, SON["id"])}
    assert rows[f.key]["stock"] == 10, rows
    assert "WAITING FOR THEIR ANSWER" in fake.prompts[-2] and pid("remember", SOMPRAZ) in fake.prompts[-2]
    assert "name='answer_pending'" in fake.prompts[-2], "answer_pending is offered only on a turn with something waiting"
    assert "name='answer_pending'" not in fake.prompts[0]


async def test_a_save_cannot_confirm_itself_in_the_turn_it_was_first_made(db, at, fake):
    await setup(db, at)
    fake.brain = [call("remember", SOMPRAZ), yes(("remember", SOMPRAZ)), say("Shall I save Sompraz 40 for you at 8:00?")]
    out = await turn(db, "yes add sompraz 40 at 8am daily for me", "wamid.t6")
    assert await med(db, SON["id"], "Sompraz 40") is None and out.actions[0].get("held")
    assert not any(a.get("tool") == "answer_pending" and a.get("ok") for a in out.actions)


async def test_a_no_saves_nothing_even_when_the_model_says_yes(db, at, fake):
    await setup(db, at)
    fake.brain = [call("remember", SOMPRAZ), say("Shall I save Sompraz 40 for you at 8:00?")]
    await turn(db, "add sompraz for me at 8", "wamid.t7")
    at("2026-10-17 12:52")
    fake.brain = [answer("no", ("remember", SOMPRAZ)), call("remember", SOMPRAZ), say("Okay, I won't.")]
    out = await turn(db, "nahi rehne do", "wamid.t8")
    assert await med(db, SON["id"], "Sompraz 40") is None
    mine = (await db.execute(select(Turn).where(Turn.id == out.turn_id))).scalar_one()
    assert [d["id"] for d in mine.meta["declined_saves"]] == [pid("remember", SOMPRAZ)], "kept as declined for the re-read"


async def test_a_change_after_the_read_back_is_read_back_again(db, at, fake):
    await setup(db, at)
    fake.brain = [call("remember", SOMPRAZ), say("Shall I save Sompraz 40 for you at 8:00?")]
    await turn(db, "add sompraz for me at 8", "wamid.t9")
    at("2026-10-17 12:52")
    other = {**SOMPRAZ, "details": {**SOMPRAZ["details"], "times": ["09:00"]}}
    fake.brain = [answer("change", ("remember", SOMPRAZ)), call("remember", other), say("Sompraz 40 at 9:00 instead. Save it?")]
    out = await turn(db, "ok but make it 9", "wamid.t10")
    assert await med(db, SON["id"], "Sompraz 40") is None and [a.get("held") for a in out.actions if a.get("tool") == "remember"] == [True]


async def test_for_a_family_member_the_read_back_names_them_and_saves_only_for_them(db, at, fake):
    await setup(db, at)
    hers = {**SOMPRAZ, "about": MAA["id"], "sentence": "Takes Sompraz 40 before breakfast."}
    fake.brain = [call("remember", hers), say("Sompraz 40 for Kamla at 8:00. Save it?")]
    held = await turn(db, "it's maa's, 8 am", "wamid.t11")
    assert held.actions and json.loads(json.dumps(held.actions[0]))["held"]
    mine = (await db.execute(select(Turn).where(Turn.id == held.turn_id))).scalar_one()
    assert mine.meta["pending_saves"][0]["for"] == "Kamla"
    at("2026-10-17 12:52")
    fake.brain = [yes(("remember", hers)), say("Saved for Kamla.")]
    await turn(db, "yes", "wamid.t12")
    assert await med(db, MAA["id"], "Sompraz 40") is not None and await med(db, SON["id"], "Sompraz 40") is None


async def test_on_the_record_without_reminders_needs_no_times(db, at, fake):
    await setup(db, at)
    plain = {**SOMPRAZ, "details": {k: v for k, v in SOMPRAZ["details"].items() if k not in ("times", "duration_days")}}
    fake.brain = [call("remember", plain), say("I'll keep Sompraz 40 on your record with no reminders. Okay?")]
    await turn(db, "just note it, no reminders", "wamid.t13")
    at("2026-10-17 12:52")
    fake.brain = [yes(("remember", plain)), say("Noted, no reminders.")]
    await turn(db, "ok", "wamid.t14")
    f = await med(db, SON["id"], "Sompraz 40")
    assert f is not None and not f.value.get("times") and "ends_on" not in f.value


# ── delete: named, and vague ─────────────────────────────────────────────────────────────────────────────────────

async def _saved(db, at, fake, args=SOMPRAZ):
    fake.brain = [call("remember", args), say("Save it?")]
    await turn(db, "add it", "wamid.s1")
    fake.brain = [yes(("remember", args)), say("Saved.")]
    await turn(db, "yes", "wamid.s2")
    assert await med(db, args["about"], args["name"]) is not None


async def test_a_named_delete_gets_one_short_confirm_then_deletes(db, at, fake):
    await setup(db, at)
    await _saved(db, at, fake)
    stop = {"domain": "medicine", "name": "Sompraz 40", "about": SON["id"], "reason": "they asked to delete it"}
    fake.brain = [call("stop", stop), say("Stop Sompraz 40 and its reminders?")]
    await turn(db, "delete sompraz", "wamid.d1")
    assert await med(db, SON["id"], "Sompraz 40") is not None
    fake.brain = [yes(("stop", stop)), say("Stopped Sompraz 40.")]
    await turn(db, "yes", "wamid.d2")
    assert await med(db, SON["id"], "Sompraz 40") is None


async def test_a_vague_delete_sees_a_numbered_list_with_stock_and_all(db, at, fake):
    await setup(db, at)
    await _saved(db, at, fake)
    fake.brain = [say("Which one?\n1. Sompraz 40\nOr all?")]
    await turn(db, "delete my medicines", "wamid.d3")
    p = fake.prompts[-1]
    assert "MEDICINES ON RECORD" in p and "] Sompraz 40 40 mg 1 tablet" in p and "1. [medicine:" in p
    assert "plus 'all'" in p and "10 left" in p, "the count the dashboard shows is in front of her"
    assert "never ask for a count" in p


DART = {"domain": "medicine", "name": "Dart", "about": SON["id"], "sentence": "Takes Dart when needed.",
        "details": {"name": "Dart", "purpose": "headache"}}


async def _two_saved(db, at, fake):
    fake.brain = [call("remember", SOMPRAZ), call("remember", DART, 2), say("Sompraz 40 at 8:00 and Dart, no reminders. Save both?")]
    await turn(db, "add both for me", "wamid.s1")
    fake.brain = [yes(("remember", SOMPRAZ), ("remember", DART)), say("Saved both for you.")]
    await turn(db, "yes", "wamid.s2")
    assert await med(db, SON["id"], "Sompraz 40") and await med(db, SON["id"], "Dart")


def _stop(name):
    return {"domain": "medicine", "name": name, "about": SON["id"], "reason": "they asked to delete it"}


async def test_a_vague_delete_then_all_stops_every_one_after_one_confirm(db, at, fake):
    await setup(db, at)
    await _two_saved(db, at, fake)
    fake.brain = [say("Which one?\n1. Dart\n2. Sompraz 40\nOr all?")]
    await turn(db, "dawai ke reminder hata do", "wamid.v1")
    assert "1. [medicine:" in fake.prompts[-1] and "2. [medicine:" in fake.prompts[-1]
    fake.brain = [call("stop", _stop("Sompraz 40")), call("stop", _stop("Dart"), 2), say("Stop Sompraz 40 and Dart for you?")]
    held = await turn(db, "all", "wamid.v2")
    assert [a.get("held") for a in held.actions] == [True, True], "'all' picks them; the stop itself still waits for the yes"
    fake.brain = [yes(("stop", _stop("Sompraz 40")), ("stop", _stop("Dart"))), say("Stopped both.")]
    await turn(db, "haan", "wamid.v3")
    assert await med(db, SON["id"], "Sompraz 40") is None and await med(db, SON["id"], "Dart") is None


async def test_a_vague_delete_then_one_number_stops_only_that_one(db, at, fake):
    await setup(db, at)
    await _two_saved(db, at, fake)
    fake.brain = [say("Which one?\n1. Dart\n2. Sompraz 40\nOr all?")]
    await turn(db, "delete medicine schedule", "wamid.w1")
    fake.brain = [call("stop", _stop("Dart")), say("Stop Dart for you?")]
    await turn(db, "1", "wamid.w2")
    fake.brain = [yes(("stop", _stop("Dart"))), say("Stopped Dart.")]
    await turn(db, "yes", "wamid.w3")
    assert await med(db, SON["id"], "Dart") is None and await med(db, SON["id"], "Sompraz 40") is not None


async def test_a_vague_delete_cancelled_stops_nothing(db, at, fake):
    await setup(db, at)
    await _two_saved(db, at, fake)
    fake.brain = [call("stop", _stop("Dart")), say("Stop Dart for you?")]
    await turn(db, "delete dart", "wamid.x1")
    fake.brain = [answer("no", ("stop", _stop("Dart"))), say("Okay, kept.")]
    await turn(db, "no wait, keep it", "wamid.x2")
    assert await med(db, SON["id"], "Dart") is not None


# ── other saves: food, reminders ─────────────────────────────────────────────────────────────────────────────────

async def test_a_reminder_is_read_back_and_set_only_after_yes(db, at, fake):
    await setup(db, at)
    rem = {"text": "Drink water", "times": ["16:00"], "repeat": "daily", "about": SON["id"]}
    fake.brain = [call("set_reminder", rem), say("A daily 4 pm reminder to drink water. Set it?")]
    held = await turn(db, "remind me to drink water at 4 daily", "wamid.r1")
    assert held.actions[0].get("held")
    fake.brain = [yes(("set_reminder", rem)), say("Set.")]
    done = await turn(db, "👍", "wamid.r2")
    assert any(a.get("tool") == "set_reminder" and a.get("ok") and not a.get("held") for a in done.actions), done.actions


async def test_a_food_log_waits_for_yes(db, at, fake):
    await setup(db, at)
    food = {"for_whom": SON["id"], "meal": "lunch", "items": [{"name": "dal", "quantity": 1, "unit": "bowl"},
                                                              {"name": "roti", "quantity": 2, "unit": "piece"}]}
    fake.brain = [call("log_food", food), say("Lunch: 1 bowl dal, 2 roti. Log it?")]
    held = await turn(db, "had dal and 2 roti for lunch", "wamid.f1")
    assert held.actions[0].get("held")
    fake.brain = [yes(("log_food", food)), say("Logged.")]
    await turn(db, "theek hai", "wamid.f2")
    sigs = [s for s in (await db.execute(select(Turn).where(Turn.family_id == FAM, Turn.role == "assistant").order_by(Turn.id))).scalars()]
    assert not (sigs[-1].meta or {}).get("pending_saves"), "nothing left waiting after the yes"


async def test_the_gate_can_be_switched_off(db, at, fake, monkeypatch):
    await setup(db, at)
    monkeypatch.setenv("SAVE_CONFIRM", "off")
    fake.brain = [call("remember", SOMPRAZ), say("Saved.")]
    await turn(db, "add sompraz for me", "wamid.o1")
    assert await med(db, SON["id"], "Sompraz 40") is not None


# ── pieces ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_a_list_counts_as_one_unit_so_a_photo_of_four_is_never_clipped():
    assert len(guards.units(LIST_REPLY)) <= 3
    assert guards.clip_reply(LIST_REPLY, cap=3).count("\n") >= 4


async def test_the_same_call_again_on_a_message_that_is_not_a_yes_stays_held(db, at, fake):
    await setup(db, at)
    fake.brain = [call("remember", SOMPRAZ), say("Shall I save Sompraz 40 for you at 8:00?")]
    await turn(db, "add sompraz for me at 8", "wamid.n1")
    at("2026-10-17 12:52")
    fake.brain = [call("remember", SOMPRAZ), say("Dart is usually for headache. Shall I save Sompraz 40 at 8:00?")]
    out = await turn(db, "what is dart for", "wamid.n2")
    assert await med(db, SON["id"], "Sompraz 40") is None and out.actions[0].get("held")


def test_a_changed_schedule_is_a_different_save():
    a = confirm.signature("remember", SOMPRAZ)
    b = confirm.signature("remember", {**SOMPRAZ, "details": {**SOMPRAZ["details"], "times": ["09:00"]}})
    c = confirm.signature("remember", {**SOMPRAZ, "sentence": "Takes Sompraz 40 daily before breakfast."})
    assert a != b and a == c


def test_course_dates():
    assert _course({"duration_days": 5, "starts_on": "2026-10-20"}) == {"duration_days": 5, "ends_on": "2026-10-24"}
    assert _course({}) == {}


def test_the_persona_never_asks_again_for_a_photo_it_has_and_confirms_saves():
    from pathlib import Path

    d = Path(loop.__file__).parent / "persona"
    core, ph, health = ((d / n).read_text() for n in ("core.md", "photos.md", "health.md"))
    assert "PHOTOS IN THIS CHAT" in core and "Nothing is saved until they say yes" in core
    assert "send it again" not in ph.split("PHOTOS IN THIS CHAT")[0][-200:]
    assert "MEDICINES ON RECORD" in health
    assert 'only say "Sure, send it"' in core, "a photo announced is waited for, not asked for again"
    assert "answer_pending" in core
    assert "usually for" not in ph and "for your flu" not in ph and "please confirm with your doctor" in ph


# ── review 17 Oct: readings alert at once ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,vital", [
    ("sugar 45", {"kind": "sugar", "value": "45", "unit": "mg/dL"}),
    ("BP 200/120", {"kind": "bp", "value": "200/120"}),
])
async def test_a_dangerous_reading_is_logged_and_alerts_in_the_same_turn_with_no_yes(db, at, fake, text, vital):
    await setup(db, at)
    fake.brain = [call("log_vital", {**vital, "about": MAA["id"]}), say("Logged. Please check on her now.")]
    out = await turn(db, text, "wamid.v-" + vital["kind"])
    logged = [a for a in out.actions if a.get("tool") == "log_vital"]
    assert logged and logged[0].get("ok") and not logged[0].get("held"), out.actions
    assert any(a.get("tool") == "alert_caregiver" and a.get("ok") for a in out.actions), "the family is alerted in this turn"
    mine = (await db.execute(select(Turn).where(Turn.id == out.turn_id))).scalar_one()
    assert not (mine.meta or {}).get("pending_saves")


async def test_diary_notes_and_habits_are_not_held(db, at, fake):
    assert {"note", "note_habit", "log_vital"}.isdisjoint(confirm.CONFIRM_FIRST)
    assert {"remember", "set_reminder", "log_food", "save_place", "add_person"} <= confirm.CONFIRM_FIRST


# ── review 17 Oct: the model answers in a structured field; the code never reads their words ─────────────────────

def test_the_confirm_code_has_no_word_list_and_never_reads_their_text():
    import inspect

    src = inspect.getsource(confirm)
    assert not hasattr(confirm, "says_yes") and not hasattr(confirm, "says_no")
    assert "user_text" not in src and "req.text" not in src


NOT_A_YES = ["that's not right", "Is it correct?", "kya ye sahi hai?", "Maa ji ko kal bukhar tha", "I'm fine"]
A_YES = ["acha", "हो गया", "galat nahi hai, theek hai"]


async def _held_sompraz(db, at, fake):
    await setup(db, at)
    fake.brain = [call("remember", SOMPRAZ), say("Sompraz 40 for you at 8:00 before breakfast for 14 days. Save it?")]
    await turn(db, "add sompraz 40 for me at 8 before breakfast, 2 weeks", "wamid.h1")
    at("2026-10-17 12:52")


@pytest.mark.parametrize("text", NOT_A_YES)
async def test_not_a_yes_saves_nothing_even_if_the_model_calls_the_save_again(db, at, fake, text):
    # Plumbing only: the scripted model does not call answer_pending (and even re-calls the save tool); the code holds it.
    await _held_sompraz(db, at, fake)
    fake.brain = [call("remember", SOMPRAZ), say("Okay. Shall I save Sompraz 40 at 8:00?")]
    out = await turn(db, text, "wamid.h2")
    assert await med(db, SON["id"], "Sompraz 40") is None
    assert [a.get("held") for a in out.actions if a.get("tool") == "remember"] == [True]


@pytest.mark.parametrize("text", A_YES)
async def test_a_yes_the_model_marks_saves_exactly_what_was_read_back(db, at, fake, text):
    # Plumbing only: the scripted model marks it yes; the real model's reading of these words is measured in eval/.
    await _held_sompraz(db, at, fake)
    fake.brain = [yes(("remember", SOMPRAZ)), say("Saved Sompraz 40 for you at 8:00.")]
    await turn(db, text, "wamid.h3")
    f = await med(db, SON["id"], "Sompraz 40")
    assert f is not None and f.value["times"] == ["08:00"]


async def test_an_unknown_id_saves_nothing(db, at, fake):
    await _held_sompraz(db, at, fake)
    fake.brain = [LLMReply(text="", tool_calls=[ToolCall(id="a0", name="answer_pending", args={"id": "s000000", "answer": "yes"})],
                           model="fake"), say("Saved.")]
    out = await turn(db, "yes", "wamid.h4")
    assert await med(db, SON["id"], "Sompraz 40") is None
    assert any(a.get("tool") == "answer_pending" and not a.get("ok") for a in out.actions)


def test_a_plural_or_small_rewording_is_the_same_save():
    food = {"for_whom": SON["id"], "meal": "lunch", "items": [{"name": "roti", "quantity": 2}, {"name": "Dal"}]}
    again = {"for_whom": SON["id"], "meal": "lunch", "items": [{"name": "rotis", "quantity": 2}, {"name": "dal "}]}
    assert confirm.signature("log_food", food) == confirm.signature("log_food", again)
    assert confirm.signature("log_food", food) != confirm.signature("log_food", {**food, "items": [{"name": "paratha"}]})


# ── review 17 Oct: a held save is confirmable only if its read-back was sent ─────────────────────────────────────

async def test_a_replaced_reply_drops_the_held_save_and_a_later_yes_saves_nothing(db, at, fake):
    await setup(db, at)
    fake.brain = [call("remember", SOMPRAZ), say("Save Sompraz 40 at 8:00?\n```python\nprint('x')\n```")] + [
        say("Save Sompraz 40 at 8:00?\n```python\nprint('x')\n```")] * 3
    out = await turn(db, "add sompraz 40 for me at 8", "wamid.z1")
    mine = (await db.execute(select(Turn).where(Turn.id == out.turn_id))).scalar_one()
    assert "Sompraz" not in out.reply, out.reply
    assert not (mine.meta or {}).get("pending_saves"), "the read-back was not sent, so nothing waits for a yes"
    at("2026-10-17 12:52")
    fake.brain = [yes(("remember", SOMPRAZ)), say("Saved.")]
    await turn(db, "yes", "wamid.z2")
    assert await med(db, SON["id"], "Sompraz 40") is None


# ── review 17 Oct: photos never claim a condition; links match the whole name ────────────────────────────────────

def test_the_reading_keeps_no_usual_use():
    text = photos.reading_text({**FOUR, "items": photos._clean_items(FOUR["items"])})
    assert all("usual_use" not in i for i in photos._clean_items(FOUR["items"]))
    assert "acidity" not in text and "fever" not in text and "Sompraz 40" in text
    assert "usual_use" not in photos.PROMPT and "used for" in photos.PROMPT


@pytest.mark.parametrize("saved,read,same", [
    ("Vitamin D3", "Vitamin B12", False), ("Sompraz 40", "Sompraz 20", False), ("Sompraz", "Sompraz 40", True),
    ("Dolo 650", "DOLO-650", True), ("Montair", "Montair LC", False), ("Telma 40", "Telma 40 mg", True),
])
def test_a_photo_link_needs_the_whole_name(saved, read, same):
    assert photos.same_medicine(saved, read) is same


async def test_a_quoted_photo_with_no_kept_reading_is_asked_for_again_never_guessed(db, at, fake):
    await setup(db, at)
    await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="user", text="", speaker_id=SON["id"],
                         meta={"images": 1, "media": ["image/jpeg"]}, message_ref="wamid.old")
    await db.commit()
    at("2026-10-17 12:55")
    fake.brain = [say("I can't see that photo now. Could you send it again or type the names?")]
    await turn(db, "which ones are these", "wamid.q1", reply_to="wamid.old")
    assert "you cannot see it now" in fake.prompts[-1] and "never guess" in fake.prompts[-1]


async def test_a_pdf_gets_no_extra_photo_read(db, at, fake):
    assert photos.start_read([{"mime": "application/pdf", "data": "aGVsbG8="}], "") is None
