"""Outcomes and feedback: said in chat, tapped on WhatsApp, logged on the dashboard, or noticed automatically."""

from datetime import timedelta

import pytest

from app.brain import tools
from app.brain.loop import TurnRequest, run_turn
from app.care import outcomes, patterns, store
from app.care.models import OpenLoop
from app.core import clock
from app.llm import router, spend
from app.llm.router import LLMReply, ToolCall
from app.sim.world import SimHost

FAM, ELDER, CG = "fam-o", "elder-o", "cg-o"
E = {"id": ELDER, "name": "Kamla", "role": "elder"}
C = {"id": CG, "name": "Asha", "role": "primary caregiver"}


def ctx(db, speaker=C, host=None):
    return tools.TurnCtx(session=db, host=host or SimHost(), family_id=FAM, elder=E, speaker=speaker, members=[E, C])


class Script:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    async def complete(self, route, **kw):
        self.calls += 1
        return self.replies.pop(0) if self.replies else LLMReply(text="Theek hai ji.", tool_calls=[], model="fake")


@pytest.fixture
def model(monkeypatch):
    holder = {}

    def use(replies=()):
        holder["m"] = Script(replies)
        router.register_provider("fake", holder["m"])
        return holder["m"]

    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    yield use
    router._providers.pop("fake", None)


async def test_said_in_chat_is_logged(db, at, model):
    at("2026-10-02 10:00")
    await store.save_roster(db, FAM, E, [E, C])
    model([LLMReply(text="", tool_calls=[ToolCall("c1", "log_outcome", {"kind": "fall", "summary": "slipped in the bathroom"})], model="fake"),
           LLMReply(text="Oh no. Kya chot lagi hai?", tool_calls=[], model="fake")])
    await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=E, speaker=E, members=[E, C], text="Main bathroom mein gir gayi", message_ref="o1"))
    got = await outcomes.recent(db, FAM, ELDER)
    assert got and got[0]["kind"] == "fall" and got[0]["source"] == "said"


async def test_tapped_button_needs_no_model(db, at, model):
    at("2026-10-02 10:00")
    await store.save_roster(db, FAM, E, [E, C])
    m = model()
    key = f"alert:red_flag:fall:2026-10-01:subj={ELDER}"
    await store.open_loop(db, family_id=FAM, subject_id=ELDER, kind="followup", title="t", detail={"key": key, "set": "outcome"},
                          owner_id=CG, wake_at=clock.now() + timedelta(hours=1), alert_rule="dashboard", dedupe_key=f"followup:{key}")
    await db.commit()
    out = await run_turn(db, SimHost(), TurnRequest(family_id=FAM, elder=E, speaker=C, members=[E, C], text=f"v2:oc:hospital_visit:{key}", message_ref="b1"))
    assert m.calls == 0 and out.model == "none" and "recovery" in out.reply.lower()
    got = await outcomes.recent(db, FAM, ELDER)
    assert got[0]["kind"] == "hospital_visit" and got[0]["source"] == "button"
    loops = [l for l in (await db.execute(OpenLoop.__table__.select())).all() if l.dedupe_key == f"followup:{key}" and l.family_id == FAM]
    assert loops and loops[0].status == "done"


async def test_feedback_mutes_a_pattern_for_that_family(db, at):
    at("2026-10-02 10:00")
    for d in (1, 3, 5, 7):
        clock.set_now(clock.now().replace(day=d))
        await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="symptom", summary="knee pain", payload={"severity": "watch"})
    clock.set_now(clock.now().replace(day=9))
    await patterns.record_new(db, FAM, ELDER)
    shown = await patterns.recent(db, FAM, [ELDER])
    assert shown and "offer_buttons" in patterns.context_block(shown, {})
    await outcomes.record_feedback(db, family_id=FAM, subject_id=ELDER, target=shown[0]["target"], vote="down", by=CG, source="button")
    assert await patterns.recent(db, FAM, [ELDER]) == []


async def test_caregiver_medicine_change_is_an_automatic_outcome(db, at):
    at("2026-10-02 10:00")
    c = ctx(db)
    await tools.run(c, "remember", {"domain": "medicine", "name": "Telma", "details": {"dose": "40 mg", "times": ["09:00"]}, "sentence": "Telma 40 at 9"})
    await tools.run(c, "remember", {"domain": "medicine", "name": "Telma", "details": {"dose": "80 mg", "times": ["21:00"]}, "sentence": "Telma 80 at 9 pm"})
    got = await outcomes.recent(db, FAM, ELDER)
    assert [g["kind"] for g in got] == ["medicine_changed"] and got[0]["source"] == "auto"


async def test_red_flag_alert_schedules_a_next_day_follow_up(db, at):
    at("2026-10-02 18:00")
    c = ctx(db, speaker=E)
    c.user_text = "I fell in the bathroom"
    out, err = await tools.run(c, "alert_caregiver", {"reason": "red_flag", "issue": "fall-bathroom", "message": "Kamla fell", "confidence": 0.95})
    assert not err
    loops = [l for l in (await db.execute(OpenLoop.__table__.select())).all() if l.kind == "followup" and l.family_id == FAM]
    assert len(loops) == 1 and loops[0].owner_id == CG and clock.ist(loops[0].wake_at).strftime("%d %H:%M") == "03 10:00"


async def test_follow_up_message_carries_buttons_in_their_language(db, at):
    at("2026-10-02 10:00")
    host = SimHost()
    c = ctx(db, speaker={"id": "saheli-scheduler", "name": "Scheduler", "role": "system"}, host=host)
    c.profiles = {CG: {"script": "latin", "roman": "indic"}}
    await tools.run(c, "send_message", {"to": CG, "text": "आशा जी, मम्मी अब कैसी हैं? बताइए।", "buttons": {"kind": "outcome", "key": "alert:x:subj=elder-o"}})
    sent = host.world.sent[0]
    assert [b["title"] for b in sent["buttons"]] == ["सब ठीक", "डॉक्टर को दिखाया", "अस्पताल"]  # Hinglish writer → Devanagari
    assert sent["buttons"][0]["id"] == "v2:oc:all_fine:alert:x:subj=elder-o"


async def test_consent_is_caregiver_only(db, at):
    at("2026-10-02 10:00")
    _, err = await tools.run(ctx(db, speaker=E), "learning_consent", {"granted": True})
    assert err
    await tools.run(ctx(db), "learning_consent", {"granted": True})
    assert (await outcomes.consent(db, FAM, ELDER))["granted"]


def test_button_ids_fit_whatsapp_limits():
    for kind in ("outcome", "visit", "feedback"):
        for b in outcomes.buttons(kind, "pattern:dose:metformin:21:00:subj=" + "x" * 300, {"script": "devanagari"}):
            assert len(b["id"]) <= 256 and len(b["title"]) <= 20


async def test_language_preference_saves_dialect_and_tells_the_backend(db, at):
    at("2026-10-02 10:00")
    host = SimHost()
    out, err = await tools.run(ctx(db, host=host), "language_preference", {"dialect": "Marwari", "about": ELDER})
    assert not err, out
    facts = [f for f in await store.facts(db, FAM, ELDER) if f.domain == "language"]
    assert facts and facts[0].value == {"dialect": "mwr", "language": "hi"} and "Marwari" in facts[0].text
    assert host.world.speech[ELDER] == {"language": "hi", "dialect": "mwr", "script": "native"}
    out, err = await tools.run(ctx(db, host=host), "language_preference", {"about": ELDER})
    assert "Marwari" in out
    # "write to me in English letters": Roman letters, dialect kept
    out, err = await tools.run(ctx(db, host=host), "language_preference", {"script": "roman", "about": ELDER})
    assert not err and host.world.speech[ELDER]["script"] == "roman" and host.world.speech[ELDER]["dialect"] == "mwr"
    # remember(domain=language) goes the same way
    out, err = await tools.run(ctx(db, host=host), "remember", {"domain": "language", "name": "Maithili", "details": {}, "sentence": "Speaks Maithili", "about": ELDER})
    assert not err and host.world.speech[ELDER]["dialect"] == "mai"


async def test_elder_sets_only_her_own_language(db, at):
    at("2026-10-02 10:00")
    _, err = await tools.run(ctx(db, speaker=E), "language_preference", {"language": "ta", "about": CG})
    assert err
    out, err = await tools.run(ctx(db, speaker=E), "language_preference", {"language": "Tamil"})
    assert not err and '"ta"' in out
    _, err = await tools.run(ctx(db), "language_preference", {"language": "Klingon", "about": ELDER})
    assert err


async def test_setup_progress_lists_what_is_missing(db, at):
    at("2026-10-02 10:00")
    out, err = await tools.run(ctx(db), "setup_progress", {"about": ELDER})
    assert not err and '"complete": false' in out and '"naming"' in out
    host = SimHost()
    await tools.run(ctx(db, host=host), "language_preference", {"dialect": "Bhojpuri", "about": ELDER})
    await tools.run(ctx(db), "remember", {"domain": "allergy", "name": "none", "details": {}, "sentence": "No known allergies", "about": ELDER})
    out, _ = await tools.run(ctx(db), "setup_progress", {"about": ELDER})
    import json as _j
    data = _j.loads(out)
    assert "language" in data["already_known"] and "allergy" in data["already_known"]
    assert [m["item"] for m in data["missing"]][:3] == ["naming", "condition", "medicine"]
