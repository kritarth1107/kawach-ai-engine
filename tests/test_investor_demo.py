"""The investor demo set stays a fixed list of first messages, off the nightly gate."""

import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "investor_demo_2026-10-16.jsonl"
PHOTOS = Path(__file__).resolve().parents[1] / "eval" / "data" / "nutrition_photos"


def test_a_short_greeting_while_setting_up_is_two_lines():
    from app.brain import guards

    hi = guards.first_hello(None, "Hi")
    assert hi == "Hi, I'm Saheli, from Kavach. Which language should we talk in?"
    assert "किस भाषा" in guards.first_hello(None, "नमस्ते")
    assert guards.first_hello(None, "Hi, I'm Anita").startswith("Hi, I'm Saheli")
    assert guards.first_hello(None, "Hi, can you remind my mom about her medicines?") == ""
    assert guards.first_hello(None, "What can you do?") == ""


def test_a_setup_pitch_does_not_answer_what_she_can_do():
    from app.brain import guards

    reply = "Would you like us to finish setting up her profile now?"
    assert guards.setup_instead(reply, "What can you do?")
    assert not guards.setup_instead(reply, "Help me set up Saheli?")
    out = guards.settle_reply(reply, None, situation="chit_chat", needs_framing=False, force_light=False,
                              held_topic_text="", actions=[], prior_assistant=False, naming=False, user_text="What can you do?")
    assert "check in a flight" in out and "room" in out
    assert len(guards.sentences(out)) <= 4


def test_a_long_medicine_answer_stays_three_sentences():
    from app.brain import guards

    long = ("Don't skip the tablet, because stopping it suddenly can make her pressure swing. "
            "If the reading is low, call her doctor. Ask how she feels. "
            "This is general information; confirm it with your doctor or pharmacist. "
            "I can also explain the brand.")
    line = guards.with_framing(None, long)
    assert len(guards.sentences(line)) <= 3
    assert "general information" in line
    assert "swing" not in line
    short = "Metformin helps. This is general information; confirm it with your doctor or pharmacist."
    assert guards.with_framing(None, short) == short


def test_a_covered_brand_is_not_a_second_medicine():
    from app.care.dedupe import medicine_already
    from app.care.models import CareFact

    saved = CareFact(family_id="f", subject_id="m", domain="medicine", key="medicine:telmisartan",
                     value={"name": "Telmisartan", "dose": "40 mg", "times": ["09:00"]}, text="Telmisartan 40 mg at 09:00",
                     status="active", source_kind="prescription")
    assert medicine_already({"same_as": "medicine:telmisartan", "covered": True}, [saved]) is saved
    assert medicine_already({"same_as": "medicine:telmisartan", "covered": False}, [saved]) is None
    assert medicine_already({"same_as": "medicine:telmisartan", "covered": False}, [saved],
                            {"name": "Telma", "dose": "40 mg", "times": []}) is saved
    assert medicine_already({"same_as": "medicine:telmisartan", "covered": False}, [saved],
                            {"name": "Telma", "dose": "40 mg", "times": ["21:00"]}) is None
    assert medicine_already({"same_as": "medicine:other", "covered": True}, [saved]) is None


def test_a_message_break_is_not_a_sentence():
    from app.brain import guards

    reply = "I'm on it. Is she all right?\n---\nShould I place this order?"
    assert len(guards.sentences(reply)) == 3
    assert guards.without_bookkeeping("Do not say you are looking it up.\n---\nLagaan is a classic. First time?") == (
        "Lagaan is a classic. First time?")


def test_who_she_is_stays_one_line_and_a_medicine_question_is_answered():
    from app.brain import guards

    pitch = "Would you like us to finish setting up her profile now?"
    who = guards.settle_reply(pitch, None, situation="chit_chat", needs_framing=False, force_light=False,
                              held_topic_text="", actions=[], prior_assistant=False, naming=False, user_text="Who are you?")
    assert who == "I'm Saheli, from Kavach."
    med = guards.settle_reply(pitch, None, situation="med_question", needs_framing=False, force_light=False,
                              held_topic_text="", actions=[], prior_assistant=False, naming=False,
                              user_text="Hi, can you remind my mom about her medicines?",
                              medicines=[("Telmisartan", "40 mg", ["09:00"])])
    assert "Telmisartan 40 mg at 09:00" in med and "check in a flight" not in med


def test_a_saved_medicine_does_not_answer_the_price_she_just_gave():
    from app.brain import guards

    out = guards.settle_reply(
        "Telmisartan is often cheaper at a Jan Aushadhi store.", None, situation="chit_chat", needs_framing=False,
        force_light=False, held_topic_text="", actions=[], prior_assistant=True, naming=False,
        user_text="Anything cheaper?", medicines=[("Telmisartan", "40 mg", ["09:00"])],
        last_assistant="Milk and bread, to be confirmed, come to 240 rupees.")
    assert "240" in out and "Telmisartan" not in out


def test_a_reading_or_a_red_flag_is_one_calm_line_and_a_week_keeps_its_facts():
    from app.brain import guards

    assert guards.noted_reading(None, "sugar 280") == "Noted: sugar 280."
    assert guards.calm_noted(None, "no one else in the family to tell").endswith("no one else in the family to tell")
    assert "skip" in guards.missed_reminder_line(None)
    facts = guards.week_facts("Medicines: 0 of 7.\nREADING: BP 128/80\nActivity: Morning walk, 20 minutes\nMood/sleep/meals: Lunch: dal and rice")
    assert len(facts) == 3


def test_a_reading_kept_only_in_the_summary_is_still_in_the_report():
    from app.care.features import report_facts_text

    text = report_facts_text("Meera", {
        "from": "2026-10-10", "to": "2026-10-16",
        "adherence": {"taken": 0, "expected": 7, "percent": 0, "byMedicine": []},
        "vitals": [{"kind": None, "value": None, "summary": "BP 128/80"}],
        "activities": [{"text": "Morning walk, 20 minutes"}],
        "symptoms": [], "mood": [{"text": "Lunch: dal and rice"}], "alerts": [], "changes": [],
    })
    assert "BP 128/80" in text and "Morning walk, 20 minutes" in text


def test_forty_first_messages_and_the_photo_exists():
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    assert len(rows) == 40
    assert len({r["id"] for r in rows}) == 40
    assert all(r["source"] == "investor-demo" and r["when"] == "gemini" and r["status"] == "open" for r in rows)
    plate = next(r for r in rows if r["id"] == "demo-photo-plate")
    assert (PHOTOS / plate["context"]["photo"]).is_file()
    hello = next(r for r in rows if r["id"] == "demo-hello-en")
    assert hello["context"]["onboarding"] is True
    assert "91337bfa" not in FIXTURE.read_text()
