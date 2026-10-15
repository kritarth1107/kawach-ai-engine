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
    assert medicine_already({"same_as": "medicine:other", "covered": True}, [saved]) is None


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
