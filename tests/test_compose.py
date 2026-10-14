"""Reminder lines are written when they go out and never repeat (founder 2026-10-13: "send new message every time ... it
still sends the same message again and again ... generate it at the time of sending only")."""

import json

import pytest

from app.care import compose, store
from app.llm import router
from app.llm.router import LLMReply

FAM = "fam-compose"
ME = {"id": "self-c", "name": "Kittu", "role": "primary caregiver"}


class Writer:
    def __init__(self, lines):
        self.lines, self.prompts = list(lines), []

    async def complete(self, route, *, messages, **kw):
        self.prompts.append(messages[0]["content"][0]["text"])
        return LLMReply(text=json.dumps({"text": self.lines.pop(0) if self.lines else ""}), tool_calls=[], model="fake")


@pytest.fixture
def writer(monkeypatch):
    def make(lines):
        w = Writer(lines)
        router.register_provider("fake", w)
        monkeypatch.setenv("MODEL_ROUTES", '{"worker": ["fake:m"]}')
        router.reset_breakers()
        return w

    yield make
    router._providers.pop("fake", None)


async def test_a_line_is_written_for_the_moment_with_their_language_and_the_lines_it_must_not_repeat(db, at, writer):
    at("2026-10-13 18:00")
    await store.save_roster(db, FAM, ME, [ME])
    await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="language", key="language:preferred",
                           value={"language": "hi", "script": "roman"}, text="Speaks Hindi in Roman letters", source_kind="elder_said")
    await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="preference", key="preference:reminder_emoji",
                           value={}, text="Wants emoji in reminders", source_kind="elder_said")
    w = writer(["Ek lambi saans lo, cigarette aaj bhi skip! 💪"])
    out = await compose.reminder_text(db, family_id=FAM, subject_id=ME["id"], purpose="Deep breath, skip the cigarette", at="18:00",
                                      recent=["Deep breath, skip the cigarette", "Cigarette ko na bolo, aap kar sakte ho"])
    assert out == {"text": "Ek lambi saans lo, cigarette aaj bhi skip! 💪", "fresh": True}
    prompt = w.prompts[0]
    assert "Hindi" in prompt and "Roman" in prompt and "Wants emoji in reminders" in prompt
    assert "- Cigarette ko na bolo, aap kar sakte ho" in prompt and "FOR: Deep breath, skip the cigarette" in prompt


async def test_a_repeat_is_rejected_and_written_again_and_twice_means_the_stored_text(db, at, writer):
    at("2026-10-13 18:00")
    await store.save_roster(db, FAM, ME, [ME])
    recent = ["Deep breath, skip the cigarette"]
    writer(["deep breath, skip the cigarette!", "Aaj ki jeet: ek aur cigarette-free shaam."])
    out = await compose.reminder_text(db, family_id=FAM, subject_id=ME["id"], purpose="Deep breath, skip the cigarette", recent=recent)
    assert out["text"] == "Aaj ki jeet: ek aur cigarette-free shaam."
    writer(["Deep breath, skip the cigarette", "Deep breath — skip the cigarette."])
    out = await compose.reminder_text(db, family_id=FAM, subject_id=ME["id"], purpose="Deep breath, skip the cigarette", recent=recent)
    assert out == {"text": None, "fresh": False}, "nothing new: the backend sends its stored text"


async def test_the_stored_slogan_is_not_sent_again_word_for_word(db, at, writer):
    at("2026-10-14 20:00")
    await store.save_roster(db, FAM, ME, [ME])
    stored = "Every hour is a new victory. Skip this cigarette and keep your streak going!"
    writer([f"Reminder: {stored}", stored])
    out = await compose.reminder_text(db, family_id=FAM, subject_id=ME["id"], purpose=stored, at="20:00", recent=[stored])
    assert out == {"text": None, "fresh": False}
    assert "everyday" in compose.PROMPT.lower() and "English stays English" in compose.PROMPT


def test_fresh_means_not_a_recent_line_and_not_the_same_opening_as_the_last():
    assert compose.fresh("Time for Metformin 500 mg", ["Metformin time!", "Your Metformin 500 mg now"])
    assert not compose.fresh("metformin time", ["Metformin time!"])
    assert not compose.fresh("Your Metformin 500 mg, please take it", ["Your Metformin 500 mg now"]), "same opening as the last line"
    assert not compose.fresh("", [])
