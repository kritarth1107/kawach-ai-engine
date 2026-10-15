"""A stopped medicine's clock time is not the answer to a different dose question.

Day 21 of a self-care mock: "What dose of Thyronorm do I take now, and at what time?" came back as
"75 mcg at 08:00". 08:00 was Amlong, filed on her by mistake on day 7 and stopped later. The active
Thyronorm fact is 06:30. The old time was still in a note, in the event that saved Amlong, and on a
schedule row the backend had not cleared.
"""

from app.brain import loop
from app.care import digest, doses, memory_index, store
from app.care.domains import fact_key
from app.learn import situations
from app.llm import router, spend
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM = "fam-dose"
ME = {"id": "self-dose", "name": "Asha Rao", "role": "primary caregiver"}


async def _thyronorm(db):
    await store.write_fact(
        db, family_id=FAM, subject_id=ME["id"], domain="medicine",
        key=fact_key("medicine", "Thyronorm"),
        value={"name": "Thyronorm", "dose": "75 mcg", "times": ["06:30"]},
        text="Thyronorm 75 mcg at 06:30", source_kind="caregiver_said", stated_by=ME["id"],
    )


async def _stopped_amlong(db):
    key = fact_key("medicine", "Amlong")
    await store.write_fact(
        db, family_id=FAM, subject_id=ME["id"], domain="medicine", key=key,
        value={"name": "Amlong", "dose": "5 mg", "times": ["08:00"]},
        text="Amlong 5 mg at 08:00", source_kind="caregiver_said", stated_by=ME["id"],
    )
    stopped = await store.stop_fact(
        db, family_id=FAM, subject_id=ME["id"], key=key, reason="filed on the wrong person",
        source_kind="caregiver_said",
    )
    assert stopped.result == "stopped"
    await store.upsert_note(
        db, family_id=FAM, subject_id=ME["id"], slug="meds", title="Medicines",
        body_md="Amlong 5 mg at 08:00 was filed here. Thyronorm is the morning tablet.",
    )
    await store.record_event(
        db, family_id=FAM, subject_id=ME["id"], kind="fact_created",
        summary="A morning tablet at 08:00, noted beside Thyronorm",
        payload={"key": key},
    )


async def test_a_stopped_medicine_time_is_not_recalled(db):
    await _thyronorm(db)
    await _stopped_amlong(db)
    hits = await memory_index.search(db, FAM, [ME["id"]], "Thyronorm time")
    texts = " | ".join(h.text for h in hits)
    assert "06:30" in texts
    assert "08:00" not in texts
    record = digest.care_record("Asha Rao", await store.facts(db, FAM, ME["id"]))
    assert "06:30" in record and "08:00" not in record
    _, names = await doses.ended_medicines(db, FAM, [ME["id"]])
    assert doses.mentions_ended_medicine("Amlong 5 mg at 08:00", names)
    assert not doses.mentions_ended_medicine("Thyronorm 75 mcg at 06:30", names)


async def test_a_stopped_medicine_is_not_a_due_reminder(db):
    await _thyronorm(db)
    await _stopped_amlong(db)
    req = loop.TurnRequest(
        family_id=FAM, elder=ME, speaker=ME, members=[ME], text="hi", message_ref="dose-r",
        reminders_soon=[
            {"id": "old", "item": fact_key("medicine", "Amlong"), "person": ME["id"], "day": "2026-10-15",
             "time": "08:00", "text": "Amlong 5 mg at 08:00", "repeat": "daily"},
            {"id": "live", "item": fact_key("medicine", "Thyronorm"), "person": ME["id"], "day": "2026-10-15",
             "time": "06:30", "text": "Thyronorm 75 mcg at 06:30", "repeat": "daily"},
        ],
    )
    block = await loop.reminders_block(db, req)
    assert "06:30" in block and "08:00" not in block


class _Then:
    """First brain draft is only the canned acknowledgement. The next one answers. A judge call is empty claims."""

    def __init__(self):
        self.brain = 0

    async def complete(self, route, **kw):
        blob = " ".join(str(m) for m in (kw.get("messages") or []))
        if "REPLY\n" in blob:
            return LLMReply(text='{"claims":[]}', tool_calls=[], model="fake")
        self.brain += 1
        text = "Okay, noted 🙏" if self.brain == 1 else "Thyronorm 75 mcg at 06:30."
        return LLMReply(text=text, tool_calls=[], model="fake")


async def test_a_dose_question_is_not_answered_with_only_an_acknowledgement(db, monkeypatch):
    question = "What dose of Thyronorm do I take now, and at what time?"
    assert situations.tag(text=question, role="caregiver", tools=[]) == "med_question"
    await store.save_roster(db, FAM, ME, [ME])
    await _thyronorm(db)
    model = _Then()
    router.register_provider("fake", model)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"], "classify": ["fake:m"]}')
    monkeypatch.setenv("BRAIN_HEDGE_AFTER_S", "0")
    router.reset_breakers()
    spend.reset()
    try:
        out = await loop.run_turn(db, SimHost(), loop.TurnRequest(
            family_id=FAM, elder=ME, speaker=ME, members=[ME], text=question, message_ref="dose-q",
        ))
    finally:
        router._providers.pop("fake", None)
        router.reset_breakers()
    assert "06:30" in out.reply
    assert out.reply.strip() != "Okay, noted 🙏"
    assert model.brain >= 2
