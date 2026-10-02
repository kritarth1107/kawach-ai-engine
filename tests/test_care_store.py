from datetime import timedelta

from app.care import store
from app.care.domains import fact_key, needs_confirmation, slug

FAM, ELDER, DAUGHTER = "fam-1", "elder-1", "cg-1"


async def _med(db, dose, source, by=None):
    return await store.write_fact(
        db,
        family_id=FAM,
        subject_id=ELDER,
        domain="medicine",
        key=fact_key("medicine", "Amlodipine"),
        value={"name": "Amlodipine", "dose": dose, "times": ["21:00"], "food_timing": "after_food"},
        text=f"Amlodipine {dose} at night after food",
        source_kind=source,
        stated_by=by,
    )


def test_slug_ignores_strength_and_form():
    assert slug("Metformin 500 mg tablet") == "metformin"
    assert fact_key("allergy", "Milk") == "allergy:milk"


def test_confirmation_rule():
    assert needs_confirmation("medicine", "elder_said", "prescription", True)
    assert not needs_confirmation("medicine", "caregiver_said", "elder_said", True)
    assert not needs_confirmation("diet", "elder_said", "caregiver_said", True)
    assert not needs_confirmation("medicine", "elder_said", None, False)


async def test_change_supersedes_and_history_answers_past_value(db, at):
    at("2026-09-01 10:00")
    await _med(db, "5 mg", "caregiver_said", DAUGHTER)
    at("2026-09-03 10:00")
    w = await _med(db, "10 mg", "prescription")
    assert w.result == "superseded" and w.previous.value["dose"] == "5 mg"
    hist = await store.fact_history(db, FAM, ELDER, "medicine:amlodipine")
    assert [(h.value["dose"], h.status) for h in hist] == [("5 mg", "superseded"), ("10 mg", "active")]
    assert hist[0].valid_to == hist[1].valid_from


async def test_same_value_is_unchanged_and_gets_confirmed(db, at):
    at("2026-09-01 10:00")
    await _med(db, "5 mg", "elder_said", ELDER)
    w = await _med(db, "5 mg", "caregiver_said", DAUGHTER)
    assert w.result == "unchanged" and w.fact.confirmed_by == DAUGHTER


async def test_elder_cannot_change_prescribed_dose_alone(db, at):
    at("2026-09-01 10:00")
    await _med(db, "5 mg", "prescription")
    w = await _med(db, "2.5 mg", "elder_said", ELDER)
    assert w.result == "pending"
    live = await store.facts(db, FAM, ELDER, domains=["medicine"], statuses=("active",))
    assert [f.value["dose"] for f in live] == ["5 mg"]
    loops = await store.live_loops(db, FAM, [ELDER])
    assert loops[0].kind == "confirm_fact"
    approved = await store.resolve_pending(db, fact_id=w.fact.id, approve=True, by=DAUGHTER)
    assert approved.status == "active" and approved.confirmed_by == DAUGHTER
    live = await store.facts(db, FAM, ELDER, domains=["medicine"], statuses=("active",))
    assert [f.value["dose"] for f in live] == ["2.5 mg"]
    assert await store.live_loops(db, FAM, [ELDER]) == []


async def test_elder_stopping_medicine_waits_for_caregiver(db, at):
    at("2026-09-01 10:00")
    await _med(db, "5 mg", "caregiver_said", DAUGHTER)
    w = await store.stop_fact(
        db, family_id=FAM, subject_id=ELDER, key="medicine:amlodipine", reason="feels dizzy", source_kind="elder_said"
    )
    assert w.result == "pending"
    assert (await store.active_fact(db, FAM, ELDER, "medicine:amlodipine")).value["dose"] == "5 mg"
    await store.resolve_pending(db, fact_id=w.fact.id, approve=False, by=DAUGHTER)
    assert (await store.active_fact(db, FAM, ELDER, "medicine:amlodipine")) is not None


async def test_caregiver_stop_is_immediate(db, at):
    at("2026-09-01 10:00")
    await _med(db, "5 mg", "caregiver_said", DAUGHTER)
    w = await store.stop_fact(
        db, family_id=FAM, subject_id=ELDER, key="medicine:amlodipine", reason="doctor stopped", source_kind="caregiver_said"
    )
    assert w.result == "stopped"
    assert await store.active_fact(db, FAM, ELDER, "medicine:amlodipine") is None


async def test_events_are_idempotent_by_ref(db, at):
    at("2026-10-02 08:00")
    a = await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="reminder_sent", summary="Metformin", ref="slot-1")
    b = await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="reminder_sent", summary="Metformin", ref="slot-1")
    assert a and b is None
    day = await store.events(db, FAM, ELDER, day="2026-10-02")
    assert len(day) == 1 and day[0].day == "2026-10-02"


async def test_recall_finds_notes_events_and_past_facts(db, at):
    at("2026-09-01 10:00")
    await _med(db, "5 mg", "caregiver_said", DAUGHTER)
    at("2026-09-03 10:00")
    await _med(db, "10 mg", "prescription")
    await store.upsert_note(
        db, family_id=FAM, subject_id=ELDER, slug="food", title="Food", body_md="Makes besan chilla every Sunday."
    )
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind="symptom", summary="Dizziness after amlodipine")
    hits = await store.recall(db, FAM, [ELDER], "amlodipine dose")
    texts = " | ".join(h.text for h in hits)
    assert "5 mg" in texts and "10 mg" in texts and "Dizziness" in texts
    hits = await store.recall(db, FAM, [ELDER], "chilla")
    assert hits and hits[0].source == "note"


async def test_open_loop_dedupes_and_due(db, at):
    now = at("2026-10-02 08:00")
    a = await store.open_loop(db, family_id=FAM, subject_id=ELDER, kind="question", title="BP?", dedupe_key="q:bp",
                              wake_at=now + timedelta(minutes=30))
    b = await store.open_loop(db, family_id=FAM, subject_id=ELDER, kind="question", title="BP today?", dedupe_key="q:bp")
    assert a.id == b.id and b.title == "BP today?"
    assert await store.due_loops(db, before=now) == []
    at("2026-10-02 08:31")
    assert [x.id for x in await store.due_loops(db, before=now + timedelta(minutes=31))] == [a.id]


async def test_turns_dedupe_and_order(db, at):
    at("2026-10-02 08:00")
    await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="hi", message_ref="wamid.1")
    await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="hi", message_ref="wamid.1")
    await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="assistant", text="Namaste!")
    turns = await store.recent_turns(db, FAM, ELDER)
    assert [t.text for t in turns] == ["hi", "Namaste!"]
