"""D-12 (2026-10-11): Saheli's memory says what is true now.

Gap 1: the meaning-search index kept a replaced dose as "(active)" next to the new one, and note lines a rewrite had
taken out stayed findable forever. Gap 2: nothing checked notes against the care record or against each other.
These tests use a tiny fake meaning model; each of the first four failed on the code before D-12.
"""

import hashlib
import json
import re

import pytest
from sqlalchemy import select

from app.brain import tools
from app.care import memory_index, memory_upkeep, store, versions
from app.care.models import CareEvent, MemoryNote, OpenLoop
from app.llm import router
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM, ELDER, CG, OTHER = "fam-truth", "e-truth", "c-truth", "o-truth"
E = {"id": ELDER, "name": "Sarla", "role": "elder"}
C = {"id": CG, "name": "Rohan", "role": "primary caregiver"}
SYN = {"sugar": "metformin", "goli": "metformin", "rehta": "lives", "kahan": "where", "purana": "old", "ghar": "house",
       "mandir": "temple"}


async def fake_embed(texts):
    """Words mapped to concepts, hashed into 64 dims: enough to find by meaning what keywords miss."""
    out = []
    for t in texts:
        v = [0.0] * memory_index.DIM
        for w in re.findall(r"\w+", t.lower()):
            w = SYN.get(w, w)
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        out.append(v)
    return out


@pytest.fixture
def meaning():
    memory_index.set_embedder(fake_embed)
    yield
    memory_index.set_embedder(None)


async def dose(db, mg: int):
    return await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="medicine", key="medicine:metformin",
                                  value={"name": "Metformin", "dose": f"{mg} mg"}, text=f"Metformin {mg} mg after dinner",
                                  source_kind="caregiver_said", stated_by=CG)


def metformin(hits) -> tuple[list[str], list[str]]:
    """Care-record values in the hits (ledger events like "[fact_superseded] …" are dated history, not values)."""
    meds = [h.text for h in hits if h.source == "fact" and "metformin" in h.text.lower()]
    return [t for t in meds if "EARLIER" not in t], meds


async def test_after_a_supersede_search_returns_exactly_one_current_value(db, at, meaning):
    at("2026-10-01 09:00")
    await dose(db, 500)
    await memory_index.nightly(db, FAM, ELDER)
    at("2026-10-05 09:00")
    w = await dose(db, 1000)
    assert w.result == "superseded"
    await memory_index.nightly(db, FAM, ELDER)  # the new value is in the index too
    await db.commit()
    current, meds = metformin(await memory_index.search(db, FAM, [ELDER], "sugar ki goli", limit=8))
    assert len(current) == 1 and "1000" in current[0], meds
    assert all("1000" in t or "EARLIER" in t for t in meds), meds
    rows = (await db.execute(select(memory_index.MemoryVector.text).where(memory_index.MemoryVector.source == "fact"))).scalars().all()
    assert sorted(rows) == ["(active) Metformin 1000 mg after dinner", "(earlier, until 2026-10-05) Metformin 500 mg after dinner"], rows


async def test_a_line_taken_out_by_a_rewrite_is_not_found_any_more(db, at, meaning):
    at("2026-10-01 09:00")
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="family", title="Family", body_md="- Her son Rahul lives in Pune with his wife")
    await memory_index.nightly(db, FAM, ELDER)
    at("2026-10-03 09:00")
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="family", title="Family", body_md="- Her son Rahul lives in Bangalore with his wife")
    await db.commit()
    hits = await memory_index.search(db, FAM, [ELDER], "beta kahan rehta hai", limit=8)
    assert not any("Pune" in h.text for h in hits), [h.text for h in hits]
    await memory_index.nightly(db, FAM, ELDER)
    hits = await memory_index.search(db, FAM, [ELDER], "beta kahan rehta hai", limit=8)
    assert any("Bangalore" in h.text for h in hits) and not any("Pune" in h.text for h in hits), [h.text for h in hits]


async def test_dated_diary_lines_stay_findable_after_they_drop_out(db, at, meaning):
    at("2026-10-01 09:00")
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="diary", title="Diary", body_md="- 2026-09-30: Went to the temple with her sister")
    await memory_index.nightly(db, FAM, ELDER)
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="diary", title="Diary", body_md="- 2026-10-01: Quiet day at home")
    hits = await memory_index.search(db, FAM, [ELDER], "mandir", limit=8)
    assert any("temple with her sister" in h.text for h in hits), "a dated diary line is history and stays findable"


async def test_forget_restore_and_undo_still_work(db, at, meaning):
    at("2026-10-01 09:00")
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="places", title="Places",
                            body_md="- Goes to the Hanuman temple every Tuesday evening\n- Her old house was in Civil Lines near the park")
    await memory_index.nightly(db, FAM, ELDER)
    assert (await memory_upkeep.forget(db, FAM, [ELDER], "old house Civil Lines", by=CG))["forgotten"] == 1
    assert not any("Civil Lines" in h.text for h in await memory_index.search(db, FAM, [ELDER], "purana ghar", limit=8))
    assert any("Hanuman temple" in h.text for h in await memory_index.search(db, FAM, [ELDER], "mandir", limit=8)), \
        "only the forgotten line leaves the index, not the rest of the note"
    eid = (await db.execute(select(CareEvent.id).where(CareEvent.family_id == FAM, CareEvent.kind == "memory_forgotten"))).scalar_one()
    assert (await memory_upkeep.restore(db, FAM, eid, by=CG, subjects=[ELDER]))["restored"] == 1
    await memory_index.nightly(db, FAM, ELDER)
    assert any("Civil Lines" in h.text for h in await memory_index.search(db, FAM, [ELDER], "purana ghar", limit=8))

    # undo a dose change: the old dose is current again and the index says so, once
    await dose(db, 500)
    await memory_index.nightly(db, FAM, ELDER)
    at("2026-10-05 09:00")
    await dose(db, 1000)
    v = await versions.latest(db, FAM, ELDER, "fact", "medicine:metformin")
    plan = await versions.plan_fact_undo(db, v, caregiver=True, actor_id=CG, confirmed=True)
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=C, members=[E, C])
    assert (await tools._apply_fact_plan(ctx, plan, "undo"))["result"] == "superseded"
    await memory_index.nightly(db, FAM, ELDER)
    current, meds = metformin(await memory_index.search(db, FAM, [ELDER], "sugar ki goli", limit=8))
    assert len(current) == 1 and "500" in current[0], meds


async def test_a_turned_down_proposal_never_comes_back_from_search(db, at, meaning):
    at("2026-10-01 09:00")
    await dose(db, 500)
    proposal = await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="medicine", key="medicine:metformin",
                                      value={"name": "Metformin", "dose": "250 mg"}, text="Metformin 250 mg after dinner",
                                      source_kind="elder_said", stated_by=ELDER)
    assert proposal.result == "pending"
    await memory_index.nightly(db, FAM, ELDER)  # indexed as "(pending)"
    await store.resolve_pending(db, fact_id=proposal.fact.id, approve=False, by=CG)
    current, meds = metformin(await memory_index.search(db, FAM, [ELDER], "sugar ki goli", limit=8))
    assert not any("250" in t for t in meds) and len(current) == 1 and "500" in current[0], meds


def scripted(answer: dict, seen: list):
    async def complete(role, **kw):
        seen.append(kw["messages"][0]["content"][0]["text"])
        return LLMReply(text=json.dumps(answer), tool_calls=[], model="fake")
    return complete


async def test_a_note_against_the_record_gives_exactly_one_question(db, at, monkeypatch):
    at("2026-10-11 20:00")
    await dose(db, 500)
    await dose(db, 1000)
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="health", title="Health",
                            body_md="- Takes Metformin 500 mg after dinner\n- Walks for twenty minutes after lunch")
    seen: list[str] = []
    monkeypatch.setattr(router, "complete", scripted({"contradictions": [{
        "record_key": "medicine:metformin", "note": "health", "line": "Takes Metformin 500 mg after dinner",
        "question": "My notes say Metformin 500 mg, but the care record says 1000 mg. Which is right now?"}]}, seen))
    assert await memory_upkeep.ask_about_contradictions(db, FAM, ELDER, CG) == 1
    assert await memory_upkeep.ask_about_contradictions(db, FAM, ELDER, CG) == 0, "asked once"
    loops = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "memory_check"))).scalars().all()
    assert len(loops) == 1 and loops[0].detail["kind"] == "contradiction" and loops[0].owner_id == CG
    assert "1000 mg" in loops[0].title and loops[0].detail["record_key"] == "medicine:metformin"
    assert "[medicine:metformin] Metformin 1000 mg after dinner" in seen[0] and "Takes Metformin 500 mg" in seen[0]
    # nothing was resolved on its own: the record and the note are as they were
    active = await store.active_fact(db, FAM, ELDER, "medicine:metformin")
    note = (await db.execute(select(MemoryNote).where(MemoryNote.family_id == FAM, MemoryNote.slug == "health"))).scalar_one()
    assert "1000" in active.text and "Metformin 500 mg" in note.body_md


async def test_a_note_naming_a_stopped_medicine_is_checked_and_nothing_is_guessed(db, at, monkeypatch):
    at("2026-10-11 20:00")
    await store.write_fact(db, family_id=FAM, subject_id=ELDER, domain="medicine", key="medicine:telma",
                           value={"name": "Telma", "dose": "40 mg"}, text="Telma 40 mg at night", source_kind="caregiver_said", stated_by=CG)
    await store.stop_fact(db, family_id=FAM, subject_id=ELDER, key="medicine:telma", reason="doctor stopped it",
                          source_kind="caregiver_said", stated_by=CG)
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="health", title="Health", body_md="- Takes Telma 40 at night for BP")
    seen: list[str] = []
    monkeypatch.setattr(router, "complete", scripted({"contradictions": [
        {"note": "health", "line": "Takes Telma 40 at night for BP", "question": "Is Telma still taken?"},  # no second side: dropped
        {"note": "health", "line": "a line that is not in any note", "record_key": "medicine:telma"}]}, seen))
    assert await memory_upkeep.ask_about_contradictions(db, FAM, ELDER, CG) == 0
    assert "CHECK THESE NOTE LINES" in seen[0] and "Takes Telma 40 at night for BP" in seen[0], "the stopped medicine is flagged to the model"


async def test_no_check_without_changes_and_none_when_the_model_is_unavailable(db, at, monkeypatch):
    at("2026-10-01 09:00")
    await dose(db, 500)
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="health", title="Health", body_md="- Takes Metformin 1000 mg")
    seen: list[str] = []
    monkeypatch.setattr(router, "complete", scripted({"contradictions": []}, seen))
    at("2026-10-04 09:00")
    assert await memory_upkeep.ask_about_contradictions(db, FAM, ELDER, CG) == 0 and not seen, "nothing changed: no model call"

    async def down(role, **kw):
        raise router.AllModelsFailed("over the cap")
    monkeypatch.setattr(router, "complete", down)
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="health", title="Health", body_md="- Takes Metformin 750 mg")
    assert await memory_upkeep.ask_about_contradictions(db, FAM, ELDER, CG) == 0


async def test_startup_cleanup_fixes_old_rows_once(db, at, meaning):
    at("2026-10-01 09:00")
    old = (await dose(db, 500)).fact
    at("2026-10-05 09:00")
    await dose(db, 1000)
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="family", title="Family", body_md="- Her son Rahul lives in Bangalore with his wife")
    # rows as the old code left them: the replaced dose still "(active)", a line no longer in its note, a dropped proposal
    vec = (await fake_embed(["x"]))[0]
    for key, source, ref, text in [("old-1", "fact", f"fact:{old.id}", f"(active) {old.text}"),
                                   ("old-2", "note", "note:family", "Family: Her son Rahul lives in Pune with his wife"),
                                   ("old-3", "fact", "fact:00000000-0000-0000-0000-000000000000", "(pending) Metformin 250 mg")]:
        db.add(memory_index.MemoryVector(key=key, family_id=FAM, subject_id=ELDER, source=source, ref=ref, text=text, at=old.recorded_at,
                                         importance=0.8, embedding=vec))
    await db.flush()
    first = await memory_index.cleanup_all(db)
    assert first == {"people": 1, "relabelled": 1, "removed": 2}, first
    assert await memory_index.cleanup_all(db) == {"people": 1, "relabelled": 0, "removed": 0}, "idempotent"
    rows = (await db.execute(select(memory_index.MemoryVector))).scalars().all()
    assert [(r.text, r.embedding is not None) for r in rows] == [("(earlier, until 2026-10-05) Metformin 500 mg after dinner", True)], \
        "relabelled in place, keeping its embedding"


async def test_two_people_with_the_same_note_name_are_kept_apart(db, at, meaning):
    at("2026-10-01 09:00")
    for who, line in [(ELDER, "- 2026-09-30: Visited the doctor for her knee"), (OTHER, "- 2026-09-30: Long walk in the park")]:
        await store.upsert_note(db, family_id=FAM, subject_id=who, slug="diary", title="Diary", body_md=line)
        await memory_index.nightly(db, FAM, who)
    await memory_index.unindex(db, FAM, "note:diary", subject_id=ELDER)
    left = (await db.execute(select(memory_index.MemoryVector.subject_id).where(memory_index.MemoryVector.family_id == FAM))).scalars().all()
    assert left == [OTHER]
