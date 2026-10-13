"""One memory is the truth (trust plan A1): what is still useful in the old stores (the backend's Mongo profile, usuals,
companion dates, voice setting, permissions, and the engine's own Instinct inbox) moves into the care record once, with
every row traceable (source_ref "import:<store>:<id>"), a rerun changing nothing, and undo through memory versions.
Every mapping row of the plan has a test here; the model's judgement (sorting the old words, the duplicate check) is a
scripted fake."""

import json
import re
import uuid

import pytest
from sqlalchemy import select

from app.care import dedupe, memory_unify, naming, rejected, store
from app.care.models import CareEvent, CareFact
from app.care.versions import MemoryVersion
from app.llm import router, spend
from app.llm.router import LLMReply

FAM = "fam-unify"
MAA = "elder-unify"
SON = "son-unify"


class Fake:
    """Sorting answers come from `sorting` (the old item's text → the model's decision); duplicate-check answers from
    `same` (the new key → the answer)."""

    def __init__(self):
        self.sorting: dict[str, dict] = {}
        self.same: dict[str, dict] = {}
        self.calls: list[str] = []

    async def complete(self, route, **kw):
        system = kw.get("system_stable") or ""
        system = " ".join(system) if isinstance(system, list) else system
        text = kw["messages"][-1]["content"][0]["text"]
        if system.startswith(memory_unify.SORT_PROMPT[:40]):
            self.calls.append("sort")
            old = text.split("\nOLD\n", 1)[1]
            items = []
            for m in re.finditer(r"^- id=(\S+?)( REJECTED)?: (.*)$", old, re.M):
                d = self.sorting.get(m.group(3), {"target": "skip"})
                items.append({"id": m.group(1), **d})
            return LLMReply(text=json.dumps({"items": items}), tool_calls=[], model="fake")
        if system.startswith(dedupe.SAME_PROMPT[:40]):
            key = re.search(r"\nNEW\n\[([^\]]+)\]", text).group(1)
            self.calls.append(f"same:{key}")
            return LLMReply(text=json.dumps(self.same.get(key, {"same_as": None})), tool_calls=[], model="fake")
        self.calls.append(system[:30])
        return LLMReply(text="{}", tool_calls=[], model="fake")


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    router.register_provider("fake", f)
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"], "judge_fast": ["fake:m"]}')
    monkeypatch.setenv("FACT_DEDUPE", "on")
    router.reset_breakers()
    spend.reset()
    yield f
    router._providers.pop("fake", None)
    router.reset_breakers()


def export(**person) -> dict:
    """What the backend tool export_legacy_memory returns for one family (one person here unless given)."""
    p = {"id": MAA, "role": "elder", "caregiver_ids": [SON], "schedules": [], **person}
    return {"family_id": FAM, "people": [p]}


def fact(fid: str, text: str, status: str = "learned", **kw) -> dict:
    return {"id": fid, "category": "health", "text": text, "status": status, "firstSeen": "2026-09-20T10:00:00Z", **kw}


async def active(db, key):
    return await store.active_fact(db, FAM, MAA, key)


async def run(db, data, apply=True):
    return await memory_unify.run_family(db, data, apply=apply)


# ── every mapping row ────────────────────────────────────────────────────────


async def test_a_caregiver_confirmed_or_edited_fact_becomes_a_pinned_caregiver_fact(db, fake):
    fake.sorting["Loves gardening in the morning"] = {"target": "fact", "domain": "preference", "name": "gardening",
                                                      "details": {"when": "morning"}, "sentence": "Loves gardening in the morning."}
    fake.sorting["Allergic to penicillin"] = {"target": "fact", "domain": "allergy", "name": "penicillin",
                                              "details": {"allergen": "penicillin"}, "sentence": "Allergic to penicillin."}
    rep = await run(db, export(profile={"id": "p1", "facts": [
        fact("f1", "Loves gardening in the morning", "caregiver_confirmed", editedBy=SON, pinned=True),
        fact("f2", "Allergic to penicillin", "caregiver_edited", editedBy=SON)]}))
    g, a = await active(db, "preference:gardening"), await active(db, "allergy:penicillin")
    assert g.source_kind == a.source_kind == "caregiver_said"
    assert g.pinned_at and a.pinned_at and g.stated_by == SON
    assert a.status == "active"  # a caregiver's health fact needs no second confirmation
    assert g.source_ref == "import:elder_profile:f1"
    assert rep["by_store"]["profile_fact"]["written_fact"] == 2


async def test_a_learned_fact_is_inferred_and_a_learned_health_fact_waits_for_a_caregiver(db, fake):
    fake.sorting["Likes ginger tea"] = {"target": "fact", "domain": "preference", "name": "ginger_tea", "sentence": "Likes ginger tea."}
    fake.sorting["Has high blood pressure"] = {"target": "fact", "domain": "condition", "name": "high blood pressure",
                                               "sentence": "Has high blood pressure."}
    await run(db, export(profile={"id": "p1", "facts": [fact("f1", "Likes ginger tea"), fact("f2", "Has high blood pressure")]}))
    tea = await active(db, "preference:ginger_tea")
    assert tea.source_kind == "inferred" and not tea.pinned_at
    assert await active(db, "condition:high_blood_pressure") is None
    pending = [f for f in await store.facts(db, FAM, MAA, statuses=("pending",)) if f.key == "condition:high_blood_pressure"]
    assert pending and pending[0].source_ref == "import:elder_profile:f2"


async def test_rejected_and_blocked_facts_go_on_the_rejected_list_so_they_never_come_back(db, fake):
    fake.sorting["Allergic to milk"] = {"target": "fact", "domain": "allergy", "name": "milk", "details": {"allergen": "milk"},
                                        "sentence": "Allergic to milk."}
    fake.sorting["Goes to the temple daily"] = {"target": "fact", "domain": "routine", "name": "temple", "sentence": "Goes to the temple daily."}
    rep = await run(db, export(profile={"id": "p1", "facts": [
        fact("f1", "Allergic to milk", "rejected", editedBy=SON, blocked=True),
        fact("f2", "Goes to the temple daily", "learned", blocked=True)]}))
    wrong = {f.key: f for f in await rejected.rejected(db, FAM, MAA)}
    assert set(wrong) == {"allergy:milk", "routine:temple"}
    assert wrong["allergy:milk"].note.startswith("rejected: ") and wrong["allergy:milk"].status == "stopped"
    assert await active(db, "allergy:milk") is None
    assert rep["by_store"]["profile_fact"]["written_rejected"] == 2
    # the duplicate check now refuses an inference that says it again (it is on the list)
    same = await dedupe.check(db, family_id=FAM, subject_id=MAA, domain="allergy", key="allergy:milk", text="Allergic to milk", value={})
    assert same and same.rejected


async def test_a_faded_fact_is_kept_as_history_in_the_earlier_note(db, fake):
    await run(db, export(profile={"id": "p1", "facts": [fact("f1", "Used to walk in the park every evening", "faded")]}))
    notes = {n.slug: n for n in await store.notes(db, FAM, [MAA])}
    assert "Used to walk in the park every evening" in notes["earlier"].body_md
    assert notes["earlier"].title == "Earlier (history only)"
    assert "2026-09-20" in notes["earlier"].body_md
    assert "sort" not in fake.calls  # history only: nothing for the model to read


async def test_the_old_name_settings_become_the_pinned_naming_fact(db, fake):
    rep = await run(db, export(profile={"id": "p1", "facts": [], "tuning": {"addressAs": "Amma", "avoidMaa": True}, "nameToUse": "Leela ji"}))
    f = await active(db, "naming:address_as")
    assert f.value == {"name": "Leela ji", "avoid": ["Maa"]}
    assert f.source_kind == "elder_said" and f.pinned_at
    assert (await naming.for_person(db, FAM, MAA)) == {"name": "Leela ji", "avoid": ["Maa"]}
    assert rep["by_store"]["profile_tuning"]["written_fact"] == 1


async def test_a_newer_naming_fact_keeps_its_name_and_only_gains_a_missing_avoid_word(db, fake):
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="naming", key="naming:address_as", value={"name": "Kamla ji", "avoid": []},
                           text="Call her Kamla ji", source_kind="caregiver_said", stated_by=SON)
    await run(db, export(profile={"id": "p1", "tuning": {"addressAs": "Amma", "avoidMaa": True}}))
    assert (await naming.for_person(db, FAM, MAA)) == {"name": "Kamla ji", "avoid": ["Maa"]}
    # nothing to add: the record already has it all
    rep = await run(db, export(profile={"id": "p2", "tuning": {"addressAs": "Amma"}}))
    assert rep["by_store"]["profile_tuning"] == {"in_engine": 1}


async def test_allergies_diet_and_medicines_from_the_old_profile(db, fake):
    fake.sorting["No salt in dal"] = {"target": "fact", "domain": "diet", "name": "low_salt", "details": {"rule": "low_salt"},
                                      "sentence": "No salt in dal."}
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="medicine", key="medicine:metformin", value={"name": "Metformin"},
                           text="Metformin 500 mg at 08:00", source_kind="caregiver_said", stated_by=SON)
    rep = await run(db, export(
        profile={"id": "p1", "allergies": ["peanuts"], "dietRules": ["No salt in dal"],
                 "medicines": [{"name": "Metformin 500mg", "time": "08:00"}, {"name": "Thyronorm", "dose": "50 mcg"},
                               {"name": "Amlodipine", "time": "21:00"}]},
        schedules=[{"title": "Thyronorm 50", "type": "MEDICINE"}]))
    assert (await active(db, "allergy:peanuts")).value == {"allergen": "peanuts"}
    assert (await active(db, "diet:low_salt")).source_kind == "import"
    pend = {f.key for f in await store.facts(db, FAM, MAA, statuses=("pending",))}
    assert "medicine:amlodipine" in pend  # nobody scheduled it: a caregiver confirms, no reminder starts on its own
    assert "medicine:thyronorm" not in pend and await active(db, "medicine:thyronorm") is None  # the backend schedule has it
    assert rep["by_store"]["profile_lists"] == {"written_fact": 2, "in_engine": 1, "in_schedules": 1, "written_pending": 1}


async def test_usuals_become_usual_preference_facts(db, fake):
    await run(db, export(usuals={"id": "u1", "items": [{"name": "Amul taaza milk", "partner": "blinkit", "count": 6}],
                                 "brandPrefs": [{"brand": "Tata", "item": "salt", "text": "Only Tata salt"}]}))
    milk, salt = await active(db, "preference:usual_amul_taaza_milk"), await active(db, "preference:usual_salt")
    assert milk.source_kind == salt.source_kind == "inferred"
    assert milk.value["partner"] == "blinkit" and salt.value["brand"] == "Tata"
    assert milk.source_ref == "import:elder_usuals:u1#item0"


async def test_companion_birthday_and_important_dates_become_occasion_facts(db, fake):
    await run(db, export(companion={"id": "c1", "birthday": "1950-03-14", "importantDates": [{"label": "Wedding anniversary", "date": "1972-11-02"}]}))
    b, w = await active(db, "occasion:birthday"), await active(db, "occasion:wedding_anniversary")
    assert b.value["date"] == "1950-03-14" and w.value["date"] == "1972-11-02"
    assert b.source_kind == w.source_kind == "caregiver_said"


async def _instinct(db) -> dict:
    from app.models.entities import Elder, Family, FamilyMemory, MemoryEntity, MemoryProfile

    fam, eld = Family(id=uuid.uuid4(), name="t"), None
    db.add(fam)
    await db.flush()
    eld = Elder(id=uuid.uuid4(), family_id=fam.id, display_name="t", slug=f"e{uuid.uuid4().hex[:6]}")
    db.add(eld)
    await db.flush()
    ent = MemoryEntity(family_id=fam.id, elder_id=eld.id, slug="person-sunita", kind="person", title="Sunita",
                       body_md="---\nslug: person-sunita\n---\n# Sunita\n- Daughter Sunita lives in Pune (2026-09-01)\n")
    db.add(ent)
    await db.flush()
    db.add_all([
        FamilyMemory(family_id=fam.id, elder_id=eld.id, content="Grew up in Varanasi by the ghats", topic="childhood"),
        FamilyMemory(family_id=fam.id, elder_id=eld.id, content="Knee pain when climbing stairs", topic="knee"),
        FamilyMemory(family_id=fam.id, elder_id=eld.id, content="Daughter Sunita lives in Pune", topic="sunita", entity_id=ent.id),
        MemoryProfile(family_id=fam.id, elder_id=eld.id, body_md="# Profile\n- Daughter Sunita lives in Pune\n- Enjoys old Hindi songs\n"),
    ])
    await db.flush()
    return {"aiFamilyId": str(fam.id), "aiElderId": str(eld.id)}


async def test_the_instinct_inbox_becomes_notes_and_a_health_statement_a_pending_fact(db, fake):
    link = await _instinct(db)
    fake.sorting.update({
        "Grew up in Varanasi by the ghats": {"target": "note", "note": "life", "name": "childhood", "sentence": "Grew up in Varanasi by the ghats."},
        "Knee pain when climbing stairs": {"target": "fact", "domain": "condition", "name": "knee pain", "sentence": "Knee pain when climbing stairs."},
        "Daughter Sunita lives in Pune (2026-09-01)": {"target": "note", "note": "about", "name": "daughter", "sentence": "Daughter Sunita lives in Pune."},
        "Daughter Sunita lives in Pune": {"target": "skip", "same_as": "x"},
        "Enjoys old Hindi songs": {"target": "note", "note": "about", "name": "music", "sentence": "Enjoys old Hindi songs."},
    })
    rep = await run(db, export(instinct=link))
    notes = {n.slug: n.body_md for n in await store.notes(db, FAM, [MAA])}
    assert "Grew up in Varanasi" in notes["life"]
    assert "Sunita lives in Pune" in notes["about"] and "old Hindi songs" in notes["about"]
    assert notes["about"].count("Sunita") == 1  # the profile repeats the entity: kept once
    knee = [f for f in await store.facts(db, FAM, MAA, statuses=("pending",)) if f.key == "condition:knee_pain"]
    assert knee and knee[0].source_kind == "inferred" and knee[0].source_ref.startswith("import:instinct_inbox:")
    assert rep["by_store"]["instinct"]["written_note"] == 3
    assert rep["by_store"]["instinct"]["written_pending"] == 1


async def test_the_voice_setting_moves_into_the_language_fact_when_the_record_has_none(db, fake):
    rep = await run(db, export(voice={"id": "v1", "language": "bn", "script": "native", "mode": "always", "updatedBy": MAA}))
    f = await active(db, "language:preferred")
    assert f.value == {"language": "bn", "script": "native", "voice": "always"} and f.source_kind == "elder_said"
    assert rep["by_store"]["voice"] == {"written_fact": 1}


async def test_the_voice_setting_joins_a_newer_language_fact_without_changing_the_language(db, fake):
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="language", key="language:preferred", value={"language": "hi"},
                           text="Speaks Hindi", source_kind="elder_said", stated_by=MAA)
    await run(db, export(voice={"id": "v1", "language": "en", "mode": "never"}))
    assert (await active(db, "language:preferred")).value == {"language": "hi", "voice": "never"}


async def test_permissions_are_compared_with_the_boundary_policy_and_never_copied(db, fake):
    data = export()
    data["permissions"] = {"groceries": True, "food": True, "medicines": True, "rides": True, "spendSoftLimitInr": None}
    assert (await run(db, data))["permissions"] == "equal"
    data["permissions"] = {"groceries": False, "spendSoftLimitInr": 300}
    rep = await run(db, data)
    assert rep["permissions"] == "differs"
    assert await store.active_fact(db, FAM, "family", "boundary:policy") is None


async def test_wellbeing_days_and_daily_snapshots_are_not_moved(db, fake):
    data = export(wellbeingDays=[{"dayKey": "2026-10-01", "mood": "low"}], dailySnapshots=[{"summary": "x"}])
    rep = await run(db, data)
    assert rep["items"] == 0 and not await store.facts(db, FAM, MAA)


# ── the duplicate check, idempotence, dry run, undo ──────────────────────────


async def test_the_duplicate_check_runs_before_every_insert_and_the_newer_fact_wins(db, fake):
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="allergy", key="allergy:groundnut", value={"allergen": "groundnut"},
                           text="Allergic to groundnut", source_kind="caregiver_said", stated_by=SON)
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="diet", key="diet:diabetic", value={},
                           text="Diabetic diet", source_kind="caregiver_said", stated_by=SON)
    fake.sorting["Peanut allergy"] = {"target": "fact", "domain": "allergy", "name": "peanut", "sentence": "Peanut allergy."}
    fake.sorting["No sugar, and only two rotis at night"] = {"target": "fact", "domain": "diet", "name": "no sugar",
                                                             "sentence": "No sugar, and only two rotis at night."}
    fake.same["allergy:peanut"] = {"same_as": "allergy:groundnut", "covered": True}
    fake.same["diet:no_sugar"] = {"same_as": "diet:diabetic", "covered": False, "sentence": "Diabetic diet: no sugar, two rotis at night."}
    rep = await run(db, export(profile={"id": "p1", "facts": [fact("f1", "Peanut allergy"), fact("f2", "No sugar, and only two rotis at night")]}))
    assert "same:allergy:peanut" in fake.calls and "same:diet:no_sugar" in fake.calls
    assert await active(db, "allergy:peanut") is None
    assert (await active(db, "diet:diabetic")).text == "Diabetic diet"  # never written over by the older profile
    notes = {n.slug: n.body_md for n in await store.notes(db, FAM, [MAA])}
    assert "two rotis at night" in notes["earlier"]  # the extra detail is kept, as history
    assert rep["by_store"]["profile_fact"] == {"in_engine": 1, "written_detail_note": 1}


async def test_a_rerun_changes_nothing(db, fake):
    fake.sorting["Likes ginger tea"] = {"target": "fact", "domain": "preference", "name": "ginger_tea", "sentence": "Likes ginger tea."}
    data = export(profile={"id": "p1", "facts": [fact("f1", "Likes ginger tea"), fact("f2", "Used to sing bhajans", "faded")],
                           "allergies": ["dust"]}, companion={"id": "c1", "birthday": "1950-03-14"})
    await run(db, data)
    facts_before = [(f.key, f.status, str(f.id)) for f in await store.facts(db, FAM, MAA, statuses=("active", "pending", "stopped"))]
    notes_before = {n.slug: (n.version, n.body_md) for n in await store.notes(db, FAM, [MAA])}
    versions_before = len((await db.execute(select(MemoryVersion).where(MemoryVersion.family_id == FAM))).scalars().all())
    rep = await run(db, data)
    assert {k: v for k, v in rep["by_store"].items()} == {"dates": {"already_imported": 1}, "profile_fact": {"already_imported": 2},
                                                          "profile_lists": {"already_imported": 1}}
    assert [(f.key, f.status, str(f.id)) for f in await store.facts(db, FAM, MAA, statuses=("active", "pending", "stopped"))] == facts_before
    assert {n.slug: (n.version, n.body_md) for n in await store.notes(db, FAM, [MAA])} == notes_before
    assert len((await db.execute(select(MemoryVersion).where(MemoryVersion.family_id == FAM))).scalars().all()) == versions_before
    assert fake.calls.count("sort") == 1  # nothing left to sort the second time


async def test_the_dry_run_writes_nothing_and_counts_what_it_would_do(db, fake):
    fake.sorting["Likes ginger tea"] = {"target": "fact", "domain": "preference", "name": "ginger_tea", "sentence": "Likes ginger tea."}
    rep = await run(db, export(profile={"id": "p1", "facts": [fact("f1", "Likes ginger tea"), fact("f2", "Used to sing", "faded")]}), apply=False)
    assert rep["by_store"]["profile_fact"] == {"write": 1, "note_line": 1}
    assert not await store.facts(db, FAM, MAA, statuses=("active", "pending", "stopped"))
    assert not await store.notes(db, FAM, [MAA])
    assert not (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM))).scalars().all()


async def test_the_report_holds_counts_only_never_what_was_said(db, fake):
    fake.sorting["Secret recipe: kadhi with extra hing"] = {"target": "note", "note": "about", "name": "food", "sentence": "Secret recipe: kadhi with extra hing."}
    rep = await run(db, export(profile={"id": "p1", "facts": [fact("f1", "Secret recipe: kadhi with extra hing")], "tuning": {"addressAs": "Amma"}},
                               companion={"id": "c1", "birthday": "1950-03-14"}), apply=False)
    flat = json.dumps(rep)
    for word in ("kadhi", "hing", "Amma", "1950", MAA):
        assert word not in flat


async def test_every_row_and_ledger_entry_carries_its_import_ref(db, fake):
    fake.sorting["Likes ginger tea"] = {"target": "fact", "domain": "preference", "name": "ginger_tea", "sentence": "Likes ginger tea."}
    await run(db, export(profile={"id": "p1", "facts": [fact("f1", "Likes ginger tea"), fact("f2", "Old line", "faded")]}))
    rows = (await db.execute(select(CareFact).where(CareFact.family_id == FAM))).scalars().all()
    assert [r.source_ref for r in rows] == ["import:elder_profile:f1"]
    refs = sorted(e.ref for e in (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "memory_import"))).scalars())
    assert refs == ["import:elder_profile:f1", "import:elder_profile:f2"]
    v = (await db.execute(select(MemoryVersion).where(MemoryVersion.family_id == FAM, MemoryVersion.kind == "fact"))).scalars().first()
    assert v.source == "import" and v.actor_id == "kavach"


async def test_undo_takes_every_import_back_through_versions_and_the_move_can_run_again(db, fake):
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="naming", key="naming:address_as", value={"name": "Kamla ji"},
                           text="Call her Kamla ji", source_kind="caregiver_said", stated_by=SON)
    await store.upsert_note(db, family_id=FAM, subject_id=MAA, slug="about", title="About", body_md="- Loves her garden")
    fake.sorting["Likes ginger tea"] = {"target": "fact", "domain": "preference", "name": "ginger_tea", "sentence": "Likes ginger tea."}
    fake.sorting["Allergic to milk"] = {"target": "fact", "domain": "allergy", "name": "milk", "sentence": "Allergic to milk."}
    fake.sorting["Has asthma"] = {"target": "fact", "domain": "condition", "name": "asthma", "sentence": "Has asthma."}
    fake.sorting["Plays carrom"] = {"target": "note", "note": "about", "name": "games", "sentence": "Plays carrom."}
    data = export(profile={"id": "p1", "facts": [fact("f1", "Likes ginger tea"), fact("f2", "Allergic to milk", "rejected", blocked=True),
                                                 fact("f3", "Has asthma"), fact("f4", "Plays carrom")],
                           "tuning": {"avoidMaa": True}})
    await run(db, data)
    assert (await naming.for_person(db, FAM, MAA))["avoid"] == ["Maa"]
    out = await memory_unify.undo_family(db, FAM)
    assert out["undone"] == 5
    assert await active(db, "preference:ginger_tea") is None
    assert not [f for f in await store.facts(db, FAM, MAA, statuses=("pending",))]
    assert not await rejected.rejected(db, FAM, MAA)
    assert (await naming.for_person(db, FAM, MAA)) == {"name": "Kamla ji", "avoid": []}  # back to what it was
    notes = {n.slug: n.body_md for n in await store.notes(db, FAM, [MAA])}
    assert notes["about"] == "- Loves her garden"  # the line the move added is gone, the family's line stays
    undos = (await db.execute(select(MemoryVersion).where(MemoryVersion.family_id == FAM, MemoryVersion.undoes.is_not(None)))).scalars().all()
    assert undos  # every undo is itself a version
    # the move can run again after an undo
    rep = await run(db, data)
    assert rep["by_store"]["profile_fact"]["written_fact"] == 1


async def test_undo_leaves_a_fact_a_person_changed_since(db, fake):
    fake.sorting["Likes ginger tea"] = {"target": "fact", "domain": "preference", "name": "ginger_tea", "sentence": "Likes ginger tea."}
    await run(db, export(profile={"id": "p1", "facts": [fact("f1", "Likes ginger tea")]}))
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="preference", key="preference:ginger_tea", value={"sugar": "none"},
                           text="Likes ginger tea without sugar", source_kind="elder_said", stated_by=MAA)
    out = await memory_unify.undo_family(db, FAM)
    assert out.get("changed_since") == 1
    assert (await active(db, "preference:ginger_tea")).text == "Likes ginger tea without sugar"


async def test_a_pinned_fact_is_never_overwritten_by_an_import(db, fake):
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="preference", key="preference:tea", value={"how": "no sugar"},
                           text="Tea without sugar", source_kind="elder_said", stated_by=MAA)
    fake.sorting["Likes tea with sugar"] = {"target": "fact", "domain": "preference", "name": "tea", "sentence": "Likes tea with sugar."}
    rep = await run(db, export(profile={"id": "p1", "facts": [fact("f1", "Likes tea with sugar")]}))
    assert (await active(db, "preference:tea")).text == "Tea without sugar"
    assert rep["by_store"]["profile_fact"] == {"in_engine": 1}


async def test_a_sorting_failure_guesses_nothing(db, fake, monkeypatch):
    async def down(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(fake, "complete", down)
    rep = await run(db, export(profile={"id": "p1", "facts": [fact("f1", "Likes ginger tea")]}))
    assert rep["by_store"]["profile_fact"] == {"unsorted": 1}
    assert not await store.facts(db, FAM, MAA)


# ── the job, the person view, the legacy-writes count ────────────────────────


class Host:
    shadow = False

    def __init__(self, data: dict):
        self.data = data
        self.calls: list[tuple[str, dict]] = []

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        self.calls.append((tool, args))
        if tool == "export_legacy_memory" and args.get("list"):
            return {"families": [{"family_id": FAM, "actor_id": MAA}, {"family_id": "other-fam", "actor_id": "x"}]}
        if tool == "export_legacy_memory":
            return self.data if family_id == FAM else {"family_id": family_id, "people": []}
        return {"ok": True}


async def test_the_job_runs_every_family_in_its_own_transaction_and_reports_counts(db, fake, monkeypatch):
    fake.sorting["Likes ginger tea"] = {"target": "fact", "domain": "preference", "name": "ginger_tea", "sentence": "Likes ginger tea."}
    host = Host(export(profile={"id": "p1", "facts": [fact("f1", "Likes ginger tea")]}))

    class Sessions:
        def __call__(self):
            return self

        async def __aenter__(self):
            return db

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(db, "commit", db.flush)  # the test's transaction is rolled back at the end
    dry = await memory_unify.run(Sessions(), host, apply=False)
    assert dry["families"]["fam-unif"]["by_store"]["profile_fact"] == {"write": 1}
    assert dry["families"]["other-fa"]["items"] == 0
    assert await active(db, "preference:ginger_tea") is None
    done = await memory_unify.run(Sessions(), host, apply=True, family="fam-")
    assert list(done["families"]) == ["fam-unif"]
    assert (await active(db, "preference:ginger_tea")).source_ref == "import:elder_profile:f1"
    undone = await memory_unify.run(Sessions(), host, apply=True, undo=True, family="fam-")
    assert undone["families"]["fam-unif"]["undone"] == 1


async def test_the_person_view_gives_the_name_rule_and_the_language_and_voice(db):
    from app.api.one_memory import person

    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="naming", key="naming:address_as", value={"name": "Leela ji", "avoid": ["Maa"]},
                           text="Call her Leela ji", source_kind="elder_said", stated_by=MAA)
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="language", key="language:preferred",
                           value={"language": "hi", "dialect": "mwr", "voice": "always"}, text="Speaks Marwari", source_kind="elder_said", stated_by=MAA)
    out = await person(FAM, MAA, db)
    assert out == {"name": "Leela ji", "avoid": ["Maa"], "language": "hi", "dialect": "mwr", "script": None, "voice": "always", "known": True}
    assert (await person(FAM, "nobody", db))["known"] is False


async def test_a_naming_or_language_change_is_pushed_to_the_backend_after_the_commit(db, monkeypatch):
    from app.care import person_sync

    monkeypatch.setenv("PERSON_SYNC", "on")
    pushed: list[tuple[str, str]] = []

    async def notify(family_id, subject_id, host=None):
        pushed.append((family_id, subject_id))
        return True

    monkeypatch.setattr(person_sync, "notify", notify)
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="naming", key="naming:address_as", value={"name": "Leela ji"},
                           text="Call her Leela ji", source_kind="elder_said", stated_by=MAA)
    await store.write_fact(db, family_id=FAM, subject_id=MAA, domain="diet", key="diet:low_salt", value={}, text="Low salt",
                           source_kind="elder_said", stated_by=MAA)
    await store.write_fact(db, family_id="shadow:" + FAM, subject_id=MAA, domain="naming", key="naming:address_as", value={"name": "X"},
                           text="X", source_kind="elder_said", stated_by=MAA)
    assert pushed == []  # never from inside the transaction
    await db.commit()
    import asyncio

    await asyncio.sleep(0)
    assert pushed == [(FAM, MAA)]  # once, only for the real family's naming change


async def test_the_legacy_writes_count_sees_a_new_instinct_row(db):
    from app.api.one_memory import legacy_writes

    before = (await legacy_writes(db, 24))["counts"]["family_memories"]
    await _instinct(db)
    out = await legacy_writes(db, 24)
    assert out["counts"]["family_memories"] == before + 3 and out["total"] >= 3
