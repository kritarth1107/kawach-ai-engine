"""The background re-read (app.care.extract) and confirm-before-save (review 17 Oct 2026): what a person said no to, or what
Saheli read back and is still waiting on, is never saved by the re-read; health it hears waits for a caregiver; a reading
is put to a caregiver, not logged; low-risk non-health facts still save."""

import json

from sqlalchemy import select

from app.brain import confirm
from app.care import extract, store
from app.care.models import CareEvent, CareFact, OpenLoop
from app.llm import router
from app.llm.router import LLMReply

FAM = "fam-reread-hold"
ME = {"id": "self-rh", "name": "Kittu", "role": "primary caregiver"}


class Fake:
    def __init__(self, out):
        self.out = out

    async def complete(self, route, **kw):
        return LLMReply(text=json.dumps(self.out), tool_calls=[], model="fake")


def use(monkeypatch, out):
    router.register_provider("fake", Fake(out))
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"]}')
    router.reset_breakers()


def fact(domain, name, sentence, details=None):
    return {"about": ME["id"], "domain": domain, "name": name, "sentence": sentence, "details": details or {}, "evidence": sentence}


def held(tool, args, key="pending_saves"):
    sig = confirm.signature(tool, args)
    return {key: [{"id": confirm.pending_id(sig), "tool": tool, "sig": sig, "args": args, "what": "x", "for": ME["name"]}]}


async def facts(db, statuses=("active",)):
    return {f.key for f in await store.facts(db, FAM, ME["id"], statuses=statuses)}


async def test_declined_and_waiting_saves_are_never_saved_by_the_re_read(db, at, monkeypatch):
    at("2026-10-17 15:00")
    await store.save_roster(db, FAM, ME, [ME])
    declined = {"domain": "dish", "name": "khichdi", "about": ME["id"], "sentence": "Likes khichdi", "details": {}}
    waiting = {"domain": "routine", "name": "gym", "about": ME["id"], "sentence": "Goes to Cult gym", "details": {}}
    await store.add_turn(db, family_id=FAM, thread_id=ME["id"], role="user", text="I like khichdi. I go to Cult gym. My sister is Asha.")
    await store.add_turn(db, family_id=FAM, thread_id=ME["id"], role="assistant", text="Save that you like khichdi?",
                         meta=held("remember", declined, "declined_saves"))
    await store.add_turn(db, family_id=FAM, thread_id=ME["id"], role="assistant", text="Save Cult gym as your gym?",
                         meta=held("remember", waiting))
    use(monkeypatch, {"facts": [fact("dish", "khichdi", "Likes khichdi"), fact("routine", "gym", "Goes to Cult gym"),
                                fact("family", "sister", "Her sister is Asha", {"name": "Asha"})],
                      "stops": [], "notes": [], "readings": []})
    try:
        counts = await extract.extract_family(db, FAM)
    finally:
        router._providers.pop("fake", None)
    saved = await facts(db)
    assert "dish:khichdi" not in saved and "routine:gym" not in saved
    assert counts.get("held") == 2
    assert any(k.startswith("family:") for k in saved), "a low-risk non-health fact still saves"


async def test_health_heard_by_the_re_read_waits_for_a_caregiver_and_a_health_stop_is_not_applied(db, at, monkeypatch):
    at("2026-10-17 15:00")
    await store.save_roster(db, FAM, ME, [ME])
    await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="medicine", key="medicine:telma_40",
                           value={"name": "Telma 40"}, text="Takes Telma 40", source_kind="caregiver_said", stated_by=ME["id"])
    await store.add_turn(db, family_id=FAM, thread_id=ME["id"], role="user", text="started dolo, stopped telma, sugar was 180 today")
    use(monkeypatch, {"facts": [fact("medicine", "Dolo 650", "Takes Dolo 650", {"name": "Dolo 650"})],
                      "stops": [{"about": ME["id"], "domain": "medicine", "name": "telma_40", "evidence": "stopped telma"}],
                      "notes": [], "readings": [{"about": ME["id"], "kind": "sugar", "value": "180", "unit": "mg/dL"}]})
    try:
        counts = await extract.extract_family(db, FAM)
    finally:
        router._providers.pop("fake", None)
    assert not [k for k in await facts(db) if k.startswith("medicine:dolo")] and [k for k in await facts(db, ("pending",)) if k.startswith("medicine:dolo")]
    assert "medicine:telma_40" in await facts(db), "a re-read never stops a medicine"
    assert counts.get("stops_held") == 1 and not counts.get("stops")
    assert not (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "vital"))).first()
    loops = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "memory_check"))).scalars().all()
    assert len(loops) == 1 and "sugar 180" in loops[0].title
    assert (await db.execute(select(CareFact).where(CareFact.family_id == FAM, CareFact.status == "pending"))).first()
