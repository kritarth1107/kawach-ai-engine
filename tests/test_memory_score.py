"""Trust plan A2: the nightly memory score (5 truths; stored, recalled, used; the minimum across people). Scripted models."""

import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care import store, versions
from app.care.models import FamilyRoster, OpenLoop
from app.core import clock
from app.learn.models import ReplyLog
from app.llm import router, spend
from app.llm.router import LLMReply
from app.quality import memory_score
from app.quality.models import MemoryScore, ReplayCase, ReplyVerdict

FAM = "fam-ms"
ELDER = {"id": "e-ms", "name": "Kamla Sharma", "role": "elder"}
CG = {"id": "c-ms", "name": "Ankit Sharma", "role": "primary caregiver"}
DAY = "2026-10-05"
GOOD = {"name_to_use": "Amma", "never_call": ["Maa"], "medicines": [{"name": "Metformin", "dose": "500 mg", "times": ["21:00"]}],
        "allergies": ["peanut"], "diet": ["no sugar"], "latest_change": {"what": "Metformin time", "new": "21:00"}}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


class Brain:
    """Answers the recall question with `answer`; the equivalence check with `same`."""

    def __init__(self, answer, same=True):
        self.answer, self.same, self.calls = answer, same, []

    async def complete(self, route, *, system_stable, system_dynamic="", messages, tools=None, **kw):
        last = "".join(c.get("text", "") for c in messages[-1]["content"])
        self.calls.append({"tools": tools, "system": "\n".join(system_stable), "dynamic": system_dynamic, "last": last})
        if "field by field" in "\n".join(system_stable):
            items = json.loads(last)
            return LLMReply(text=json.dumps({"items": [{"field": i["field"], "same": self.same} for i in items]}), tool_calls=[], model="fake")
        return LLMReply(text=json.dumps(self.answer), tool_calls=[], model="fake")


@pytest.fixture
def brain(monkeypatch):
    def use(answer=GOOD, same=True):
        b = Brain(answer, same)
        router.register_provider("fake", b)
        return b

    monkeypatch.setenv("MODEL_ROUTES", json.dumps({r: ["fake:b"] for r in ("brain", "judge", "judge_fast", "extract")}))
    spend.reset()
    router.reset_breakers()
    yield use
    router._providers.pop("fake", None)


async def fact(db, key, value, text, *, source="caregiver_said", by="c-ms", subject="e-ms"):
    with versions.attribution(actor_id=by, source="whatsapp"):
        return await store.write_fact(db, family_id=FAM, subject_id=subject, domain=key.split(":")[0], key=key, value=value, text=text,
                                      source_kind=source, stated_by=by)


async def setup(db, at, *, members=(ELDER, CG)):
    at("2026-09-20 10:00")
    await store.save_roster(db, FAM, ELDER, list(members))
    await fact(db, "naming:address_as", {"name": "Amma", "avoid": ["Maa"]}, "Call her Amma, never Maa", source="elder_said", by="e-ms")
    await fact(db, "medicine:metformin", {"name": "Metformin", "dose": "500 mg", "times": ["08:00"]}, "Metformin 500 mg at 08:00")
    await fact(db, "allergy:peanut", {"allergen": "peanut"}, "Allergic to peanuts")
    await fact(db, "diet:no_sugar", {"rule": "no sugar"}, "No sugar")
    at("2026-10-01 10:00")
    await fact(db, "medicine:metformin", {"times": ["21:00"]}, "Metformin 500 mg at 21:00")  # the latest correction
    at("2026-10-06 02:30")
    await db.flush()
    return await db.get(FamilyRoster, FAM)


async def reply(db, text, *, to="e-ms", said="", at_="2026-10-05 10:00"):
    from datetime import datetime

    when = datetime.fromisoformat(at_).replace(tzinfo=clock.IST)
    r = ReplyLog(family_id=FAM, thread_id=to, at=when, kind="reply", situation="chit_chat", speaker_role="elder", lang="latin-indic",
                 user_text=said, text=text, text_len=len(text), tools=[])
    db.add(r)
    await db.flush()
    return r


async def test_all_truths_stored_recalled_and_no_conversation_is_no_data(db, at, brain):
    r = await setup(db, at)
    b = brain()
    out = await memory_score.score_family(db, r, DAY)
    p = out["per_person"][0]
    assert p["stored"] == 100.0 and p["recalled"] == 100.0, p["failures"]
    assert p["used"] is None and p["score"] == 100.0 and p["unknowns"] == []  # no messages: "no data", never 100
    assert all(not c["tools"] for c in b.calls)  # no tools in the recall check
    assert "[medicine:metformin] Metformin 500 mg at 21:00" in b.calls[0]["system"]  # exactly the turn's care record
    fam = (await db.execute(select(MemoryScore).where(MemoryScore.subject_id == "family"))).scalars().one()
    assert fam.score == 100.0 and fam.used is None


async def test_each_missing_truth_is_unknown_and_one_question_not_a_failure(db, at, brain):
    at("2026-10-05 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, CG])
    await fact(db, "medicine:telma", {"name": "Telma", "dose": "40 mg", "times": ["09:00"]}, "Telma 40 mg at 09:00")
    at("2026-10-06 02:30")
    r = await db.get(FamilyRoster, FAM)
    brain({"medicines": [{"name": "Telma", "dose": "40 mg", "times": ["09:00"]}]})
    out = await memory_score.score_family(db, r, DAY)
    p = out["per_person"][0]
    assert p["unknowns"] == ["name", "allergies", "diet"] and not [f for f in p["failures"] if f["part"] != "stored" or not f["ok"] and "allerg" in f["check"]]
    assert p["recalled"] == 100.0 and p["asked"] == "allergies"
    loops = (await db.execute(select(OpenLoop).where(OpenLoop.kind == "memory_check"))).scalars().all()
    assert len(loops) == 1 and loops[0].owner_id == "c-ms" and "allergies" in loops[0].title.lower()
    out2 = await memory_score.score_family(db, r, DAY)  # one question at a time: nothing new while it is open
    assert out2["per_person"][0]["asked"] is None
    assert len((await db.execute(select(OpenLoop).where(OpenLoop.kind == "memory_check"))).scalars().all()) == 1


async def test_a_duplicate_fails_stored(db, at, brain):
    r = await setup(db, at)
    brain()
    groups = {"e-ms": [{"domain": "allergy", "keep": "allergy:peanut", "drop": ["allergy:groundnut"], "text": "x"}]}
    out = await memory_score.score_family(db, r, DAY, groups=groups)
    p = out["per_person"][0]
    assert any(f["check"] == "allergies.once" for f in p["failures"] if f["part"] == "stored") and p["stored"] < 100


async def test_a_correction_not_pinned_fails_stored(db, at, brain):
    r = await setup(db, at)
    row = await store.active_fact(db, FAM, "e-ms", "medicine:metformin")
    row.pinned_at = None
    await db.flush()
    brain()
    p = (await memory_score.score_family(db, r, DAY))["per_person"][0]
    assert [f["check"] for f in p["failures"] if f["part"] == "stored"] == ["correction.pinned"]


async def test_context_missing_a_truth_fails_stored(db, at, brain):
    r = await setup(db, at)
    brain()

    async def lossy(session, roster, sid, text):  # a retriever or ranking bug: the allergy never reaches the brain
        stable, dynamic, msgs = await memory_score.real_context(session, roster, sid, text)
        return [stable[0], "\n".join(ln for ln in stable[1].splitlines() if "allergy:peanut" not in ln)], dynamic, msgs

    p = (await memory_score.score_family(db, r, DAY, context_fn=lossy))["per_person"][0]
    stored = [f for f in p["failures"] if f["part"] == "stored"]
    assert [f["check"] for f in stored] == ["allergies.in_context"] and "allergy:peanut" in stored[0]["why"]


@pytest.mark.parametrize("field,wrong", [
    ("name_to_use", {"name_to_use": "Kamla"}),
    ("never_call", {"never_call": []}),
    ("medicines", {"medicines": [{"name": "Metformin", "dose": "500 mg", "times": ["08:00"]}]}),  # the old time
    ("medicines", {"medicines": [{"name": "Metformin", "dose": "1000 mg", "times": ["21:00"]}]}),
    ("medicines", {"medicines": [*GOOD["medicines"], {"name": "Telma", "dose": "40 mg", "times": ["09:00"]}]}),  # an invented one
    ("allergies", {"allergies": ["shellfish"]}),
    ("diet", {"diet": []}),
    ("latest_change", {"latest_change": None}),
])
async def test_a_recall_answer_wrong_in_each_field(db, at, brain, field, wrong):
    r = await setup(db, at)
    brain({**GOOD, **wrong})
    p = (await memory_score.score_family(db, r, DAY))["per_person"][0]
    bad = [f["field"] for f in p["failures"] if f["part"] == "recalled"]
    assert bad == [field] and p["recalled"] < 100
    case = (await db.execute(select(ReplayCase).where(ReplayCase.kind == f"recall_{field}"))).scalars().one()
    assert case.assertion[0]["type"] == "recall" and case.message.startswith("[Memory check")


async def test_free_text_fields_settled_by_the_equivalence_check(db, at, brain):
    r = await setup(db, at)
    brain({**GOOD, "diet": ["avoid sweets"]}, same=False)
    p = (await memory_score.score_family(db, r, DAY))["per_person"][0]
    assert {f["field"] for f in p["failures"] if f["part"] == "recalled"} == {"diet", "latest_change"}


async def test_a_real_reply_using_an_avoided_name_fails_used(db, at, brain):
    r = await setup(db, at)
    await reply(db, "Maa, aapki dawai ka time ho gaya")
    await reply(db, "Amma, Metformin raat 9 baje (21:00) 🙏", at_="2026-10-05 20:50")
    brain()
    p = (await memory_score.score_family(db, r, DAY))["per_person"][0]
    used = [f for f in p["failures"] if f["part"] == "used"]
    assert p["used"] == 50.0 and len(used) == 1 and "never-use name 'Maa'" in used[0]["problems"][0]
    case = (await db.execute(select(ReplayCase).where(ReplayCase.kind == "used_name"))).scalars().one()
    assert {"type": "not_contains", "words": ["Maa"]} in case.assertion and case.bad_reply.startswith("Maa,")


async def test_a_wrong_medicine_time_or_dose_fails_used(db, at, brain):
    r = await setup(db, at)
    await reply(db, "Amma, Metformin 8:00 am pe le lijiye", to="c-ms")  # about the elder, to the caregiver
    await reply(db, "Amma, Metformin 1000 mg le lijiye", at_="2026-10-05 11:00")
    brain()
    p = (await memory_score.score_family(db, r, DAY))["per_person"][0]
    probs = [x for f in p["failures"] if f["part"] == "used" for x in f["problems"]]
    assert any("21:00" in x for x in probs) and any("1000 mg" in x for x in probs) and p["used"] == 0.0


async def test_a_reply_using_a_rejected_fact_fails_used(db, at, brain):
    r = await setup(db, at)
    at("2026-10-03 10:00")
    with versions.attribution(actor_id="saheli", source="nightly"):
        w = await store.write_fact(db, family_id=FAM, subject_id="e-ms", domain="allergy", key="allergy:shellfish", value={"allergen": "shellfish"},
                                   text="Allergic to shellfish", source_kind="inferred")
    pending = w.row if w.result == "pending" else None
    if pending is None:  # a new inferred health fact is active; a caregiver stops it as wrong
        with versions.attribution(actor_id="c-ms", source="whatsapp"):
            await store.stop_fact(db, family_id=FAM, subject_id="e-ms", key="allergy:shellfish", reason="rejected: not true", source_kind="caregiver_said",
                                  stated_by="c-ms")
    else:
        with versions.attribution(actor_id="c-ms", source="dashboard"):
            await store.resolve_pending(db, fact_id=pending.id, approve=False, by="c-ms")
    at("2026-10-06 02:30")
    await reply(db, "Shellfish se door rahiye, Amma")
    await reply(db, "Haan Amma, shellfish wali baat galat thi", said="kya mujhe shellfish se allergy hai?", at_="2026-10-05 12:00")
    brain({**GOOD, "latest_change": {"what": "shellfish allergy is wrong", "new": "no shellfish allergy"}})
    p = (await memory_score.score_family(db, r, DAY))["per_person"][0]
    used = [f for f in p["failures"] if f["part"] == "used"]
    assert len(used) == 1 and "which the family said is wrong" in used[0]["problems"][0]  # the second was asked about it


async def test_judge_paraphrase_counts_in_used(db, at, brain):
    r = await setup(db, at)
    rr = await reply(db, "Aapki maa ji ko yaad dilaya")
    db.add(ReplyVerdict(reply_id=rr.id, family_id=FAM, thread_id="e-ms", at=rr.at, day=DAY, passed=False, issues=["wrong_name"],
                        verdict={"issues": {"wrong_name": {"found": True, "words": ["maa ji"]}}}, judged_at=clock.now()))
    await db.flush()
    brain()
    p = (await memory_score.score_family(db, r, DAY))["per_person"][0]
    assert p["used"] == 0.0 and "judge: wrong_name (maa ji)" in p["failures"][-1]["problems"]


async def test_family_score_is_the_minimum_across_people(db, at, brain):
    r = await setup(db, at)
    at("2026-09-21 10:00")
    await fact(db, "medicine:telma", {"name": "Telma", "dose": "40 mg", "times": ["09:00"]}, "Telma 40 mg at 09:00", subject="c-ms", by="c-ms")
    await fact(db, "allergy:none", {}, "No allergies", subject="c-ms", by="c-ms")
    at("2026-10-06 02:30")
    await reply(db, "Ankit ji, Telma 10:00 baje", to="c-ms")  # the caregiver's own medicine, wrong time

    class Two(Brain):
        async def complete(self, route, **kw):
            last = "".join(c.get("text", "") for c in kw["messages"][-1]["content"])
            if "(id c-ms)" in last:
                self.answer = {"medicines": [{"name": "Telma", "dose": "40 mg", "times": ["09:00"]}], "allergies": ["none"]}
            else:
                self.answer = GOOD
            return await super().complete(route, **kw)

    router.register_provider("fake", Two(GOOD))
    out = await memory_score.score_family(db, r, DAY)
    by = {p["subject"]: p for p in out["per_person"]}
    assert by["e-ms"]["score"] == 100.0 and by["c-ms"]["used"] == 0.0 and by["c-ms"]["score"] == 0.0
    assert out["score"] == 0.0  # one bad person is not hidden by an average
    rows = {m.subject_id: m for m in (await db.execute(select(MemoryScore))).scalars()}
    assert set(rows) == {"e-ms", "c-ms", "family"} and rows["family"].score == 0.0


async def test_nobody_without_a_record_is_scored(db, at, brain):
    at("2026-10-05 10:00")
    await store.save_roster(db, FAM, {"id": "", "name": ""}, [CG])
    r = await db.get(FamilyRoster, FAM)
    brain()
    assert (await memory_score.score_family(db, r, DAY))["people"] == 0


def test_dose_and_name_checks_read_only_her_text():
    t = memory_score.Truths(subject_id="x", name="Kamla", naming={"name": "Amma", "avoid": ["Maa"], "key": "naming:address_as", "text": ""},
                            medicines=[{"key": "medicine:telma", "name": "Telma", "dose": "40 mg", "times": ["09:00"], "text": "Telma 40 mg at 09:00"}])
    assert memory_score.used_problems("Amma, Telma 40 mg at 9:00 am", "Maa ki dawai?", to_subject=True, t=t) == []
    assert memory_score.used_problems("Maa, Telma 20 mg", "", to_subject=True, t=t) == ["gives Telma as 20 mg, but the record says 40 mg",
                                                                                   "used the never-use name 'Maa'"]
    assert memory_score.used_problems("Maa ko Telma de dijiye", "", to_subject=False, t=t) == []  # to a caregiver about "Maa": the judge decides
