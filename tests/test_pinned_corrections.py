"""Corrections apply at once and stay pinned until a person changes them; what the family said is wrong never comes back
(trust plan A3; founder 2026-10-13: "'Don't call me Maa' must hold forever").

Before: the nightly re-read wrote naming, diet and preference facts as `inferred` and silently replaced a person's
correction, a naming update could drop the avoid list, a rejected fact could be proposed again, and reminder lines did not
check avoided names."""

import json

import pytest

from app.brain import tools
from app.care import compose, dedupe, digest, extract, naming, rejected, store
from app.llm import router, spend
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM = "fam-pin"
MAA = {"id": "elder-pin", "name": "Kamla Devi", "role": "elder"}
SON = {"id": "son-pin", "name": "Ravi Verma", "role": "primary caregiver"}


class Fake:
    def __init__(self):
        self.answers: dict[str, list[str]] = {}
        self.calls: list[str] = []

    async def complete(self, route, **kw):
        system = kw.get("system_stable") or ""
        system = " ".join(system) if isinstance(system, list) else system
        key = next((k for k in self.answers if system.startswith(k)), None)
        self.calls.append(key or system[:30])
        out = self.answers[key].pop(0) if key and self.answers[key] else "{}"
        return LLMReply(text=out, tool_calls=[], model="fake")


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    router.register_provider("fake", f)
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"], "judge_fast": ["fake:m"], "worker": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    yield f
    router._providers.pop("fake", None)
    router.reset_breakers()


def ctx(db, speaker=MAA):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=MAA, speaker=speaker, members=[MAA, SON], message_ref="m1")


async def write(db, domain, key, value, text, source, by=None):
    return await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain=domain, key=key, value=value, text=text,
                                  source_kind=source, stated_by=by)


# ── pinned ───────────────────────────────────────────────────────────────────

async def test_what_a_person_says_is_pinned_and_an_inference_cannot_change_it(db):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    w = await write(db, "naming", "naming:address_as", {"name": "Kamla ji", "avoid": ["Maa"]}, "Call her Kamla ji, never Maa",
                    "elder_said", MAA["id"])
    assert w.fact.pinned_at is not None and w.fact.pinned_by == MAA["id"]
    kept = await write(db, "naming", "naming:address_as", {"name": "Maa"}, "Call her Maa", "inferred")
    assert kept.result == "kept_pinned"
    now = await store.active_fact(db, FAM, MAA["id"], "naming:address_as")
    assert now.value == {"name": "Kamla ji", "avoid": ["Maa"]}
    assert (await store.stop_fact(db, family_id=FAM, subject_id=MAA["id"], key="naming:address_as", reason="x",
                                  source_kind="inferred")).result == "kept_pinned"


async def test_a_caregivers_diet_correction_survives_the_nightly_re_read(db):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await write(db, "diet", "diet:sugar", {"rule": "no sugar"}, "No sugar at all", "caregiver_said", SON["id"])
    assert (await write(db, "diet", "diet:sugar", {"rule": "a little sugar is fine"}, "A little sugar is fine", "inferred")).result == "kept_pinned"
    assert (await store.active_fact(db, FAM, MAA["id"], "diet:sugar")).text == "No sugar at all"


async def test_names_to_avoid_only_grow_until_the_person_lifts_one(db):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await write(db, "naming", "naming:address_as", {"name": "Kamla ji", "avoid": ["Maa"]}, "never Maa", "elder_said", MAA["id"])
    await write(db, "naming", "naming:address_as", {"name": "Kamla ji", "avoid": ["Amma"]}, "never Amma either", "caregiver_said", SON["id"])
    assert set((await naming.for_person(db, FAM, MAA["id"]))["avoid"]) == {"Maa", "Amma"}
    await write(db, "naming", "naming:address_as", {"name": "Maa", "unavoid": ["Maa"]}, "You can call me Maa again", "elder_said", MAA["id"])
    got = await naming.for_person(db, FAM, MAA["id"])
    assert got == {"name": "Maa", "avoid": ["Amma"]}


async def test_a_caregiver_approval_pins_the_fact(db):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await write(db, "medicine", "medicine:metformin", {"name": "Metformin", "dose": "500 mg"}, "Metformin 500 mg", "caregiver_said", SON["id"])
    p = await write(db, "medicine", "medicine:metformin", {"dose": "1000 mg"}, "Metformin 1000 mg", "elder_said", MAA["id"])
    assert p.result == "pending"
    row = await store.resolve_pending(db, fact_id=p.fact.id, approve=True, by=SON["id"])
    assert row.status == "active" and row.pinned_by == SON["id"]


# ── rejected ─────────────────────────────────────────────────────────────────

async def test_a_rejected_proposal_is_never_written_by_a_re_read_but_a_person_can_say_it_again(db, fake, monkeypatch):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    p = await write(db, "allergy", "allergy:peanut", {"allergen": "peanut"}, "Allergic to peanuts", "inferred")
    assert p.result in ("pending", "created")
    if p.result == "created":  # a first health fact from a re-read is pending in extract; force the same here
        p.fact.status = "pending"
        await db.flush()
    await store.resolve_pending(db, fact_id=p.fact.id, approve=False, by=SON["id"])
    wrong = await rejected.rejected(db, FAM, MAA["id"])
    assert [r.key for r in wrong] == ["allergy:peanut"] and "peanut" in rejected.terms(wrong)
    # the exact key from an inference: refused with no model call
    same = await dedupe.check(db, family_id=FAM, subject_id=MAA["id"], domain="allergy", key="allergy:peanut", text="peanut allergy", value={})
    assert same is not None and same.rejected and fake.calls == []
    # the re-read hears it again: nothing is proposed
    await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="user", text="mujhe moongphali se allergy hai", speaker_id=MAA["id"])
    fake.answers[extract.EXTRACT_PROMPT[:40]] = [json.dumps({"facts": [{"domain": "allergy", "name": "peanut", "about": MAA["id"],
                                                                        "details": {}, "sentence": "Allergic to peanuts", "evidence": "x"}]})]
    await extract.extract_family(db, FAM)
    assert not [f for f in await store.facts(db, FAM, MAA["id"], domains=["allergy"]) if f.status in ("active", "pending")]
    # the person themselves says it again: a new fact from a person
    out = await tools.remember(ctx(db, SON), {"domain": "allergy", "name": "peanut", "details": {"allergen": "peanut"},
                                             "sentence": "Allergic to peanuts (the doctor confirmed)", "about": MAA["id"]})
    assert out["result"] in ("created", "superseded")


async def test_a_stop_marked_wrong_goes_on_the_rejected_list(db):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await write(db, "condition", "condition:diabetes", {"name": "diabetes"}, "Has diabetes", "caregiver_said", SON["id"])
    await tools.stop(ctx(db, SON), {"domain": "condition", "name": "diabetes", "reason": "she never had diabetes", "wrong": True, "about": MAA["id"]})
    wrong = await rejected.rejected(db, FAM, MAA["id"])
    assert [r.key for r in wrong] == ["condition:diabetes"] and "diabetes" in rejected.terms(wrong)
    # an ordinary stop (it ended) is not a rejection
    await write(db, "diet", "diet:fasting", {"rule": "fasting"}, "Fasting on Tuesdays", "caregiver_said", SON["id"])
    await tools.stop(ctx(db, SON), {"domain": "diet", "name": "fasting", "reason": "she stopped fasting", "about": MAA["id"]})
    assert [r.key for r in await rejected.rejected(db, FAM, MAA["id"])] == ["condition:diabetes"]


async def test_the_care_record_shows_the_name_rule_pins_and_what_never_to_use(db):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await write(db, "naming", "naming:address_as", {"name": "Kamla ji", "avoid": ["Maa"]}, "Call her Kamla ji", "elder_said", MAA["id"])
    await write(db, "condition", "condition:diabetes", {"name": "diabetes"}, "Has diabetes", "caregiver_said", SON["id"])
    await tools.stop(ctx(db, SON), {"domain": "condition", "name": "diabetes", "reason": "never had it", "wrong": True, "about": MAA["id"]})
    rec = digest.care_record(MAA["name"], await store.facts(db, FAM, MAA["id"]), rejected=await rejected.rejected(db, FAM, MAA["id"]))
    assert "ALWAYS: call them Kamla ji. NEVER call them Maa" in rec
    assert "PINNED" in rec
    assert "Never say, assume or use" in rec and "Has diabetes" in rec


async def test_the_reply_guard_catches_a_rejected_fact_and_an_avoided_name(db):
    from app.brain import loop

    await store.save_roster(db, FAM, MAA, [MAA, SON])
    c = ctx(db, SON)
    req = loop.TurnRequest(family_id=FAM, elder=MAA, speaker=SON, members=[MAA, SON], text="how is she today?", message_ref="r1")
    problems = await loop.guard_problems(db, req, c, "She is fine; keep her diabetes diet going.", avoid=["Maa"], unreachable=[],
                                         wrong_terms=["diabetes"])
    assert any("diabetes" in p for p in problems)
    req2 = loop.TurnRequest(family_id=FAM, elder=MAA, speaker=SON, members=[MAA, SON], text="does she have diabetes?", message_ref="r2")
    problems2 = await loop.guard_problems(db, req2, c, "No, diabetes is not on her record.", avoid=[], unreachable=[], wrong_terms=["diabetes"])
    assert not any("family said is wrong" in p for p in problems2)  # they asked about it themselves
    assert naming.uses_avoided("Good morning Maa!", ["Maa"]) == ["Maa"] and naming.uses_avoided("Mausam accha hai", ["Maa"]) == []


async def test_a_reminder_line_never_uses_an_avoided_name(db, fake):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await write(db, "naming", "naming:address_as", {"name": "Kamla ji", "avoid": ["Maa"]}, "Call her Kamla ji", "elder_said", MAA["id"])
    fake.answers[compose.PROMPT[:40]] = [json.dumps({"text": "Maa, time for your 8 o'clock tablet"}),
                                         json.dumps({"text": "Kamla ji, time for your 8 o'clock tablet"})]
    out = await compose.reminder_text(db, family_id=FAM, subject_id=MAA["id"], purpose="Metformin 500 mg", at="08:00")
    assert out["text"] == "Kamla ji, time for your 8 o'clock tablet"


async def test_a_duplicate_merge_keeps_the_pinned_fact(db, fake):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    await write(db, "allergy", "allergy:milk_inferred", {}, "Milk allergy", "inferred")
    await write(db, "allergy", "allergy:milk", {"reaction": "rash"}, "Allergic to milk (rash)", "caregiver_said", SON["id"])
    group = dedupe.Group("allergy", keep="allergy:milk_inferred", drop=["allergy:milk"], text="Allergic to milk (rash)", details={})
    await dedupe.apply(db, FAM, MAA["id"], [group], [])
    active = [f.key for f in await store.facts(db, FAM, MAA["id"], domains=["allergy"], statuses=("active",))]
    assert active == ["allergy:milk"]


# ── "Don't call me Maa" for 30 days ───────────────────────────────────────────

async def test_dont_call_me_maa_holds_for_30_days_of_re_reads_merges_and_reminders(db, fake, at):
    await store.save_roster(db, FAM, MAA, [MAA, SON])
    at("2026-10-01 09:00")
    await tools.remember(ctx(db, MAA), {"domain": "naming", "name": "address_as", "details": {"name": "Kamla ji", "avoid": ["Maa"]},
                                        "sentence": "Call her Kamla ji; she does not want to be called Maa"})
    for day in range(1, 31):
        at(f"2026-10-{day:02d} 21:00" if day <= 31 else "2026-10-31 21:00")
        # the family keeps saying "Maa" when talking about her; the re-read hears it every night
        await store.add_turn(db, family_id=FAM, thread_id=SON["id"], role="user", text=f"Maa ne aaj dawai li? ({day})", speaker_id=SON["id"])
        fake.answers[extract.EXTRACT_PROMPT[:40]] = [json.dumps({"facts": [{"domain": "naming", "name": "address_as", "about": MAA["id"],
                                                                            "details": {"name": "Maa", "avoid": []},
                                                                            "sentence": "The family calls her Maa", "evidence": "x"}]})]
        await extract.extract_family(db, FAM)
        fake.answers[compose.PROMPT[:40]] = [json.dumps({"text": f"Maa, tablet time ({day})"}),
                                             json.dumps({"text": f"Kamla ji, tablet time ({day})"})]
        line = (await compose.reminder_text(db, family_id=FAM, subject_id=MAA["id"], purpose="Metformin 500 mg", at="21:00"))["text"]
        assert line and not naming.uses_avoided(line, ["Maa"]), f"day {day}: {line}"
    final = await naming.for_person(db, FAM, MAA["id"])
    assert final == {"name": "Kamla ji", "avoid": ["Maa"]}
    rec = digest.care_record(MAA["name"], await store.facts(db, FAM, MAA["id"]))
    assert "NEVER call them Maa" in rec
