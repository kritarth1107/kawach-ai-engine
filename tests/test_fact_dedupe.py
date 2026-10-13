"""One fact, one entry (plan D3). Kavach NUdge found one person's single allergy saved under four keys, a home twice,
and wake/sleep/gym both as routine facts and as activities; every turn and nudge carried all of them."""

import json

import pytest
from sqlalchemy import select

from app.brain import tools
from app.care import dedupe, digest, store
from app.care.versions import MemoryVersion
from app.llm import router, spend
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM = "fam-dedupe"
ME = {"id": "self-d", "name": "Kittu Sharma", "role": "primary caregiver"}


class Fake:
    """Answers by which job is asking (the start of its system prompt)."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.answers: dict[str, list[str]] = {}
        self.fail = False

    async def complete(self, route, **kw):
        system = kw.get("system_stable") or ""
        system = " ".join(system) if isinstance(system, list) else system
        text = kw["messages"][-1]["content"][0]["text"]
        if self.fail:
            raise RuntimeError("model down")
        key = next((k for k in self.answers if system.startswith(k)), None)
        self.calls.append((key or system[:40], text))
        out = self.answers[key].pop(0) if key and self.answers[key] else "{}"
        return LLMReply(text=out, tool_calls=[], model="fake")


SAME = dedupe.SAME_PROMPT[:40]
GROUP = dedupe.GROUP_PROMPT[:40]
ROUTINE = dedupe.ROUTINE_PROMPT[:40]


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


def ctx(db):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=ME, speaker=ME, members=[ME], message_ref="m1")


async def active(db, domain):
    return [f for f in await store.facts(db, FAM, ME["id"], domains=[domain], statuses=("active",))]


async def seed_allergy(db):
    await store.save_roster(db, FAM, ME, [ME])
    await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="allergy", key="allergy:cucumber_onion",
                           value={"items": ["cucumber", "roasted onion"]}, text="Allergic to cucumber and roasted onion.",
                           source_kind="caregiver_said", stated_by=ME["id"])


# ── at write time ────────────────────────────────────────────────────────────

async def test_the_same_allergy_in_other_words_is_not_saved_again(db, fake):
    await seed_allergy(db)
    fake.answers[SAME] = [json.dumps({"same_as": "allergy:cucumber_onion", "covered": True})]
    out = await tools.remember(ctx(db), {"domain": "allergy", "name": "cucumber and roasted onions",
                                         "details": {}, "sentence": "Kittu is allergic to cucumber and roasted onions."})
    rows = await active(db, "allergy")
    assert [r.key for r in rows] == ["allergy:cucumber_onion"] and out["key"] == "allergy:cucumber_onion"
    assert rows[0].text == "Allergic to cucumber and roasted onion."  # nothing lost, nothing added
    assert "allergy:cucumber_onion" in fake.calls[0][1] and "NEW" in fake.calls[0][1]


async def test_more_detail_about_the_same_thing_updates_the_saved_fact(db, fake):
    await seed_allergy(db)
    fake.answers[SAME] = [json.dumps({"same_as": "allergy:cucumber_onion", "covered": False,
                                      "sentence": "Allergic to cucumber and roasted onion; cucumber gives a rash."})]
    await tools.remember(ctx(db), {"domain": "allergy", "name": "cucumber", "details": {"reaction": "rash"},
                                   "sentence": "Cucumber gives Kittu a rash."})
    rows = await active(db, "allergy")
    assert len(rows) == 1 and rows[0].key == "allergy:cucumber_onion"
    assert rows[0].text == "Allergic to cucumber and roasted onion; cucumber gives a rash."
    assert rows[0].value["reaction"] == "rash" and rows[0].value["items"] == ["cucumber", "roasted onion"]


async def test_a_different_thing_gets_its_own_fact(db, fake):
    await seed_allergy(db)
    fake.answers[SAME] = [json.dumps({"same_as": None})]
    await tools.remember(ctx(db), {"domain": "allergy", "name": "peanuts", "details": {}, "sentence": "Allergic to peanuts."})
    assert sorted(r.key for r in await active(db, "allergy")) == ["allergy:cucumber_onion", "allergy:peanuts"]


async def test_a_made_up_key_or_a_model_failure_never_loses_the_fact(db, fake):
    await seed_allergy(db)
    fake.answers[SAME] = [json.dumps({"same_as": "allergy:nonsense", "covered": True})]
    await tools.remember(ctx(db), {"domain": "allergy", "name": "dust", "details": {}, "sentence": "Allergic to dust."})
    fake.fail = True
    await tools.remember(ctx(db), {"domain": "allergy", "name": "pollen", "details": {}, "sentence": "Allergic to pollen."})
    assert sorted(r.key for r in await active(db, "allergy")) == ["allergy:cucumber_onion", "allergy:dust", "allergy:pollen"]


async def test_no_model_call_for_a_first_fact_a_same_key_or_an_unchecked_domain(db, fake):
    await store.save_roster(db, FAM, ME, [ME])
    await tools.remember(ctx(db), {"domain": "home", "name": "raipur home", "details": {}, "sentence": "Has a home in Raipur."})
    await tools.remember(ctx(db), {"domain": "home", "name": "raipur home", "details": {"area": "Labhandih"},
                                   "sentence": "Has a home in Labhandih, Raipur."})
    await tools.remember(ctx(db), {"domain": "medicine", "name": "Metformin", "details": {"dose": "500 mg"}, "sentence": "Metformin 500 mg."})
    assert fake.calls == []


async def test_the_re_read_uses_the_same_check(db, fake):
    from app.care import extract

    await seed_allergy(db)
    await store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain="home", key="home:raipur_address",
                           value={"area": "Labhandih"}, text="Has a home in Raipur at Sunita Park, Labhandih.", source_kind="caregiver_said")
    await store.add_turn(db, family_id=FAM, thread_id=ME["id"], role="user", text="mera ghar labhandih raipur mein hai", speaker_id=ME["id"])
    fake.answers[extract.EXTRACT_PROMPT[:40]] = [json.dumps({"facts": [{"domain": "home", "name": "raipur home", "about": ME["id"],
                                                                        "details": {}, "sentence": "Kittu has a home in Labhandih, Raipur.",
                                                                        "evidence": "x"}]})]
    fake.answers[SAME] = [json.dumps({"same_as": "home:raipur_address", "covered": True})]
    await extract.extract_family(db, FAM)
    homes = await active(db, "home")
    assert [h.key for h in homes] == ["home:raipur_address"]
    assert any(k == SAME for k, _ in fake.calls)


# ── nightly ──────────────────────────────────────────────────────────────────

async def seed_duplicates(db):
    await store.save_roster(db, FAM, ME, [ME])
    w = lambda d, k, v, t, s="caregiver_said": store.write_fact(db, family_id=FAM, subject_id=ME["id"], domain=d, key=k, value=v, text=t,
                                                                 source_kind=s, stated_by=ME["id"])  # noqa: E731
    await w("allergy", "allergy:cucumber_onion", {}, "Allergic to cucumber and roasted onion.", "elder_said")
    await w("allergy", "allergy:cucumber", {"reaction": "rash"}, "Allergy to cucumber (rash).", "elder_said")
    await w("allergy", "allergy:roasted_onion", {}, "Allergy to roasted onion.", "elder_said")
    await w("allergy", "allergy:cucumber_and_roasted_onions", {}, "Allergic to cucumber and roasted onions.", "inferred")
    await w("medicine", "medicine:thyronorm", {"name": "Thyronorm", "times": ["07:00"]}, "Thyronorm 50 at 07:00")
    await w("medicine", "medicine:thyroxine", {"name": "Thyroxine", "times": ["07:00"]}, "Thyroxine 50 mcg at 07:00")
    await w("routine", "routine:wake", {"time": "06:30"}, "Wakes up around 06:30")
    await w("routine", "routine:daily_brief", {"time": "22:00"}, "Daily brief at 22:00")
    await w("activity", "activity:wake_up", {"type": "wake_up", "label": "Wake up", "kind": "routine", "days": [], "start": "06:30"},
            "Wake up (every day, 06:30)", "inferred")


def answers(fake):
    fake.answers[GROUP] = [
        json.dumps({"groups": [{"keep": "allergy:cucumber_onion", "drop": ["allergy:cucumber", "allergy:roasted_onion",
                                                                            "allergy:cucumber_and_roasted_onions"],
                                "sentence": "Allergic to cucumber (rash) and roasted onion.", "details": {"items": ["cucumber", "roasted onion"]}}]}),
        json.dumps({"groups": [{"keep": "medicine:thyronorm", "drop": ["medicine:thyroxine"], "sentence": "Thyronorm 50 mcg at 07:00", "details": {}}]}),
    ]
    fake.answers[ROUTINE] = [json.dumps({"covered": [{"routine": "routine:wake", "activity": "activity:wake_up"},
                                                     {"routine": "routine:daily_brief", "activity": "activity:wake_up"}]})]


async def test_the_dry_run_lists_duplicates_and_changes_nothing(db, fake):
    await seed_duplicates(db)
    answers(fake)
    out = await dedupe.merge_family(db, FAM, apply_changes=False)
    found = out[ME["id"]]["found"]
    assert any("keep allergy:cucumber_onion" in x and "allergy:roasted_onion" in x for x in found)
    assert any("medicine" in x and "(report only)" in x for x in found)
    assert any("routine:wake is already activity:wake_up" in x for x in found)
    assert not any("daily_brief" in x for x in found)  # a setting is never a routine duplicate
    assert len(await active(db, "allergy")) == 4 and "applied" not in out[ME["id"]]


async def test_the_merge_keeps_one_fact_with_every_detail_and_can_be_undone(db, fake):
    await seed_duplicates(db)
    answers(fake)
    out = await dedupe.merge_family(db, FAM, apply_changes=True)
    assert out[ME["id"]]["applied"] == {"merged": 1, "stopped": 3, "routine_marked": 1}
    allergies = await active(db, "allergy")
    assert [a.key for a in allergies] == ["allergy:cucumber_onion"]
    a = allergies[0]
    assert a.text == "Allergic to cucumber (rash) and roasted onion." and a.value["reaction"] == "rash"
    assert a.source_kind == "elder_said"  # the strongest source in the group, so no confirmation is asked for
    # medicines are never merged by code
    assert sorted(m.key for m in await active(db, "medicine")) == ["medicine:thyronorm", "medicine:thyroxine"]
    # every stop is a version that undo can bring back
    stops = (await db.execute(select(MemoryVersion).where(MemoryVersion.family_id == FAM, MemoryVersion.op == "stop"))).scalars().all()
    assert sorted(v.target for v in stops) == ["allergy:cucumber", "allergy:cucumber_and_roasted_onions", "allergy:roasted_onion"]
    assert all("merged" in (v.reason or "") or "same fact" in (v.reason or "") for v in stops)


async def test_a_routine_shown_as_an_activity_is_listed_once_and_keeps_its_time(db, fake):
    await seed_duplicates(db)
    answers(fake)
    await dedupe.merge_family(db, FAM, apply_changes=True)
    wake = await store.active_fact(db, FAM, ME["id"], "routine:wake")
    assert wake is not None and wake.value["time"] == "06:30" and wake.value["as_activity"] == "activity:wake_up"
    record = digest.care_record(ME["name"], await store.facts(db, FAM, ME["id"]))
    assert "routine:wake]" not in record and "activity:wake_up]" in record and "routine:daily_brief]" in record


async def test_nightly_mode_is_a_dry_run_until_switched_on(monkeypatch):
    monkeypatch.delenv("FACT_MERGE", raising=False)
    assert dedupe.merge_mode() == "dryrun"
    monkeypatch.setenv("FACT_MERGE", "on")
    assert dedupe.merge_mode() == "on"
    monkeypatch.setenv("FACT_MERGE", "nonsense")
    assert dedupe.merge_mode() == "dryrun"
