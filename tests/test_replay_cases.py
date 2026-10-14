"""Trust plan C3: replay cases from every source, anonymised; the pre-deploy runner's choice, checks and 2-of-3 rule;
the fix loop; a reintroduced bug blocks the deploy (fake runner)."""

import json
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care import store, versions
from app.care.models import FamilyRoster
from app.core import clock
from app.learn.models import ReplyLog
from app.quality import cases, replay
from app.quality.models import ReplayCase
from eval import replay_cases as runner

FAM = "fam-rc"
ELDER = {"id": "e-rc", "name": "Savitri Kulkarni", "role": "elder"}
CG = {"id": "c-rc", "name": "Mohan Kulkarni", "role": "primary caregiver"}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


async def family(db, at):
    at("2026-10-05 09:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, CG])
    with versions.attribution(actor_id="e-rc", source="whatsapp"):
        await store.write_fact(db, family_id=FAM, subject_id="e-rc", domain="naming", key="naming:address_as", value={"name": "Aaji", "avoid": ["Maa"]},
                               text="Call her Aaji, never Maa", source_kind="elder_said", stated_by="e-rc")
        await store.write_fact(db, family_id=FAM, subject_id="e-rc", domain="contact", key="contact:neighbour", value={"name": "Shalini Deshpande"},
                               text="Neighbour Shalini Deshpande, phone 9876543210", source_kind="caregiver_said", stated_by="c-rc")
    await store.add_turn(db, family_id=FAM, thread_id="e-rc", role="user", text="Shalini ko bolo main theek hoon", speaker_id="e-rc")
    await db.flush()
    return await db.get(FamilyRoster, FAM)


async def test_guard_rewrites_and_corrections_become_anonymised_cases(db, at, sessions):
    from app.learn import corrections

    await family(db, at)
    at("2026-10-05 10:00")
    await corrections.note_guard(db, family_id=FAM, thread_id="e-rc", draft="Maa, Shalini ji ne 9876543210 pe call kiya tha",
                                 problems=["uses Maa, a name they asked never to use", "mentions the time 10:00, which is not in memory"], user_text="koi aaya?")
    r = ReplyLog(family_id=FAM, thread_id="e-rc", at=clock.now(), kind="reply", situation="chit_chat", speaker_role="elder", lang="latin-indic",
                 user_text="Shalini kab aayegi?", text="Shalini kal subah 9 baje aayengi", text_len=10, tools=[], corrected=True)
    db.add(r)
    await db.flush()
    await store.add_turn(db, family_id=FAM, thread_id="e-rc", role="user", text="galat, wo aaj aayegi", speaker_id="e-rc")
    await db.commit()
    got = await corrections.collect(sessions)
    assert got.get("guard") == 1 and got.get("chat") == 1
    made = await cases.from_corrections(sessions, since=clock.now() - timedelta(hours=1))
    assert made["guard_rewrite"] == 1 and made["chat_correction"] == 1
    rows = {c.source: c for c in (await db.execute(select(ReplayCase))).scalars()}
    g = rows["guard"]
    whole = json.dumps({"c": g.context, "m": g.message, "b": g.bad_reply, "a": g.assertion, "r": g.rubric}, ensure_ascii=False)
    assert g.anonymised and "Shalini" not in whole and "Savitri" not in whole and "9876543210" not in whole and "e-rc" not in whole
    assert {"type": "not_contains", "words": ["Maa"]} in g.assertion and g.bad_reply.startswith("Maa,")  # what is checked is kept
    assert g.context["facts"] and g.context["speaker"]["id"] == "p1"
    assert rows["correction"].bad_reply.endswith("aayengi") and "[PERSON" in rows["correction"].bad_reply


async def test_probe_failures_become_world_cases(db):
    scores = {"examples": [{"probe": "w0034-q12", "family": "w0034", "language": "mwr", "topic": "stop_calling_me", "text": "Maa mat bolo",
                            "reply": "Ji Maa", "fails": ["used the word 'Maa' after being asked not to", "expected one of ['remember'], used no tools"]}],
              "judge_failures": [{"probe": "w0100-q2", "family": "w0100", "issues": ["made_up_fact"], "note": "invented a visit"}]}
    assert await cases.from_probes(db, scores, "2026-10-14") == 2
    assert await cases.from_probes(db, scores, "2026-10-15") == 0  # the same failure never makes two cases
    c = (await db.execute(select(ReplayCase).where(ReplayCase.source_ref == "probe:w0034-q12"))).scalars().one()
    assert c.context["world"] == {"family": "w0034", "probe": "w0034-q12"}
    assert {"type": "not_contains", "words": ["Maa"]} in c.assertion and {"type": "must_tool", "tools": ["remember"]} in c.assertion


async def test_the_fix_loop_flips_open_to_fixed_only_on_master(db, at):
    at("2026-10-14 09:00")
    await replay.add(db, source="probe", source_ref="probe:x", kind="k", day="2026-10-14", context={}, assertion=[{"type": "grounded"}])
    c = (await db.execute(select(ReplayCase))).scalars().one()
    await cases.apply_results(db, [{"id": str(c.id), "passed": True, "runs": [{"passed": True}]}], commit="abc", branch="feature")
    await db.refresh(c)
    assert c.status == "open" and c.last_result["passed"] is True
    await cases.apply_results(db, [{"id": str(c.id), "passed": True}], commit="def", branch="master")
    await db.refresh(c)
    assert c.status == "fixed" and c.fixed_commit == "def"
    await cases.apply_results(db, [{"id": str(c.id), "passed": False}], commit="ghi", branch="master")
    await db.refresh(c)
    assert c.status == "fixed"  # a fixed case that fails stays fixed: it blocks the deploy


def test_choice_keeps_every_fixed_case_then_one_per_kind():
    cs = [{"id": f"f{i}", "status": "fixed", "kind": "a", "created_at": f"2026-10-{i + 1:02d}"} for i in range(5)]
    cs += [{"id": f"o{i}", "status": "open", "kind": k, "created_at": "2026-10-01"} for i, k in enumerate(["x", "x", "x", "y", "z"])]
    got = runner.choose(cs, 8)
    assert [c["id"] for c in got[:5]] == ["f4", "f3", "f2", "f1", "f0"] and {c["kind"] for c in got[5:]} == {"x", "y", "z"}
    assert len(runner.choose(cs, 100)) == 10


def test_code_checks_on_her_reply():
    assert runner.code_check({"type": "not_contains", "words": ["Maa"]}, "Ji Maa, theek hai", []) == "used ['Maa']"
    assert runner.code_check({"type": "not_contains", "words": ["Maa"]}, "Ji Aaji, theek hai", []) is None
    assert runner.code_check({"type": "no_tool", "tools": ["start_task"]}, "", ["start_task"]) == "called ['start_task']"
    assert runner.code_check({"type": "must_tool", "tools": ["send_message"]}, "", ["remember"])
    assert runner.code_check({"type": "max_sentences", "n": 1}, "One. Two.", [])
    assert runner.code_check({"type": "medicine_time", "medicine": "Metformin", "times": ["08:00"]}, "Metformin 9:00 pm pe lijiye", [])


async def test_two_of_three_and_a_reintroduced_bug_blocks(monkeypatch, tmp_path):
    outcomes = {"fixed-1": [True, False, True], "fixed-2": [False, True, False], "open-1": [False, False, False]}

    async def fake_once(sessions, case, n):
        return {"passed": outcomes[case["id"]][n], "why": "" if outcomes[case["id"]][n] else "used ['Maa']"}

    monkeypatch.setattr(runner, "run_once", fake_once)
    import time

    got = {c: await runner.run_case(None, {"id": c, "status": c.split("-")[0], "kind": "k", "source": "s"}, time.monotonic() + 60, 100.0)
           for c in outcomes}
    assert got["fixed-1"]["passed"] is True and len(got["fixed-1"]["runs"]) == 3
    assert got["fixed-2"]["passed"] is False  # 2 of 3 failed: the old bug is back
    assert got["open-1"]["passed"] is False and len(got["open-1"]["runs"]) == 2  # decided after two

    async def fake_case(sessions, c, deadline, cap):
        return got[c["id"]]

    monkeypatch.setattr(runner, "run_case", fake_case)

    class Eng:
        async def dispose(self):
            pass

    async def prepare(engine):
        return None

    monkeypatch.setattr(runner, "create_async_engine", lambda url, **kw: Eng())

    async def answers():
        return None

    monkeypatch.setattr(runner, "preflight", answers)  # never a real model call in tests
    import eval.world.seed_engine as se

    monkeypatch.setattr(se, "prepare", prepare)
    monkeypatch.setattr(se, "assert_local", lambda url: None)
    code = await runner.main_async([{"id": c, "status": c.split("-")[0], "kind": "k"} for c in outcomes], 3, "postgresql://localhost/x", 100.0, tmp_path)
    res = json.loads((tmp_path / "results.json").read_text())
    assert code == 1 and res["summary"]["fixed_failed"] == 1 and res["summary"]["open_failed"] == 1  # an open case never blocks
    assert "BLOCKED" in (tmp_path / "summary.md").read_text()


def test_founder_and_seed_cases_are_invented_and_runnable():
    cs = runner.load_cases([])
    ids = {c["id"] for c in cs}
    assert {"founder-cigarette_en", "founder-relay", "seed-effort-forgot_med", "seed-effort-dont_call_maa"} <= ids
    forgot = next(c for c in cs if c["id"] == "seed-effort-forgot_med")
    assert forgot["status"] == "open" and forgot["safety"] and {"type": "no_medical_advice"} in forgot["assertion"]
    assert all(runner.message_of(c) for c in cs) and all(c["context"]["facts"] for c in cs)


async def test_run_for_playbook_refuses_outside_the_test_project(monkeypatch, sessions):
    from app.learn import replay as learn_replay

    monkeypatch.setenv("GCP_PROJECT_ID", "kavach-care")
    with pytest.raises(learn_replay.ReplayUnavailable):
        await learn_replay.run_for_playbook(sessions, None)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("REPLAY_DATABASE_URL", raising=False)
    with pytest.raises(learn_replay.ReplayUnavailable):
        await learn_replay.run_for_playbook(sessions, None)


async def test_run_for_playbook_reports_regressions_and_fixes(monkeypatch, sessions):
    from app.learn import replay as learn_replay

    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("REPLAY_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_replay")
    cand = type("PB", (), {"version": 9, "lessons": {"general": ["be brief"]}, "examples": {}})()
    seen = []

    async def fake_run(cases, url, cap, *, parallel=20, playbook=None, deadline_s=None):
        seen.append(playbook)
        live = playbook is not cand
        return [{"id": "a", "passed": True if live else False}, {"id": "b", "passed": False if live else True}, {"id": "c", "passed": True},
                {"id": "d", "passed": None}]

    monkeypatch.setattr(runner, "run_cases", fake_run)
    monkeypatch.setattr(runner, "load_cases", lambda argv: [{"id": x, "status": "fixed", "kind": "k"} for x in "abcd"])

    async def answers():
        return None

    monkeypatch.setattr(runner, "preflight", answers)
    got = await learn_replay.run_for_playbook(sessions, cand, budget_inr=50)
    assert got["regressions"] == ["a"] and got["fixed"] == ["b"] and got["skipped"] == ["d"] and got["n"] == 4 and got["playbook"] == 9
    assert seen[1] is cand  # the candidate replaced the live playbook in the second run


def test_misread_cases_cover_each_failure_in_the_main_languages():
    cases = [json.loads(line) for line in runner.MISREAD.read_text().splitlines() if line.strip()]
    kinds = {c["kind"] for c in cases}
    assert {"asking_vs_did", "one_option_who", "ongoing", "joke_vs_fact", "voice_note"} <= kinds
    for kind in kinds:
        langs = {c["lang"] for c in cases if c["kind"] == kind}
        assert {"en", "hi", "hinglish", "bn"} <= langs, kind
    assert all(c["status"] == "open" and "91337bfa" not in json.dumps(c) for c in cases)
    assert all(c["id"].startswith("misread-") for c in cases)
