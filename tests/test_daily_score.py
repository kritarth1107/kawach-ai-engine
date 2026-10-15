"""Trust plan C1: the daily score (three checks of 3 Oct, their minimum), the full-memory judge on every real reply,
memory rebuilt as of a moment, the morning email, the nightly world score's endpoint, replay cases. Scripted models."""

import json
from datetime import date, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care import store, versions
from app.care.models import FamilyRoster
from app.core import clock
from app.learn.models import ReplyLog
from app.llm import router, spend
from app.llm.router import LLMReply
from app.quality import asof, daily, judge, nightly, report
from app.quality.models import DailyScore, EvalScore, ReplayCase, ReplyVerdict

FAM = "fam-ds"
ELDER = {"id": "e-ds", "name": "Kamla Sharma", "role": "elder"}
CG = {"id": "c-ds", "name": "Ankit Sharma", "role": "primary caregiver"}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


@pytest.fixture(autouse=True)
def clean():
    spend.reset()
    router.reset_breakers()
    yield
    spend.reset()


class Judge:
    """A fake judge: answer(system_text, user_text) -> reply text. Records every prompt."""

    def __init__(self, answer):
        self.answer, self.seen = answer, []

    async def complete(self, route, *, system_stable, system_dynamic="", messages, **kw):
        sys_text = "\n".join(system_stable)
        user = "".join(c.get("text", "") for c in messages[-1]["content"])
        self.seen.append({"system": sys_text, "user": user, "effort": kw.get("effort")})
        return LLMReply(text=self.answer(sys_text, user), tool_calls=[], model="fake-judge", usage={"in": 1000, "out": 200})


@pytest.fixture
def fake_judge(monkeypatch):
    def use(answer):
        j = Judge(answer)
        router.register_provider("fake", j)
        return j

    monkeypatch.setenv("MODEL_ROUTES", json.dumps({r: ["fake:j"] for r in ("judge", "judge_fast", "brain", "extract")}))
    yield use
    router._providers.pop("fake", None)


def verdict(**found) -> str:
    issues = {n: {"found": n in found, "words": found.get(n, []), "note": "x" if n in found else ""} for n in judge.ISSUES}
    return json.dumps({"issues": issues, "summary": "bad" if found else "good"})


async def roster(db):
    await store.save_roster(db, FAM, ELDER, [ELDER, CG])
    await db.flush()
    return await db.get(FamilyRoster, FAM)


async def fact(db, key, value, text, source="caregiver_said", by="c-ds", subject="e-ds"):
    with versions.attribution(actor_id=by, source="whatsapp"):
        return await store.write_fact(db, family_id=FAM, subject_id=subject, domain=key.split(":")[0], key=key, value=value, text=text,
                                      source_kind=source, stated_by=by)


async def log_reply(db, text, *, to="e-ds", said="", trace=None, kind="reply", situation="chit_chat", tools=None):
    r = ReplyLog(family_id=FAM, thread_id=to, at=clock.now(), kind=kind, situation=situation, speaker_role="elder", lang="latin-indic",
                 user_text=said, text=text, text_len=len(text), tools=list(tools or []), trace=trace or [])
    db.add(r)
    await store.add_turn(db, family_id=FAM, thread_id=to, role="assistant", text=text)
    await db.flush()
    return r


# ── memory as of a moment ─────────────────────────────────────────────────────


async def test_memory_rebuilt_as_of_a_timestamp(db, at):
    t1 = at("2026-10-01 09:00")
    await fact(db, "medicine:metformin", {"name": "Metformin", "dose": "500 mg", "times": ["08:00"]}, "Metformin 500 mg at 08:00")
    await store.upsert_note(db, family_id=FAM, subject_id="e-ds", slug="about", title="About", body_md="Likes jasmine tea.", actor_id="saheli", source="saheli")
    t2 = at("2026-10-02 09:00")
    await fact(db, "medicine:metformin", {"times": ["09:00"]}, "Metformin 500 mg at 09:00")
    t3 = at("2026-10-03 09:00")
    with versions.attribution(actor_id="saheli", source="nightly"):
        await store.write_fact(db, family_id=FAM, subject_id="e-ds", domain="medicine", key="medicine:metformin", value={"dose": "1000 mg"},
                               text="Metformin 1000 mg at 09:00", source_kind="inferred")  # a weaker source: waits as pending
    await store.upsert_note(db, family_id=FAM, subject_id="e-ds", slug="about", title="About", body_md="Likes ginger tea.", actor_id="saheli", source="saheli")
    t4 = at("2026-10-04 09:00")
    with versions.attribution(actor_id="c-ds", source="whatsapp"):
        await store.stop_fact(db, family_id=FAM, subject_id="e-ds", key="medicine:metformin", reason="doctor stopped it", source_kind="caregiver_said",
                              stated_by="c-ds")
    await db.flush()

    m1 = await asof.memory_at(db, FAM, ["e-ds"], t1 + timedelta(minutes=1))
    assert [f.text for f in m1.active("e-ds")] == ["Metformin 500 mg at 08:00"] and "jasmine" in m1.notes_text()
    m2 = await asof.memory_at(db, FAM, ["e-ds"], t2 + timedelta(minutes=1))
    assert [f.text for f in m2.active("e-ds")] == ["Metformin 500 mg at 09:00"] and not m2.pending("e-ds")
    m3 = await asof.memory_at(db, FAM, ["e-ds"], t3 + timedelta(minutes=1))
    assert [f.text for f in m3.pending("e-ds")] == ["Metformin 1000 mg at 09:00"] and "ginger" in m3.notes_text()
    assert "PENDING" in m3.record("e-ds", "Kamla")
    m4 = await asof.memory_at(db, FAM, ["e-ds"], t4 + timedelta(minutes=1))
    assert not m4.active("e-ds")
    before = await asof.memory_at(db, FAM, ["e-ds"], t1 - timedelta(minutes=1))
    assert not before.active("e-ds") and "tea" not in before.notes_text()


# ── the full-memory judge on every reply ──────────────────────────────────────


async def test_judge_input_has_the_care_record_and_notes(db, at, sessions, fake_judge):
    at("2026-10-05 09:00")
    r0 = await roster(db)
    await fact(db, "medicine:metformin", {"name": "Metformin", "dose": "500 mg", "times": ["08:00"]}, "Metformin 500 mg at 08:00")
    await fact(db, "naming:address_as", {"name": "Amma", "avoid": ["Maa"]}, "Call her Amma, never Maa", source="elder_said", by="e-ds")
    await store.upsert_note(db, family_id=FAM, subject_id="e-ds", slug="about", title="About", body_md="Her grandson Ishaan lives in Pune.",
                            actor_id="saheli", source="saheli")
    await store.add_turn(db, family_id=FAM, thread_id="e-ds", role="user", text="Subah ki goli le li", speaker_id="e-ds")
    at("2026-10-05 10:00")
    r = await log_reply(db, "Maa, Ishaan ne aaj phone kiya?", said="Subah ki goli le li",
                        trace=[{"role": "model", "calls": [{"name": "log_dose", "args": {"medicine": "Metformin"}}]}])
    await db.commit()
    j = fake_judge(lambda s, u: verdict(wrong_name=["Maa"], made_up_fact=["Ishaan ne aaj phone kiya"]))
    out = await judge.judge_replies(sessions, FAM, "2026-10-05")
    assert out["graded"] == 1 and out["failed"] == 1 and out["errors"] == 0 and out["of"] == 1 and out["skipped"] == 0
    p = j.seen[0]
    assert "[medicine:metformin] Metformin 500 mg at 08:00" in p["system"]  # the care record as it stood
    assert "NEVER call them Maa" in p["system"] and "Her grandson Ishaan lives in Pune." in p["system"]  # naming and the notes
    assert "HOUSEHOLD" in p["system"] and "Subah ki goli le li" in p["user"] and "log_dose" in p["user"]  # chat and tool calls
    assert p["effort"] == "medium"
    v = (await db.execute(select(ReplyVerdict))).scalars().one()
    assert v.passed is False and set(v.issues) == {"wrong_name", "made_up_fact"} and v.reply_id == r.id
    await db.refresh(r)
    assert r.judge_pass is False and "Maa" in (r.judge_note or "")  # the learning loop reads the new verdict
    # not judged twice
    assert (await judge.judge_replies(sessions, FAM, "2026-10-05"))["graded"] == 0
    # a failure becomes a replay case with what must hold
    assert await nightly.judge_cases(sessions, r0, "2026-10-05") == 1
    case = (await db.execute(select(ReplayCase))).scalars().one()
    assert case.source == "judge" and case.bad_reply.startswith("Maa,") and {"type": "not_contains", "words": ["Maa"]} in case.assertion
    ctx = json.dumps(case.context, ensure_ascii=False)
    assert "[medicine:metformin] Metformin 500 mg at 08:00" in case.context["rendered"]  # the record as it stood, medicine names kept
    assert any(f["key"] == "naming:address_as" and f["value"]["avoid"] == ["Maa"] for f in case.context["facts"])  # structured, to seed a family
    assert case.anonymised and "Kamla" not in ctx and "Ankit" not in ctx and "e-ds" not in ctx and "Ishaan" not in ctx  # real names and ids gone
    assert case.context["speaker"]["id"] == "p1" and case.message == "Subah ki goli le li"


async def test_ordinary_replies_are_sampled_and_an_alert_is_always_graded(db, at, sessions, fake_judge, monkeypatch):
    monkeypatch.setenv("JUDGE_PER_FAMILY", "2")
    monkeypatch.setenv("JUDGE_DAILY_CAP", "90")
    at("2026-10-05 10:00")
    await roster(db)
    for i in range(4):
        await log_reply(db, f"Theek hai {i}")
    await log_reply(db, "Maine unhe bata diya", situation="emergency", tools=["alert_caregiver"])
    await db.commit()
    fake_judge(lambda s, u: verdict())
    out = await judge.judge_replies(sessions, FAM, "2026-10-05")
    assert out["graded"] == 3 and out["skipped"] == 2 and out["failed"] == 0
    bands = {r.text: r.judge_band for r in (await db.execute(select(ReplyLog))).scalars()}
    assert bands["Maine unhe bata diya"] == "always"
    assert sum(1 for b in bands.values() if b == "sample") == 2
    assert sum(1 for b in bands.values() if b == "skip") == 2


async def test_unparseable_judge_answer_is_not_a_pass(db, at, sessions, fake_judge):
    at("2026-10-05 10:00")
    await roster(db)
    await log_reply(db, "Theek hai ji")
    await db.commit()
    fake_judge(lambda s, u: "I think it's fine")
    out = await judge.judge_replies(sessions, FAM, "2026-10-05")
    assert out["graded"] == 0 and out["errors"] == 1
    assert not (await db.execute(select(ReplyVerdict))).scalars().all()


# ── check 1: the reminder was actually sent ──────────────────────────────────


def test_check1_maths_held_excluded():
    due = "2026-10-05T08:00:00+05:30"
    slots = [
        {"familyId": FAM, "kind": "dose_reminder", "dueAt": due, "sentAt": "2026-10-05T08:01:00+05:30", "deliveredAt": "2026-10-05T08:01:05+05:30"},
        {"familyId": FAM, "kind": "dose_reminder", "dueAt": due, "sentAt": "2026-10-05T08:09:00+05:30", "deliveredAt": "2026-10-05T08:09:05+05:30"},
        {"familyId": FAM, "kind": "dose_reminder", "dueAt": due, "held": True, "heldReason": "done"},
        {"familyId": FAM, "kind": "reminder", "dueAt": due, "sentAt": "2026-10-05T08:10:00+05:30", "deliveredAt": None},
        {"familyId": FAM, "kind": "reminder", "dueAt": due, "sentAt": "2026-10-05T08:10:00+05:30", "deliveredAt": "2026-10-05T08:10:02+05:30"},
        {"familyId": FAM, "kind": "dose_reminder", "dueAt": due, "failedAt": "2026-10-05T08:00:30+05:30", "error": "131047"},
        {"familyId": FAM, "kind": "dose_reminder", "dueAt": due},
    ]
    got = daily.reminders(slots)[FAM]
    assert (got["num"], got["den"]) == (2, 6) and got["held"] == {"done": 1}
    whys = sorted(f["what"] for f in got["failures"])
    assert whys == ["failed: 131047", "not sent", "sent 9 min late", "sent, no delivered receipt from WhatsApp"]
    # the reminder ledger has no WhatsApp receipts: "delivered" is Meta's acceptance there
    old = daily.reminders([{"familyId": FAM, "kind": "reminder", "dueAt": due, "sentAt": "2026-10-05T08:02:00+05:30"}], receipts=False)[FAM]
    assert (old["num"], old["den"]) == (1, 1)


# ── check 2: nothing she was told is missing from memory ──────────────────────


async def test_check2_told_and_saved_is_remembered_told_and_not_saved_is_missing(db, at, sessions, fake_judge):
    at("2026-10-05 09:00")
    await roster(db)
    await store.add_turn(db, family_id=FAM, thread_id="e-ds", role="user", text="Mujhe moongphali se allergy hai", speaker_id="e-ds")
    await store.add_turn(db, family_id=FAM, thread_id="e-ds", role="user", text="Mera beta Ravi har Sunday aata hai", speaker_id="e-ds")
    await fact(db, "allergy:peanut", {"allergen": "peanut"}, "Allergic to peanuts", source="elder_said", by="e-ds")
    await db.commit()

    def answer(system, user):  # the judge reads the memory it is given
        return json.dumps({"items": [
            {"what": "peanut allergy", "kind": "fact", "said": "moongphali se allergy", "remembered": "[allergy:peanut]" in system},
            {"what": "son Ravi visits on Sundays", "kind": "person", "said": "beta Ravi", "remembered": "Ravi" in system},
        ], "invented": []})

    fake_judge(answer)
    out = await daily.family_checks(sessions, FAM, "2026-10-05")
    assert out["remembered"] == {"num": 1, "den": 2, "errors": 0}
    row = (await db.execute(select(DailyScore).where(DailyScore.check == "remembered"))).scalars().one()
    assert row.value == 50.0 and row.failures[0]["what"] == "not remembered: son Ravi visits on Sundays"


# ── check 3: a correction held ────────────────────────────────────────────────


async def test_check3_a_correction_then_the_old_value_is_not_held(db, at, sessions, fake_judge):
    at("2026-10-05 09:00")
    await roster(db)
    await fact(db, "medicine:telma", {"name": "Telma", "dose": "40 mg", "times": ["08:00"]}, "Telma 40 mg at 08:00")
    at("2026-10-05 10:00")
    await fact(db, "medicine:telma", {"times": ["21:00"]}, "Telma 40 mg at 21:00")  # a caregiver's change
    await fact(db, "naming:address_as", {"name": "Amma", "avoid": ["Maa"]}, "Call her Amma, never Maa", source="elder_said", by="e-ds")
    at("2026-10-05 10:05")
    await log_reply(db, "Ji Amma, ab Telma raat 9 baje.", to="c-ds")  # the next reply to the caregiver who corrected: held
    at("2026-10-05 19:00")
    await log_reply(db, "Maa, Telma 08:00 ko le lijiye", to="e-ds")  # the old time and the avoided name
    await db.commit()

    def answer(system, user):
        data = json.loads(user)
        old = data["CORRECTION"]["old"]
        return json.dumps({"messages": [{"n": m["n"], "relevant": "Telma" in m["text"], "held": not ("08:00" in m["text"] and "08:00" in old),
                                         "words": "Telma 08:00" if "08:00" in m["text"] else ""} for m in data["MESSAGES"]]})

    fake_judge(answer)
    got = await daily.corrections_held(sessions, FAM, "2026-10-05")
    by = {c["key"]: c for c in got["checked"]}
    assert by["medicine:telma"]["held"] is False  # the judge saw the old time
    assert by["naming:address_as"]["held"] is False and "used 'Maa'" in by["naming:address_as"]["words"]  # code: the avoided name
    assert (got["num"], got["den"]) == (0, 2)


async def test_check3_held_when_her_messages_use_the_new_value(db, at, sessions, fake_judge):
    at("2026-10-05 09:00")
    await roster(db)
    await fact(db, "medicine:telma", {"name": "Telma", "dose": "40 mg", "times": ["08:00"]}, "Telma 40 mg at 08:00")
    at("2026-10-05 10:00")
    await fact(db, "medicine:telma", {"times": ["21:00"]}, "Telma 40 mg at 21:00")
    at("2026-10-05 20:55")
    await log_reply(db, "Telma ka time ho gaya, raat 9 baje 🙏", to="e-ds")
    await db.commit()
    fake_judge(lambda s, u: json.dumps({"messages": [{"n": 1, "relevant": True, "held": True}]}))
    got = await daily.corrections_held(sessions, FAM, "2026-10-05")
    assert (got["num"], got["den"]) == (1, 1)


# ── the minimum rule, check 1 from the backend, everyone's total ─────────────


class Host:
    def __init__(self, slots, source="nudge_log"):
        self.slots, self.source, self.calls = slots, source, []

    async def call(self, tool, args, **kw):
        self.calls.append((tool, args))
        if tool == "delivery_day":
            return {"ok": True, "result": {"source": self.source, "receipts": self.source == "message_deliveries", "slots": self.slots}}
        return {"ok": True, "result": {"sent": True, "to": 1}}


async def test_overall_is_the_minimum_and_no_data_is_not_100(db, at, sessions):
    at("2026-10-06 03:00")
    day = "2026-10-05"
    await daily.put(db, day, FAM, "remembered", 80.0, 10, [], {"num": 8})
    await daily.put(db, day, FAM, "corrections", None, 0, [], {"num": 0})
    await daily.put(db, day, "fam-2", "remembered", 100.0, 5, [], {"num": 5})
    await daily.put(db, day, "fam-2", "corrections", 50.0, 2, [{"what": "x"}], {"num": 1})
    await db.commit()
    due = "2026-10-05T08:00:00+05:30"
    host = Host([{"familyId": FAM, "kind": "reminder", "dueAt": due, "sentAt": due},
                 {"familyId": "fam-2", "kind": "reminder", "dueAt": due, "sentAt": due}, {"familyId": "fam-2", "kind": "reminder", "dueAt": due}])
    out = await daily.finish(sessions, host, day)
    rows = {(r.family_id, r.check): r for r in (await db.execute(select(DailyScore).where(DailyScore.day == date(2026, 10, 5)))).scalars()}
    assert rows[(FAM, "reminders")].value == 100.0
    assert rows[(FAM, "overall")].value == 80.0  # corrections had no data: left out, never counted as 100
    assert rows[("fam-2", "overall")].value == 50.0  # the lowest check
    assert rows[("all", "reminders")].value == pytest.approx(66.7) and rows[("all", "remembered")].value == pytest.approx(86.7)
    assert rows[("all", "overall")].value == 50.0 and out["overall"] == 50.0
    assert rows[("all", "reminders")].detail["source"] == "nudge_log" and rows[("all", "reminders")].detail["receipts"] is False


async def test_check1_error_is_reported_not_scored(db, at, sessions):
    at("2026-10-06 03:00")

    class Broken:
        async def call(self, tool, args, **kw):
            return {"error": "Unknown tool: delivery_day"}

    out = await daily.finish(sessions, Broken(), "2026-10-05")
    assert out["reminders_error"] and out["reminders"] is None


# ── the 08:00 email ───────────────────────────────────────────────────────────


async def test_morning_email_lists_every_failure_and_sends_once(db, at, sessions):
    at("2026-10-06 08:00")
    day = "2026-10-05"
    await daily.put(db, day, "all", "overall", 50.0, 3, [], {})
    await daily.put(db, day, "all", "reminders", 100.0, 4, [], {"num": 4, "held": {"done": 1}, "source": "nudge_log", "receipts": False})
    await daily.put(db, day, "all", "remembered", 87.5, 8, [{"family": "fam-ds", "person": "Kamla", "what": "not remembered: son Ravi"}], {"num": 7})
    await daily.put(db, day, "all", "corrections", 50.0, 2, [{"family": "fam-ds", "person": "Kamla", "what": "correction of naming:address_as not held",
                                                              "words": ["used 'Maa'"]}], {"num": 1})
    await daily.put(db, day, "all", "replies", 90.0, 10, [{"family": "fam-ds", "person": "Kamla", "what": "wrong_name", "words": "Maa"}], {"num": 9})
    db.add(EvalScore(run_id="123", kind="nightly", day=date(2026, 10, 5), passed=False, commit="abc", branch="master",
                     scores={"judge": {"overall": {"n": 900, "pct": 94.0}, "languages": {"hi": {"n": 40, "pct": 87.5}}}, "probes": {"selected": 1000, "executed": 990}},
                     failures=["hi: judge 87.5% < 90% (n=40)"], created_at=clock.now()))
    await db.commit()
    subject, text, html = await report.email(db, day)
    assert subject.startswith("ALERT: Saheli daily score 2026-10-05: 50%")
    assert "not remembered: son Ravi" in text and "correction of naming:address_as not held" in text and "wrong_name" in text
    assert "hi 87.5% (40)" in text and "FAILED" in text and "1 held by rule" in text and "Meta accepted" in text
    assert "/daily-score?day=2026-10-05" in text and "<div>" in html
    host = Host([])
    assert (await report.send(sessions, host, day))["sent"] is True
    assert host.calls[0][0] == "team_report" and host.calls[0][1]["subject"] == subject
    assert (await report.send(sessions, host, day))["sent"] is False and len(host.calls) == 1  # a retry sends nothing twice


# ── the nightly world score endpoint ─────────────────────────────────────────


async def test_eval_scores_need_the_nightly_service_account(monkeypatch):
    from fastapi import HTTPException
    from starlette.requests import Request

    from app.api import quality

    req = Request({"type": "http", "headers": [(b"authorization", b"Bearer x.y.z")]})
    monkeypatch.delenv("EVAL_POSTER_SA", raising=False)
    with pytest.raises(HTTPException) as e:
        await quality.verify_eval_poster(req)
    assert e.value.status_code == 503
    monkeypatch.setenv("EVAL_POSTER_SA", "nightly@kavach-care.iam.gserviceaccount.com")
    monkeypatch.setattr(quality, "caller_email", lambda h: "someone@else.com")
    with pytest.raises(HTTPException) as e:
        await quality.verify_eval_poster(req)
    assert e.value.status_code == 403
    monkeypatch.setattr(quality, "caller_email", lambda h: "nightly@kavach-care.iam.gserviceaccount.com")
    assert await quality.verify_eval_poster(req) == "nightly@kavach-care.iam.gserviceaccount.com"


async def test_a_failing_night_is_stored_and_emailed(db, at, monkeypatch):
    from app.api import quality
    from app.brain import host as host_mod

    at("2026-10-06 02:30")
    sent = []

    class Live:
        async def call(self, tool, args, **kw):
            sent.append((tool, args))
            return {"ok": True, "result": {"sent": True}}

    monkeypatch.setattr(host_mod, "LiveHost", Live)
    body = quality.EvalIn(run_id="999", day="2026-10-05", passed=False, commit="abc123", branch="master",
                          scores={"judge": {"overall": {"pct": 80.0, "n": 100}}}, failures=["judge pass 80% < 95%"])
    out = await quality.post_eval_scores(body, db, "nightly@x")
    assert out["stored"] is True and sent and sent[0][0] == "team_report" and "FAILED" in sent[0][1]["subject"]
    row = (await db.execute(select(EvalScore))).scalars().one()
    assert row.passed is False and row.failures == ["judge pass 80% < 95%"]
    # the same run posted again replaces it
    body.passed = True
    await quality.post_eval_scores(body, db, "nightly@x")
    db.expire_all()
    assert (await db.execute(select(EvalScore))).scalars().one().passed is True


# ── inside the 02:00 dream ────────────────────────────────────────────────────


async def test_the_dream_runs_the_daily_score_after_the_merge(db, at, sessions, fake_judge, monkeypatch):
    from sqlalchemy import delete

    from app.care import dream
    from app.care.models import CareEvent

    monkeypatch.setenv("DAILY_SCORE", "on")
    monkeypatch.setenv("FACT_MERGE", "dryrun")
    await db.execute(delete(FamilyRoster).where(FamilyRoster.family_id != FAM))
    at("2026-10-05 09:00")
    await roster(db)
    await fact(db, "allergy:peanut", {"allergen": "peanut"}, "Allergic to peanuts")
    await store.add_turn(db, family_id=FAM, thread_id="e-ds", role="user", text="moongphali se allergy hai", speaker_id="e-ds")
    at("2026-10-05 09:01")
    await log_reply(db, "Ji, yaad rakhungi 🙏", said="moongphali se allergy hai")
    await db.commit()

    def answer(system, user):
        if "keep a person's care record free of duplicates" in system or "duplicates" in system.lower() and "groups" in system.lower():
            return json.dumps({"groups": []})
        if "SAHELI'S MESSAGE TO GRADE" in user:
            return verdict()
        if "MEMORY AFTER THE CONVERSATION" in system:
            return json.dumps({"items": [{"what": "peanut allergy", "kind": "fact", "said": "moongphali", "remembered": True}], "invented": []})
        if "[Memory check" in user:
            return json.dumps({"allergies": ["peanut"], "never_call": [], "medicines": []})
        return "Quiet day."

    fake_judge(answer)
    at("2026-10-06 02:00")
    stats = await dream.dream_all(sessions, host=Host([{"familyId": FAM, "kind": "dose_reminder", "dueAt": "2026-10-05T08:00:00+05:30",
                                                        "sentAt": "2026-10-05T08:00:20+05:30", "deliveredAt": "2026-10-05T08:00:25+05:30"}]))
    q = stats["quality"]
    assert q["reminders"] == 100.0 and q["remembered"] == 100.0 and q["replies"] == 100.0 and q["overall"] == 100.0
    mark = (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "dream"))).scalars().one()
    fq = mark.payload["quality"]
    assert fq["merged"] is True and fq["judge"]["graded"] == 1 and fq["memory"]["people"] == 1 and not fq["errors"]
    assert await nightly.merged_tonight(db, FAM, "2026-10-05")  # the 02:15 consolidate leaves this family's merge alone


async def test_morning_email_has_the_lessons_section(db, at):
    from app.learn.models import PlaybookVersion, RuleProposal

    at("2026-10-06 08:00")
    await daily.put(db, "2026-10-05", "all", "overall", 90.0, 3, [], {})
    db.add(PlaybookVersion(version=31, status="live", lessons={"general": ["Ask before assuming a relation."]}, examples={}, created_at=clock.now()))
    db.add(PlaybookVersion(version=32, status="canary", lessons={"general": ["In Bengali, use apni for elders."]}, examples={}, created_at=clock.now(),
                           canary={"families": 3, "score_delta": 0.012}))
    db.add(PlaybookVersion(version=30, status="rolled_back", lessons={"general": ["Keep replies under 2 lines."]}, examples={}, created_at=clock.now()))
    db.add(RuleProposal(created_at=clock.now(), situation="general", rule="Tell them the safe insulin dose", why="touches a safety rule", status="dropped",
                        decided_by="auto"))
    await db.commit()
    _subject, text, _html = await report.email(db, "2026-10-05")
    assert "Lessons (Saheli's self-learning):" in text
    assert "live: v31: Ask before assuming a relation." in text and "in testing: canary: v32: In Bengali" in text and "score_delta 0.012" in text
    assert "rolled back: v30" in text and "dropped by the safety gate: Tell them the safe insulin dose (why: touches a safety rule)" in text
