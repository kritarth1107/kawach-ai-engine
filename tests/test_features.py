"""Care views shared by WhatsApp tools and the dashboard: refills, emergency card, appointments,
family tasks, wellbeing, report and spending."""

from datetime import timedelta

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import tools, wake
from app.care import features, store
from app.care.domains import fact_key
from app.core import clock
from app.sim.world import SimHost
from app.tasks.models import Task

FAM, ELDER, CG, BRO = "fam-f", "elder-f", "cg-f", "bro-f"


def ctx(db, speaker=CG, role="caregiver"):
    return tools.TurnCtx(
        session=db, host=SimHost(), family_id=FAM, elder={"id": ELDER, "name": "Kamla"},
        speaker={"id": speaker, "name": "Asha" if speaker == CG else "Kamla", "role": role},
        members=[{"id": ELDER, "name": "Kamla"}, {"id": CG, "name": "Asha"}, {"id": BRO, "name": "Ravi"}],
    )


async def call(c, name, args):
    import json

    out, err = await tools.run(c, name, args)
    assert not err, out
    return json.loads(out)


async def test_stock_counts_down_and_flags_refill(db, at):
    at("2026-10-01 08:00")
    c = ctx(db)
    await call(c, "remember", {"domain": "medicine", "name": "Amlodipine", "details": {"dose": "5 mg", "times": ["08:00", "20:00"]}, "sentence": "Amlodipine 5 mg twice"})
    rows = await call(c, "medicine_stock", {})
    assert rows["medicines"][0]["stock"] is None and rows["unknown"] == ["Amlodipine"]
    await call(c, "set_stock", {"medicine": "amlodipine", "count": 14})
    for when in ("2026-10-01 08:10", "2026-10-01 20:10", "2026-10-02 08:10", "2026-10-02 20:10"):  # four doses (the same dose said twice is one)
        at(when)
        await call(ctx(db, ELDER, "elder"), "log_dose", {"medicine": "Amlodipine", "outcome": "taken"})
    row = (await features.stock(db, FAM, ELDER))[0]
    assert row["stock"] == 10 and row["daysLeft"] == 5 and row["low"]
    low = await features.refill_candidates(db)
    assert [(f, s, r["key"]) for f, s, r in low] == [(FAM, ELDER, fact_key("medicine", "Amlodipine"))]
    # Restocking clears it.
    await call(c, "set_stock", {"medicine": "Amlodipine", "count": 60})
    assert not (await features.stock(db, FAM, ELDER))[0]["low"]


async def test_self_care_logs_on_the_caregiver_not_the_elder(db, at):
    at("2026-10-02 09:00")
    c = ctx(db)
    await call(c, "remember", {"domain": "medicine", "name": "Vitamin D", "details": {"times": ["09:00"]}, "sentence": "Vitamin D at 9", "about": CG})
    await call(c, "log_dose", {"medicine": "Vitamin D", "outcome": "taken", "about": CG})
    await call(c, "log_vital", {"kind": "bp", "value": "128/82", "about": CG})
    mine = await store.events(db, FAM, CG, day=clock.ist_day())
    elders = await store.events(db, FAM, ELDER, day=clock.ist_day())
    assert {e.kind for e in mine} >= {"dose_taken", "vital"} and not [e for e in elders if e.kind in ("dose_taken", "vital")]
    assert [f.key for f in await store.facts(db, FAM, CG, domains=["medicine"])] == ["medicine:vitamin_d"]


async def test_emergency_card_and_link(db, at):
    at("2026-10-02 09:00")
    c = ctx(db)
    await call(c, "remember", {"domain": "profile", "name": "blood_group", "details": {"value": "B+"}, "sentence": "Blood group B+"})
    await call(c, "remember", {"domain": "allergy", "name": "Penicillin", "details": {"allergen": "penicillin", "reaction": "rash"}, "sentence": "Allergic to penicillin"})
    await call(c, "remember", {"domain": "contact", "name": "Asha", "details": {"name": "Asha", "phone": "+91 90000 00000", "relation": "daughter", "emergency": True}, "sentence": "Call Asha first"})
    out = await call(c, "emergency_card", {})
    assert "Blood Group: B+" in out["text"] and "penicillin (rash)" in out["text"] and "Asha (daughter)" in out["text"]
    assert out["link"] == f"https://kavach.test/e/{ELDER}" and out["missing"] == []


async def test_appointment_wakes_evening_before_and_two_hours_before(db, at):
    at("2026-10-02 09:00")
    c = ctx(db)
    out = await call(c, "remember", {"domain": "appointment", "name": "Dr Iyer", "details": {"doctor": "Dr Iyer", "when": "2026-10-05T11:00", "place": "Apollo"}, "sentence": "Dr Iyer 5 Oct 11:00"})
    assert len(out["reminders"]) == 3  # evening before, 2 h before, and the follow-up after the visit
    loops = [l for l in await store.live_loops(db, FAM, [ELDER]) if l.kind == "appointment"]
    assert sorted(clock.ist(l.wake_at).strftime("%d %H:%M") for l in loops) == ["04 19:00", "05 09:00"]
    q = await call(c, "add_doctor_question", {"appointment": "Dr Iyer", "question": "Can the BP dose be lowered?"})
    assert q["questions"] == ["Can the BP dose be lowered?"]
    team = await call(c, "care_team", {})
    assert team["upcoming"][0]["doctor"] == "Dr Iyer" and team["upcoming"][0]["questions"] == ["Can the BP dose be lowered?"]


async def test_family_task_reminds_assignee_once(db, at):
    now = at("2026-10-02 09:00")
    c = ctx(db)
    out = await call(c, "assign_family_task", {"title": "Call Maa tonight", "to": BRO, "due": "2026-10-02T20:00"})
    tasks = (await call(c, "family_tasks", {}))["tasks"]
    assert tasks[0]["assignee"] == BRO and tasks[0]["status"] == "open" and out["due"] == "02 Oct 20:00"
    loop = [l for l in await store.live_loops(db, FAM, [ELDER]) if l.kind == "family_task"][0]
    assert loop.detail["max_wakes"] == 1 and clock.ist(loop.wake_at).strftime("%H:%M") == "20:00"
    prompt = await wake.wake_prompt(db, loop, ELDER)
    assert "assigned to person bro-f" in prompt and "send_message" in prompt


async def test_one_shot_wake_does_not_repeat(db, at, monkeypatch):
    from app.llm import router
    from app.llm.router import LLMReply

    class Quiet:
        async def complete(self, route, **kw):
            return LLMReply(text="none", tool_calls=[], model="fake")

    router.register_provider("quiet", Quiet())
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["quiet:m"], "extract": ["quiet:m"]}')
    router.reset_breakers()
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    now = at("2026-10-02 19:00")
    await store.save_roster(db, FAM, {"id": ELDER, "name": "Kamla"}, [{"id": ELDER, "name": "Kamla"}, {"id": BRO, "name": "Ravi"}])
    loop = await features.add_family_task(db, family_id=FAM, subject_id=ELDER, title="Call Maa", assignee=BRO, due=now + timedelta(minutes=5), by=CG)
    await db.commit()
    at("2026-10-02 19:06")
    assert (await wake.wake_due(sessions, lambda fid: SimHost()))["ran"] == 1
    await db.refresh(loop)
    assert loop.wake_at is None and loop.status == "open"
    router._providers.pop("quiet", None)


async def test_wellbeing_and_report(db, at):
    at("2026-09-30 09:00")
    c = ctx(db)
    await call(c, "remember", {"domain": "medicine", "name": "Metformin", "details": {"times": ["08:00"]}, "sentence": "Metformin at 8"})
    e = ctx(db, ELDER, "elder")
    for day in ("2026-09-30", "2026-10-01", "2026-10-02"):
        at(f"{day} 08:30")
        await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="ho gaya", speaker_id=ELDER)
        await call(e, "log_dose", {"medicine": "Metformin", "outcome": "taken"})
    await call(e, "log_event", {"kind": "mood", "summary": "Felt low, missed Ravi", "severity": "watch"})
    await call(e, "log_vital", {"kind": "bp", "value": "150/95"})
    w = await call(c, "wellbeing", {"days": 7})
    assert w["streak"] == 3 and "Felt low" in w["summary"]
    r = await features.report(db, FAM, ELDER, 7)
    assert r["adherence"]["taken"] == 3 and r["adherence"]["byMedicine"][0]["name"] == "Metformin"
    assert r["vitals"][0]["value"] == "150/95" and r["mood"][0]["text"].startswith("Felt low")
    text = (await call(c, "weekly_report", {}))
    assert "3 of 7 doses" in text["summary"] and text["link"].endswith(f"recipient={ELDER}&days=7")


async def test_spending_counts_only_placed_orders(db, at):
    import uuid

    now = at("2026-10-02 09:00")
    for status, placed, total, cancel in (("done", True, "₹283", False), ("done", True, "₹120", True), ("failed", False, "₹500", False)):
        db.add(Task(id=uuid.uuid4(), family_id=FAM, subject_id=ELDER, requested_by=CG, service="instamart", kind="order", goal="atta",
                    details={}, status=status, phase="place", result={"placed": placed, "total": total}, history=[], runs=0,
                    cancel_requested=cancel, created_at=now, updated_at=now, deadline_at=now))
    await db.flush()
    s = await call(ctx(db), "spending", {"month": "2026-10"})
    assert s["total"] == 283 and s["count"] == 1 and s["byService"] == {"instamart": 283}
