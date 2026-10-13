"""Regular nudges (Kavach NUdge's plan §5): a personal message after each hour of chat silence while they are awake, never
mid-chat, with the unanswered ladder, caregivers every 3 hours, early-bird messages for late risers, a data question 5
minutes after an unanswered reminder, and shadow mode that sends nothing."""

import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care import nudges, store
from app.care.models import NudgeLedger
from app.llm import router, spend
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM = "fam-nudge"
MAA = {"id": "maa-n", "name": "Kamla", "role": "elder"}
SON = {"id": "son-n", "name": "Ravi", "role": "primary caregiver", "relation": "son", "relation_to": "maa-n"}
DOC = {"id": "doc-n", "name": "Dr Rao", "role": "family doctor"}
LINES = ["How was the walk in the park this morning?", "Did the new plant on the balcony get some sun today?",
         "What are you cooking for dinner tonight?", "Is the knee feeling any better after resting?",
         "Any good news from your sister today?", "Which song have you been humming lately?",
         "Did you get some rest this afternoon?", "How is the reading going with that new book?",
         "Was the tea at the neighbour's place nice?", "Did the grandchildren call you today?",
         "How did the temple visit go?", "Is the new pillow helping you sleep better?"]


class Writer:
    """The composer model, faked: a different line each call unless a test scripts its answers."""

    def __init__(self):
        self.calls, self.script, self.on_call, self.seen, self.usage = 0, [], None, [], {}

    async def complete(self, route, **kw):
        system = kw.get("system_stable") or ""
        head = " ".join(system if isinstance(system, list) else [system])
        if head.startswith("You read how a person took"):
            self.reacted = getattr(self, "reacted", 0) + 1
            self.replies_seen = getattr(self, "replies_seen", []) + [kw["messages"][-1]["content"][0]["text"]]
            return LLMReply(text=json.dumps({"reaction": getattr(self, "reaction", "neutral")}), tool_calls=[], model="fake", usage=self.usage)
        if head.startswith("You grade one WhatsApp message"):
            self.graded = getattr(self, "graded", 0) + 1
            return LLMReply(text=json.dumps(getattr(self, "grade", {"pass": True, "issues": []})), tool_calls=[], model="fake", usage=self.usage)
        if head.startswith("You check a WhatsApp reply from Saheli"):
            self.grounded = getattr(self, "grounded", 0) + 1
            return LLMReply(text=json.dumps({"claims": getattr(self, "claims", [])}), tool_calls=[], model="fake", usage=self.usage)
        if head.startswith("You check one WhatsApp message"):
            self.verified = getattr(self, "verified", 0) + 1
            return LLMReply(text=json.dumps(getattr(self, "verdict", {"ok": True})), tool_calls=[], model="fake", usage=self.usage)
        self.calls += 1
        self.seen.append(kw["messages"][-1]["content"][0]["text"])
        if self.on_call:
            await self.on_call()
        if self.script:
            return LLMReply(text=self.script.pop(0), tool_calls=[], model="fake", usage=self.usage)
        return LLMReply(text=json.dumps({"option_id": "o1", "text": LINES[(self.calls - 1) % len(LINES)], "field": None,
                                         "expects_reply": True}), tool_calls=[], model="fake", usage=self.usage)


@pytest.fixture
def writer(monkeypatch):
    w = Writer()
    router.register_provider("fake", w)
    monkeypatch.setenv("MODEL_ROUTES", '{"worker": ["fake:m"], "brain": ["fake:m"], "judge_fast": ["fake:m"], "judge": ["fake:m"]}')
    monkeypatch.delenv("NUDGE_REGULAR", raising=False)
    router.reset_breakers()
    spend.reset()
    nudges._mode.update(value=None, at=0.0)
    nudges._everyone.update(at=0.0, rates={})
    yield w
    nudges._mode.update(value=None, at=0.0)  # the flag cache never leaks into another test
    router._providers.pop("fake", None)
    router.reset_breakers()


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


async def family(db, members=(MAA, SON)):
    await store.save_roster(db, FAM, MAA, list(members))
    await db.commit()


async def wrote(db, who, text="I went to the temple today, it felt lovely"):
    await store.add_turn(db, family_id=FAM, thread_id=who["id"], role="user", text=text, speaker_id=who["id"])
    await db.commit()


def sent_to(host, who) -> list[str]:
    return [m["text"] for m in host.world.sent if m.get("to") == who["id"]]


async def rows(db, who=MAA) -> list[NudgeLedger]:
    return list((await db.execute(select(NudgeLedger).where(NudgeLedger.family_id == FAM, NudgeLedger.person_id == who["id"])
                                  .order_by(NudgeLedger.planned_at))).scalars())


async def run(sessions, host, mode="on"):
    return await nudges.run(sessions, lambda f: host, mode=mode)


# 1. the silence clock
async def test_a_nudge_comes_after_an_hour_of_silence_and_not_a_minute_before(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db)
    await wrote(db, MAA)
    host = SimHost()
    at("2026-10-14 10:59")
    await run(sessions, host)
    assert sent_to(host, MAA) == []
    at("2026-10-14 11:00")
    await run(sessions, host)
    assert sent_to(host, MAA) == [LINES[0]]
    assert [(r.status, r.kind) for r in await rows(db)][-1][0] == "sent"
    turn = (await store.recent_turns(db, FAM, MAA["id"]))[-1]
    assert turn.role == "assistant" and turn.meta["nudge"]["id"] and turn.message_ref.startswith("nudge:")


@pytest.mark.parametrize("source", ["saheli_turn", "reminder_event", "backend_outbound"])
async def test_any_message_from_saheli_counts_as_that_hours_message(db, at, sessions, writer, source):
    at("2026-10-14 10:00")
    await family(db)
    await wrote(db, MAA)
    host = SimHost()
    at("2026-10-14 10:30")
    if source == "saheli_turn":
        await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="assistant", text="Okay, take care of yourself")
    elif source == "reminder_event":
        await store.record_event(db, family_id=FAM, subject_id=MAA["id"], kind="reminder_sent", summary="reminder: walk at 10:30",
                                 payload={"custom": True})
    else:
        host.world.outbound.append({"to": MAA["id"], "text": "Your schedule for today", "at": "2026-10-14T10:30:00+05:30"})
    await db.commit()
    at("2026-10-14 11:00")
    await run(sessions, host)
    assert sent_to(host, MAA) == []
    at("2026-10-14 11:30")
    await run(sessions, host)
    assert len(sent_to(host, MAA)) == 1


# 2. never mid-chat
async def test_a_message_from_them_while_the_nudge_is_written_drops_it(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db)
    await wrote(db, MAA)
    host = SimHost()
    at("2026-10-14 11:00")

    async def they_write():
        await wrote(db, MAA, "I just had my lunch, dear")

    writer.on_call = they_write
    out = await run(sessions, host)
    assert sent_to(host, MAA) == []
    assert out.get("talking") == 1
    assert [r.reason for r in await rows(db)] == ["talking"]


# 3. busy, paused, window closed, asleep
@pytest.mark.parametrize("state,reason", [("flow", "busy"), ("paused", "paused"), ("closed", "window_closed"), ("night", "asleep")])
async def test_busy_paused_closed_or_asleep_means_no_nudge_for_the_right_reason(db, at, sessions, writer, state, reason):
    at("2026-10-14 10:00")
    await family(db)
    await wrote(db, MAA)
    host = SimHost()
    if state == "flow":
        host.world.flows[MAA["id"]] = "ride:await_choice"
    elif state == "paused":
        host.world.paused = True
    elif state == "closed":
        host.world.closed.add(MAA["id"])
    at("2026-10-14 22:00" if state == "night" else "2026-10-14 11:30")
    out = await run(sessions, host)
    assert sent_to(host, MAA) == []
    assert out["skipped"].get(reason) == 1
    logged = [r.reason for r in await rows(db)]
    assert logged == ([] if reason == "asleep" else [reason])  # a night is not logged every run


# 4. the unanswered ladder (D1)
async def test_unanswered_nudges_slow_down_then_stop_until_they_write_or_the_next_morning(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db)
    await wrote(db, MAA)
    host = SimHost()

    async def at_(when: str) -> int:
        at(f"2026-10-{when}")
        await run(sessions, host)
        return len(sent_to(host, MAA))

    assert await at_("14 11:00") == 1  # an hour after their message
    assert await at_("14 12:00") == 2  # one unanswered: still hourly
    assert await at_("14 13:00") == 2  # two unanswered: every 2 hours
    assert await at_("14 14:00") == 3
    assert await at_("14 16:00") == 3  # three unanswered: every 3 hours
    assert await at_("14 17:00") == 4
    assert await at_("14 20:00") == 4  # four unanswered: stopped
    assert await at_("15 09:00") == 5  # the next morning's first nudge
    assert await at_("15 11:00") == 5
    assert await at_("15 15:00") == 5
    await wrote(db, MAA, "Yes dear, all is fine here")  # 15:00: a reply resets the ladder
    assert await at_("15 16:00") == 6
    assert [r.unanswered_run for r in await rows(db) if r.status == "sent"][-1] == 0


# 5. caregivers every 3 hours; doctors never; silent caregivers never
async def test_caregivers_every_three_hours_doctors_and_silent_caregivers_never(db, at, sessions, writer):
    at("2026-09-10 10:00")
    await family(db, (MAA, SON, DOC, {"id": "niece-n", "name": "Asha", "role": "caregiver"}))
    await wrote(db, {"id": "niece-n"}, "Kaisi hain maa?")  # more than 30 days ago
    at("2026-10-14 10:00")
    await wrote(db, SON, "How is Maa today?")
    await wrote(db, DOC, "Please keep the BP log")
    host = SimHost()
    at("2026-10-14 12:00")
    await run(sessions, host)
    assert sent_to(host, SON) == []
    at("2026-10-14 13:00")
    await run(sessions, host)
    assert len(sent_to(host, SON)) == 1
    assert sent_to(host, DOC) == [] and sent_to(host, {"id": "niece-n"}) == []


# 6. late risers: early-bird messages
async def routine(db, who, type_, start):
    await store.write_fact(db, family_id=FAM, subject_id=who["id"], domain="activity", key=f"activity:{type_}",
                           value={"type": type_, "kind": "routine", "start": start}, text=f"{type_} {start}", source_kind="caregiver_said")
    await db.commit()


async def test_a_late_riser_gets_at_most_two_soft_morning_messages_then_the_first_one_at_wake_up(db, at, sessions, writer):
    at("2026-10-13 23:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    await routine(db, MAA, "sleep", "03:00")
    await routine(db, MAA, "wake_up", "13:30")
    host = SimHost()
    sent = []
    for t in ("07:45", "08:20", "08:30", "09:30", "10:30", "11:00", "12:30", "13:10", "13:30"):
        at(f"2026-10-14 {t}")
        before = len(sent_to(host, MAA))
        await run(sessions, host)
        if len(sent_to(host, MAA)) > before:
            sent.append(t)
    assert sent == ["08:30", "10:30", "13:30"]  # 2 early birds 2 h apart (none in the first 5 h of sleep), then at wake-up
    kinds = [r.kind for r in await rows(db) if r.status == "sent"]
    assert kinds[:2] == ["early_bird", "early_bird"] and kinds[2] != "early_bird"
    assert "expects_reply false" in writer.seen[0]


async def test_a_late_riser_who_answers_early_is_awake_and_gets_the_hourly_cadence(db, at, sessions, writer):
    at("2026-10-13 23:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    await routine(db, MAA, "sleep", "03:00")
    await routine(db, MAA, "wake_up", "13:30")
    host = SimHost()
    at("2026-10-14 08:30")
    await run(sessions, host)  # early bird 1
    at("2026-10-14 09:00")
    await wrote(db, MAA, "I woke up early today")
    at("2026-10-14 10:00")
    await run(sessions, host)
    assert len(sent_to(host, MAA)) == 2
    assert [r.kind for r in await rows(db) if r.status == "sent"][-1] != "early_bird"


# 7. the reminder follow-up
async def reminder(db, at_text, *, custom=True, about=None, rid="rem-q"):
    from app.api.brain import EventIn, push_event

    payload = {"custom": custom, "reminderId": rid, "slot": "18:00", "dateKey": at_text[:10], "preview": "Stay smoke-free this evening 🚭",
               **({"aboutKey": about} if about else {})}
    await push_event(EventIn(family_id=FAM, subject_id=MAA["id"], kind="reminder_sent", summary="reminder: no cigarette at 18:00",
                             payload=payload, ref=f"nudge:{rid}:{at_text}:sent"), db)


async def test_five_minutes_after_an_unanswered_reminder_one_data_question(db, at, sessions, writer):
    at("2026-10-14 17:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    at("2026-10-14 18:00")
    await reminder(db, "2026-10-14 18:00")
    thread = await store.recent_turns(db, FAM, MAA["id"])
    assert thread[-1].meta.get("via") == "reminder" and "smoke-free" in thread[-1].text  # the reminder is in the chat (§3.8)
    at("2026-10-14 18:04")
    out = await nudges.reminder_followups(sessions, lambda f: host, mode="on")
    assert out["sent"] == 0
    at("2026-10-14 18:05")
    out = await nudges.reminder_followups(sessions, lambda f: host, mode="on")
    assert out["sent"] == 1 and len(sent_to(host, MAA)) == 1
    assert [r.kind for r in await rows(db)] == ["reminder_followup"]
    assert "smoke-free" in writer.seen[-1]
    at("2026-10-14 18:08")
    assert (await nudges.reminder_followups(sessions, lambda f: host, mode="on"))["sent"] == 0  # one per reminder
    await run(sessions, host)
    assert len(sent_to(host, MAA)) == 1  # the reminder and its follow-up are this hour's messages


async def test_no_follow_up_after_a_reply_or_for_a_medicine_reminder(db, at, sessions, writer):
    at("2026-10-14 17:00")
    await family(db, (MAA,))
    host = SimHost()
    at("2026-10-14 18:00")
    await reminder(db, "2026-10-14 18:00", rid="rem-a")
    at("2026-10-14 18:02")
    await wrote(db, MAA, "No smoking for me today")
    at("2026-10-14 18:06")
    out = await nudges.reminder_followups(sessions, lambda f: host, mode="on")
    assert out["sent"] == 0 and out["skipped"].get("answered") == 1
    at("2026-10-15 18:00")
    await reminder(db, "2026-10-15 18:00", rid="rem-m", about="medicine:metformin")
    at("2026-10-15 18:06")
    out = await nudges.reminder_followups(sessions, lambda f: host, mode="on")
    assert out["reminders"] == 0 and sent_to(host, MAA) == []  # medicines keep their own follow-up (D3)


def test_after_three_unanswered_follow_ups_only_every_other_reminder_gets_one():
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)

    def row(i, status="sent", reason="", replied=False):
        return NudgeLedger(kind="reminder_followup", status=status, reason=reason, planned_at=t0 - timedelta(hours=i),
                           sent_at=t0 - timedelta(hours=i) if status == "sent" else None, replied_at=t0 if replied else None)

    assert not nudges.followup_ladder_skips([row(1), row(2)])
    three = [row(1), row(2), row(3)]
    assert nudges.followup_ladder_skips(three)
    assert not nudges.followup_ladder_skips([row(0, "skipped", "ladder"), *three])  # skipped last time: this one goes
    assert nudges.followup_ladder_skips([row(0), row(1, "skipped", "ladder"), *three])
    assert not nudges.followup_ladder_skips([row(1), row(2), row(3, replied=True)])


# 8. the last 5 messages either way
async def test_the_context_has_the_last_five_messages_either_way_in_order_with_reminders(db, at, sessions, writer):
    at("2026-10-14 09:00")
    await family(db, (MAA,))
    for i, (role, text) in enumerate([("user", "one"), ("assistant", "two"), ("user", "three"), ("assistant", "four")]):
        at(f"2026-10-14 09:{10 + i:02d}")
        await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role=role, text=text)
    await db.commit()
    gate = {"last_outbound_preview": [{"text": "four", "at": "2026-10-14T09:13:30+05:30"},  # the same message, recorded by both
                                      {"text": "Walk reminder: shoes on!", "at": "2026-10-14T09:20:00+05:30"}]}
    got = await nudges.last_messages(db, FAM, MAA["id"], gate)
    assert [(who, t) for _, who, t in got] == [("Saheli", "two"), ("them", "three"), ("Saheli", "four"), ("Saheli", "Walk reminder: shoes on!")][-5:] \
        or [(who, t) for _, who, t in got] == [("them", "one"), ("Saheli", "two"), ("them", "three"), ("Saheli", "four"), ("Saheli", "Walk reminder: shoes on!")]
    assert len(got) == 5 and got[-1][2] == "Walk reminder: shoes on!" and [t for _, _, t in got].count("four") == 1


# 9. the check, and at most 2 retries
def ctx(**kw):
    base = dict(person=nudges.Person(FAM, MAA["id"], "Kamla", "elder", True, True), text="", known="walk park evening", recent=[],
                saheli_recent=["How was the walk in the park this morning?"], profile=None,
                options=[nudges.Option(id="o1", kind="reflect", why="the evening")], max_len=220)
    return nudges.Context(**{**base, **kw})


def test_the_check_rejects_repeats_long_texts_two_questions_wrong_script_and_invented_numbers():
    ok = {"option_id": "o1", "text": "What made you smile this evening?"}
    assert nudges.check(ok, ctx()) == []
    assert any("too close" in p for p in nudges.check({**ok, "text": "How was the walk in the park this morning?!"}, ctx()))
    assert any("too long" in p for p in nudges.check({**ok, "text": "x" * 300}, ctx()))
    assert any("more than one question" in p for p in nudges.check({**ok, "text": "Did you eat? Did you walk?"}, ctx()))
    assert any("reading" in p for p in nudges.check({**ok, "text": "Your BP was 150/95 yesterday, well done."}, ctx()))
    assert any("option_id" in p for p in nudges.check({**ok, "option_id": "o9"}, ctx()))
    deva = {"script": "devanagari", "lang": "hi"}
    assert nudges.check({**ok, "text": "What made you smile this evening, Kamla ji?"}, ctx(profile=deva))  # they read Devanagari


async def test_a_draft_that_keeps_failing_is_tried_three_times_then_skipped(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    bad = json.dumps({"option_id": "o1", "text": "Did you eat? Did you walk? Did you sleep?"})
    writer.script = [bad, bad, bad]
    host = SimHost()
    at("2026-10-14 11:00")
    out = await run(sessions, host)
    assert writer.calls == 3 and sent_to(host, MAA) == []
    assert out["skipped"].get("guard") == 1
    assert (await rows(db))[-1].reason.startswith("guard:")


# 10. their reply answers it
async def test_a_reply_to_a_nudge_that_asked_for_something_is_saved_and_answers_it(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    writer.script = [json.dumps({"option_id": "o1", "text": "How did you sleep last night, one to five?", "field": "sleep"})]
    host = SimHost()
    at("2026-10-14 11:00")
    await run(sessions, host)
    hint = await nudges.reply_hint(db, FAM, MAA["id"])
    fld = (await rows(db))[-1].field  # the picked option's field
    assert fld and f"asked about {fld}" in hint and "save it" in hint
    at("2026-10-14 11:10")
    await nudges.note_reply(db, FAM, MAA["id"])
    await db.commit()
    assert (await rows(db))[-1].replied_at is not None
    at("2026-10-14 18:00")
    assert await nudges.reply_hint(db, FAM, MAA["id"]) == ""  # older than 6 hours


# 11. refs never collide
async def test_every_nudge_has_its_own_ref_and_a_second_run_in_the_same_minute_adds_nothing(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    at("2026-10-14 11:00")
    await run(sessions, host)
    await run(sessions, host)
    at("2026-10-14 12:00")
    await run(sessions, host)
    sent = [r for r in await rows(db) if r.status == "sent"]
    assert len(sent) == 2 and len({r.ref for r in sent}) == 2 and len(sent_to(host, MAA)) == 2


async def test_a_check_in_retried_the_same_day_gets_its_own_ref(db, at, monkeypatch):
    from app.brain import wake
    from app.care import checkins

    refs = []

    async def fake_turn(sessions, host, fid, prompt, ref, deliver_to=None, still_true=None):
        refs.append(ref)

    async def plan(*a, **kw):
        return "feelings", "[Check-in] ask"

    monkeypatch.setattr(wake, "system_turn", fake_turn)
    monkeypatch.setattr(checkins, "plan", plan)
    await family(db, (MAA,))
    sess = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    at("2026-10-14 10:30")
    await checkins.run(sess, lambda f: SimHost())
    at("2026-10-14 16:30")
    await checkins.run(sess, lambda f: SimHost())
    assert len(refs) == 2 and refs[0] != refs[1]


# 12. shadow and off
async def test_shadow_writes_and_logs_but_sends_nothing_and_keeps_its_own_cadence(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    at("2026-10-14 11:00")
    await run(sessions, host, mode="shadow")
    at("2026-10-14 11:05")
    await run(sessions, host, mode="shadow")
    assert host.world.sent == []
    logged = await rows(db)
    assert [r.status for r in logged] == ["shadow"] and logged[0].text == LINES[0]
    at("2026-10-14 12:00")
    await run(sessions, host, mode="shadow")
    assert [r.status for r in await rows(db)] == ["shadow", "shadow"]


async def test_off_does_nothing_and_the_mode_comes_from_the_backend_flag(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    at("2026-10-14 11:00")
    out = await nudges.run(sessions, lambda f: host)  # no mode given: the flag decides (off by default)
    assert out["mode"] == "off" and host.world.sent == [] and await rows(db) == [] and writer.calls == 0
    host.world.nudge_mode = "on"
    nudges._mode["at"] -= 2 * nudges.MODE_TTL  # a flag change takes effect within a minute (the plan's kill switch)
    out = await nudges.run(sessions, lambda f: host)
    assert out["mode"] == "on" and len(sent_to(host, MAA)) == 1
    assert nudges.cached_mode() == "on"  # the old check-ins step aside


async def test_someone_who_never_wrote_or_not_in_24_hours_gets_no_nudge_even_if_the_backend_calls_the_chat_open(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA, {"id": "dad-n", "name": "Mohan", "role": "elder"}))
    host = SimHost()
    at("2026-10-14 12:00")
    out = await run(sessions, host)
    assert host.world.sent == [] and out["skipped"].get("window_closed") == 1  # the care recipient never wrote
    await wrote(db, MAA)
    at("2026-10-15 11:55")
    out = await run(sessions, host)
    assert host.world.sent == [] and out["skipped"].get("window_closed") == 1  # her last message was 25 hours ago
    host.world.inbound[MAA["id"]] = "2026-10-15T09:00:00+05:30"  # a Done tap the backend saw (no engine turn)
    out = await run(sessions, host)
    assert len(sent_to(host, MAA)) == 1


async def test_the_reviewer_sends_back_a_draft_that_says_something_happened_when_it_did_not(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    writer.verdict = {"ok": False, "problems": ["invented: says the walk is done, it is still ahead"]}
    host = SimHost()
    at("2026-10-14 11:00")
    out = await run(sessions, host)
    assert sent_to(host, MAA) == [] and writer.calls == 3 and writer.verified == 3
    assert out["skipped"].get("guard") == 1 and "invented" in (await rows(db))[-1].reason
    assert "says the walk is done" in writer.seen[-1]  # the rewrite was told why


def test_a_long_reference_fits_its_column_and_the_same_set_stays_equal():
    many = "doses:" + ",".join(f"sched-{i:04d}-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" for i in range(12))
    short = nudges.short_ref(many)
    assert len(short) <= 160 and short == nudges.short_ref(many) and short != nudges.short_ref(many + ",x")
    assert nudges.short_ref("doses:a,b") == "doses:a,b"


async def test_a_nudge_about_many_unmarked_doses_is_logged_not_lost(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    for i in range(12):  # twelve items past their time and unmarked
        sid = f"sched-{i:04d}-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        host.world.schedules[sid] = {"scheduleId": sid, "title": f"Tablet {i}", "time": "08:00", "active": True}

    async def schedule(tool, args, **kw):
        if tool == "get_today_schedule":
            return {"items": [{"scheduleId": s, "title": r["title"], "time": "08:00", "status": "missed"} for s, r in host.world.schedules.items()]}
        return await SimHost.call(host, tool, args, **kw)

    host.call = schedule
    writer.script = [json.dumps({"option_id": "o1", "text": "Did the morning tablets happen today?", "field": "doses"})]
    at("2026-10-14 11:00")
    await run(sessions, host, mode="shadow")
    logged = [r for r in await rows(db) if r.status == "shadow"]
    assert len(logged) == 1 and logged[0].kind == "doses" and len(logged[0].about_ref) <= 160


async def test_every_ledger_row_that_cost_a_model_call_carries_what_it_cost(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    writer.usage = {"in": 5000, "out": 250}
    host = SimHost()
    at("2026-10-14 11:00")
    await run(sessions, host, mode="shadow")
    one = spend.cost_inr("fake", writer.usage)
    logged = await rows(db)
    assert [r.status for r in logged] == ["shadow"] and logged[0].cost_inr == pytest.approx(2 * one, rel=1e-3)  # writer + reviewer
    writer.verdict = {"ok": False, "problems": ["not_personal: fits anyone"]}
    at("2026-10-14 12:00")
    await run(sessions, host, mode="shadow")
    skipped = [r for r in await rows(db) if r.status == "skipped"]
    assert len(skipped) == 1 and skipped[0].cost_inr == pytest.approx(6 * one, rel=1e-3)  # three tries, each written and reviewed


# 13. Saheli learns from how they took it (self-learning)
async def answered(db, at, sessions, writer, host, reply: str, reaction: str, when="2026-10-14 11:10"):
    """A nudge goes at 11:00; they reply; the reply is read."""
    at("2026-10-14 11:00")
    await run(sessions, host)
    at(when)
    await wrote(db, MAA, reply)
    await nudges.note_reply(db, FAM, MAA["id"])
    await db.commit()
    writer.reaction = reaction
    return await nudges.read_reactions(sessions)


async def test_every_nudge_keeps_the_context_it_was_written_from(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA, "Going to the temple fair with Sunita this evening")
    at("2026-10-14 11:00")
    await run(sessions, SimHost(), mode="shadow")
    row = (await rows(db))[-1]
    assert row.status == "shadow" and "temple fair with Sunita" in row.context and "OPTIONS" in row.context


async def test_a_reply_asking_saheli_to_stop_stops_her_own_messages_for_thirty_days(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    out = await answered(db, at, sessions, writer, host, "Please don't keep messaging me like this", "stop")
    assert out["stop"] == 1 and (await rows(db))[-1].reaction == "stop"
    assert "don't keep messaging me" in writer.replies_seen[-1]  # the model read their words; no word list did
    at("2026-10-14 13:00")
    out = await run(sessions, host)
    assert len(sent_to(host, MAA)) == 1 and out["skipped"].get("asked_stop") == 1
    at("2026-11-14 10:00")
    await wrote(db, MAA, "Good morning")
    at("2026-11-14 11:00")
    await run(sessions, host)
    assert len(sent_to(host, MAA)) == 2  # thirty days later, the hourly nudge comes back


async def test_a_question_they_did_not_want_rests_and_too_many_messages_doubles_the_silence(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    writer.script = [json.dumps({"option_id": "o1", "text": "What was the best part of your day so far?", "field": "reflect:morning"})]
    host = SimHost()
    at("2026-10-14 11:00")
    await run(sessions, host)
    row = (await rows(db))[-1]
    row.kind = "reflect"  # an evening-reflection kind of question
    await db.commit()
    await answered(db, at, sessions, writer, host, "Why do you keep asking me this? Leave it.", "unwelcome")
    lr = await nudges.learned(db, FAM, MAA["id"], nudges.clock.now())
    assert "reflect" in lr.unwelcome  # light talk rests as a kind
    shown = await store.active_fact(db, FAM, MAA["id"], "learned:nudges_avoid_reflect")
    assert shown and "Didn't want to be asked about how-was-your-day questions" in shown.text  # visible, undoable
    p = nudges.Person(FAM, MAA["id"], "Kamla", "elder", True, True)
    ck = nudges.Clock(None, None, 0, False)
    at("2026-10-14 18:00")
    opts = await nudges.candidates(db, p, ck, nudges.clock.now(), facts=[], acts=[], schedule={}, reminders=[])
    assert not any(o.kind == "reflect" for o in opts)
    # "too many messages": the next nudge waits two hours instead of one
    writer.script = []
    at("2026-10-15 10:00")
    await wrote(db, MAA)
    at("2026-10-15 11:00")
    await run(sessions, host)
    n = len(sent_to(host, MAA))
    at("2026-10-15 11:05")
    await wrote(db, MAA, "Bahut zyada messages aate hain")
    await nudges.note_reply(db, FAM, MAA["id"])
    await db.commit()
    writer.reaction = "fewer"
    await nudges.read_reactions(sessions)
    at("2026-10-15 12:05")
    await run(sessions, host)
    assert len(sent_to(host, MAA)) == n
    at("2026-10-15 13:05")
    await run(sessions, host)
    assert len(sent_to(host, MAA)) == n + 1


async def test_one_reply_to_several_nudges_is_read_once_and_kinds_they_answer_score_higher(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    at("2026-10-14 11:00")
    await run(sessions, host)
    at("2026-10-14 12:00")
    await run(sessions, host)
    at("2026-10-14 12:30")
    await wrote(db, MAA, "So sorry, I was out. All good!")
    await nudges.note_reply(db, FAM, MAA["id"])
    await db.commit()
    writer.reaction = "welcome"
    out = await nudges.read_reactions(sessions)
    sent = [r for r in await rows(db) if r.status == "sent"]
    assert out["read"] == 1 and writer.reacted == 1 and [r.reaction for r in sent] == ["neutral", "welcome"]
    lr = nudges.Learned(rates={"reflect": 0.9, "know": 0.1})
    assert lr.weight("reflect") > 0 > lr.weight("know") and lr.weight("delight") == 0


async def test_real_nudges_are_graded_against_their_context_and_a_made_up_fact_is_an_error(db, at, sessions, writer, caplog):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    at("2026-10-14 11:00")
    await run(sessions, SimHost(), mode="shadow")
    writer.grade = {"pass": False, "issues": ["made_up"], "note": "says the doctor called; nothing says so"}
    out = await nudges.review(sessions)
    row = (await rows(db))[-1]
    assert out["graded"] == 1 and row.grade["pass"] is False and row.grade["issues"] == ["made_up"]
    assert any("nudge review: made_up" in r.message for r in caplog.records if r.levelname == "ERROR")
    assert (await nudges.review(sessions))["graded"] == 0  # graded once


async def test_a_good_answered_nudge_becomes_one_tuning_example_and_a_bad_one_does_not(db, at, sessions, writer):
    from eval import export_nudge_tuning as ex

    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    await answered(db, at, sessions, writer, host, "Haan beta, bahut accha tha!", "welcome")
    assert (await nudges.review(sessions))["graded"] == 1
    got = await ex.rows_to_learn_from(sessions, 30, "neutral")
    assert len(got) == 1
    e = ex.example(got[0])
    answer = json.loads(e["contents"][1]["parts"][0]["text"])
    assert e["contents"][0]["parts"][0]["text"] == got[0].context and answer["text"] == got[0].text
    assert f"{answer['option_id']} [{got[0].kind}]" in got[0].context
    row = (await rows(db))[-1]
    row.grade = {"pass": False, "issues": ["made_up"]}
    await db.commit()
    assert await ex.rows_to_learn_from(sessions, 30, "neutral") == []  # a graded failure never teaches


async def test_an_unmarked_item_is_asked_about_only_two_hours_after_its_time_by_name_and_once_a_day(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    p = nudges.Person(FAM, MAA["id"], "Kamla", "elder", True, True)
    sched = {"items": [{"scheduleId": "s1", "title": "Thyronorm 50", "time": "10:00", "status": "missed"}]}

    async def doses(when: str, ck=None) -> list:
        at(when)
        opts = await nudges.candidates(db, p, ck or nudges.Clock(None, None, 0, False), nudges.clock.now(), facts=[], acts=[],
                                       schedule=sched, reminders=[])
        return [o for o in opts if o.kind == "doses"]

    assert await doses("2026-10-14 11:00") == []  # the backend's own missed-dose follow-up is still asking
    got = await doses("2026-10-14 12:00")
    assert len(got) == 1 and "Thyronorm 50 at 10:00" in got[0].why and "naming each item" in got[0].why
    await nudges._log(db, p, nudges.clock.now(), kind="doses", status="sent", text="Did the 10 am Thyronorm happen?", fld="doses",
                      about=got[0].about_ref, sent=True)
    await db.commit()
    sched["items"].append({"scheduleId": "s2", "title": "Ecosprin", "time": "13:00", "status": "missed"})
    ck = nudges.Clock(None, None, 0, False, rows=await nudges.ledger_rows(db, FAM, MAA["id"]))
    assert await doses("2026-10-14 16:00", ck) == []  # asked once today already


async def test_the_ledger_keeps_the_options_field_not_the_writers_words_for_it(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    writer.script = [json.dumps({"option_id": "o1", "text": "How did the temple visit go today?", "field": "whether the temple trip went well"})]
    at("2026-10-14 11:00")
    await run(sessions, host, mode="shadow")
    row = (await rows(db))[-1]
    assert row.status == "shadow" and row.field != "whether the temple trip went well"
    assert f"[{row.kind}]" in row.context and (row.field is None or f"(field: {row.field})" in row.context)


async def test_the_flag_is_read_every_run_even_when_nobody_is_due_so_old_check_ins_stay_off(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    host.world.nudge_mode = "on"
    at("2026-10-14 10:20")  # within her hour: nobody is due, so no person reaches the gate
    out = await nudges.run(sessions, lambda f: host)
    assert out["mode"] == "on" and nudges.cached_mode() == "on" and host.world.sent == []
    nudges._mode["at"] -= 20 * nudges.MODE_TTL  # long after the last read
    assert nudges.cached_mode() is None
    at("2026-10-14 10:25")
    await nudges.run(sessions, lambda f: host)
    assert nudges.cached_mode() == "on"



# 14. trust plan: avoided names, rejected facts, no invention, learned hours, visible and undoable learnings
async def test_a_nudge_never_uses_an_avoided_name_or_something_the_family_said_is_wrong(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="naming", key="naming:address_as",
                           value={"name": "Amma", "avoid": ["Maa"]}, text="Call her Amma, never Maa", source_kind="caregiver_said")
    await db.commit()
    p = nudges.Person(FAM, MAA["id"], "Kamla", "elder", True, True)
    ctx = await nudges.build_context(db, SimHost(), p, nudges.Clock(None, None, 0, False), nudges.clock.now())
    assert "NEVER CALL THEM: Maa" in ctx.text and ctx.avoid == ["Maa"]
    bad = {"option_id": ctx.options[0].id, "text": "Maa, how was the walk in the park?"}
    assert any("never to be called" in x for x in nudges.check(bad, ctx))
    ctx.never = ["Metformin"]
    bad = {"option_id": ctx.options[0].id, "text": "Is the metformin suiting you?"}
    assert any("family said is wrong" in x for x in nudges.check(bad, ctx))


async def test_an_invented_detail_found_by_the_no_invention_check_is_rewritten(db, at, sessions, writer, monkeypatch):
    monkeypatch.setenv("GROUNDING_CHECK", "on")
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    writer.claims = [{"kind": "event", "quote": "Rohit called", "supported": False}]
    at("2026-10-14 11:00")
    out = await run(sessions, SimHost(), mode="shadow")
    assert out["skipped"].get("guard") == 1 and writer.grounded == 3 and "Rohit called" in writer.seen[-1]
    writer.claims = []
    at("2026-10-14 12:00")
    await run(sessions, SimHost(), mode="shadow")
    assert [r.status for r in await rows(db)][-1] == "shadow"
    p = nudges.Person(FAM, MAA["id"], "Kamla", "elder", True, True)
    await store.add_turn(db, family_id=FAM, thread_id=MAA["id"], role="assistant", text="Rohit called you, right?")
    await db.commit()
    ctx = await nudges.build_context(db, SimHost(), p, nudges.Clock(None, None, 0, False), nudges.clock.now())
    assert "Rohit called" not in ctx.memory and "Now:" in ctx.memory  # her own earlier lines are never a source


async def test_a_learned_bad_hour_holds_the_nudge(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    await store.write_fact(db, family_id=FAM, subject_id=MAA["id"], domain="learned", key="learned:timing_messages",
                           value={"avoid_hours": [11], "prefer_hours": [16, 17]}, text="Rarely answers 10–12", source_kind="learned")
    await db.commit()
    host = SimHost()
    at("2026-10-14 11:00")
    out = await run(sessions, host)
    assert sent_to(host, MAA) == [] and out["skipped"].get("learned_hour") == 1
    at("2026-10-14 12:00")
    await run(sessions, host)
    assert len(sent_to(host, MAA)) == 1


async def test_undoing_a_learned_stop_lets_nudges_come_back(db, at, sessions, writer):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    host = SimHost()
    await answered(db, at, sessions, writer, host, "Bas karo, itne message mat bhejo", "stop")
    fact = await store.active_fact(db, FAM, MAA["id"], "learned:nudges_pause")
    assert fact and "not to message on her own" in fact.text
    at("2026-10-14 13:00")
    assert (await run(sessions, host))["skipped"].get("asked_stop") == 1
    fact.status = "stopped"  # the caregiver pressed Undo on the dashboard
    await db.commit()
    at("2026-10-14 14:00")
    await run(sessions, host)
    assert len(sent_to(host, MAA)) == 2


async def test_kinds_they_answer_become_one_visible_learning_that_weights_the_picks(db, at, sessions, writer):
    at("2026-10-01 10:00")
    await family(db, (MAA,))
    p = nudges.Person(FAM, MAA["id"], "Kamla", "elder", True, True)
    for i in range(6):
        for kind, answered_ in (("reflect", True), ("know", False)):
            t = nudges.clock.now() + nudges.timedelta(hours=i * 3 + (1 if kind == "know" else 0))
            r = await nudges._log(db, p, t, kind=kind, status="sent", text=f"{kind} {i}", sent=True)
            r.replied_at = t + nudges.timedelta(minutes=10) if answered_ else None
    await db.commit()
    at("2026-10-03 10:00")
    out = await nudges.learn_topics(sessions)
    fact = await store.active_fact(db, FAM, MAA["id"], "learned:nudges_topics")
    assert out["saved"] == 1 and fact.value["answered"] == ["reflect"] and fact.value["rarely"] == ["know"]
    assert "Answers how-was-your-day questions most" in fact.text
    lr = await nudges.learned(db, FAM, MAA["id"], nudges.clock.now())
    assert lr.weight("reflect") > 0 > lr.weight("know")
    assert (await nudges.learn_topics(sessions))["saved"] == 0  # unchanged: no new version



async def test_a_fresh_process_grades_at_once_even_when_its_clock_starts_near_zero(db, at, sessions, writer, monkeypatch):
    at("2026-10-14 10:00")
    await family(db, (MAA,))
    await wrote(db, MAA)
    at("2026-10-14 11:00")
    await run(sessions, SimHost(), mode="shadow")
    monkeypatch.setattr(nudges, "_review", {"at": None})
    monkeypatch.setattr(nudges._time, "monotonic", lambda: 12.0)  # a container started 12 s ago (live 13 Oct: never graded)
    out = await nudges.learn(sessions)
    assert out["review"]["graded"] == 1
    assert "review" not in await nudges.learn(sessions)  # then once an hour
