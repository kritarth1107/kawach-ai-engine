"""What is said on WhatsApp reaches memory and the dashboard (founder 2026-10-13: "most data chatted around in WhatsApp
with Saheli doesn't get structured in memory properly, and moods in dashboard should be also changed regularly").

- The 30-minute re-read kept nothing from 4 to 13 Oct 2026 (its commit ran after the session closed), and the nightly one
  read 120 turns, so a busy family fell a day further behind each day. Now it keeps what it reads and catches up in batches.
- "Takes no regular medicines" was filed in the never-order list: "has none" is now a structured answer (a <domain>:none
  fact), never a medicine to remind or an item to avoid, ended by a real one.
- Home's mood came only from the backend's nightly snapshot ("—" all day): moods now carry a score and a word, are
  logged in the conversation or by the re-read, and Home shows the latest."""

import json

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import tools
from app.care import extract, mood, store
from app.care.models import CareEvent, Turn
from app.core.security import verify_api_secret
from app.db import session as db_session
from app.db.session import get_db
from app.llm import router
from app.llm.router import LLMReply
from app.main import app
from app.sim.world import SimHost

FAM = "fam-mem"
ELDER = {"id": "elder-m", "name": "Kamla", "role": "elder"}
SON = {"id": "son-m", "name": "Ravi", "role": "primary caregiver"}
SELF = {"id": "self-m", "name": "Arjun", "role": "primary caregiver"}


class Reader:
    """The re-read model: answers with whatever the test sets."""

    def __init__(self):
        self.out: dict = {}
        self.calls = 0

    async def complete(self, route, **kw):
        self.calls += 1
        return LLMReply(text=json.dumps(self.out), tool_calls=[], model="fake")


@pytest.fixture
def reader(monkeypatch):
    r = Reader()
    router.register_provider("fake", r)
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"], "brain": ["fake:m"]}')
    router.reset_breakers()
    yield r
    router._providers.pop("fake", None)


@pytest.fixture
async def api(db, monkeypatch, reader):
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    monkeypatch.setattr(db_session, "SessionLocal", sessions)  # jobs open their own sessions

    async def _db():
        yield db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[verify_api_secret] = lambda: None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c
    app.dependency_overrides.clear()


def ctx(db, host, person, members, said=""):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=person, speaker=person, members=members, user_text=said)


async def call(c, name: str, args: dict) -> tuple[dict, bool]:
    out, err = await tools.run(c, name, args)
    return json.loads(out), err


async def test_the_30_minute_re_read_keeps_what_it_read_and_catches_up_in_batches(api, db, at, reader, monkeypatch):
    at("2026-10-13 10:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    monkeypatch.setattr(extract, "BATCH", 3)
    for i in range(7):
        await store.add_turn(db, family_id=FAM, thread_id=ELDER["id"], role="user", text=f"baat {i}", speaker_id=ELDER["id"])
    await db.commit()
    reader.out = {"facts": [{"domain": "home", "name": "cook", "details": {"name": "Radha"}, "sentence": "Cook Radha comes at 11",
                             "evidence": "Radha 11 baje aati hai"}]}
    r = await api.post("/v2/jobs/extract")
    assert r.status_code == 200
    db.expire_all()
    left = (await db.execute(select(Turn.id).where(Turn.family_id == FAM, Turn.extracted.is_(False)))).scalars().all()
    assert not left, "every turn was read and that was kept (it was rolled back every 30 minutes)"
    assert reader.calls == 3, "7 turns in batches of 3: one run caught up"
    assert any(f.key == "home:cook" for f in await store.facts(db, FAM, ELDER["id"])), "what it heard is in memory"


async def test_has_none_is_an_answer_never_a_medicine_or_an_item_to_avoid(db, at):
    at("2026-10-13 01:14")
    await store.save_roster(db, FAM, SELF, [SELF])
    host = SimHost()
    c = ctx(db, host, SELF, [SELF], "I take no regular medicines")
    out, _ = await call(c, "remember", {"domain": "medicine", "name": "no medicines", "none": True, "details": {},
                                        "sentence": "Takes no regular medicines."})
    facts = {f.key: f for f in await store.facts(db, FAM, SELF["id"])}
    assert "medicine:none" in facts and facts["medicine:none"].status == "active"
    assert not [k for k in facts if k.startswith("no_order:")], "not in the never-order list"
    assert not host.world.schedules and "reminders" not in str(out), "no reminder for 'none'"
    refused, err = await call(c, "remember", {"domain": "diet", "name": "x", "none": True, "details": {}, "sentence": "No diet rules."})
    assert err and "refused" in refused
    # a real one ends "none"; then "none" is refused while one is listed
    await call(c, "remember", {"domain": "medicine", "name": "Amoxicillin", "details": {"dose": "250 mg", "times": ["09:00", "21:00"]},
                                    "sentence": "Amoxicillin 250 mg twice a day"})
    facts = {f.key: f for f in await store.facts(db, FAM, SELF["id"], statuses=("active", "pending", "stopped"))}
    assert facts["medicine:none"].status == "stopped" and facts["medicine:amoxicillin"].status == "active"
    again, _ = await call(c, "remember", {"domain": "medicine", "name": "none", "none": True, "details": {}, "sentence": "No medicines."})
    assert again["saved"] is False and "Amoxicillin" in again["why"]


async def test_the_re_read_files_none_as_none_and_skips_it_when_real_ones_are_listed(db, at, reader):
    at("2026-10-13 10:00")
    await store.save_roster(db, FAM, SELF, [SELF])
    await store.add_turn(db, family_id=FAM, thread_id=SELF["id"], role="user", text="koi allergy nahi hai, koi dawai bhi nahi", speaker_id=SELF["id"])
    await store.write_fact(db, family_id=FAM, subject_id=SELF["id"], domain="medicine", key="medicine:metformin", value={"name": "Metformin"},
                           text="Metformin 500 mg", source_kind="caregiver_said")
    reader.out = {"facts": [{"domain": "allergy", "none": True, "sentence": "No known allergies.", "evidence": "koi allergy nahi"},
                            {"domain": "medicine", "none": True, "sentence": "Takes no medicines.", "evidence": "koi dawai nahi"},
                            {"domain": "no_order", "name": "medicines", "none": True, "sentence": "No medicines.", "evidence": "x"}]}
    await extract.extract_family(db, FAM)
    keys = {f.key: f.status for f in await store.facts(db, FAM, SELF["id"], statuses=("active", "pending"))}
    assert keys.get("allergy:none") == "pending", "a health 'none' from a re-read waits for a confirm like any other"
    assert "medicine:none" not in keys, "Metformin is listed: not 'none'"


async def test_mood_has_a_score_and_a_word_a_repeat_is_not_logged_twice_and_a_change_is(db, at):
    at("2026-10-13 09:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    c = ctx(db, SimHost(), ELDER, [ELDER, SON], "aaj mann bahut accha hai")
    out, _ = await call(c, "log_event", {"kind": "mood", "summary": "Cheerful, enjoyed the morning walk", "mood_score": 5, "mood_word": "Cheerful"})
    assert out["logged"]
    at("2026-10-13 09:20")
    again, _ = await call(c, "log_event", {"kind": "mood", "summary": "Still happy", "mood_score": 5, "mood_word": "happy"})
    assert again["logged"] is False
    at("2026-10-13 09:30")
    await call(c, "log_event", {"kind": "mood", "summary": "Worried about her knee pain", "mood_score": 2, "mood_word": "worried"})
    now = await mood.now_for(db, FAM, ELDER["id"])
    assert now["score"] == 2 and now["word"] == "worried" and now["source"] == "chat"
    # the event that shows it can carry the mood: a visit that made her happy
    at("2026-10-13 19:00")
    await call(c, "log_event", {"kind": "social", "summary": "Daughter visited, had dinner together", "mood_score": 5, "mood_word": "happy"})
    assert (await mood.now_for(db, FAM, ELDER["id"]))["word"] == "happy"
    kinds = [e.kind for e in (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM).order_by(CareEvent.id))).scalars()]
    assert kinds[-2:] == ["mood", "social"]
    rows = (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "mood").order_by(CareEvent.id))).scalars().all()
    assert len(rows) == 3 and rows[1].payload["severity"] == "watch"


async def test_the_re_read_notes_a_mood_saheli_missed_and_not_one_she_noted(db, at, reader):
    at("2026-10-13 15:00")
    await store.save_roster(db, FAM, SELF, [SELF])
    await store.add_turn(db, family_id=FAM, thread_id=SELF["id"], role="user", text="bahut thak gaya hoon aaj, kuch accha nahi lag raha", speaker_id=SELF["id"])
    reader.out = {"moods": [{"about": SELF["id"], "score": 2, "word": "tired", "summary": "Very tired and low today", "evidence": "bahut thak gaya"}]}
    counts = await extract.extract_family(db, FAM)
    assert counts.get("moods") == 1
    assert (await mood.now_for(db, FAM, SELF["id"]))["source"] == "reread"
    # next conversation: Saheli noted the mood herself, so the re-read adds none
    at("2026-10-13 18:00")
    await store.add_turn(db, family_id=FAM, thread_id=SELF["id"], role="user", text="ab theek hoon, gym ho gaya", speaker_id=SELF["id"])
    await call(ctx(db, SimHost(), SELF, [SELF]), "log_event", {"kind": "mood", "summary": "Better after the gym", "mood_score": 4, "mood_word": "better"})
    reader.out = {"moods": [{"about": SELF["id"], "score": 4, "word": "fine", "summary": "Fine after gym"}]}
    counts = await extract.extract_family(db, FAM)
    assert not counts.get("moods")
    assert (await mood.now_for(db, FAM, SELF["id"]))["word"] == "better"
    # a backlog read a day late: that day's mood is not logged as today's
    await store.add_turn(db, family_id=FAM, thread_id=SELF["id"], role="user", text="aaj bahut udaas hoon", speaker_id=SELF["id"])
    at("2026-10-14 12:00")
    reader.out = {"moods": [{"about": SELF["id"], "score": 2, "word": "sad", "summary": "Sad"}]}
    counts = await extract.extract_family(db, FAM)
    assert not counts.get("moods")


async def test_home_shows_the_latest_mood_of_the_last_day(api, db, at):
    at("2026-10-13 09:00")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    r = (await api.get(f"/v2/dash/{FAM}/{ELDER['id']}/home")).json()
    assert r["mood"] is None
    await mood.record(db, family_id=FAM, subject_id=ELDER["id"], score=4, word="calm", summary="Calm after her bhajan")
    await db.commit()
    r = (await api.get(f"/v2/dash/{FAM}/{ELDER['id']}/home")).json()
    assert r["mood"]["word"] == "calm" and r["mood"]["score"] == 4
    at("2026-10-14 10:00")
    r = (await api.get(f"/v2/dash/{FAM}/{ELDER['id']}/home")).json()
    assert r["mood"] is None, "a day-old mood is not today's"
