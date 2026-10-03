"""Anonymiser and scoring-signal regressions from the Data Curator's corpus spot-check
(OpenBot/Shared/saheli-lab/reviews/corpus-spotcheck.md). Simulated names only."""

from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care import store
from app.core import clock
from app.learn import scoring, situations
from app.learn.anonymise import anonymise, indic_key, latin_key, leaks, looks_like_junk
from app.learn.models import ReplyLog

NAMES = ["Neha Joshi", "Savitri Gupta", "Gopal", "Venkatesh Iyer", "Subrata Banerjee", "Gurpreet Kaur", "Kamla", "Rahul", "Amit"]
MEDS = ["Glimepiride", "Lantus Glargine", "Shelcal"]


@pytest.mark.parametrize("latin,native", [
    ("Neha", "नेहा"), ("Joshi", "जोशी"), ("Savitri", "सावित्री"), ("Gupta", "गुप्ता"), ("Kamla", "कमला"), ("Amit", "अमित"),
    ("Subrata", "সুব্রত"), ("Gurpreet", "ਗੁਰਪ੍ਰੀਤ"), ("Gopal", "கோபால்"), ("Joshi", "ஜோஷி"), ("Rahul", "రాహుల్"), ("Vijay", "विजय"),
])
def test_same_name_same_sound_in_every_script(latin, native):
    assert latin_key(latin) == indic_key(native)


@pytest.mark.parametrize("text,gone", [
    ("नेहा को परेशान मत करना", "नेहा"),
    ("डॉक्टर जोशी के साथ आपकी अपॉइंटमेंट", "जोशी"),
    ("मम्मीजी (सावित्री गुप्ता)", "सावित्री"),
    ("ektu gopal ke bolbo", "gopal"),
    ("Gopaler kotha bolo", "Gopal"),
    ("இன்று கோபாலுக்கு போன் செய்தேன்", "கோபால"),
    ("সুব্রত আসবে", "সুব্রত"),
    ("ਗੁਰਪ੍ਰੀਤ ਆ ਰਹੀ ਹੈ", "ਗੁਰਪ੍ਰੀਤ"),
    ("इंसुलिन और ग्लिमेपिराइड जारी रखने को बोला है", "ग्लिमेपिराइड"),
    ("Glargine raat ko", "Glargine"),
    ("gluconorm and ecosprin taken", "gluconorm"),  # not in this family's list: the shared brand list
])
def test_names_and_medicines_in_any_script_and_case_are_scrubbed(text, gone):
    out = anonymise(text, names=NAMES, medicines=MEDS)
    assert gone.lower() not in out.lower()
    assert leaks(out, names=NAMES, medicines=MEDS) == []


@pytest.mark.parametrize("text", [
    "Tell me one simple Roti dinner item. Dalma okay na?",
    "Til chikki Baba sathi safe aahe",
    "clinic on Tuesdays and Fridays",
    "शुभ रात्रि बेटा, खुश रहो",
    "Shab-ba-khair Ammi",
    "Hip pain, Physio said Ghutne ke liye walker",
    "Adaab! Alhamdulillah sab theek",
    "Wait what hospital?? Emergency what?!",
    "household doorbell beta-blockers dahi lane the colony",
    "Faiz ki nazm suni, Kaku aali",
])
def test_ordinary_words_keep_their_meaning(text):
    assert anonymise(text, names=NAMES, medicines=MEDS) == text


def test_addresses_still_scrubbed():
    out = anonymise("Flat 4B, MG Road; house no. 12, Lajpat Nagar", names=[])
    assert "4B" not in out and "MG" not in out and "Lajpat" not in out and "12" not in out


def test_leaks_catches_unscrubbed_script_names_and_medicines():
    assert "name:Neha" in leaks("नेहा आई", names=NAMES)
    assert "medicine:Glimepiride" in leaks("ग्लिमेपिराइड ली", medicines=MEDS)


def test_junk_never_enters():
    assert looks_like_junk("das_actors = orchestrator.registry")
    assert looks_like_junk("see /tmp/simulated_users/da-bijay_11.txt .raw_output")
    assert looks_like_junk('The user prompt says "NOW: day 3"')
    assert not looks_like_junk("Now: subah ki goli le li")


@pytest.mark.parametrize("text,tone", [
    ("Radhe Radhe 🙏", "neutral"), ("All good 👍", "neutral"), ("Thank you beta, bahut accha laga 🙏", "warm"),
    ("Why again? Told you at 7.50 itself", "annoyed"), ("dont disturb repeatedly", "annoyed"), ("useless fellow, the attendant", "neutral"),
    ("remote useless ho gaya", "neutral"), ("আবার কেন মেসেজ?", "annoyed"),
])
def test_tone_signals(text, tone):
    assert scoring.tone_of(text) == tone


@pytest.mark.parametrize("text", ["Shelcal? Shelcal konachi ahe?", "I already took it", "Wait what hospital??", "ভুল বলছো", "ye meri nahi hai"])
def test_corrections_in_more_languages(text):
    assert scoring.CORRECTED.search(text)


def test_kha_chuka_is_not_a_correction():
    assert not scoring.CORRECTED.search("Haan main kha chuka hoon")


def test_alert_without_red_flag_when_fine_is_not_an_emergency():
    assert situations.tag(text="तबीयत बिल्कुल ठीक है", role="elder", tools=["alert_caregiver"]) != "emergency"
    assert situations.tag(text="Mujhe chakkar aa raha hai aur seene mein dard", role="elder", tools=[]) == "emergency"
    assert situations.tag(text="Gir gayi thi", role="elder", tools=["alert_caregiver"]) == "emergency"


async def test_reaction_credited_only_to_last_message_in_a_burst(db, at):
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    t0 = at("2026-10-02 10:00")
    for i, txt in enumerate(["Mummy ji, BP check kiya?", "Aur sugar bhi dekh lijiye"]):
        db.add(ReplyLog(family_id="fam-b", thread_id="e-b", at=t0 + timedelta(minutes=i), kind="proactive", situation="followup",
                        text=txt, text_len=len(txt), tools=[], sent_hour=10))
    await db.commit()
    clock.set_now(t0 + timedelta(minutes=5))
    await store.add_turn(db, family_id="fam-b", thread_id="e-b", role="user", text="Why again? Told you already", speaker_id="e-b")
    await db.commit()
    clock.set_now(t0 + timedelta(hours=7))
    await scoring.score_pending(sessions)
    first, second = (await db.execute(select(ReplyLog).where(ReplyLog.family_id == "fam-b").order_by(ReplyLog.at))).scalars().all()
    assert first.replied and first.tone is None and second.tone == "annoyed" and second.corrected


async def test_late_answer_to_a_reply_is_a_new_topic(db, at):
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    t0 = at("2026-10-02 10:00")
    db.add(ReplyLog(family_id="fam-n", thread_id="e-n", at=t0, kind="reply", situation="chit_chat", text="Accha ji", text_len=8, tools=[], sent_hour=10))
    await db.commit()
    clock.set_now(t0 + timedelta(hours=4))
    await store.add_turn(db, family_id="fam-n", thread_id="e-n", role="user", text="Aaj kya banau?", speaker_id="e-n")
    await db.commit()
    clock.set_now(t0 + timedelta(hours=7))
    await scoring.score_pending(sessions)
    row = (await db.execute(select(ReplyLog).where(ReplyLog.family_id == "fam-n"))).scalars().one()
    assert row.replied is False
