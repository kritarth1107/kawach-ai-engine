"""Reply language for people with no saved setting (live 2026-10-11: a caregiver who writes English said "Leave it" and got
the answer in Hindi, the language of the elder in the same family). A message too short to tell keeps the language
Saheli has been writing to them in; one in an Indian script is answered in that script. Scripts only, no word lists."""

from app.brain import guards as g
from app.brain.loop import writing_profiles
from app.care import store

FAM = "fam-rl"
ROHAN = {"id": "cg-rl", "name": "Rohan Mehta", "role": "primary caregiver"}
SARLA = {"id": "el-rl", "name": "Sarla", "role": "elder"}

HINDI_DRAFT = "ठीक है, ककड़ी का ऑर्डर नहीं करते हैं। क्या मैं किसी और चीज़ में मदद करूँ?"
ENGLISH_DRAFT = "Okay, I won't order the cucumber. Anything else I can help with?"


def short_english(**extra) -> dict:
    p = g.profile(["Order cucumber", "Yes save it", "At night??", "Leave it"])
    assert p == {"script": "latin", "roman": None}, p  # each is too short to tell English from Hinglish by itself
    return {**p, "now": "latin", **extra}


def test_a_short_reply_keeps_english_when_saheli_has_been_writing_english():
    p = short_english(replied="latin")
    assert g.target(p) == "latin"
    assert g.language_problems(HINDI_DRAFT, p), "a switch to Hindi nobody asked for is sent back"
    assert not g.language_problems(ENGLISH_DRAFT, p)
    assert g.describe(p) == "English"


def test_without_any_reply_yet_the_model_decides_from_their_message():
    p = short_english()
    assert g.target(p) is None and not g.language_problems(HINDI_DRAFT, p)
    assert "never another family member's language" in g.describe(p)


def test_a_short_reply_after_hindi_replies_is_left_to_the_model():
    # Indian-script replies are never forced onto a Roman-letter writer: one wrong reply must not lock a language in.
    p = short_english(replied="devanagari")
    assert g.target(p) is None and not g.language_problems(ENGLISH_DRAFT, p) and not g.language_problems(HINDI_DRAFT, p)


def test_a_voice_note_in_hindi_is_answered_in_hindi():
    p = {**short_english(replied="latin"), "now": "devanagari"}
    assert g.target(p) == "devanagari"
    assert g.language_problems(ENGLISH_DRAFT, p)
    assert not g.language_problems(HINDI_DRAFT, p)


def test_saved_settings_and_long_messages_work_as_before():
    elder = {"script": "latin", "roman": None, "now": "latin", "replied": "latin", "saved": {"language": "hi"}}
    assert g.target(elder) == "devanagari", "a saved language still wins over English replies"
    hinglish = {**g.profile(["Haan beta, dawai le li hai", "kya hua aaj"]), "now": "latin", "replied": "latin"}
    assert g.target(hinglish) == "indic"
    english = {**g.profile(["Thanks, I will be there by six.", "Please remind him about the walk"]), "now": "latin"}
    assert g.target(english) == "latin"


def test_no_bare_name_opener_mid_conversation():
    names = ["Rohan Mehta", "Rohan"]
    out = g.tidy_opening("Rohan, your profile lists a cucumber allergy. Is this for Mummy?", greeted_today=True,
                         mid_conversation=True, names=names)
    assert out == "Your profile lists a cucumber allergy. Is this for Mummy?"
    kept = "Rohan, your profile lists a cucumber allergy. Is this for Mummy?"
    assert g.tidy_opening(kept, greeted_today=False, mid_conversation=False, names=names) == kept
    possessive = "Rohan's order is on the way and should come by 8 pm."
    assert g.tidy_opening(possessive, greeted_today=True, mid_conversation=True, names=names) == possessive
    other = "Okay, I have saved it for you."
    assert g.tidy_opening(other, greeted_today=True, mid_conversation=True, names=names) == other


def test_a_transcript_does_not_override_a_saved_language():
    p = {"script": "latin", "roman": "english", "saved": {"language": "en"}, "now": "devanagari", "transcript": True}
    assert g.target(p) == "latin"
    assert not g.language_problems(ENGLISH_DRAFT, p)
    assert g.language_problems(HINDI_DRAFT, p)
    assert "they just wrote" not in g.describe(p)


async def test_a_voice_transcript_does_not_become_how_they_write(db, at):
    at("2026-10-14 14:32")
    await store.add_turn(db, family_id=FAM, thread_id=ROHAN["id"], role="user",
                         text="Please remind me about the walk tonight", meta={"channel": "whatsapp"})
    await store.add_turn(db, family_id=FAM, thread_id=ROHAN["id"], role="user", text="मैंने अंग्रेजी में बात की",
                         meta={"voice": True})
    await db.commit()
    prof = (await writing_profiles(db, FAM, [ROHAN]))[ROHAN["id"]]
    assert prof.get("transcript") is not True and prof["now"] == "latin", prof
    assert g.target(prof) == "latin" and not g.language_problems(ENGLISH_DRAFT, prof)
    # nothing saved, and the only message is a transcript: answered in that script this turn, not learned as how they type
    await store.add_turn(db, family_id=FAM, thread_id=SARLA["id"], role="user", text="दवाई ले ली क्या", meta={"voice": True})
    await db.commit()
    heard = (await writing_profiles(db, FAM, [SARLA]))[SARLA["id"]]
    assert heard.get("transcript") is True and g.target(heard) == "devanagari", heard
    await store.write_fact(db, family_id=FAM, subject_id=ROHAN["id"], domain="language", key="language:preferred",
                           value={"language": "en"}, text="Speaks English", source_kind="elder_said", replace=True)
    await db.commit()
    saved = (await writing_profiles(db, FAM, [ROHAN]))[ROHAN["id"]]
    assert saved["saved"]["language"] == "en" and saved.get("transcript") is not True, saved
    assert g.target(saved) == "latin" and not g.language_problems(ENGLISH_DRAFT, saved)


async def test_profiles_read_the_script_of_saheli_replies(db, at):
    at("2026-10-11 19:30")
    for role, text in [("user", "Can you remind me about the gym at 7:30"), ("assistant", "Done, I will remind you at 7:30 PM."),
                       ("user", "Order cucumber"), ("assistant", "Your profile lists a cucumber allergy. Is this for Mummy?"),
                       ("user", "Leave it")]:
        await store.add_turn(db, family_id=FAM, thread_id=ROHAN["id"], role=role, text=text)
    await db.commit()
    prof = (await writing_profiles(db, FAM, [ROHAN]))[ROHAN["id"]]
    assert prof["replied"] == "latin" and prof["now"] == "latin", prof
    assert g.target(prof) == "latin" and g.language_problems(HINDI_DRAFT, prof)
