"""Health hard turns (trust plan C6 A/B of 13 Oct: 6 of 8 health cases failed at both effort levels).

The failures were behaviour, not effort: no "when to see / check with the doctor" line, replies opening like a leaflet
("According to general medical guidelines…"), "please don't take your missed dose now" (her own dose advice), an offer to
check a medicine combination herself, and "I will check with your family" to an adult managing their own care, who has no
one to check with. The fixes are in the standing rules (persona, reply check) and in the household block, which now says
SELF CARE from the roster (data), so every family, role and language gets them. The real-model proof is eval/hard_effort_ab.py
and the replay seeds (eval/fixtures/replay_seeds.jsonl, kind health_* / safety_medical_advice / correction_apology).
"""

from app.brain.persona import PERSONA, REPLY_FORMAT
from app.care import digest

SELF = {"id": "s1", "name": "Anita Rao", "role": "primary caregiver"}
ELDER = {"id": "e1", "name": "Kamla Devi", "role": "elder"}
SON = {"id": "c1", "name": "Rohit Verma", "role": "primary caregiver", "relation": "son", "relation_to": "e1"}


def test_household_says_self_care_when_the_person_cared_for_is_their_own_caregiver():
    block = digest.household(SELF, [SELF], SELF["id"])
    assert "SELF CARE: Anita Rao manages their own care" in block
    assert "never say you will check with the family" in block


def test_household_self_care_follows_the_roster_role_even_when_the_elder_entry_says_elder():
    # the backend may send the care recipient as {"role": "elder"} while the roster member entry holds the caregiver role
    block = digest.household({**SELF, "role": "elder"}, [SELF, SON], SELF["id"])
    assert "SELF CARE" in block


def test_household_has_no_self_care_line_for_an_elder_or_their_caregiver():
    assert "SELF CARE" not in digest.household(ELDER, [ELDER, SON], ELDER["id"])
    assert "SELF CARE" not in digest.household(ELDER, [ELDER, SON], SON["id"])
    # a caregiver who also keeps their own record, speaking about the elder, is not self care for this turn
    assert "SELF CARE" not in digest.household(ELDER, [ELDER, SELF], SELF["id"])


def test_every_health_answer_carries_when_to_see_the_doctor():
    assert "ends with one short line on when to see or call the doctor" in PERSONA
    assert "says in one short line that it is general information and to confirm with their doctor or pharmacist" in REPLY_FORMAT


def test_no_dose_decisions_of_her_own_and_no_invented_reason_for_a_missed_reminder():
    assert "Never tell anyone to skip a dose, leave it for tomorrow or take two" in PERSONA
    assert "Never invent a reason the ledger does not show" in PERSONA
    assert "never skip, delay, double or stop a dose" in REPLY_FORMAT
    assert "never offer to check a combination or a dose yourself" in PERSONA


def test_self_care_dose_questions_go_to_their_own_doctor_not_the_family():
    assert "(SELF CARE in HOUSEHOLD) has no one to check with" in PERSONA


def test_health_answers_sound_like_a_person_not_a_leaflet():
    assert '"According to general medical guidelines…"' in PERSONA
    assert 'never as a quote ("according to guidelines", "medical sources say")' in PERSONA


def test_a_correction_gets_a_brief_sorry():
    assert "when you had it wrong, say a brief sorry in a few words" in PERSONA


def test_a_health_turn_asks_for_the_doctor_line_even_without_a_search():
    from app.brain import loop

    ok = {"ok": True}
    assert loop.health_turn([{"tool": "web_search", **ok}])
    assert loop.health_turn([{"tool": "log_event", "args": {"kind": "symptom"}, **ok}])  # dizziness, loose motions (no search)
    assert loop.health_turn([{"tool": "log_vital", **ok}])  # a sugar reading
    assert not loop.health_turn([{"tool": "log_event", "args": {"kind": "mood"}, **ok}, {"tool": "remember", **ok}])
    assert not loop.health_turn([{"tool": "web_search", "ok": False}])


def test_red_flags_get_112_now_not_if_it_gets_worse():
    assert "now for chest pain, breathlessness, signs of stroke, fainting, heavy bleeding" in PERSONA
    assert 'never "if it gets worse" for these' in PERSONA


async def test_web_search_failure_still_answers_what_is_well_known(db, monkeypatch):
    from app.brain import tools, websearch
    from app.care import store
    from app.sim.world import SimHost

    async def broken(q):
        raise TimeoutError()

    monkeypatch.setattr(websearch, "search", broken)
    await store.save_roster(db, "fam-ws", SELF, [SELF])
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id="fam-ws", elder=SELF, speaker=SELF, members=[SELF], message_ref="ws1",
                        user_text="what are the side effects of amlodipine?")
    out, err = await tools.run(ctx, "web_search", {"question": "common side effects of amlodipine"})
    assert err and "Answer from what is well known" in out and "never mention looking it up" in out


def test_extra_rules_and_low_effort_only_on_health_turns():
    from app.brain import grounding

    base = grounding._system(True, False)
    health = grounding._system(True, True)
    assert "dose_change" not in base and "doctor_note_missing" not in base  # those two stay on health turns only
    assert "followup_missing" in base and "unwanted_apology" in base and "tease_as_question" in base
    assert "framing_missing" in base and "question_unanswered" in base
    assert "6. doctor_note_missing" in health and '7. For each advice claim also add "dose_change"' in health
    assert grounding._system(False, True).count("5. doctor_note_missing") == 1
    assert grounding.effort_for(False) == "minimal" and grounding.effort_for(True) == "low"
