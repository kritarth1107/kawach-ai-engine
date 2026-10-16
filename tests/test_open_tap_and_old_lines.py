"""A tap on the notice is not words of theirs, an old message is not sent again, and one meal is not a dish.

Live 16 Oct 13:35: an elder tapped "read message" under the notice. The system gave the 08:30 breakfast nudge word for
word, then Saheli asked about a bhindi sabji she had never been asked about: the same line, word for word, as a message
from 14 Oct. It came from a "dish" fact saved on 13 Oct, when she answered "bhindi" to "what sabji today?".
"""

from app.brain import guards, tools
from app.brain.loop import TurnRequest, turn_context
from app.care import store
from app.care.domains import DOMAINS

FAM = "fam-tap"
ELDER = {"id": "e-tap", "name": "Kamla", "role": "elder"}
SON = {"id": "c-tap", "name": "Ravi", "role": "primary caregiver"}


def test_a_message_from_days_ago_is_not_sent_again():
    old = "भिंडी री सब्जी बणती देखी, काई रो स्वाद आज थाने सबसे जास्तो लभो?"
    assert guards.old_message_again(old, [("14 Oct", old)]) == "14 Oct"
    assert guards.old_message_again("आज दोपहर रो खाणो कांई बणियो, मम्मी? थोड़ो आराम भी कर लियो?", [("14 Oct", old)]) is None
    assert guards.old_message_again("ठीक है सा", [("14 Oct", "ठीक है सा")]) is None  # short acks may repeat


def test_one_meal_is_not_a_dish_fact():
    assert "One meal they had is a food log, not a dish" in DOMAINS["dish"]
    remember = next(s for s in tools.specs() if s.name == "remember")
    assert "never a dish fact" in remember.description


async def test_a_tap_on_the_notice_is_explained_to_the_brain(db, at):
    at("2026-10-16 13:35")
    await store.save_roster(db, FAM, ELDER, [ELDER, SON])
    req = TurnRequest(family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER, SON], text="open:waiting", message_ref="m-tap")
    ctx_text = "\n".join(await turn_context(db, req))
    assert "a tap on \"read message\"" in ctx_text and "fits this time of day" in ctx_text
    plain = "\n".join(await turn_context(db, TurnRequest(family_id=FAM, elder=ELDER, speaker=ELDER, members=[ELDER, SON], text="namaste",
                                               message_ref="m-plain")))
    assert "a tap on \"read message\"" not in plain

