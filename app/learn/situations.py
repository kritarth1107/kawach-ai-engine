"""What kind of moment a message is, so lessons and examples are learned per situation.

Rules only (no model): the person's words, who they are, which tools Saheli used, and for scheduled turns
the prompt that woke her. About twenty situations; "chit_chat" is the catch-all.
"""

from __future__ import annotations

import re

from app.brain import guards

SITUATIONS = {
    "emergency": "A red flag, fall, or urgent health worry",
    "med_question": "A question about medicines, food with medicines, or symptoms",
    "dose_report": "Reporting a dose taken or missed",
    "vital_report": "Sharing a reading (BP, sugar, temperature…)",
    "symptom": "Telling about pain or feeling unwell",
    "low_mood": "Sad, lonely, worried or upset",
    "grief": "Missing someone who died or is away (incl. memory loss)",
    "annoyed": "Annoyed or rude with Saheli, or wants fewer messages",
    "injection": "Trying to change Saheli's rules",
    "setup": "Setting up or changing the care record (medicines, routines, names)",
    "relay": "Asking Saheli to tell someone in the family something",
    "order": "Ordering, rides, or a running order",
    "caregiver_update": "A caregiver asking how the elder is",
    "caregiver_stressed": "A caregiver who is tired, stressed or overwhelmed",
    "followup": "Saheli following up on something (scheduled)",
    "checkin": "The weekly check-in with a caregiver",
    "appointment": "Doctor appointments and visits",
    "refill": "Medicines running out",
    "thanks": "Thanks, greetings, small talk ending",
    "chit_chat": "Ordinary conversation",
}

_R = lambda p: re.compile(p, re.I)  # noqa: E731
GRIEF = _R(r"\b(miss|yaad aa|chale gaye|chali gayi|passed away|nahi rahe|nahi rahi|guzar|swarg|late (husband|wife)|kahan hai|kab aayeg|kab aayegi)\b|याद आ|गुज़र|नहीं रहे")
LOW_MOOD = _R(r"\b(sad|lonely|alone|akela|akeli|udaas|upset|worried|anxious|tension|cry|ro rah|depress|hopeless|dukhi|pareshan|ghabra|dar lag)\w*|उदास|अकेल|दुखी|परेशान|घबरा")
# Complaints about Saheli or her messages ("useless fellow" about the attendant is not one).
ANNOYED = _R(
    r"\b(bas karo|band karo|chup|stop (it|messag|remind|sending|asking)|annoying|irritat|pareshan mat (karo|kar)|shut up|mat bhejo|too many messages|"
    r"kitni baar|baar baar|bar bar|why again|told you|don'?t disturb|dont disturb|disturb mat|repeatedly|how many times|enough now|"
    r"(useless|bekaar|faltu) (message|msg|app|reminder|baat|saheli)|you are useless|dimag mat|kiti vela|tras deu nako|abar keno|birokto|birakto|"
    r"malli enduku|visugu|thirumba (thirumba|yen))\b|"
    r"परेशान मत|बार बार|बार-बार|फिर से क्यों|আবার কেন|বিরক্ত|ବାରମ୍ବାର|చిరాకు|தொந்தரவு|ತೊಂದರೆ ಕೊಡಬೇಡ"
)
FINE = _R(r"\b(all good|i'?m (fine|ok|okay)|theek hoon|thik hoon|sab theek|bilkul theek|feeling better|much better)\b|बिल्कुल ठीक|ठीक हूँ|ठीक हूं|सब ठीक")
STRESS = _R(r"\b(tired|exhausted|overwhelmed|stressed|burn(ed|t) out|can'?t cope|thak gay|thak gayi|no time|sleepless|neend nahi)\b")
MED_Q = _R(r"\?.*|\b(can i|kya main|kya mai|should i|le sakt|kha sakt|pee sakt)\b")
MED_WORDS = _R(r"\b(tablet|goli|dawai|dawa|medicine|dose|syrup|insulin|crocin|paracetamol|dolo|combiflam)\b|दवा|गोली")
DOSE = _R(r"\b(le li|kha li|took|taken|li hai|kha liya|le liya|missed|bhool gay|bhool gayi|nahi li)\b|ले ली|खा ली")
VITAL = _R(r"\b(bp|sugar|pressure|temperature|temp|fever|oxygen|spo2|pulse|weight)\b.{0,15}\d|\d{2,3}\s*/\s*\d{2,3}|शुगर|बीपी")
SYMPTOM = _R(r"\b(pain|dard|ache|chakkar|dizzy|cough|khansi|fever|bukhar|vomit|ulti|swelling|sujan|tired|weak|kamzori|can'?t sleep)\b|दर्द|चक्कर")
RELAY = _R(r"\b(tell|bolo|bata do|bataiye|keh do|message|inform|let .{0,15} know|ko bolo|ko bata)\b")
ORDER = _R(r"\b(order|mangwa|mangao|deliver|cab|ride|uber|ola|rapido|auto book|swiggy|zepto|instamart|blinkit|apollo|1mg|pharmeasy)\b")
THANKS = _R(r"^\W*(thanks?|thank you|shukriya|dhanyavad|ok+|okay|theek hai|thik hai|good night|gn|good morning|gm|namaste|radhe radhe|jai shri krishna|bye)\W*$")
UPDATE_Q = _R(r"\b(how is|how was|kaisi hain|kaise hain|kaisa hai|did (she|he)|any update|sab theek|kya haal)\b")
SETUP_TOOLS = {"remember", "stop", "set_reminder", "confirm_change", "set_stock"}


def tag(*, text: str, role: str, tools: list[str], prompt: str = "") -> str:
    """role: elder | caregiver | system. tools: names of tools used in the turn."""
    t = text or ""
    used = set(tools or [])
    if role == "system":
        p = prompt or t
        for key, words in (("followup", "follow-up"), ("checkin", "check-in"), ("appointment", "appointment"), ("refill", "refill"),
                           ("order", "task update"), ("followup", "wake-up")):
            if words in p.lower():
                return key
        return "followup"
    if guards.INJECTION.search(t):
        return "injection"
    if guards.RED_FLAG_WORDS.search(t) or guards.critical_reading(t):
        return "emergency"
    # An alert without a red flag in the words (an unanswered check, an order to approve) is an emergency only when
    # the person did not say they are fine.
    if any(a.startswith("alert_caregiver") for a in used) and not FINE.search(t) and not THANKS.search(t):
        return "emergency"
    if ANNOYED.search(t):
        return "annoyed"
    if GRIEF.search(t):
        return "grief"
    if role == "caregiver" and STRESS.search(t):
        return "caregiver_stressed"
    if LOW_MOOD.search(t):
        return "low_mood"
    if used & {"start_task", "task_input", "cancel_task", "past_orders"} or ORDER.search(t):
        return "order"
    if "log_vital" in used or VITAL.search(t):
        return "vital_report"
    if "log_dose" in used or (DOSE.search(t) and MED_WORDS.search(t)):
        return "dose_report"
    if MED_WORDS.search(t) and MED_Q.search(t):
        return "med_question"
    if SYMPTOM.search(t):
        return "symptom"
    if used & SETUP_TOOLS:
        return "setup"
    if "send_message" in used or RELAY.search(t):
        return "relay"
    if role == "caregiver" and UPDATE_Q.search(t):
        return "caregiver_update"
    if THANKS.search(t):
        return "thanks"
    return "chit_chat"


def lang_of(profile: dict | None) -> str:
    p = profile or {}
    if not p:
        return ""
    if p.get("script") != "latin":
        return p.get("script") or ""
    return f"latin-{p.get('roman') or 'mixed'}"
