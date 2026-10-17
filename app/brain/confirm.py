"""Nothing is saved until the person said yes to it (founder 2026-10-17: "be sure before adding anything, do not assume
or guess; confirm everything, not just medicine but food and other stuff, before saving").

The rule lives in the tool contract, not in a reply guard:
  1. A save tool called in a person's WhatsApp turn does not write. It returns what it WOULD save and for whom, and the
     call is kept on Saheli's reply (turn meta "pending_saves").
  2. Her reply reads it back and asks one yes/no (all held items of the turn in one read-back).
  3. In a later turn the same call (same tool, same key fields) runs only when that read-back was in one of her last
     messages and their message is a yes (in any of the usual words; said_yes=true is also taken when a model sends it), and is not a no.
said_yes is not put in the tool schemas: the cold prompt prefix has a token budget (tests/test_gemini_prefix.py).
A call cannot confirm itself in the turn it was first made. Scheduled turns, the dashboard (its own dialogs) and a call
that answers a tool's own question (log_food answer_to) are not held.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta

from app.core import clock

# Tools that write something about a person: held for a yes.
CONFIRM_FIRST = frozenset({
    "remember", "note", "stop", "forget", "set_reminder", "change_reminder", "set_stock", "log_food", "change_food",
    "log_vital", "save_place", "add_person", "note_habit", "activity", "nutrition_target", "pantry", "save_family_skill",
    "add_doctor_question", "assign_family_task",
})
PENDING_HOURS = 6
PENDING_TURNS = 2  # her last two messages: a yes can come after one short side question

# The fields that say WHAT is saved. A yes confirms a call whose key fields match the read-back.
_SIG = {
    "remember": ("domain", "name", "about"), "note": ("about", "topic"), "stop": ("domain", "name", "about"),
    "forget": ("what", "about"), "set_reminder": ("about", "times", "in_minutes", "item", "repeat", "date", "days", "until"),
    "change_reminder": ("id", "action", "about", "times", "date", "time"), "set_stock": ("medicine", "count", "about"),
    "log_food": ("for_whom", "meal", "skipped"), "change_food": ("entry_id", "last", "for_whom", "delete"),
    "log_vital": ("kind", "name", "value", "about"), "save_place": ("name", "place", "address"),
    "add_person": ("name", "phone"), "note_habit": ("what", "about"), "activity": ("action", "type", "key", "label", "about"),
    "nutrition_target": ("for_whom", "remove"), "pantry": ("action", "item", "for_whom"), "save_family_skill": ("id", "about"),
    "add_doctor_question": ("appointment", "about"), "assign_family_task": ("title", "to"),
}

_NO = re.compile(
    r"(?i)(^|\b)(no|nope|nah|nahi|nahin|nai|mat|don'?t|do not|galat|wrong|not now|later|baad mein|ruko|wait|rehne do|abhi nahi)(\b|$)"
    r"|नहीं|नही|मत|रुको|गलत|रहने दो")
# "no problem, go ahead" is not a no
_NOT_A_NO = re.compile(r"(?i)\b(no problem|no worries|no issues?|koi baat nahi(n)?|no doubt)\b|कोई बात नहीं")
_YES = re.compile(
    r"(?i)(^|\b)(yes|yeah|yea|yep|yup|ya|yaa|yah|ok|okay|okk|okie|k|sure|haan|haa|han|ha|hanji|haanji|ji|jee|theek|thik|thick|"
    r"sahi|correct|right|done|confirm|confirmed|go ahead|kar do|kardo|kr do|krdo|save|save it|add|add it|please do|do it|"
    r"bilkul|zaroor|haan ji|perfect|fine|yes please)(\b|$)|👍|✅|👌|🙏|हाँ|हां|हा\b|जी|ठीक|सही|कर दो|करदो|बिल्कुल|ज़रूर|जरूर")


def says_no(text: str) -> bool:
    return bool(_NO.search(_NOT_A_NO.sub(" ", text or "")))


def says_yes(text: str) -> bool:
    return bool(_YES.search(text or "")) and not says_no(text)


def _norm(v):
    if isinstance(v, str):
        return re.sub(r"[^a-z0-9:]+", "_", v.strip().lower()).strip("_")
    if isinstance(v, list):
        return sorted(_norm(x) for x in v)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in sorted(v.items())}
    return v


def signature(name: str, args: dict) -> str:
    keys = _SIG.get(name, ("about",))
    sig = {k: _norm(args.get(k)) for k in keys if args.get(k) not in (None, "", [], {})}
    if name == "log_food":  # what was eaten: the item names
        sig["items"] = sorted(_norm(str((i or {}).get("name") or "")) for i in (args.get("items") or []) if isinstance(i, dict))
    if name == "remember" and args.get("domain") == "medicine":  # the schedule read back: another time or length is asked again
        d = args.get("details") if isinstance(args.get("details"), dict) else {}
        sig["schedule"] = {k: _norm(d.get(k)) for k in ("times", "dose", "duration_days", "ends_on") if d.get(k) not in (None, "", [])}
    return name + ":" + json.dumps(sig, sort_keys=True, ensure_ascii=False)


def exempt(name: str, args: dict) -> bool:
    """Calls that save nothing yet, or answer a question the tool itself asked."""
    if name == "remember" and args.get("domain") in ("language", "naming"):
        return True  # how to talk to them: their own instruction about the chat, acted on at once
    if name == "change_reminder" and args.get("action") in ("skip", "unskip"):
        return True  # "I already took it": this time is skipped, nothing about them is saved
    if name == "log_food" and (args.get("answer_to") or args.get("cancel") or args.get("from_photo") or args.get("label_photo")
                               or args.get("from_order_task")):
        return True  # the food tool's own question flow (a plate photo is asked about by the tool, nothing saved yet)
    if name in ("nutrition_target",) and args.get("view"):
        return True
    if name == "pantry" and str(args.get("action") or "") in ("list", "view", "show"):
        return True
    if name == "activity" and args.get("action") in ("pause", "resume"):
        return True
    return False


def readback(ctx, name: str, args: dict) -> tuple[str, str]:
    """(what would be saved, for whom) in plain words for the model to read back."""
    who = ""
    about = args.get("about") if "about" in args else args.get("for_whom")
    try:
        if about:
            who = ctx.member_name(ctx.subject(about))
        else:
            who = ctx.speaker.get("name") or "the speaker"
    except Exception:  # noqa: BLE001 — an unknown name is refused when the save runs
        who = str(about or ctx.speaker.get("name") or "")
    finally:
        ctx.resolved_subject = None
    d = args.get("details") or {}
    if name == "remember" and args.get("domain") == "medicine":
        bits = [args.get("sentence") or args.get("name") or ""]
        if d.get("times"):
            bits.append("reminders at " + ", ".join(map(str, d["times"])))
        elif not d.get("times"):
            bits.append("no reminders")
        if d.get("duration_days"):
            bits.append(f"for {d['duration_days']} days")
        if d.get("ends_on"):
            bits.append(f"until {d['ends_on']}")
        what = "; ".join(b for b in bits if b)
    elif name == "remember":
        what = args.get("sentence") or f"{args.get('domain')}: {args.get('name')}"
    elif name == "stop":
        what = f"stop {args.get('domain')} {args.get('name')}" + (" and its reminders" if args.get("domain") == "medicine" else "")
    elif name == "set_reminder":
        when = ", ".join(args.get("times") or []) or (f"in {args['in_minutes']} min" if args.get("in_minutes") else "")
        what = f"reminder \"{args.get('text')}\" {when} ({args.get('repeat') or 'once'})" + (f" until {args['until']}" if args.get("until") else "")
    elif name == "change_reminder":
        what = f"{args.get('action')} reminder {args.get('id')}" + (f" to {', '.join(args.get('times') or [])}" if args.get("times") else "")
    elif name == "log_food":
        items = ", ".join(f"{(i or {}).get('quantity') or ''} {(i or {}).get('unit') or ''} {(i or {}).get('name') or ''}".strip()
                          for i in (args.get("items") or []) if isinstance(i, dict))
        what = ("skipped " + str(args.get("meal") or "meal")) if args.get("skipped") else f"food log: {items or 'the meal'}" + (f" ({args['meal']})" if args.get("meal") else "")
    elif name == "set_stock":
        what = f"{args.get('count')} {args.get('medicine')} on hand"
    elif name == "log_vital":
        what = f"{args.get('name') or args.get('kind')} {args.get('value')} {args.get('unit') or ''}".strip()
    else:
        shown = {k: v for k, v in args.items() if k not in ("about", "for_whom", "said_yes") and v not in (None, "", [], {})}
        what = f"{name}: " + json.dumps(shown, ensure_ascii=False)[:240]
    return what.strip()[:400], who


def pending_from(turns: list, now=None) -> list[dict]:
    """The saves her recent messages read back and asked about (newest message first, at most PENDING_TURNS of hers)."""
    now = now or clock.now()
    mine = [t for t in turns if t.role == "assistant"][-PENDING_TURNS:]
    out: list[dict] = []
    for t in mine:
        if t.at < now - timedelta(hours=PENDING_HOURS):
            continue
        for p in (t.meta or {}).get("pending_saves") or []:
            if isinstance(p, dict) and p.get("sig"):
                out.append(p)
    return out


def check(ctx, name: str, args: dict) -> dict | None:
    """None: run the tool now. A dict: the save is held; this is the tool's result for the model."""
    if not getattr(ctx, "confirm_gate", False) or name not in CONFIRM_FIRST or ctx.is_system or ctx.confirmed:
        return None
    if ctx.channel not in ("whatsapp", "call") or exempt(name, args):
        return None
    sig = signature(name, args)
    asked = {p["sig"] for p in getattr(ctx, "pending_before", []) or []}
    text = ctx.user_text or ""
    if sig in asked and not says_no(text) and (args.get("said_yes") is True or says_yes(text)):
        ctx.confirmed_saves.add(sig)
        return None
    what, who = readback(ctx, name, args)
    clean = {k: v for k, v in args.items() if k != "said_yes"}
    if not any(p["sig"] == sig for p in ctx.pending_saves):
        ctx.pending_saves.append({"tool": name, "sig": sig, "args": clean, "what": what, "for": who})
    why = ("Nothing is saved yet: they have not said yes to this exact save." if sig not in asked else
           "Nothing is saved: their message is not a clear yes to it.")
    return {
        "saved": False, "waiting_for_yes": True, "would_save": what, "for": who,
        "next": (why + " In your reply, read back in one short line what will be saved and for whom (every item you are holding "
                 "this turn, together, one read-back), and ask one yes/no question. Do not say it is saved, noted, added or set. "
                 "When they say yes, call this tool again with the same arguments."),
    }


def waiting_block(pending: list[dict]) -> str:
    """For the next turn: the saves her last message asked about."""
    if not pending:
        return ""
    lines = []
    for p in pending[:10]:
        lines.append(f"  - {p.get('tool')} for {p.get('for') or 'them'}: {p.get('what')}  args={json.dumps(p.get('args') or {}, ensure_ascii=False)[:500]}")
    return ("WAITING FOR THEIR YES (your last message read these back; NOTHING is saved yet):\n" + "\n".join(lines) + "\n"
            "If their message says yes (any language: yes, haan, ok, theek hai, kar do, 👍), call each one again NOW with exactly "
            "these args, then say in one line what was saved and for whom. Yes to only some: call only those. "
            "A change (another time, another person, fewer days): call it with the change; it is read back once more. "
            "No, or a new topic: save nothing and do not ask again.")

