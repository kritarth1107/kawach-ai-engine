"""Nothing is saved until the person said yes to it (founder 2026-10-17: "be sure before adding anything, do not assume
or guess; confirm everything, not just medicine but food and other stuff, before saving").

The rule lives in the tool contract, not in a reply guard, and never in a word list on what people wrote:
  1. A save tool called in a person's WhatsApp turn does not write. It returns what it WOULD save, for whom, and an id,
     and the call is kept on Saheli's reply (turn meta "pending_saves") only when that reply was really sent.
  2. Her reply reads it back and asks one yes/no.
  3. In a LATER turn the model reads their answer and calls answer_pending(id, yes | no | change). On yes the code runs
     exactly the saved arguments that were read back (so what is saved is what they agreed to); on no it is dropped and
     kept as declined (the background re-read never saves it either); on change the model calls the tool again with the
     change, which is read back once more.
The model decides what their words mean; the code only checks that the id was read back in an earlier sent message.
answer_pending is sent only on turns that have something waiting, so the fixed prompt prefix does not grow.
Not held: scheduled turns, the dashboard (its own dialogs), readings (a dangerous reading must alert at once), diary notes
and habits, and calls that answer a tool's own question (log_food answer_to).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import timedelta

from app.brain.tools.base import ToolRefused, TurnCtx, tool
from app.core import clock

# Tools that write something about a person: held for a yes. log_vital is not here: a reading the person reported is
# not a guess, and a dangerous one alerts the family in the same turn. note and note_habit are low-risk diary lines.
CONFIRM_FIRST = frozenset({
    "remember", "stop", "forget", "set_reminder", "change_reminder", "set_stock", "log_food", "change_food",
    "save_place", "add_person", "activity", "nutrition_target", "pantry", "save_family_skill",
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



def _word(w: str) -> str:
    """One token of a structured value, singular: 'rotis' and 'roti' are the same item (not a list of what people say)."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _norm(v):
    if isinstance(v, str):
        return "_".join(_word(w) for w in re.split(r"[^a-z0-9:]+", v.strip().lower()) if w)
    if isinstance(v, list):
        return sorted((_norm(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in sorted(v.items())}
    return v


def signature(name: str, args: dict) -> str:
    keys = _SIG.get(name, ("about",))
    sig: dict = {k: _norm(args.get(k)) for k in keys if args.get(k) not in (None, "", [], {})}
    if name == "log_food":  # what was eaten: the item names
        sig["items"] = sorted(str(_norm(str((i or {}).get("name") or ""))) for i in (args.get("items") or []) if isinstance(i, dict))
    if name == "remember" and args.get("domain") == "medicine":  # the schedule read back: another time or length is asked again
        d: dict = (args.get("details") or {}) if isinstance(args.get("details"), dict) else {}
        sig["schedule"] = {k: _norm(d.get(k)) for k in ("times", "dose", "duration_days", "ends_on") if d.get(k) not in (None, "", [])}
    return name + ":" + json.dumps(sig, sort_keys=True, ensure_ascii=False)


def pending_id(sig: str) -> str:
    return "s" + hashlib.sha1(sig.encode()).hexdigest()[:6]


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
    """The saves her recent SENT messages read back and asked about (at most PENDING_TURNS of hers, newest first wins)."""
    now = now or clock.now()
    mine = [t for t in turns if t.role == "assistant"][-PENDING_TURNS:]
    out: dict[str, dict] = {}
    for t in mine:
        if t.at < now - timedelta(hours=PENDING_HOURS):
            continue
        for p in (t.meta or {}).get("pending_saves") or []:
            if isinstance(p, dict) and p.get("sig") and p.get("tool") in CONFIRM_FIRST:
                out[p.get("id") or pending_id(p["sig"])] = {**p, "id": p.get("id") or pending_id(p["sig"])}
    return list(out.values())


def declined_from(turns: list, now=None, hours: int = 48) -> list[dict]:
    """What they said no to lately (turn meta "declined_saves"), for the background re-read."""
    now = now or clock.now()
    out = []
    for t in turns:
        if t.role == "assistant" and t.at >= now - timedelta(hours=hours):
            out += [p for p in (t.meta or {}).get("declined_saves") or [] if isinstance(p, dict)]
    return out


def check(ctx, name: str, args: dict) -> dict | None:
    """None: run the tool now. A dict: the save is held (or was already saved after their yes this turn)."""
    if not getattr(ctx, "confirm_gate", False) or name not in CONFIRM_FIRST or ctx.is_system or ctx.confirmed:
        return None
    if ctx.channel not in ("whatsapp", "call") or exempt(name, args):
        return None
    sig = signature(name, args)
    if getattr(ctx, "confirm_running", None) == sig:
        return None  # answer_pending is running the read-back's own arguments after their yes
    if sig in ctx.confirmed_saves:
        return {"saved": True, "already": True, "note": "Already saved after their yes in this turn; do not save it again."}
    what, who = readback(ctx, name, args)
    clean = {k: v for k, v in args.items() if k != "said_yes"}
    pid = pending_id(sig)
    if not any(p["sig"] == sig for p in ctx.pending_saves):
        ctx.pending_saves.append({"id": pid, "tool": name, "sig": sig, "args": clean, "what": what, "for": who})
    waiting = any(p["sig"] == sig for p in getattr(ctx, "pending_before", []) or [])
    return {
        "saved": False, "waiting_for_yes": True, "id": pid, "would_save": what, "for": who,
        "next": (("This is already waiting for their answer: if their message answers it, call answer_pending with id " + pid
                  + ". " if waiting else "Nothing is saved yet. ")
                 + "In your reply, read back in one short line what will be saved and for whom (every item you are holding "
                 "this turn, together), and ask one yes/no question. Do not say it is saved, noted, added or set."),
    }


def waiting_block(pending: list[dict]) -> str:
    """For the next turn: the saves her last sent message asked about, with their ids."""
    if not pending:
        return ""
    lines = [f"  - id {p['id']}: {p.get('tool')} for {p.get('for') or 'them'}: {p.get('what')}" for p in pending[:10]]
    return ("WAITING FOR THEIR ANSWER (your last message read these back; NOTHING is saved yet):\n" + "\n".join(lines) + "\n"
            "Read their message. If it agrees to one (in any words or language), call answer_pending with that id and answer "
            "yes: exactly what was read back is saved; then say in one line what was saved and for whom. If it refuses, "
            "answer no and do not ask again. If it changes something (another time, person, amount), answer change and call "
            "the save tool with the change (it is read back once more). A question back, unclear words, or another topic is "
            "none of these: call nothing for it, and answer what they said.")


@tool(
    "answer_pending",
    "Their answer to a save you read back in an earlier message (listed in WAITING FOR THEIR ANSWER). yes: saves exactly "
    "what was read back. no: nothing is saved. change: they want it different; then call the save tool with the change.",
    {"id": {"type": "string", "description": "the id from WAITING FOR THEIR ANSWER"},
     "answer": {"type": "string", "enum": ["yes", "no", "change"]}},
    ["id", "answer"],
)
async def answer_pending(ctx: TurnCtx, a: dict) -> dict:
    from app.brain.tools.base import run

    pid, answer = str(a.get("id") or ""), str(a.get("answer") or "")
    if any(p["id"] == pid for p in ctx.pending_saves):
        raise ToolRefused("That save was read back in this same turn: wait for their answer in their next message.")
    p = next((x for x in getattr(ctx, "pending_before", []) or [] if x.get("id") == pid), None)
    if p is None:
        raise ToolRefused("No save with that id is waiting (only ones your earlier sent message read back). Nothing was saved.")
    if pid in ctx.answered_ids:
        return {"result": "already answered in this turn"}
    ctx.answered_ids.add(pid)
    if answer == "no":
        ctx.declined_saves.append({k: p.get(k) for k in ("id", "tool", "sig", "args", "what", "for")})
        return {"saved": False, "declined": True, "next": "Nothing saved. Say so in a few words; do not ask about it again."}
    if answer == "change":
        return {"saved": False, "next": "Call the save tool with what they changed; it is read back once more."}
    if answer != "yes":
        raise ToolRefused("answer is yes, no or change.")
    ctx.confirmed_saves.add(p["sig"])
    ctx.confirm_running = p["sig"]
    try:
        out, err = await run(ctx, p["tool"], dict(p.get("args") or {}))
    finally:
        ctx.confirm_running = None
    ctx.pending_saves[:] = [x for x in ctx.pending_saves if x["sig"] != p["sig"]]
    try:
        result = json.loads(out)
    except json.JSONDecodeError:
        result = {"raw": out[:300]}
    if err:
        ctx.confirmed_saves.discard(p["sig"])
    return {"saved": not err, "what": p.get("what"), "for": p.get("for"), "result": result}
