"""The tools framework: TurnCtx, the registry (@tool), run(), specs and the shared field definitions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.brain.host import ToolHost
from app.care import store
from app.llm.router import ToolSpec

logger = logging.getLogger(__name__)


@dataclass
class TurnCtx:
    session: AsyncSession
    host: ToolHost
    family_id: str
    elder: dict  # {"id", "name"}
    speaker: dict  # {"id", "name", "role": "elder" | "caregiver" | ...}
    members: list[dict]
    message_ref: str | None = None
    alerts: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    # Filled by the loop for the guards (app.brain.guards): what the person wrote, what Saheli may draw on,
    # medicine times on record, and how each person writes.
    user_text: str = ""
    # A scheduled turn's prompt (the task update, the loop that woke her) and this turn's tool results ("name: output", the
    # loop fills them as tools run): logged with each message she starts, so the full-memory judge grades it against the
    # data it was about (13 Oct: an order's options read as "made up" because the judge saw neither).
    prompt: str = ""
    tool_texts: list[str] = field(default_factory=list)
    send_checks: dict = field(default_factory=dict)  # no-invention checks of send_message per recipient this turn
    known: list[str] = field(default_factory=list)
    meds: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)
    buttons: list[dict] = field(default_factory=list)  # one-tap WhatsApp buttons attached to the reply
    buttons_spec: tuple | None = None  # (kind, key) of those buttons: titled in the final reply's script
    situation: str = "chit_chat"  # learning: what kind of moment this is (app.learn.situations)
    playbook_version: int = 0
    arm: str = "live"
    judge_band: str = ""  # sample | always | skip, set once per turn (app.quality.sample)
    judged_text: str = ""  # the draft already sent to the live judge; the same words are not sent again
    judged_claims: list = field(default_factory=list)
    tool_groups: set[str] = field(default_factory=set)  # groups need_tools asked for; the next step sends them
    channel: str = "whatsapp"  # whatsapp | dashboard: where a change came from, for memory history
    # Set only by the dashboard, after the caregiver confirmed in a dialog: an undo may then end or restart a medicine
    # at once. Never set from a model's tool arguments.
    confirmed: bool = False
    # The message was a voice note the speech engine was not sure about: health changes from it wait for a confirmation.
    voice_unsure: bool = False
    # The message is a voice-note transcript. Its script and the speech engine's language are not a request to switch.
    voice: bool = False
    last_assistant: str = ""  # her previous message in this chat, for a follow-up that left it
    # People this turn sent a message to, and people the task system is telling about an order in this turn (one message each).
    messaged: set = field(default_factory=set)
    once: set = field(default_factory=set)  # side effects already done in this turn (a pin saved to a place)
    resolved_subject: str | None = None  # set by subject() for this tool call; the result says who it was for
    told_by_tasks: set = field(default_factory=set)
    # A WhatsApp location pin in this message, and whether the reply asks for one (a "Send location" button)
    location: dict | None = None
    location_request: bool = False
    # The media attached to this message ([{"mime", "data"}] base64), for tools that read a photo themselves (log_food:
    # a meal or a nutrition label)
    images: list[dict] = field(default_factory=list)
    # Set when a meal was just saved and water is still an open gap, so the reply asks it.
    pending_gap: str = ""
    # Her last message asked something, and this message is a short answer to it. Setup and naming stay closed.
    held_topic: str = ""
    # The grounding check already said so: the code writes the line, the model does not get another chance to drop it.
    needs_framing: bool = False
    force_light: bool = False
    # True when an earlier message of hers is in the thread (an apology can stand only then).
    prior_assistant: bool = False
    # A scheduled turn's person (TurnRequest.about): a save with no about in that turn is theirs.
    system_about: str | None = None
    # Saves wait for the person's yes (app.brain.confirm). On only in a person's live turn (the loop sets it).
    confirm_gate: bool = False
    pending_before: list[dict] = field(default_factory=list)  # saves her recent messages read back and asked about
    pending_saves: list[dict] = field(default_factory=list)  # saves held in this turn (kept on her reply)
    confirmed_saves: set = field(default_factory=set)
    # Their recent photo messages (turns with a kept reading): a saved medicine notes the photo it was read from.
    photo_turns: list = field(default_factory=list)
    # The speech engine scored this voice note low (not just a short one): a yes in it does not confirm a health change.
    voice_low_score: bool = False

    @property
    def elder_id(self) -> str:
        return self.elder["id"]

    @property
    def speaker_is_elder(self) -> bool:
        return self.speaker.get("id") == self.elder_id

    @property
    def is_system(self) -> bool:
        return self.speaker.get("role") == "system"

    @property
    def self_care(self) -> bool:
        """An adult managing their own care: the person cared for is the one speaking and holds a caregiver role.

        What they say about themselves carries a caregiver's weight (a dose their doctor changed is saved, not left
        waiting for a confirmation nobody else can give; world probe judge, 2026-10-13). Alerts still never go to them
        about themselves (founder rule): is_caregiver stays False for alert routing."""
        return self.speaker_is_elder and "caregiver" in str(self.speaker.get("role", "")).lower()

    @property
    def is_caregiver(self) -> bool:
        """A primary or co-caregiver. The dashboard is caregivers-only, so a dashboard actor is one even on their own
        self-care record; on WhatsApp a family member with view access or a family doctor is not."""
        if self.channel == "dashboard":
            return True
        return not (self.speaker_is_elder or self.is_system) and "caregiver" in str(self.speaker.get("role", "")).lower()

    @property
    def source_kind(self) -> str:
        if self.is_system:
            return "inferred"
        return "elder_said" if self.speaker_is_elder and not self.self_care else "caregiver_said"

    @property
    def actor_id(self) -> str:
        return self.elder_id if self.is_system else (self.speaker.get("id") or self.elder_id)

    def member_name(self, subject_id: str) -> str:
        for person in [self.elder, *self.members]:
            if person.get("id") == subject_id:
                return (person.get("name") or "").strip() or subject_id
        return subject_id

    def subject(self, about: str | None) -> str:
        """Who this is about. Blank is the care recipient. A name counts only when one member has it.
        Someone who is not a member is never saved as the care recipient."""
        from app.care.members import UnknownMember, resolve_member

        try:
            found = resolve_member(about, elder=self.elder, members=self.members)
        except UnknownMember as exc:
            if exc.matches:
                raise ToolRefused(f"More than one member is called {exc.about}. Ask which person, and use their id.") from exc
            raise ToolRefused(
                f"{exc.about} is not a member; medicines, doses, readings and reminders can be saved only for members. "
                f"Save a note about {exc.about} under the speaker (no reminder), or ask whether to add {exc.about}."
            ) from exc
        self.resolved_subject = found
        return found

    def own(self, about: str | None) -> str:
        """Whose a save is: the person named, else the speaker's own. Nothing is saved on someone the message is not about
        (founder 2026-10-16: his "Good night" was saved as his mother's sleep). Only a scheduled turn and the dashboard,
        which act on the person on the page, keep the care recipient as the default, unless a scheduled turn is about one
        person (TurnRequest.about), whose it then is."""
        if self.is_system and not str(about or "").strip() and self.system_about:
            self.resolved_subject = self.system_about
            return self.system_about
        if str(about or "").strip() or self.is_system or self.channel == "dashboard":
            return self.subject(about)
        me = self.speaker.get("id")
        if not me:
            return self.subject(about)
        if me != self.elder_id and me not in {m.get("id") for m in self.members}:
            raise ToolRefused("The speaker is not in HOUSEHOLD, so nothing is saved as theirs. If it is about a member, give "
                              "about; otherwise save nothing.")
        self.resolved_subject = me
        return me


Handler =Callable[[TurnCtx, dict], Awaitable[Any]]
_TOOLS: dict[str, tuple[ToolSpec, Handler]] = {}


def tool(name: str, description: str, properties: dict, required: list[str]):
    schema = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}

    def wrap(fn: Handler) -> Handler:
        _TOOLS[name] = (ToolSpec(name, description, schema), fn)
        return fn

    return wrap


def names() -> list[str]:
    """Every tool's name (a reply that writes one of them as code did nothing: app.brain.guards.leaked_tool_call)."""
    return list(_TOOLS)


def specs() -> list[ToolSpec]:
    return [spec for spec, _ in _TOOLS.values()]


# Tools that only do something once a task is already running, or once a scheduled turn is speaking.
# Never chosen from the words someone typed: the situation tag, an open task, onboarding, or need_tools.
_WHEN_TASK = {"task_input", "order_feedback", "spending", "done_in_app", "cancel_task", "task_status"}
_WHEN_SYSTEM = {"weekly_report", "service_status"}

# A normal turn sends only _CORE. The situation tag (already computed), an open task, onboarding, or a photo
# adds a group. need_tools adds one the model asked for. Every tool is in exactly one place.
_CORE = frozenset({
    "remember", "recall", "search_records", "confirm_change", "whats_pending",
    "log_dose", "log_vital", "log_event", "stop", "note",
    "set_reminder", "reminders", "today_schedule", "reminder_log",
    "send_message", "alert_caregiver", "web_search",
    "undo_change", "open_loop", "need_tools",
})
_GROUPS = {
    "order": frozenset({
        "start_task", "places", "save_place", "map_link", "ask_location", "remove_place", "task_input",
        "ways_there", "note_trip", "accept_trip", "read_trip", "boundaries", "set_boundaries",
        "cancel_task", "done_in_app", "task_status", "order_feedback", "spending", "past_orders", "offer_buttons",
        "fact_still_true",
    }),
    "nutrition": frozenset({
        "log_food", "food_log", "change_food", "nutrition_target", "pantry", "nutrition_settings", "nutrition_report",
    }),
    "setup": frozenset({
        "setup_progress", "finish_setup", "add_person", "my_name", "daily_brief", "language_preference",
    }),
    "memory": frozenset({
        "forget", "memory_changes", "list_skills", "save_family_skill", "forget_skill", "note_habit",
        "learning_consent", "close_loop", "gap_asked",
    }),
    "health": frozenset({
        "lab_trend", "health_record", "find_lab_test", "find_doctor", "add_doctor_question", "wellbeing",
        "set_stock", "medicine_stock", "emergency_card", "care_team", "log_outcome", "assign_family_task",
        "family_tasks", "patterns", "change_reminder",
    }),
    "activity": frozenset({"activity", "log_activity"}),
    "chat": frozenset({"sing", "voice_replies", "feedback"}),
    "system": frozenset({"weekly_report", "service_status"}),
}
_HEALTH_SITUATIONS = frozenset({
    "emergency", "med_question", "dose_report", "vital_report", "symptom", "low_mood", "grief",
    "appointment", "refill", "injection",
})
TOOL_GROUPS = frozenset(_GROUPS)


def groups_for(*, situation: str = "", open_tasks: bool = False, system: bool = False,
               onboarding: bool = False, images: bool = False, extra: set[str] | None = None) -> set[str]:
    """Which tool groups this call sends. The situation tag is already decided; nothing here reads the person's words."""
    groups: set[str] = set(extra) if extra else set()
    if situation == "order" or open_tasks:
        groups.add("order")
    if situation == "setup" or onboarding:
        groups.add("setup")
        groups.add("memory")
    if situation in _HEALTH_SITUATIONS:
        groups.add("health")
    if system:
        groups.add("system")
        groups.add("health")
    if images:
        groups.add("nutrition")
        groups.add("photos")
    return {g for g in groups if g in _GROUPS or g == "photos"}


def specs_for(*, open_tasks: bool = False, system: bool = False, situation: str = "",
              onboarding: bool = False, images: bool = False, extra: set[str] | None = None) -> list[ToolSpec]:
    """Core tools, then only the groups this turn needs. Core stays a stable prefix."""
    groups = groups_for(situation=situation, open_tasks=open_tasks, system=system,
                        onboarding=onboarding, images=images, extra=extra)
    allow = set(_CORE)
    for name in groups:
        allow |= _GROUPS.get(name, frozenset())
    skip: set[str] = set()
    if not open_tasks:
        skip |= _WHEN_TASK
        if "order" in groups:
            # A booked ride or a placed order is no longer a live task. The cancel, an in-app cancel, and
            # "has it come?" still have to be callable. task_input stays off until a task is running.
            skip -= {"cancel_task", "done_in_app", "order_feedback", "task_status"}
    if not system:
        skip |= _WHEN_SYSTEM
    rows = [spec for spec, _ in _TOOLS.values() if spec.name in allow and spec.name not in skip]
    rows.sort(key=lambda s: (s.name not in _CORE, s.name))
    return rows


@tool(
    "need_tools",
    "Call this once when the next step needs a tool that is not listed on this call. "
    "group: order (buy, ride, trip, flight, a place), nutrition (a meal, a food change, a nutrition target or report), "
    "setup (onboarding, the daily brief, a person, a language change), memory (forget, history of a change, a family skill), "
    "health (a lab, a doctor, stock, the care team, a dose-time change), activity (a walk or a nap), "
    "chat (they asked you to sing, or to change voice replies). The next step has that group. Then call the tool.",
    {"group": {"type": "string", "enum": ["order", "nutrition", "setup", "memory", "health", "activity", "chat"]}},
    ["group"],
)
async def need_tools(ctx: TurnCtx, args: dict) -> dict:
    group = str(args.get("group") or "")
    if group not in _GROUPS or group == "system":
        return {"error": "group must be order, nutrition, setup, memory, health, activity or chat"}
    ctx.tool_groups.add(group)
    # Counted later: a group asked on more than 3% of a situation's turns joins that situation.
    logger.info("need_tools situation=%s group=%s", ctx.situation, group)
    return {"ok": True, "group": group}


def arg_problems(schema: dict, args: Any) -> list[str]:
    """What is wrong with a call against its tool's schema: a missing required field or a value outside an enum (a field the
    tool does not take is left alone). Models other than Gemini do not always hold to the schema (live 2026-10-14: stop with
    domain "outcome" eight times, each one a crash); when a tool fails on such a call, the model gets these back instead."""
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    props = schema.get("properties") or {}
    out = [f"{k} is required" for k in schema.get("required") or [] if args.get(k) is None]
    for k, v in args.items():
        spec = props.get(k)
        if spec is None:
            continue
        allowed = spec.get("enum")
        if allowed and v is not None and not isinstance(v, (dict, list)) and v not in allowed:
            out.append(f"{k} must be one of {', '.join(map(str, allowed))} (got {v!r})")
        items = spec.get("items") or {}
        if spec.get("type") == "array" and isinstance(v, list) and items.get("enum"):
            bad = [x for x in v if x not in items["enum"]]
            if bad:
                out.append(f"{k} items must be from {', '.join(map(str, items['enum']))} (got {bad!r})")
    return out


async def _release_record_locks(ctx) -> None:
    """A refused or failed tool must drop a record lock. A schema check uses a ctx with no session."""
    session = getattr(ctx, "session", None)
    if session is None:
        return
    await store.release_record_locks(session, commit=False)


async def run(ctx: TurnCtx, name: str, args: dict) -> tuple[str, bool]:
    """Run a tool; returns (json result, is_error). Errors go back to the model, never to the user."""
    if name not in _TOOLS:
        ctx.actions.append({"tool": name, "args": args, "ok": False, "unknown": True, "error": f"unknown tool {name}"})
        return json.dumps({"error": f"unknown tool {name}"}), True
    from app.brain import confirm

    held = confirm.check(ctx, name, args if isinstance(args, dict) else {})
    if held is not None:
        ctx.actions.append({"tool": name, "args": args, "ok": False, "held": True})
        return json.dumps(held, ensure_ascii=False), False
    # A short voice "haan" to her read-back is the check an unsure voice note asks for (the read-back named exactly what
    # changes); only a low speech score still makes a health change wait.
    relax_voice = bool(getattr(ctx, "voice_unsure", False) and not getattr(ctx, "voice_low_score", False) and isinstance(args, dict)
                       and confirm.signature(name, args) in (getattr(ctx, "confirmed_saves", None) or ()))
    if isinstance(args, dict) and "said_yes" in args:
        args = {k: v for k, v in args.items() if k != "said_yes"}
    if relax_voice:
        ctx.voice_unsure = False
    try:
        from app.care import versions

        ctx.resolved_subject = None
        with versions.attribution(actor_id="saheli" if ctx.is_system else (ctx.speaker.get("id") or ""),
                                  source="saheli" if ctx.is_system else ctx.channel):
            async with ctx.session.begin_nested():
                out = await _TOOLS[name][1](ctx, args)
        if isinstance(out, dict) and ctx.resolved_subject:
            out.setdefault("subject_id", ctx.resolved_subject)
            out.setdefault("for", ctx.member_name(ctx.resolved_subject))
            addressed = ctx.elder_id if ctx.is_system else (ctx.speaker.get("id") or ctx.elder_id)
            if ctx.resolved_subject != addressed:
                out["for_other"] = True
        blob = json.dumps(out, default=str, ensure_ascii=False)
        kept: dict = {"subject": ctx.resolved_subject} if ctx.resolved_subject else {}
        if name == "today_schedule" and isinstance(out, dict):
            # The trace keeps 400 characters, which cuts a day's list into invalid JSON. The names stay whole.
            kept["medicines"] = [
                {"name": str(m.get("name") or ""), "dose": m.get("dose"), "time": m.get("time")}
                for m in (out.get("medicines") or []) if isinstance(m, dict) and m.get("name")
            ][:12]
        ctx.actions.append({"tool": name, "args": args, "ok": True, "result": blob[:400], **kept})
        return blob, False
    except ToolRefused as exc:
        await _release_record_locks(ctx)
        ctx.actions.append({"tool": name, "args": args, "ok": False, "refused": str(exc)})
        return json.dumps({"refused": str(exc)}), True
    except Exception as exc:  # noqa: BLE001 — the model decides what to tell the person
        await _release_record_locks(ctx)
        # A call outside the tool's schema (a model that ignores enums): the model gets the allowed values, not a crash.
        # Tools that refuse a bad call themselves keep their own, more specific answer (it is a ToolRefused above).
        problems = arg_problems(_TOOLS[name][0].schema, args)
        if problems:
            logger.warning("tool %s called outside its schema: %s", name, "; ".join(problems))
            ctx.actions.append({"tool": name, "args": args, "ok": False, "refused": "; ".join(problems)})
            return json.dumps({"refused": "Nothing was done. Fix the call: " + "; ".join(problems)}, ensure_ascii=False), True
        logger.exception("tool %s failed", name)
        ctx.actions.append({"tool": name, "args": args, "ok": False, "error": str(exc)[:200]})
        return json.dumps({"error": "this did not work right now", "detail": str(exc)[:200]}), True
    finally:
        if relax_voice:
            ctx.voice_unsure = True


class ToolRefused(Exception):
    pass


ABOUT = {"type": "string", "description": "Who it is about: their id from HOUSEHOLD, or their name when only one member has that name. "
         "Leave out for the care recipient. A relative who is not a member is refused: never saved on someone else."}
# A save (a log, a fact, a plan): blank is the speaker's own, never the care recipient (founder 2026-10-16).
ABOUT_OWN = {"type": "string", "description": "Whose it is: their id from HOUSEHOLD, or their name when only one member has it. "
             "Leave out only for the speaker's own: what the person writing did, felt, ate, measured, plans or wants ('good "
             "night' is their own sleep). Give another person's id only when the message is about that person: they name or "
             "mean them, or answer your question about them. Never save one person's things on another."}
# The tools whose blank about is the speaker's own (TurnCtx.own); log_food has its own for_whom with the same rule.
SAVES_FOR_SPEAKER = frozenset({"remember", "stop", "note", "fact_still_true", "note_habit", "log_dose", "log_vital", "log_event",
                               "activity", "log_activity", "open_loop", "note_trip", "accept_trip", "set_stock",
                               "add_doctor_question", "log_outcome", "feedback", "save_family_skill"})


def acts_catalog_line() -> str:
    from app.care.activities import catalog_line

    return catalog_line()
# Who an order or a ride is for (founder 2026-10-12: a caregiver's own biscuits, ordered to his own home, were recorded as his
# mother's and she was asked whether they came). Always filled by the brain, never a default.
FOR_WHOM = {"type": "string", "description": "Person id from HOUSEHOLD this order or ride is for: who will receive and use it (whose "
            "groceries, food or medicine it is; who rides). Read it from their words and the conversation; never assume the care "
            "recipient: a caregiver ordering something for themselves gives their own id. If it is not clear, give \"\": for an order, "
            "start_task then asks whom it is for in the same line as the delivery address."}
