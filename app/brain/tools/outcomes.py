"""Outcomes, feedback and learning consent."""

from __future__ import annotations

from app.brain.tools.base import ABOUT, ABOUT_OWN, ToolRefused, TurnCtx, tool
from app.brain.tools.memory import _sync_medicine
from app.brain.tools.orders import _task

# ── outcomes, feedback, consent ────────────────────────────────────────────────


@tool(
    "log_outcome",
    "Record what happened: a fall, a hospital or emergency visit, a doctor visit, a medicine the doctor changed, "
    "recovery, or all fine after a worry. Use it whenever someone tells you one of these (alerts still follow their "
    "own rules). These are the facts Saheli learns from.",
    {
        "kind": {"type": "string", "enum": ["fall", "hospital_visit", "er_visit", "doctor_visit", "medicine_changed", "recovered", "all_fine", "other"]},
        "summary": {"type": "string", "description": "One line in plain words, e.g. 'slipped in the bathroom, knee hurts'"},
        "about": ABOUT_OWN,
    },
    ["kind", "summary"],
)
async def log_outcome(ctx: TurnCtx, a: dict) -> dict:
    from app.care import outcomes

    fresh = await outcomes.record_outcome(ctx.session, family_id=ctx.family_id, subject_id=ctx.own(a.get("about")), kind=a["kind"],
                                          summary=a["summary"], source="said", actor_id=ctx.speaker.get("id"),
                                          ref=ctx.message_ref and f"oc:{ctx.message_ref}:{a['kind']}")
    return {"logged": fresh, "kind": a["kind"]}


@tool(
    "offer_buttons",
    "Attach one-tap answers to your reply. kind 'feedback' (👍/👎) after you mention a pattern, with key "
    "'pattern:<pattern key>:subj=<person id>' as given in PATTERNS NOTICED; kind 'outcome' (all fine / saw doctor / "
    "hospital) or 'visit' (no change / medicine changed / later) when you ask how something turned out; kind 'order' (yes / "
    "change / cancel) with key = the task id when you ask the one confirm of an order in your reply; kind 'later' (yes / any "
    "amount / no) with key = the task id when you ask the one question of a booking set for later.",
    {"kind": {"type": "string", "enum": ["outcome", "visit", "feedback", "order", "later"]}, "key": {"type": "string"}},
    ["kind", "key"],
)
async def offer_buttons(ctx: TurnCtx, a: dict) -> dict:
    from app.care import outcomes
    from app.tasks import runtime

    if a["kind"] == "order":
        # A confirm is offered only while the order waits for that very confirm (founder 2026-10-12: "after success it
        # asked me whether to place"): never for an order already being placed, placed, or changed since.
        tid = str(a["key"]).partition(":")[0]
        task = await _task(ctx, tid)
        fp = str((task.details or {}).get("cart_fp") or "")
        if task.status != "awaiting_confirm" or task.input_needed != "confirm" or not fp:
            raise ToolRefused(f"This order is not waiting for a yes any more ({runtime.describe(task)}). Do not ask to confirm "
                              "it; say where it stands.")
        a = {**a, "key": f"{task.id}:{fp[:8]}"}  # the buttons carry the cart check, so a tap never places another cart
    if a["kind"] == "later":
        from app.tasks import later

        task = await _task(ctx, str(a["key"]).partition(":")[0])
        if task.input_needed != "later":
            raise ToolRefused(f"This is not waiting for the question for later ({runtime.describe(task)}).")
        a = {**a, "key": f"{task.id}:{later.suggest_max(runtime._later_amount(task)) or ''}"}  # yes = up to the suggested limit
    ctx.buttons = outcomes.buttons(a["kind"], a["key"], ctx.profiles.get(ctx.speaker.get("id")))
    ctx.buttons_spec = (a["kind"], a["key"])  # titled again in the reply's own script when it is final (loop)
    return {"attached": [b["title"] for b in ctx.buttons]}


@tool(
    "feedback",
    "Record that someone found something you said useful or not (when they say so in words): a pattern, an alert or a "
    "suggestion. target is the key it was given (for patterns: 'pattern:<key>:subj=<id>').",
    {"target": {"type": "string"}, "vote": {"type": "string", "enum": ["up", "down"]}, "about": ABOUT_OWN},
    ["target", "vote"],
)
async def feedback(ctx: TurnCtx, a: dict) -> dict:
    from app.care import outcomes

    await outcomes.record_feedback(ctx.session, family_id=ctx.family_id, subject_id=ctx.own(a.get("about")), target=a["target"],
                                   vote=a["vote"], by=ctx.speaker.get("id"), source="said")
    return {"recorded": True}


@tool(
    "learning_consent",
    "A caregiver says yes or no to sharing this family's anonymised data to make Saheli better for every family. "
    "Their own memory keeps learning either way. Only caregivers can decide.",
    {"granted": {"type": "boolean"}},
    ["granted"],
)
async def learning_consent(ctx: TurnCtx, a: dict) -> dict:
    from app.care import outcomes

    if ctx.speaker_is_elder or ctx.is_system:
        raise ToolRefused("Only a caregiver can decide this; ask them.")
    return await outcomes.set_consent(ctx.session, ctx.family_id, ctx.elder_id, granted=bool(a["granted"]), by=ctx.speaker.get("id") or "")


@tool(
    "forget",
    "Remove something they asked to delete, take back, or said was a joke or never true. One call hides matching "
    "logged events and memory-note lines, and stops matching care-record facts (a medicine, an allergy, a condition, "
    "a rule). Use this, not stop, when they say delete it, remove it from the system, forget it, or that it was a joke. "
    "stop is only for a real fact that ended (a medicine they no longer take). what is the words of the saved event or "
    "fact, not their whole sentence, with its time or number when it has one ('sleep at 04:04'). never_true when they say "
    "it was a joke, a mistake, or never happened. When the words match more than one memory, nothing is removed and the "
    "matches come back: call again with refs of only the ones they meant.",
    {"what": {"type": "string", "description": "Words of the saved event or fact to remove, e.g. 'chest wound' or 'fight with Rahul'"},
     "refs": {"type": "array", "items": {"type": "string"}, "description": "The refs from an earlier forget's matches: exactly these are removed"},
     "never_true": {"type": "boolean", "description": "true when they say it was a joke, never true, or saved by mistake, so it is never said again"},
     "about": ABOUT},
    ["what"],
)
async def forget_tool(ctx: TurnCtx, a: dict) -> dict:
    from app.care import memory_upkeep

    subjects = [ctx.subject(a.get("about"))]
    if not ctx.speaker_is_elder and subjects[0] == ctx.elder_id:
        subjects.append("family")
    out = await memory_upkeep.forget(ctx.session, ctx.family_id, subjects, a["what"], by=ctx.speaker.get("id") or "",
                                     source_kind="caregiver_said" if ctx.is_caregiver else ctx.source_kind,
                                     never_true=bool(a.get("never_true")), refs=a.get("refs") or None, ask_if_many=True)
    for med in out.get("stopped_medicines") or []:
        await _sync_medicine(ctx, med["subject"], med["key"], med.get("value") or {}, active=False)
    return out
