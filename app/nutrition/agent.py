"""The nutrition agent (plan §N1): a worker inside the engine. Saheli stays the only voice: the agent answers her log_food
tool with a saved meal, or with ONE question for her to ask (how many, which one, a photo or the label, who ate it, did
you eat the order, is this photo right), and she asks the person.

Its rules:
- Structured fields only. The brain reads people's words; the agent never does. AI calls judge food data (the lookup's
  normaliser, vision, the allergy and rule check); code only checks data.
- A packaged item or a drink with no quantity is one serving, saved as an estimate, and the reply names the size and
  asks them to say if it was more. A counted food with no number is asked once. A plate or a katori with no amount
  stays a usual-portion estimate.
- At most 2 follow-up questions per meal (quantity, which, label, photo); then the best estimate is saved, marked
  `estimate`. A photo is always confirmed first (founder Q4); an order's food only after a yes or part (founder Q5);
  who ate it is never guessed.
- Medicines and supplements never go in the food log (plan §N6.3).
- An allergy on record that a food matches is flagged on the entry, and Saheli mentions it gently (plan §N12.2).
- A strict target or food rule that a meal goes past is pointed out once; a guide shows progress only (founder Q6).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.core import clock
from app.llm import router
from app.nutrition import entries, jobs, lookup, nudge_hooks, pantry, photos, settings, targets, turn, vision
from app.nutrition import nutrients as N
from app.nutrition import portions as P
from app.nutrition.models import NutritionEntry, NutritionFood, NutritionJob

logger = logging.getLogger(__name__)

CHECK_PROMPT = """You check foods someone just ate against their allergies and their food rules.

ALLERGIES lists what they are allergic to (with the reaction when known). RULES lists food rules someone set for them,
each with a key. FOODS lists what they ate, with ingredients and declared allergens when known.

For each food: allergy = the allergies it clearly contains or usually contains (peanut chutney contains peanut; a plain
boiled egg contains egg); rules = the keys of the rules whose food it falls under (a boiled egg falls under "max 2 eggs a
day"; sweet tea falls under "no sugar"; plain dal does not fall under "no sugar"). Empty lists when none.

Return only JSON: {"items": [{"i": 0, "allergy": [], "rules": []}]}"""


@dataclass
class Who:
    """Who is telling and where (built by the brain tool from TurnCtx, or by the dashboard)."""

    family_id: str
    speaker_id: str
    channel: str = "whatsapp"  # whatsapp | dashboard | scheduler
    images: list[dict] = field(default_factory=list)
    essential: bool = True  # a person is waiting for this answer
    message_ref: str | None = None
    names: dict[str, str] = field(default_factory=dict)  # household id → name
    caregiver: bool = False  # the speaker is a caregiver (or the dashboard)


ASK_FIELDS = ("eaten_at", "minutes_ago", "meal", "for_whom", "from_photo", "label_photo", "confidence", "confirmed", "unsure", "cancel",
              "consumed", "portion_eaten", "skipped", "from_order_task", "usual")
ITEM_FIELDS = ("name", "quantity", "unit", "brand", "restaurant", "from_order_task", "barcode", "notes", "said", "food_id")


def clean_item(it: dict) -> dict | None:
    if not isinstance(it, dict):
        return None
    out = {}
    for k in ITEM_FIELDS:
        v = it.get(k)
        if v in (None, "", []):
            continue
        if k == "quantity":
            n = N.number(v)
            if n is None or n <= 0 or n > 5000:
                continue
            v = n
        elif k == "unit":
            if v not in P.UNITS:
                continue
        else:
            v = str(v).strip()[:200]
        out[k] = v
    return out if out.get("name") or out.get("from_order_task") or out.get("barcode") or out.get("food_id") else None


def clean_ask(a: dict) -> dict:
    out = {k: a[k] for k in ASK_FIELDS if k in a and a[k] is not None}
    out["items"] = [x for x in (clean_item(i) for i in a.get("items") or []) if x][:15]
    return out


def merge(job: NutritionJob, a: dict) -> None:
    """An answer to the agent's question: the brain sends the fields it learned (and the full items when they changed)."""
    new = clean_ask(a)
    ask = dict(job.ask or {})
    for k, v in new.items():
        if k == "items":
            if v:
                ask["items"] = v
                job.draft = {k2: v2 for k2, v2 in (job.draft or {}).items() if k2 in ("photo", "photo_ref", "label_food", "asked")}
            continue
        ask[k] = v
    job.ask = ask
    job.answers = [*(job.answers or []), {"at": clock.now().isoformat(), "args": {k: v for k, v in a.items() if k != "answer_to"}}][-10:]
    job.state = "resolving"
    job.updated_at = clock.now()


def meal_of(ask: dict, at: datetime) -> str:
    m = ask.get("meal")
    return m if isinstance(m, str) and m in P.MEALS else P.meal_for(at)


def parse_at(v) -> datetime | None:
    if not v:
        return None
    try:
        at = datetime.fromisoformat(str(v).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return at.replace(tzinfo=clock.IST) if at.tzinfo is None else at


def _minutes_ago(v) -> float | None:
    """A number of minutes the brain passed. Not a reading of anyone's words."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        n = float(v)
    elif isinstance(v, str):
        try:
            n = float(v.strip())
        except ValueError:
            return None
    else:
        return None
    if n < 0 or n > 18 * 60:
        return None
    return n


def eaten_time(ask: dict | None, *, now: datetime | None = None) -> datetime:
    """When they ate: a clock time the brain named, else minutes_ago before the message, else the message time."""
    named = parse_at((ask or {}).get("eaten_at"))
    if named is not None:
        return named
    base = now or clock.now()
    ago = _minutes_ago((ask or {}).get("minutes_ago"))
    if ago is None:
        return base
    return base - timedelta(minutes=ago)


# A missing number is one of these. Decided from the food record (kind and tags), never from what they typed.
_DRINK_TAGS = frozenset({"tea", "coffee", "water", "sugary_drink", "alcohol"})


def one_standard_serving(food) -> bool:
    """One pack, or one drink, when they gave no number."""
    if food is None:
        return False
    if getattr(food, "kind", None) == "packaged":
        return True
    return bool(set(getattr(food, "tags", None) or []) & _DRINK_TAGS)


def _fill_one(it: dict, food) -> None:
    unit = food.default_unit if getattr(food, "default_unit", None) in P.UNITS else "serving"
    qty = float(getattr(food, "typical_qty", None) or 1) if unit in ("g", "ml") else 1.0
    it["quantity"] = qty
    it["unit"] = unit
    it["_assumed"] = True


def _assumed_flags(food) -> dict:
    unit = food.default_unit if getattr(food, "default_unit", None) in P.UNITS else None
    size = getattr(food, "serving_g", None) or ((food.units or {}).get(unit) if unit else None)
    out: dict = {"assumed_one": True}
    if not size:
        return out
    drink = bool(set(food.tags or []) & _DRINK_TAGS) or unit in {"glass", "cup", "ml"}
    out["serving_ml" if drink else "serving_g"] = float(size)
    return out


def _saved_line(r: NutritionEntry) -> tuple[str, bool]:
    flags = r.flags or {}
    if flags.get("usual"):
        return f"{entries.line(r)} (your usual)", True
    if not flags.get("assumed_one"):
        return entries.line(r), False
    name = r.name_said or r.name
    q = r.quantity
    qn = int(q) if q is not None and float(q).is_integer() else q
    if r.unit not in ("g", "ml") and flags.get("serving_ml"):
        return f"{qn} {name} ({N.fmt(flags['serving_ml'])} ml)", True
    if r.unit not in ("g", "ml") and flags.get("serving_g"):
        return f"{qn} {name} ({N.fmt(flags['serving_g'])} g)", True
    return entries.line(r), True


def _with_facts(text: str, flags: dict) -> str:
    """The food's own nutrient facts, in the saved line (never advice)."""
    facts = [str(f) for f in (flags or {}).get("facts") or []]
    if not facts:
        return text
    bit = ", ".join(facts)
    if text.endswith(")"):
        return text[:-1] + f", {bit})"
    return f"{text} ({bit})"


def _say(rows: list[NutritionEntry], tracking: str, relevant: set[str] | None = None) -> str:
    food = [r for r in rows if r.kind == "food"]
    if not food:
        return "Noted: " + ", ".join(entries.line(r) for r in rows) + "."
    if any(r.confidence == "pending" for r in food):
        what = ", ".join(entries.line(r) for r in food)
        known = [r for r in food if r.confidence != "pending"]
        phrase = N.highlight_phrase(entries.totals(known), relevant) if known else ""
        lead = f"Saved: {what}, {phrase}." if phrase else f"Saved: {what}."
        line = f"{lead} Saved, adding the numbers in a minute."
        if any((r.flags or {}).get("kind_open") for r in food):
            return line[:-1] + " — tell me if it was a different one."
        return line
    parts = [_saved_line(r) for r in food]
    what = ", ".join(_with_facts(text, r.flags or {}) for (text, _), r in zip(parts, food))
    tot = entries.totals(food)
    nums = "" if tracking == "counts" else N.highlight_phrase(tot, relevant)
    base = f"Saved: {what}, {nums}." if nums else f"Saved: {what}."
    more = any(assumed for _, assumed in parts)
    kind = any((r.flags or {}).get("kind_open") for r in food)
    if more and kind:
        return base[:-1] + " — tell me if it was more or a different one."
    if more:
        return base[:-1] + " — tell me if it was more."
    if kind:
        return base[:-1] + " — tell me if it was a different one."
    return base


def _curiosity(rows: list[NutritionEntry], tracking: str) -> str:
    """At most one follow-up, and only when it fits. Never advice. A correction already asked is the one question."""
    food = [r for r in rows if r.kind == "food"]
    if not food or tracking == "counts":
        return ""
    if all("water" in ((r.flags or {}).get("tags") or []) for r in food):
        return ""
    facts = {t for r in food for t in ((r.flags or {}).get("facts") or [])}
    heavy = bool(facts & {"fried", "high fat", "high sodium"}) or N.kcal(entries.totals(food)) >= 600
    if heavy:
        return "How are you feeling after that?"
    if len(food) == 1 and not (food[0].flags or {}).get("usual"):
        return "Is this a usual one for you?"
    return ""


FOLLOW_HOW = ("Ask this one question after the save line, in their language. Never a second question and never advice. "
              "If they say it is their usual, call log_food with make_usual true and no new items. "
              "If they say they liked it or how they feel, call remember with domain preference, in their words. Then stop.")


# ── order meals (plan §N6.1): eaten only after a yes ─────────────────────────


async def _task(session: AsyncSession, family_id: str, task_id: str):
    from app.tasks.models import Task

    try:
        t = await session.get(Task, uuid.UUID(str(task_id)))
    except ValueError:
        return None
    return t if t and t.family_id == family_id else None


def order_items(task) -> list[dict]:
    """The dishes that came: the placed cart, else what was picked (with the restaurant)."""
    from app.specialists import channels

    r, d = task.result or {}, task.details or {}
    chosen = [c for c in d.get("chosen") or [] if isinstance(c, dict) and c.get("name")]
    rest = next((str(c.get("restaurant")) for c in chosen if c.get("restaurant")), None)
    placed = [i for i in r.get("items") or [] if isinstance(i, dict) and i.get("name")] if (d.get("cart_fp") or r.get("placed")) else []
    if placed:
        return [{"name": str(i["name"])[:200], "quantity": float(i.get("qty") or 1), "unit": "serving", "from_order_task": str(task.id),
                 **({"restaurant": str(i.get("restaurant") or rest)[:160]} if (i.get("restaurant") or rest) else {})} for i in placed]
    asked = d.get("items") or []
    return [{"name": str(c["name"])[:200], "quantity": float(channels._qty_for(asked, c)), "unit": "serving", "from_order_task": str(task.id),
             **({"restaurant": str(c["restaurant"])[:160]} if c.get("restaurant") else {})} for c in chosen]


def _delivered(task) -> bool:
    r = task.result or {}
    if not (r.get("placed") or r.get("delivered")):
        return False
    if r.get("delivered") or (r.get("feedback") or {}).get("arrived") in ("yes", "partly"):
        return True
    return task.updated_at <= clock.now() - timedelta(hours=2)


async def order_waiting(session: AsyncSession, family_id: str, subject_id: str | None = None, *, hours: int = 48) -> list[dict]:
    """Delivered restaurant meals nobody has said were eaten or not (the dashboard's "Waiting: did they eat it?")."""
    from app.tasks.models import Task
    from app.tasks.skills import SKILLS

    q = select(Task).where(Task.family_id == family_id, Task.kind == "order", Task.created_at >= clock.now() - timedelta(hours=hours))
    if subject_id:
        q = q.where(Task.subject_id == subject_id)
    tasks = [t for t in (await session.execute(q)).scalars() if SKILLS.get(t.service, {}).get("kind") == "food" and _delivered(t)]
    if not tasks:
        return []
    ids = [str(t.id) for t in tasks]
    logged = set((await session.execute(select(NutritionEntry.source_ref).where(
        NutritionEntry.family_id == family_id, NutritionEntry.source_ref.in_(ids)))).scalars())
    answered = set()
    for j in (await session.execute(select(NutritionJob).where(NutritionJob.family_id == family_id, NutritionJob.state == "dropped"))).scalars():
        tid = (j.result or {}).get("order_task")
        if tid:
            answered.add(tid)
    out = []
    for t in tasks:
        if str(t.id) in logged or str(t.id) in answered:
            continue
        items = order_items(t)
        out.append({"task_id": str(t.id), "subject_id": t.subject_id, "what": ", ".join(i["name"] for i in items)[:200] or t.goal,
                    "restaurant": next((i.get("restaurant") for i in items if i.get("restaurant")), None), "service": t.service,
                    "since": (t.updated_at or t.created_at).isoformat()})
    return out


async def log_order_meal(session: AsyncSession, task, *, consumed: str, portion_eaten: str | None = None, eaten_by: str | None = None,
                         by: str, channel: str = "whatsapp") -> dict:
    """A consumption answer for a delivered restaurant order, from code (aftercare, the dashboard's Yes / Part / No)."""
    who = Who(family_id=task.family_id, speaker_id=by, channel=channel, essential=channel != "scheduler", caregiver=channel == "dashboard")
    a = {"items": [], "from_order_task": str(task.id), "consumed": consumed, "portion_eaten": portion_eaten}
    return await handle(session, who, a, subject_id=eaten_by or task.subject_id)


# ── the agent ────────────────────────────────────────────────────────────────


_ONCE = frozenset({"photo", "confirm", "who_ate", "consumed"})


def _already_asked(job: NutritionJob, reason: str, question: str) -> bool:
    """The same question text, or a second confirm / photo / who / eaten question, for this one job."""
    for row in (job.draft or {}).get("asked") or []:
        if not isinstance(row, dict):
            continue
        if row.get("question") == question:
            return True
        if reason in _ONCE and row.get("reason") == reason:
            return True
    return False


def _note_asked(job: NutritionJob, reason: str, question: str) -> None:
    draft = dict(job.draft or {})
    draft["asked"] = [*list(draft.get("asked") or []), {"reason": reason, "question": question}]
    job.draft = draft


def _counts_they_accepted(items: list[dict], reading: dict) -> list[dict]:
    """They named the dish and did not change the number. That offered number is the count."""
    photo = [i for i in (reading.get("items") or []) if isinstance(i, dict)]
    out = []
    for it in items:
        row = dict(it)
        if not row.get("quantity"):
            src = next((p for p in photo if p.get("name") == row.get("name") or vision._other(p) == row.get("name")), None)
            if src is None and len(photo) == 1:
                src = photo[0]
            if src and src.get("quantity"):
                row["quantity"] = src["quantity"]
                if not row.get("unit") and src.get("unit"):
                    row["unit"] = src["unit"]
            if src and src.get("grams"):
                row["_grams"] = src["grams"]
        row["_photo"] = True
        out.append(row)
    return out


def _photo_settled(job: NutritionJob, ask: dict) -> bool:
    """They already said this photo was eaten, or whose diary it is. A confirm on the photo's own message is not that."""
    if ask.get("consumed") in ("yes", "part"):
        return True
    whom = ask.get("for_whom")
    if isinstance(whom, str) and whom.strip():
        return True
    return bool(ask.get("confirmed") and (job.answers or []))


def _still_looking(job: NutritionJob) -> dict:
    """The turn is out of time. The background save is the one line they hear."""
    job.state, job.updated_at = "resolving", clock.now()
    return {"result": "looking_up", "job_id": str(job.id),
            "say": "Tell them once, in a few words, that you are checking its nutrition and will tell them when it is saved "
                   "(no numbers yet). If you already said that, say nothing."}


class NutritionAgent:
    """One job, one step at a time: returns a saved meal, a question, "looking_up" or "dropped"."""

    def __init__(self, session: AsyncSession, who: Who) -> None:
        self.session, self.who = session, who

    def name_of(self, pid: str) -> str:
        return self.who.names.get(pid) or "them"

    async def run(self, job: NutritionJob, *, deadline: float | None, background: bool = False, estimates_only: bool = False) -> dict:
        s, who = self.session, self.who
        ask = dict(job.ask or {})
        draft = dict(job.draft or {})
        items: list[dict] = list(ask.get("items") or [])
        if job.state == "saved":
            return {"result": "saved", "already": True, "job_id": str(job.id), "say": "",
                    "how": "Already in the food log. Say nothing about it."}
        job.state = "resolving"
        if ask.get("cancel"):
            await self._close_photo(draft)
            jobs.finish(job, "dropped", {"why": "cancelled"})
            return {"result": "dropped", "job_id": str(job.id), "why": "Nothing was saved (they said never mind or did not know)."}

        # an order's food: only after a yes or part (founder Q5)
        task_id = ask.get("from_order_task") or next((i.get("from_order_task") for i in items if i.get("from_order_task")), None)
        task = None
        if task_id:
            task = await _task(s, who.family_id, str(task_id))
            if task is None:
                jobs.finish(job, "dropped", {"why": "no such order"})
                return {"result": "dropped", "job_id": str(job.id), "why": "That order is not this family's; nothing was saved."}
            from app.tasks.skills import SKILLS

            if SKILLS.get(task.service, {}).get("kind") != "food":
                # groceries go to the pantry and are counted when someone says they ate them; medicines never (founder Q5)
                jobs.finish(job, "dropped", {"why": "not a restaurant order", "order_task": str(task.id)})
                return {"result": "dropped", "job_id": str(job.id),
                        "why": "That order was groceries or medicines, not a cooked meal: nothing was logged as eaten. Log what they "
                               "actually ate from it as food (log_food items), and a pack at home is counted down."}
            dishes = ", ".join(i["name"] for i in order_items(task))[:120] or task.goal
            if ask.get("consumed") not in ("yes", "part", "no"):
                q = self._question(job, "consumed", f"Did {self._you(job)} eat the {dishes} that came? All of it?",
                                   ["All", "Half", "A little", "No / someone else"])
                if q:
                    return q
                jobs.finish(job, "dropped", {"why": "not eaten", "order_task": str(task.id)})
                return {"result": "dropped", "job_id": str(job.id), "why": "Not eaten: nothing logged from this order."}
            if ask["consumed"] == "no":
                jobs.finish(job, "dropped", {"why": "not eaten", "order_task": str(task.id)})
                return {"result": "dropped", "job_id": str(job.id), "why": "Not eaten: nothing logged from this order."}
            if not items:
                items = order_items(task)
                ask["items"] = items
                job.ask = ask

        if ask.get("for_whom") == "" and not ask.get("from_photo"):
            # One person in the household is not a guess. Asking them "who ate it" only repeats.
            # A photo asks eaten and whose diary in the one confirm, not a second who question.
            people = [(i, n) for i, n in who.names.items() if n]
            if len(people) > 1:
                q = self._question(job, "who_ate", "Who ate it?", [n for _, n in people][:5])
                if q:
                    return q
            if len(people) == 1:
                job.subject_id = people[0][0]
            ask = dict(ask)
            ask.pop("for_whom", None)
            job.ask = ask

        tracking = (await settings.get(s, who.family_id, job.subject_id))["tracking"]
        sizes = (await settings.get(s, who.family_id, job.subject_id))["sizes"]

        if ask.get("skipped") and not items:
            at = eaten_time(ask)
            slot = await entries.assign_slot(s, who.family_id, job.subject_id, at, named=ask.get("meal") if isinstance(ask.get("meal"), str) else None)
            meal = slot.meal
            if not await jobs.claim(s, job):
                return {"result": "saved", "already": True, "job_id": str(job.id), "say": "",
                        "how": "Already in the food log. Say nothing about it."}
            e = await entries.save_skipped(s, family_id=who.family_id, subject_id=job.subject_id, logged_by=who.speaker_id, meal=meal, eaten_at=at,
                                           source=self._source(job, ask), source_ref=job.source_ref, job_id=job.id, day=slot.day)
            jobs.finish(job, "saved", {"entries": [str(e.id)], "skipped": meal})
            return {"result": "saved", "job_id": str(job.id), "saved": [], "skipped": meal, "say": f"Noted: {meal} skipped."}

        # a label photo: the pack's panel becomes the family's food (plan §N4.4)
        if ask.get("label_photo") and not draft.get("label_food"):
            if who.images:
                label = await vision.read_label(who.images, essential=who.essential)
                if label:
                    food = None
                    if label.get("barcode"):
                        off = await lookup.off_barcode(label["barcode"])
                        if off and not label.get("per_100g"):
                            label = {**off, **{k: v for k, v in label.items() if v}}
                    food = await lookup.save_food(s, {**label, "source": "label"}, family_id=who.family_id,
                                                  alias=(items[0]["name"] if items else label.get("name")))
                    draft["label_food"] = str(food.id)
                    if items:
                        items[0] = {**items[0], "food_id": str(food.id)}
                    else:
                        items = [{"name": food.name, "food_id": str(food.id), **({"brand": food.brand} if food.brand else {})}]
                    ask["items"] = items
                    job.ask, job.draft = ask, draft
            if not draft.get("label_food"):
                q = self._question(job, "label", "I couldn't read the label. Could you send a clear photo of the nutrition table on the pack?") if jobs.can_ask(job) else None
                if q:
                    return q
                ask["label_photo"] = False

        # a meal photo: one question, the reading and whether it was eaten (and whose diary), then save only on a clear answer
        if ask.get("from_photo"):
            if who.images and not draft.get("photo"):
                reading = await vision.read_meal(who.images, hints=", ".join(i.get("name", "") for i in items), essential=who.essential)
                if reading and reading.get("food"):
                    draft["photo"] = reading
                    draft["photo_ref"] = await photos.put(who.family_id, f"pending/{job.id}", who.images[0])
                    job.draft = draft
            if not draft.get("photo"):
                if items:
                    ask["from_photo"] = False  # their words are what we have
                else:
                    q = self._question(job, "photo", "I couldn't make out the food in the photo. What was it, and how much?") if jobs.can_ask(job) else None
                    if q:
                        return q
                    jobs.finish(job, "dropped", {"why": "photo unreadable"})
                    return {"result": "dropped", "job_id": str(job.id), "why": "The photo could not be read and no food was named; nothing saved."}
            elif ask.get("consumed") == "no":
                await self._close_photo(draft)
                jobs.finish(job, "dropped", {"why": "not eaten"})
                return {"result": "dropped", "job_id": str(job.id), "why": "Not eaten: nothing was saved."}
            elif not _photo_settled(job, ask):
                # A confirm flag on the same message as the photo does not count: they have not seen the reading yet.
                about = await self._photo_numbers(job, draft["photo"])
                q = self._photo_question(job, draft["photo"], about)
                if q:
                    return q
            if draft.get("photo") and vision.open_choice(draft["photo"]) and not items:
                # Two names were offered. A yes that picks neither is not a save.
                await self._close_photo(draft)
                jobs.finish(job, "dropped", {"why": "which dish"})
                return {"result": "dropped", "job_id": str(job.id),
                        "why": "They did not say which dish; nothing was saved."}
            if draft.get("photo") and items and not any(i.get("quantity") for i in items):
                items = _counts_they_accepted(items, draft["photo"])
                ask["items"] = items
                job.ask = ask
            elif draft.get("photo") and not items:
                items = [{"name": i["name"], **({"quantity": i["quantity"]} if i.get("quantity") else {}), **({"unit": i["unit"]} if i.get("unit") else {}),
                          "_grams": i.get("grams"), "_photo": True} for i in draft["photo"]["items"]]
                ask["items"] = items
                job.ask = ask

        learned = await self._usuals(job.subject_id)
        at_guess = eaten_time(ask)
        meal_guess = (await entries.assign_slot(
            s, who.family_id, job.subject_id, at_guess, named=ask.get("meal") if isinstance(ask.get("meal"), str) else None)).meal
        if ask.get("usual") and not items and not ask.get("from_photo"):
            rows = nudge_hooks.usual_for_meal(learned, meal_guess)
            if not rows:
                jobs.finish(job, "dropped", {"why": "no usual"})
                return {"result": "dropped", "job_id": str(job.id),
                        "why": "No usual meal is learned from their own logs yet; nothing saved."}
            items = [nudge_hooks.as_item(r) for r in rows]
        elif items and not ask.get("from_photo"):
            items = nudge_hooks.stamp(learned, items, meal_guess, expand=bool(ask.get("usual")))
        if items:
            ask["items"] = items
            job.ask = ask

        if not items:
            jobs.finish(job, "dropped", {"why": "nothing named"})
            return {"result": "dropped", "job_id": str(job.id), "why": "No food was named; nothing saved."}

        # A known food (their usual, or a name already in the catalogue) is saved with no model call.
        clean = [{k: v for k, v in i.items() if not str(k).startswith("_")} for i in items]
        stock = await pantry.in_stock(s, who.family_id, job.subject_id)
        cached = await lookup.resolve(s, who.family_id, clean, pantry_rows=stock, cache_only=True)
        found = cached if all(r.food is not None or r.medicine for r in cached) else None

        # A flavour fold is a second model call. The person's reply does not wait for it: the brain already sends one
        # item for two flavours. The background look-up still folds before it searches.
        if found is None and background and len(items) > 1 and not ask.get("from_photo") and not ask.get("from_order_task") and not estimates_only:
            items = await lookup.fold_descriptions(items, essential=who.essential, timeout_s=4.0)
            ask["items"] = items
            job.ask = ask
            clean = [{k: v for k, v in i.items() if not str(k).startswith("_")} for i in items]

        if found is None and not background and not estimates_only:
            # One catalogue hit, or one estimate. The reply does not wait on the web: a miss with no numbers is
            # saved pending, and the background fill looks it up, then estimates only if that look-up is empty.
            found = await lookup.reply_lookup(s, who.family_id, clean, pantry_rows=stock, essential=who.essential,
                                               no_questions=not jobs.can_ask(job))
        elif found is None:
            try:
                found = await lookup.resolve(s, who.family_id, clean, pantry_rows=stock, deadline=deadline,
                                             essential=who.essential, no_questions=not jobs.can_ask(job),
                                             estimates_only=estimates_only)
            except lookup.Slow:
                if not estimates_only:
                    # One estimate for the whole meal, not another search and not one call per food. If that also
                    # does not fit, the background save is the one line they hear.
                    estimated = await self.run(job, deadline=time.monotonic() + 3, background=background, estimates_only=True)
                    if estimated.get("result") == "saved":
                        return estimated
                    job.needs = None
                    job.questions = 0
                return _still_looking(job)
        medicines = [r.item.get("name") for r in found if r.medicine]
        keep = [(i, r) for i, r in zip(items, found) if not r.medicine]
        if medicines and not keep:
            jobs.finish(job, "dropped", {"why": "medicine"})
            return {"result": "dropped", "job_id": str(job.id), "medicine": medicines,
                    "why": "These are medicines or supplements, not food: use log_dose for them (never the food log)."}
        for it, r in keep:
            if r.assumed_kind:
                it["_kind_open"] = True
            if r.needs == "label" and jobs.can_ask(job):
                q = self._question(job, "label", f"Which {it['name']} was it? Or send a photo of the label (the nutrition table on the back).",
                                   item=it["name"])
                if q:
                    return q
                it["_kind_open"] = True
            if r.needs == "photo" and r.food is None:
                # A photo, or a food they already named, is never asked for as another photo. "What's in it" is only
                # a home-cooked dish of unknown make, and only once. Anything else is saved, with one correction.
                home = (r.decision or {}).get("kind") == "recipe" and not ask.get("from_photo") and not it.get("notes")
                if home and jobs.can_ask(job):
                    q = self._question(job, "photo", f"What's in the {it['name']}, roughly?", item=it["name"])
                    if q:
                        return q
                if estimates_only:
                    # The web search has not run yet. An empty estimate is not the save; the background look-up is.
                    return _still_looking(job)
                # No numbers yet. Save the food they named and fill the numbers after. Never as 0 kcal.
                # The kind is still open, so the one correction stays on the line.
                it["_pending"] = True
                it["_kind_open"] = True
        # how much: one pack or one drink is saved; a counted food with no number is asked once
        # (unless they don't know, or a photo showed it). A plate or a katori stays a usual-portion estimate.
        # A meal that already has an amount is saved; one counted item with no number does not hold the rest back.
        told = any(i.get("quantity") or i.get("_grams") for i, _r in keep)
        for it, r in keep:
            if it.get("quantity") or it.get("_grams") or ask.get("unsure") or r.food is None:
                continue
            usual = None if it.get("_photo") else nudge_hooks.match_usual(learned, str(it.get("name") or ""), meal_guess)
            if usual and usual.get("quantity"):
                it["quantity"] = float(usual["quantity"])
                if usual.get("unit") in P.UNITS:
                    it["unit"] = usual["unit"]
                it["_usual"] = True
                it["_assumed"] = True
                continue
            if one_standard_serving(r.food):
                _fill_one(it, r.food)
                continue
            if r.food.countable and told:
                _fill_one(it, r.food)
                continue
            if r.food.countable and jobs.can_ask(job) and ask.get("confidence") != "photo":
                opts = await self._quantity_options(job.subject_id, r.food)
                q = self._question(job, "quantity", f"How many {it['name']}?", opts, item=it["name"])
                if q:
                    return q
                _fill_one(it, r.food)

        # save
        at = eaten_time(ask)
        if at > clock.now() + timedelta(minutes=10):
            at = clock.now()
        if at < clock.now() - timedelta(days=30):
            jobs.finish(job, "dropped", {"why": "too old"})
            return {"result": "dropped", "job_id": str(job.id), "why": "That is more than 30 days ago; the food log keeps the last 30 days from chat."}
        if not await jobs.claim(s, job):
            return {"result": "saved", "already": True, "job_id": str(job.id), "say": "",
                    "how": "Already in the food log. Say nothing about it."}
        only_water = bool(keep) and all(r.food is not None and "water" in (r.food.tags or []) for _, r in keep)
        slot = await entries.assign_slot(s, who.family_id, job.subject_id, at,
                                         named=ask.get("meal") if isinstance(ask.get("meal"), str) else None, only_water=only_water)
        meal = slot.meal
        conf_base = "photo" if ask.get("from_photo") and draft.get("photo") and not any(not i.get("_photo") for i in items) else (
            "label" if draft.get("label_food") else ("estimate" if ask.get("confidence") == "guess" else "said"))
        to_save: list[entries.Item] = []
        for it, r in keep:
            conf = conf_base
            if r.food is not None and r.food.source == "label" and conf == "said":
                conf = "label"
            if it.get("_assumed") and conf != "photo":
                conf = "estimate"
            elif not it.get("quantity") and not it.get("_grams"):
                conf = "estimate" if r.food is not None else "unquantified"
            if r.food is None:
                conf = "pending"
            assumed = {"usual": True} if it.get("_usual") else (
                _assumed_flags(r.food) if it.get("_assumed") and r.food is not None else {})
            if conf == "pending":
                assumed = {**assumed, "numbers": "pending"}
            if it.get("_kind_open"):
                assumed = {**assumed, "kind_open": True}
            to_save.append(entries.Item(name=it.get("name") or (r.food.name if r.food else "food"), food=r.food, quantity=it.get("quantity"),
                                        unit=it.get("unit"), name_said=it.get("said") or it.get("name") or "", confidence=conf,
                                        grams=it.get("_grams"), pantry=r.pantry, note=it.get("notes"), flags=assumed))
        flags = await self._checks(job.subject_id, to_save)
        for x, f in zip(to_save, flags):
            x.flags.update(f)
        replaced = []
        if job.subject_id == who.speaker_id:
            replaced = await entries.replace_caregiver_estimate(s, who.family_id, job.subject_id, slot.day, meal, by=who.speaker_id)
        consumed = ask.get("consumed") if task is not None else "yes"
        rows = await entries.save_meal(
            s, family_id=who.family_id, subject_id=job.subject_id, logged_by=who.speaker_id, items=to_save, meal=meal, eaten_at=at,
            source=self._source(job, ask, task=task), source_ref=str(task.id) if task is not None else job.source_ref, job_id=job.id,
            photo_ref=draft.get("photo_ref"), consumed=consumed or "yes", portion=ask.get("portion_eaten"), sizes=sizes, day=slot.day)
        for x in to_save:
            if x.pantry is not None:
                await pantry.use(s, x.pantry, float(x.quantity or 1) if x.unit in (None, "piece", x.pantry.unit) else 1.0)
        out = await self._result(job, rows, tracking, replaced=replaced, medicines=medicines)
        if any((r.flags or {}).get("refine") for r in rows):
            out["refine"] = True
        notice = "pending" if background else "reply"
        out["notice"] = notice
        jobs.finish(job, "saved", {"entries": [str(r.id) for r in rows], "meal_id": str(rows[0].meal_id) if rows else None,
                                   "say": out.get("say"), "today": out.get("today") or "", "notice": notice,
                                   **({"order_task": str(task.id)} if task else {})})
        return out

    # ── helpers ──

    async def _usuals(self, subject: str) -> dict:
        fact = await store.active_fact(self.session, self.who.family_id, subject, nudge_hooks.USUAL_KEY)
        return fact.value if fact and isinstance(fact.value, dict) else {}

    def _you(self, job: NutritionJob) -> str:
        return "you" if job.subject_id == self.who.speaker_id else self.name_of(job.subject_id)

    def _source(self, job: NutritionJob, ask: dict, task=None) -> str:
        if task is not None:
            return "order"
        if self.who.channel == "dashboard":
            return "dashboard"
        if self.who.channel == "scheduler":
            return "nudge"
        if ask.get("label_photo"):
            return "label"
        if ask.get("from_photo"):
            return "photo"
        return "caregiver" if job.subject_id != self.who.speaker_id else "chat"

    def _question(self, job: NutritionJob, reason: str, question: str, options: list[str] | None = None, item: str | None = None) -> dict | None:
        """One question. The same question, and a confirm or a photo question, is never sent twice for this job."""
        if _already_asked(job, reason, question):
            return None
        _note_asked(job, reason, question)
        n = jobs.ask_needs(job, reason, question, options, item)
        how = ("Say only the say sentence, in their language and script. Do not mention options, ids, or the words Options or Reply, "
               "and do not add a health warning of your own. Nothing is saved yet. When they answer, call log_food with "
               f"answer_to=\"{job.id}\" and the answer as fields (the full items if they changed; unsure=true if they don't know "
               "the amount; cancel=true if they say never mind). If this message already answers it, call log_food now and do not ask.")
        if reason == "confirm":
            how += (" I ate it is confirmed=true and consumed=yes. A person is for_whom that household id and consumed=yes. "
                    "Just sharing, or they did not eat it, is consumed=no. A number they say is the count; if they accept the "
                    "number in the question, pass that quantity. When two dishes are named, pass the one they choose, and "
                    "'or tell me what it was' is the food they name. Do not pass a dish they did not choose. Do not ask this again.")
        return {"result": "needs_input", "job_id": str(job.id), "say": question, "needs": [{**n, "job_id": str(job.id)}], "how": how}

    def _photo_question(self, job: NutritionJob, reading: dict, about: str) -> dict | None:
        """The reading, and whether it was eaten or is for someone else's diary, in one question."""
        desc = vision.describe(reading)
        others = [n for i, n in self.who.names.items() if n and i != self.who.speaker_id]
        named = []
        if vision.open_choice(reading):
            it = next((x for x in reading.get("items") or [] if vision._other(x)), None)
            if it:
                named = [str(it["name"]), vision._other(it)]
        if others:
            text = f"Looks like {desc}{about}. Did you eat this, or is it for someone else — like {others[0]}'s diary?"
            opts = [*named, "I ate it", *others[:2], "Just sharing"]
        else:
            text = f"Looks like {desc}{about}. Did you eat this, or are you just sharing the photo?"
            opts = [*named, "I ate it", "Just sharing"]
        return self._question(job, "confirm", text, opts[:6])

    async def _quantity_options(self, subject: str, food: NutritionFood) -> list[str]:
        opts: list[str] = []
        learned = await store.active_fact(self.session, self.who.family_id, subject, "learned:nudges_meals_portions")
        usual = ((learned.value or {}).get("items") or {}).get(food.name) if learned else None
        if isinstance(usual, dict) and usual.get("quantity"):
            opts.append(f"the usual {P.qty_label(N.number(usual['quantity']), usual.get('unit'))}".strip())
        return opts + ["1", "2", "3", "4", "more"]

    async def _close_photo(self, draft: dict) -> None:
        if draft.get("photo_ref"):
            await photos.delete(draft["photo_ref"])

    async def _checks(self, subject: str, items: list[entries.Item]) -> list[dict]:
        """Allergies and food rules on record, judged by a model (only when the person has any)."""
        s = self.session
        allergies = [f for f in await store.facts(s, self.who.family_id, subject, domains=["allergy"], statuses=("active",))
                     if not f.key.endswith(":none")]
        _, rules = await targets.current(s, self.who.family_id, subject)
        out: list[dict] = [{} for _ in items]
        foods = [(i, x) for i, x in enumerate(items) if x.food is not None or x.name]
        if not foods or not (allergies or rules):
            return out
        body = json.dumps({
            "ALLERGIES": [{"allergen": (f.value or {}).get("allergen") or f.key.split(":", 1)[-1], "reaction": (f.value or {}).get("reaction")}
                          for f in allergies],
            "RULES": [{"key": r["key"], "rule": r.get("rule"), "food": r.get("food")} for r in rules],
            "FOODS": [{"i": i, "name": x.food.name if x.food else x.name, "ingredients": (x.food.ingredients or "")[:400] if x.food else "",
                       "allergens": list(x.food.allergens or []) if x.food else []} for i, x in foods]}, ensure_ascii=False)
        try:
            got = await lookup.ask("nutrition", CHECK_PROMPT, body, essential=self.who.essential, max_tokens=1200)
        except router.AllModelsFailed:
            return out
        keys = {r["key"] for r in rules}
        for row in (got.get("items") if isinstance(got, dict) else None) or []:
            if isinstance(row, dict) and isinstance(row.get("i"), int) and 0 <= row["i"] < len(items):
                f: dict = {}
                if row.get("allergy"):
                    f["allergy"] = [str(a)[:40] for a in row["allergy"]][:4]
                hits = [k for k in row.get("rules") or [] if k in keys]
                if hits:
                    f["rules"] = hits
                out[row["i"]] = f
        return out

    async def _photo_numbers(self, job: NutritionJob, reading: dict) -> str:
        """', about 215 kcal, 14 g protein' for a photo, once the foods are known. Empty when they are not."""
        items = []
        for it in reading.get("items") or []:
            if not it.get("name"):
                continue
            row = {"name": it["name"]}
            if it.get("quantity"):
                row["quantity"] = it["quantity"]
            if it.get("unit"):
                row["unit"] = it["unit"]
            if it.get("grams"):
                row["_grams"] = it["grams"]
            items.append(row)
        if not items:
            return ""
        try:
            found = await lookup.resolve(self.session, self.who.family_id,
                                         [{k: v for k, v in i.items() if not str(k).startswith("_")} for i in items],
                                         cache_only=True)
        except lookup.Slow:
            return ""
        sizes = (await settings.get(self.session, self.who.family_id, job.subject_id))["sizes"]
        tot: dict = {}
        for i, r in zip(items, found):
            if r.food is None:
                continue
            _g, vals, _e = entries.compute(r.food, i.get("quantity"), i.get("unit"), sizes=sizes, grams=i.get("_grams"))
            tot = N.add(tot, vals)
        phrase = N.highlight_phrase(tot, await targets.focus(self.session, self.who.family_id, job.subject_id))
        return f", about {phrase}" if phrase else ""

    async def _result(self, job: NutritionJob, rows: list[NutritionEntry], tracking: str, *, replaced: list[str], medicines: list) -> dict:
        s, who = self.session, self.who
        day = rows[0].day if rows else clock.ist_day()
        today_rows = await entries.on_day(s, who.family_id, job.subject_id, day)
        tot = entries.totals(today_rows)
        tgts, rules = await targets.current(s, who.family_id, job.subject_id)
        prog = targets.progress(tot, tgts)
        say = _say(rows, tracking, await targets.focus(s, who.family_id, job.subject_id))
        out: dict = {"result": "saved", "job_id": str(job.id), "meal": rows[0].meal if rows else None, "day": day,
                     "saved": [{"id": str(r.id), "name": r.name, "said": r.name_said, "qty": P.qty_label(r.quantity, r.unit),
                                "kcal": None if r.confidence == "pending" else N.kcal(r.nutrients),
                                "protein_g": None if r.confidence == "pending" else N.grams_of(r.nutrients, "protein_g"),
                                "confidence": r.confidence} for r in rows],
                     "say": say,
                     "totals_today": {k: round(v, 1) for k, v in tot.items() if k in N.MACROS or k == "water_ml"},
                     "undo": "To correct it: change_food with entry_id (or last=true) and set; to remove it: change_food delete=true."}
        if "tell me if" not in say:
            question = _curiosity(rows, tracking)
            if question:
                out["follow"] = question
                out["follow_how"] = FOLLOW_HOW
        line = targets.today_line(prog) if tracking == "full" else None
        if line and not all(r.confidence == "pending" for r in rows if r.kind == "food"):
            out["today"] = line
        if tracking == "counts":
            out["note"] = "They chose counts only: say what was saved without nutrient numbers."
        if any(r.confidence in ("estimate", "unquantified") for r in rows):
            out["estimated"] = [entries.line(r) for r in rows if r.confidence in ("estimate", "unquantified")]
        allergy = [(r, a) for r in rows for a in (r.flags or {}).get("allergy") or []]
        if allergy:
            r, a = allergy[0]
            out["allergy"] = (f"{r.name} may contain {a}, and {self._you(job)} {'have' if self._you(job) == 'you' else 'has'} a {a} allergy on record. "
                              "Mention it gently and ask how they feel; any symptom follows the red-flag rule.")
        devs = await self._deviations(job, rows, today_rows, tot, tgts, rules)
        if devs:
            out["deviation"] = devs[0] + " Point it out once, gently and as a plain fact (whose rule it is); no shaming, no advice of your own."
        check = self._check(rows)
        if check:
            out["check"] = check
        if replaced:
            out["replaced"] = f"This replaces what a caregiver logged for this {rows[0].meal}: " + ", ".join(replaced) + "."
        if medicines:
            out["not_food"] = f"{', '.join(medicines)}: medicines or supplements go to log_dose, not the food log."
        if await self._first_time(job):
            # no page to point at: the Food & nutrition page is switched on only after the founder approves it
            out["first_time"] = ("This is the first food you have kept for them with this person: say once, in a short line, that you now keep "
                                 "track of their food (what, how much, nutrition; they can ask you for totals or counts any time) and that they "
                                 "can tell you anytime to stop.")
        return out

    async def _deviations(self, job, rows, today_rows, tot, tgts, rules) -> list[str]:
        out = []
        by_key = {r["key"]: r for r in rules}
        hit_keys = {k for r in rows for k in (r.flags or {}).get("rules") or []}
        for k in hit_keys:
            rule = by_key.get(k)
            if not rule or rule.get("how") != "strict":
                continue
            count = sum((e.quantity or 1) if e.unit in (None, "piece") else 1 for e in today_rows if k in (e.flags or {}).get("rules", []))
            who = targets.WHO.get(rule.get("set_by") or "", "someone")
            if rule.get("none"):
                out.append(f"This goes against the rule '{rule.get('rule')}' set by {who}.")
            elif rule.get("max_per_day") and count > float(rule["max_per_day"]):
                out.append(f"That makes {N.fmt(count)} today, past the rule '{rule.get('rule')}' set by {who}.")
        caused: dict[str, list[str]] = {}  # entry id → the deviations it is part of (only those entries are marked)
        for r in rows:
            for k in (r.flags or {}).get("rules") or []:
                if k in by_key and by_key[k].get("how") == "strict":
                    caused.setdefault(str(r.id), []).extend(m for m in out if by_key[k].get("rule") and f"'{by_key[k].get('rule')}'" in m)
        for t in tgts:
            if t.get("how") != "strict":
                continue
            # a ceiling (sugar at most 25 g), a range's top, or a strict "about" (kcal about 1,600: past it by more than 10 %)
            if t.get("sense") in ("max", "range") and t.get("max"):
                limit = float(t["max"])
            elif t.get("sense") == "target" and targets.goal_of(t):
                limit = float(targets.goal_of(t) or 0) * 1.1
            else:
                continue
            so_far = float(tot.get(t["nutrient"]) or 0)
            before = so_far - sum(float((r.nutrients or {}).get(t["nutrient"]) or 0) for r in rows)
            if so_far > limit >= before:
                label, unit = N.LABELS.get(t["nutrient"], (t["field"], ""))
                shown = float(t["max"]) if t.get("sense") in ("max", "range") else float(targets.goal_of(t) or 0)
                msg = (f"{label} today is {N.fmt(so_far)} {unit}, past the {N.fmt(shown)} {unit} a day set by "
                       f"{targets.WHO.get(t.get('set_by') or '', 'someone')}.")
                out.append(msg)
                for r in rows:
                    if float((r.nutrients or {}).get(t["nutrient"]) or 0) > 0:
                        caused.setdefault(str(r.id), []).append(msg)
        if caused:
            for r in rows:
                if caused.get(str(r.id)):
                    r.flags = {**(r.flags or {}), "deviation": list(dict.fromkeys(caused[str(r.id)]))[:3]}
            await self.session.flush()
        return out

    def _check(self, rows: list[NutritionEntry]) -> str | None:
        """Plan §N4.1: one quick check only when a guessed portion could move the meal's kcal by more than 25%."""
        tot = N.kcal(entries.totals(rows))
        if tot <= 0:
            return None
        for r in rows:
            if r.confidence == "estimate" and N.kcal(r.nutrients) > 0.25 * tot and len(rows) > 1:
                return (f"The {r.name_said or r.name} amount was a usual portion ({P.qty_label(r.quantity, r.unit) or 'one serving'}); if it fits, ask in "
                        "a few words whether that was right (one quick check, not a quiz).")
        return None

    async def _first_time(self, job: NutritionJob) -> bool:
        if self.who.channel != "whatsapp":
            return False
        # once per person told: the elder when they log their own food, a caregiver the first time they log someone's
        ref = f"nutrition_intro:{job.subject_id}:{self.who.speaker_id}"
        got = await store.record_event(self.session, family_id=self.who.family_id, subject_id=job.subject_id, kind="nutrition_intro",
                                       summary="Saheli said she keeps track of food", ref=ref)
        return got is not None


# Entry points live in turn.py. Names stay here so callers and tests are unchanged.
expire = turn.expire
fill_missing = turn.fill_missing
handle = turn.handle
refine_saved = turn.refine_saved
run_background = turn.run_background
tell = turn.tell
