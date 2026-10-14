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
from app.nutrition import entries, jobs, lookup, pantry, photos, settings, targets, vision
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


ASK_FIELDS = ("eaten_at", "meal", "for_whom", "from_photo", "label_photo", "confidence", "confirmed", "unsure", "cancel", "consumed",
              "portion_eaten", "skipped", "from_order_task")
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
                job.draft = {k2: v2 for k2, v2 in (job.draft or {}).items() if k2 in ("photo", "photo_ref", "label_food")}
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


def _say(rows: list[NutritionEntry], tracking: str) -> str:
    food = [r for r in rows if r.kind == "food"]
    if not food:
        return "Noted: " + ", ".join(entries.line(r) for r in rows) + "."
    parts = [_saved_line(r) for r in food]
    what = ", ".join(_with_facts(text, r.flags or {}) for (text, _), r in zip(parts, food))
    tot = entries.totals(food)
    if tracking == "counts" or not tot.get("energy_kcal"):
        base = f"Saved: {what}."
    else:
        base = f"Saved: {what}, {N.fmt(N.kcal(tot))} kcal, {N.fmt(N.grams_of(tot, 'protein_g'))} g protein."
    if any(assumed for _, assumed in parts):
        return base[:-1] + " — tell me if it was more."
    return base


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
                return self._question(job, "consumed", f"Did {self._you(job)} eat the {dishes} that came? All of it?",
                                      ["All", "Half", "A little", "No / someone else"])
            if ask["consumed"] == "no":
                jobs.finish(job, "dropped", {"why": "not eaten", "order_task": str(task.id)})
                return {"result": "dropped", "job_id": str(job.id), "why": "Not eaten: nothing logged from this order."}
            if not items:
                items = order_items(task)
                ask["items"] = items
                job.ask = ask

        if ask.get("for_whom") == "":
            # One person in the household is not a guess. Asking them "who ate it" only repeats.
            people = [(i, n) for i, n in who.names.items() if n]
            if len(people) > 1:
                return self._question(job, "who_ate", "Who ate it?", [n for _, n in people][:5])
            if len(people) == 1:
                job.subject_id = people[0][0]
            ask = dict(ask)
            ask.pop("for_whom", None)
            job.ask = ask

        tracking = (await settings.get(s, who.family_id, job.subject_id))["tracking"]
        sizes = (await settings.get(s, who.family_id, job.subject_id))["sizes"]

        if ask.get("skipped") and not items:
            at = parse_at(ask.get("eaten_at")) or clock.now()
            meal = meal_of(ask, at)
            e = await entries.save_skipped(s, family_id=who.family_id, subject_id=job.subject_id, logged_by=who.speaker_id, meal=meal, eaten_at=at,
                                           source=self._source(job, ask), source_ref=job.source_ref, job_id=job.id)
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
                if jobs.can_ask(job):
                    return self._question(job, "label", "I couldn't read the label. Could you send a clear photo of the nutrition table on the pack?")
                ask["label_photo"] = False

        # a meal photo: read, then always confirmed first (founder Q4)
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
                elif jobs.can_ask(job):
                    return self._question(job, "photo", "I couldn't make out the food in the photo. What was it, and how much?")
                else:
                    jobs.finish(job, "dropped", {"why": "photo unreadable"})
                    return {"result": "dropped", "job_id": str(job.id), "why": "The photo could not be read and no food was named; nothing saved."}
            elif not ask.get("confirmed"):
                return self._question(job, "confirm", f"Looks like {vision.describe(draft['photo'])}. Right?", ["Yes", "Change"])
            elif not items or not any(i.get("quantity") for i in items):
                items = [{"name": i["name"], **({"quantity": i["quantity"]} if i.get("quantity") else {}), **({"unit": i["unit"]} if i.get("unit") else {}),
                          "_grams": i.get("grams"), "_photo": True} for i in draft["photo"]["items"]]

        if not items:
            jobs.finish(job, "dropped", {"why": "nothing named"})
            return {"result": "dropped", "job_id": str(job.id), "why": "No food was named; nothing saved."}

        # find each food
        stock = await pantry.in_stock(s, who.family_id, job.subject_id)
        try:
            found = await lookup.resolve(s, who.family_id, [{k: v for k, v in i.items() if not k.startswith("_")} for i in items],
                                         pantry_rows=stock, deadline=deadline, essential=who.essential, no_questions=not jobs.can_ask(job),
                                         estimates_only=estimates_only)
        except lookup.Slow:
            if background and not estimates_only:
                # the search did not fit: save an estimate now (under ~30 s) and refine the numbers after they are told
                return await self.run(job, deadline=time.monotonic() + 10, background=True, estimates_only=True)
            job.state, job.updated_at = "resolving", clock.now()
            return {"result": "looking_up", "job_id": str(job.id),
                    "say": "Tell them once, in a few words, that you are checking its nutrition and will tell them when it is saved "
                           "(no numbers yet). If you already said that, say nothing."}
        medicines = [r.item.get("name") for r in found if r.medicine]
        keep = [(i, r) for i, r in zip(items, found) if not r.medicine]
        if medicines and not keep:
            jobs.finish(job, "dropped", {"why": "medicine"})
            return {"result": "dropped", "job_id": str(job.id), "medicine": medicines,
                    "why": "These are medicines or supplements, not food: use log_dose for them (never the food log)."}
        for it, r in keep:
            if r.needs == "which" and jobs.can_ask(job):
                return self._question(job, "which", f"Which {it['name']}?", r.options, item=it["name"])
            if r.needs == "label" and jobs.can_ask(job):
                return self._question(job, "label", f"Which {it['name']} was it? Or send a photo of the label (the nutrition table on the back).",
                                      item=it["name"])
            if r.needs == "photo" and r.food is None and jobs.can_ask(job):
                return self._question(job, "photo", f"What's in the {it['name']}, roughly? Or send a photo.", item=it["name"])
        # how much: one pack or one drink is saved; a counted food with no number is asked once
        # (unless they don't know, or a photo showed it). A plate or a katori stays a usual-portion estimate.
        for it, r in keep:
            if it.get("quantity") or it.get("_grams") or ask.get("unsure") or r.food is None:
                continue
            if one_standard_serving(r.food):
                _fill_one(it, r.food)
                continue
            if r.food.countable and jobs.can_ask(job) and ask.get("confidence") != "photo":
                opts = await self._quantity_options(job.subject_id, r.food)
                return self._question(job, "quantity", f"How many {it['name']}?", opts, item=it["name"])

        # save
        at = parse_at(ask.get("eaten_at")) or clock.now()
        if at > clock.now() + timedelta(minutes=10):
            at = clock.now()
        if at < clock.now() - timedelta(days=30):
            jobs.finish(job, "dropped", {"why": "too old"})
            return {"result": "dropped", "job_id": str(job.id), "why": "That is more than 30 days ago; the food log keeps the last 30 days from chat."}
        meal = meal_of(ask, at)
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
            assumed = _assumed_flags(r.food) if it.get("_assumed") and r.food is not None else {}
            to_save.append(entries.Item(name=it.get("name") or (r.food.name if r.food else "food"), food=r.food, quantity=it.get("quantity"),
                                        unit=it.get("unit"), name_said=it.get("said") or it.get("name") or "", confidence=conf,
                                        grams=it.get("_grams"), pantry=r.pantry, note=it.get("notes"), flags=assumed))
        flags = await self._checks(job.subject_id, to_save)
        for x, f in zip(to_save, flags):
            x.flags.update(f)
        replaced = []
        if job.subject_id == who.speaker_id:
            replaced = await entries.replace_caregiver_estimate(s, who.family_id, job.subject_id, clock.ist_day(at), meal, by=who.speaker_id)
        consumed = ask.get("consumed") if task is not None else "yes"
        rows = await entries.save_meal(
            s, family_id=who.family_id, subject_id=job.subject_id, logged_by=who.speaker_id, items=to_save, meal=meal, eaten_at=at,
            source=self._source(job, ask, task=task), source_ref=str(task.id) if task is not None else job.source_ref, job_id=job.id,
            photo_ref=draft.get("photo_ref"), consumed=consumed or "yes", portion=ask.get("portion_eaten"), sizes=sizes)
        for x in to_save:
            if x.pantry is not None:
                await pantry.use(s, x.pantry, float(x.quantity or 1) if x.unit in (None, "piece", x.pantry.unit) else 1.0)
        out = await self._result(job, rows, tracking, replaced=replaced, medicines=medicines)
        if any((r.flags or {}).get("refine") for r in rows):
            out["refine"] = True
        jobs.finish(job, "saved", {"entries": [str(r.id) for r in rows], "meal_id": str(rows[0].meal_id) if rows else None,
                                   "say": out.get("say"), "today": out.get("today"), **({"order_task": str(task.id)} if task else {})})
        return out

    # ── helpers ──

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

    def _question(self, job: NutritionJob, reason: str, question: str, options: list[str] | None = None, item: str | None = None) -> dict:
        n = jobs.ask_needs(job, reason, question, options, item)
        return {"result": "needs_input", "job_id": str(job.id), "say": question, "needs": [{**n, "job_id": str(job.id)}],
                "how": ("Say only the say sentence, in their language and script. Do not mention options, ids, or the words Options or Reply, "
                        "and do not add a health warning of your own. Nothing is saved yet. When they answer, call log_food with "
                        f"answer_to=\"{job.id}\" and the answer as fields (the full items if they changed; unsure=true if they don't know "
                        "the amount; cancel=true if they say never mind). If this message already answers it, call log_food now and do not ask.")}

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

    async def _result(self, job: NutritionJob, rows: list[NutritionEntry], tracking: str, *, replaced: list[str], medicines: list) -> dict:
        s, who = self.session, self.who
        day = rows[0].day if rows else clock.ist_day()
        today_rows = await entries.on_day(s, who.family_id, job.subject_id, day)
        tot = entries.totals(today_rows)
        tgts, rules = await targets.current(s, who.family_id, job.subject_id)
        prog = targets.progress(tot, tgts)
        out: dict = {"result": "saved", "job_id": str(job.id), "meal": rows[0].meal if rows else None, "day": day,
                     "saved": [{"id": str(r.id), "name": r.name, "said": r.name_said, "qty": P.qty_label(r.quantity, r.unit), "kcal": N.kcal(r.nutrients),
                                "protein_g": N.grams_of(r.nutrients, "protein_g"), "confidence": r.confidence} for r in rows],
                     "say": _say(rows, tracking),
                     "totals_today": {k: round(v, 1) for k, v in tot.items() if k in N.MACROS or k == "water_ml"},
                     "undo": "To correct it: change_food with entry_id (or last=true) and set; to remove it: change_food delete=true."}
        line = targets.today_line(prog) if tracking == "full" else None
        if line:
            out["today"] = line
        if tracking == "counts":
            out["note"] = "They chose counts only: say what was saved without kcal or protein numbers."
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


# ── entry points ─────────────────────────────────────────────────────────────


def _attach(job: NutritionJob) -> dict:
    """A repeat log_food for a meal already pending or saved: no new job and no new entries."""
    if job.state in ("queued", "resolving"):
        return {"result": "looking_up", "already": True, "job_id": str(job.id), "say": "",
                "how": "This meal is already being saved. Say nothing. Do not say you are checking, and do not call log_food again for it."}
    if job.state == "needs_input":
        n = job.needs or {}
        return {"result": "needs_input", "already": True, "job_id": str(job.id), "needs": [n], "say": n.get("question") or "",
                "how": "This question is already waiting. Ask it only if you have not asked it yet. Pass the answer with answer_to."}
    return {"result": "saved", "already": True, "job_id": str(job.id), "say": "",
            "how": "Already in the food log. Say nothing about it."}


async def handle(session: AsyncSession, who: Who, args: dict, *, subject_id: str) -> dict:
    """log_food: a new meal, or an answer to the agent's question (answer_to)."""
    if args.get("answer_to"):
        job = await jobs.get(session, who.family_id, str(args["answer_to"]))
        if job is None or job.state not in jobs.LIVE:
            return {"result": "closed", "why": "That question is closed (saved, dropped or over 2 hours old). If they are telling you what "
                                               "they ate, call log_food again without answer_to."}
        merge(job, args)
        if args.get("for_whom"):
            job.subject_id = subject_id
            ask = dict(job.ask)
            ask.pop("for_whom", None)
            job.ask = ask
    else:
        ask = clean_ask(args)
        if ask.get("items") and not any(ask.get(k) for k in ("from_photo", "label_photo", "skipped", "from_order_task")):
            existing = await jobs.same_recent(session, who.family_id, subject_id, who.speaker_id, ask)
            if existing is not None:
                return _attach(existing)
        job = await jobs.create(session, family_id=who.family_id, subject_id=subject_id, requested_by=who.speaker_id, ask=ask,
                                source_ref=who.message_ref, channel=who.channel)
    out = await NutritionAgent(session, who).run(job, deadline=time.monotonic() + jobs.inline_seconds())
    await session.flush()
    if out.get("result") == "looking_up":
        jobs.kick_nutrition()
    return out


async def run_background(session: AsyncSession, job: NutritionJob) -> dict:
    """A slow look-up finishing after the turn (no photo bytes here: a photo was read in the turn)."""
    roster = await store.roster(session, job.family_id)
    names = {m.get("id"): m.get("name") for m in ([roster.elder, *roster.members] if roster else []) if m.get("id")}
    who = Who(family_id=job.family_id, speaker_id=job.requested_by, channel=job.channel or "whatsapp", essential=False, names=names)
    return await NutritionAgent(session, who).run(job, deadline=time.monotonic() + 20, background=True)


async def expire(session: AsyncSession, job: NutritionJob) -> dict:
    """2 hours without an answer (plan §N1.3): food they said they ate is kept as unquantified; an order's food, a photo
    nobody confirmed, or "who ate it?" saves nothing."""
    ask = job.ask or {}
    reason = (job.needs or {}).get("reason")
    items = [i for i in ask.get("items") or [] if i.get("name")]
    if reason in ("consumed", "who_ate", "confirm") or not items:
        if (job.draft or {}).get("photo_ref"):
            await photos.delete((job.draft or {})["photo_ref"])
        jobs.finish(job, "dropped", {"why": f"no answer ({reason})"})
        return {"result": "dropped"}
    at = parse_at(ask.get("eaten_at")) or job.created_at
    meal = meal_of(ask, at)
    rows = await entries.save_meal(
        session, family_id=job.family_id, subject_id=job.subject_id, logged_by=job.requested_by,
        items=[entries.Item(name=i["name"], food=None, quantity=i.get("quantity"), unit=i.get("unit"), name_said=i.get("said") or i["name"],
                            confidence="unquantified") for i in items],
        meal=meal, eaten_at=at, source="dashboard" if job.channel == "dashboard" else ("caregiver" if job.requested_by != job.subject_id else "chat"),
        source_ref=job.source_ref, job_id=job.id, reason="no answer in 2 hours: kept without numbers")
    jobs.finish(job, "saved", {"entries": [str(r.id) for r in rows], "unquantified": True})
    return {"result": "saved", "unquantified": True}


async def tell(sessions, host_for, job: NutritionJob, out: dict) -> None:
    """Saheli tells the person what a background look-up found (one system turn, like an order update)."""
    from app.brain.host import LiveHost, ShadowHost
    from app.brain.wake import system_turn

    host = host_for(job.family_id) if host_for else (ShadowHost() if job.family_id.startswith("shadow:") else LiveHost())
    if out.get("result") == "saved":
        what = f"{out.get('say')} {out.get('today') or ''}".strip()
        extra = " ".join(str(out[k]) for k in ("allergy", "deviation") if out.get(k))
        fallback = what
        prompt = (f"[Nutrition update] The food {job.requested_by} told you about is now saved: {what} {extra} Tell them in one short line with "
                  f"send_message (to {job.requested_by}), in their language, numbers exactly as given. Then reply none.")
    else:
        n = (out.get("needs") or [{}])[0]
        fallback = str(n.get("question") or "")
        prompt = (f"[Nutrition update] To save the food {job.requested_by} told you about, ask them only this, with send_message (to "
                  f"{job.requested_by}), in their language: \"{n.get('question')}\". Do not read out options or ids. When they answer, "
                  f"call log_food with answer_to=\"{job.id}\". Then reply none.")
    try:
        await system_turn(sessions, host, job.family_id, prompt, f"nutrition:{job.id}:{out.get('result')}", deliver_to=job.requested_by,
                          fallback=fallback)
    except Exception:  # noqa: BLE001
        logger.exception("nutrition update not told family=%s", job.family_id)


async def refine_saved(sessions, job_id) -> None:
    """After they have been told, replace an estimate with a web look-up when one comes back. No second message."""
    async with sessions() as session:
        rows = list((await session.execute(select(NutritionEntry).where(
            NutritionEntry.job_id == job_id, NutritionEntry.status == "active"))).scalars())
        changed = False
        for e in rows:
            if not (e.flags or {}).get("refine") or e.food_id is None:
                continue
            food = await session.get(NutritionFood, e.food_id)
            if food is None:
                continue
            try:
                got = await lookup.web({"name": e.name_said or e.name}, food.name, timeout_s=lookup.WEB_S)
            except lookup.Slow:
                continue
            if not got or not N.check_per_100g(got.get("per_100g") or {}).ok:
                continue
            food.per_100g = N.clean(got["per_100g"])
            food.source = str(got.get("source") or "web")[:12]
            food.source_urls = list(got.get("source_urls") or [])[:4]
            chk = N.check_per_100g(food.per_100g)
            food.checked = {**chk.to_dict(), "rough": False}
            food.confidence = 0.8
            food.updated_at = clock.now()
            grams, vals, _est = entries.compute(food, e.quantity, e.unit)
            if not vals:
                continue
            e.nutrients, e.grams = vals, grams
            flags = dict(e.flags or {})
            flags.pop("refine", None)
            flags.pop("rough", None)
            facts = N.fact_tags(food.per_100g)
            if facts:
                flags["facts"] = facts
            else:
                flags.pop("facts", None)
            e.flags = flags
            e.version = (e.version or 1) + 1
            e.updated_at = clock.now()
            await entries.record(session, e, "write", reason="web look-up replaced the first estimate")
            changed = True
        if changed:
            await session.commit()
