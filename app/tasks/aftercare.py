"""After an order is placed: did it come, and how was it (founder 2026-10-11).

After every order Saheli asks whether it came and was all right. When the person says it has not come and it is long past
the store's time, the caregiver is told. Once it came, she asks how it was in a short chat, one question at a time: did
they like it, is it something they usually have or a first try, should she get it again. The answers go into the care
memory and the product picker, so the next order picks what they like and never what they did not.

The steps live on the order's "delivery" open loop (detail.stage):
- arrival: woken a little after the store's delivery time, or moved on by the store's own status saying it was delivered.
  No answer → once more later, together with "how was it?".
- feedback: a while after it came (food sooner, groceries later): the first question. Each answer comes back through the
  order_feedback tool, whose result says what is still worth asking (from data: what was bought before, what is not known
  yet), at most MAX_QUESTIONS answers, then it closes.
Medicines and rides stop after the arrival step. Code only checks data (times, the store's status, earlier orders); the
brain reads people's words into the tool's fields.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.care.domains import fact_key, slug
from app.care.models import OpenLoop, Turn
from app.core import clock
from app.tasks.models import Task

logger = logging.getLogger(__name__)

FOOD = ("swiggy", "zomato")
PROMISE_DEFAULT = {"ride": 15, "food": 45, "grocery": 30, "medicine": 24 * 60}  # minutes, when the store gave no time
LATE_AFTER = {"ride": timedelta(minutes=20), "food": timedelta(minutes=30), "grocery": timedelta(minutes=30), "medicine": timedelta(hours=6)}
FEEDBACK_AFTER = {"food": timedelta(minutes=40), "grocery": timedelta(hours=3)}  # kinds with a "how was it?" step
MAX_QUESTIONS = 3  # answers in the "how was it?" chat
KEEP_ARRIVAL = timedelta(days=2)  # a follow-up nobody answers closes by itself
KEEP_FEEDBACK = timedelta(days=1)
TALKS_WITHIN = timedelta(days=30)  # the person an order is for is asked only if they talk to Saheli
NEXT = {"arrival": "confirm the order arrived", "feedback": "say how they liked the order"}
VALID = {"liked": ("yes", "no", "mixed"), "usual": ("usual", "new"), "again": ("yes", "no")}


def kind_of(task: Task) -> str:
    """ride | medicine | food | grocery."""
    if task.kind == "ride":
        return "ride"
    from app.specialists.agents import specialist_for

    try:
        if specialist_for(task.service).name == "pharmacy":
            return "medicine"
    except ValueError:
        pass
    return "food" if task.service in FOOD else "grocery"


def items_of(task: Task) -> list[dict]:
    """What was ordered, as the store's cart showed it, numbered for the brain: [{n, name, qty, restaurant}]."""
    from app.tasks.runtime import _label

    r, d = task.result or {}, task.details or {}
    rows = ([i for i in r.get("items") or [] if isinstance(i, dict) and i.get("name")]
            or [i for i in d.get("chosen") or [] if isinstance(i, dict) and i.get("name")]
            or [i for i in d.get("items") or [] if isinstance(i, dict) and i.get("name")])
    restaurant = next((str(c["restaurant"]) for c in d.get("chosen") or [] if isinstance(c, dict) and c.get("restaurant")), None)
    return [{"n": n, "name": _label(i), "product": str(i.get("name")), "qty": i.get("qty") or 1,
             "restaurant": i.get("restaurant") or restaurant} for n, i in enumerate(rows[:8], 1)]


def _numbered(task: Task) -> str:
    rows = items_of(task)
    return "; ".join(f"{i['n']}. {i['name']}" + (f" x{i['qty']}" if str(i["qty"]) not in ("1", "") else "") for i in rows) or task.goal


def promised_minutes(task: Task) -> int:
    """The store's own delivery time from when it was placed ("12 mins", "10-15 min", "1 hr"), else a usual one per kind."""
    eta = str((task.result or {}).get("eta") or "").lower()
    m = re.search(r"(\d+)\s*(?:-|to)?\s*(\d+)?\s*(min|mins|minute|minutes|hr|hrs|hour|hours)\b", eta)
    if m:
        n = int(m.group(2) or m.group(1))
        return n * 60 if m.group(3).startswith("h") else n
    return PROMISE_DEFAULT[kind_of(task)]


def times(task: Task, detail: dict | None = None) -> tuple[datetime, datetime, datetime]:
    """(placed, should have come by, long overdue from)."""
    d = detail or {}
    tr = (task.details or {}).get("tracking") or {}
    placed_text = (task.result or {}).get("placed_at") or d.get("placed_at") or tr.get("placed_at")
    placed = datetime.fromisoformat(placed_text) if placed_text else task.updated_at
    promised = placed + timedelta(minutes=promised_minutes(task))
    return placed, promised, promised + LATE_AFTER[kind_of(task)]


def _hm(at: datetime) -> str:
    local, today = clock.ist(at), clock.ist()
    if local.date() == today.date():
        return local.strftime("%H:%M")
    return local.strftime("%d %b %H:%M")


def store_says(task: Task) -> str:
    """What the store's own status last said about this order (delivery tracking), in one sentence."""
    r, tr = task.result or {}, (task.details or {}).get("tracking") or {}
    if r.get("delivered"):
        at = f" at {_hm(datetime.fromisoformat(r['tracked_at']))}" if r.get("tracked_at") else ""
        return f"The store says it was delivered{at}."
    when = f" ({_hm(datetime.fromisoformat(tr['last_at']))})" if tr.get("last_at") else ""
    if tr.get("state") == "on_the_way":
        return f"The store's last status{when}: on the way" + (f", about {r['eta_now']} left." if r.get("eta_now") else ".")
    if tr.get("state") == "preparing":
        return f"The store's last status{when}: being packed; nobody is on the way yet."
    if tr:
        return "The store has not said where it is."
    return "This store gives no live status."


async def who_to_ask(session: AsyncSession, task: Task) -> str:
    """The person the order is for, if they talk to Saheli; else whoever asked for it."""
    if task.subject_id == task.requested_by:
        return task.subject_id
    talks = await session.scalar(select(func.count()).select_from(Turn).where(
        Turn.family_id == task.family_id, Turn.thread_id == task.subject_id, Turn.role == "user", Turn.at >= clock.now() - TALKS_WITHIN))
    return task.subject_id if talks else task.requested_by


def title(task: Task, stage: str) -> str:
    from app.tasks.skills import SKILLS

    label = SKILLS[task.service]["label"]
    what = "the cab came" if task.kind == "ride" else ("the order arrived" if stage == "arrival" else "how they liked the order")
    return f"Check {what}: {label}: {_numbered(task)} [task {task.id}]"[:300]


async def loop_for(session: AsyncSession, task: Task) -> OpenLoop | None:
    return (await session.execute(select(OpenLoop).where(
        OpenLoop.family_id == task.family_id, OpenLoop.dedupe_key == f"delivery:{task.id}", OpenLoop.status == "open"))).scalar_one_or_none()


async def open_followup(session: AsyncSession, task: Task) -> OpenLoop:
    """When an order or ride is placed: one follow-up that asks, a little after the store's time, whether it came."""
    from app.tasks.runtime import _eta_wait

    now, kind, r = clock.now(), kind_of(task), task.result or {}
    task.result = {**r, "placed_at": r.get("placed_at") or now.isoformat()}
    detail = {"stage": "arrival", "task_id": str(task.id), "placed_at": now.isoformat(), "next_action": NEXT["arrival"],
              "order_id": r.get("order_id") or r.get("ride_id"),
              # no answer to "has it come?": once more later, together with "how was it?" (kinds with that step)
              "max_wakes": 2 if kind in FEEDBACK_AFTER else 1,
              "rewake_minutes": int(FEEDBACK_AFTER.get(kind, timedelta(hours=1)).total_seconds() // 60),
              "expire_at": (now + KEEP_ARRIVAL).isoformat()}
    return await store.open_loop(
        session, family_id=task.family_id, subject_id=task.subject_id, kind="delivery", title=title(task, "arrival"),
        owner_id=await who_to_ask(session, task), wake_at=now + _eta_wait(task), alert_rule="dashboard",
        dedupe_key=f"delivery:{task.id}", detail=detail,
    )


def _to_feedback(loop: OpenLoop, task: Task, at: datetime, *, chatting: bool = False) -> None:
    """It came: next, how was it (now if they are already talking about it, else a while later)."""
    wait = FEEDBACK_AFTER[kind_of(task)]
    loop.detail = {**(loop.detail or {}), "stage": "feedback", "wakes": 0, "max_wakes": 1, "next_action": NEXT["feedback"],
                   "feedback_from": at.isoformat(), "expire_at": (at + wait + KEEP_FEEDBACK).isoformat()}
    loop.title = title(task, "feedback")
    loop.wake_at = None if chatting else at + wait
    loop.updated_at = clock.now()


async def store_delivered(session: AsyncSession, task: Task, at: datetime) -> bool:
    """The store says it was delivered. True when the update to the person who asked should also ask whether they got
    it all right (it is for them); otherwise the person it is for is asked later, with "how was it?"."""
    loop = await loop_for(session, task)
    if not loop:
        return False
    asks_now = loop.owner_id == task.requested_by and (loop.detail or {}).get("stage") != "feedback"
    loop.detail = {**(loop.detail or {}), "store_delivered_at": at.isoformat()}
    if (loop.detail or {}).get("stage") == "feedback":
        return False  # they already said it came
    if kind_of(task) in FEEDBACK_AFTER:
        _to_feedback(loop, task, at)
    else:
        loop.wake_at = None if asks_now else at + timedelta(minutes=15)
    return asks_now


async def bought_before(session: AsyncSession, task: Task) -> dict[str, int]:
    """How often each product of this order was ordered before through Saheli (placed orders, last 6 months)."""
    rows = (await session.execute(select(Task).where(
        Task.family_id == task.family_id, Task.kind == "order", Task.status == "done", Task.id != task.id,
        Task.created_at < task.created_at, Task.created_at >= task.created_at - timedelta(days=180)))).scalars()
    counts: dict[str, int] = {}
    for t in rows:
        if not (t.result or {}).get("placed"):
            continue
        for k in {slug(i["product"]) for i in items_of(t)}:
            counts[k] = counts.get(k, 0) + 1
    return counts


def next_question(task: Task, fb: dict, bought: dict[str, int]) -> str | None:
    """What is still worth asking about this order, from what is known (None: nothing, or enough asked)."""
    if int(fb.get("answers") or 0) >= MAX_QUESTIONS:
        return None
    rows, state = items_of(task), fb.get("items") or {}
    s = lambda i: state.get(str(i["n"])) or {}  # noqa: E731
    unknown = [i for i in rows if not s(i).get("liked") and s(i).get("usual") != "usual"]  # a usual buy they keep getting
    if unknown:
        names = ", ".join(i["name"] for i in unknown[:3])
        return f"how {names} {'was' if len(unknown) == 1 else 'were'} and whether they liked it."
    for i in rows:
        if s(i).get("liked") in ("no", "mixed") and not s(i).get("again"):
            return f"kindly, what was not right about {i['name']}, and whether to get something else next time instead."
    for i in rows:
        if s(i).get("liked") == "yes" and not s(i).get("usual") and not bought.get(slug(i["product"])):
            return f"whether {i['name']} is something they usually have, or something new they tried."
    restaurant = next((i["restaurant"] for i in rows if i.get("restaurant")), None)
    if kind_of(task) == "food" and restaurant and not (fb.get("order") or {}).get("again"):
        return f"whether they would like food from {restaurant} again."
    for i in rows:
        if s(i).get("liked") == "yes" and s(i).get("usual") == "new" and not s(i).get("again"):
            return f"whether to get {i['name']} again next time."
    return None


def _known(task: Task, bought: dict[str, int]) -> str:
    return "; ".join(f"{i['name']} was ordered {bought[slug(i['product'])]} times before (a usual buy)"
                     for i in items_of(task) if bought.get(slug(i["product"])))


async def _names(session: AsyncSession, family_id: str) -> dict[str, str]:
    roster = await store.roster(session, family_id)
    return {m.get("id"): (m.get("name") or "").split(" ")[0] for m in ([roster.elder, *roster.members] if roster else [])}


async def wake_prompt(session: AsyncSession, loop: OpenLoop) -> str | None:
    """The scheduled turn for an order's follow-up; None when nothing is left to ask (the loop is then closed)."""
    from app.tasks.skills import SKILLS

    d = loop.detail or {}
    try:
        task = await session.get(Task, uuid.UUID(str(d.get("task_id"))))
    except ValueError:
        task = None
    if not task or task.status == "cancelled" or (task.result or {}).get("store_cancelled"):
        return None
    r, kind, owner = task.result or {}, kind_of(task), loop.owner_id or task.subject_id
    fb = r.get("feedback") or {}
    label = SKILLS[task.service]["label"]
    names = await _names(session, task.family_id)
    for_whom = "" if task.requested_by == owner else f" ({names.get(task.requested_by) or 'the family'} ordered it for them)"
    placed, promised, _ = times(task, d)
    if d.get("stage") == "feedback":
        if fb.get("answers"):
            return None  # they already talked about it
        bought = await bought_before(session, task)
        first = next_question(task, fb, bought)
        if not first:
            return None
        came = datetime.fromisoformat(d.get("feedback_from") or d.get("store_delivered_at") or placed.isoformat())
        check = ("" if fb.get("arrived") in ("yes", "partly") else
                 " The store said it was delivered, but they have not said they got it: make sure of that in the same message.")
        known = _known(task, bought)
        return (f"[Order feedback] The {label} order for person {owner}{for_whom} ({_numbered(task)}) came at {_hm(came)}.{check} "
                f"Learn what they think of it, for next time: send person {owner} one short, warm question with send_message: {first} "
                f"It starts a short chat: whenever they answer, save it with order_feedback (task_id {task.id}; items by their "
                "number) and ask only what its result suggests, one question per message, while they are happy to talk."
                + (f" Known already, do not ask: {known}." if known else "") + " Reply none.")
    if fb.get("arrived") in ("yes", "partly"):
        return None
    wakes = int(d.get("wakes") or 0)
    total = f", {r['total']}" if r.get("total") else ""
    if wakes == 0:
        q = {"ride": "did the cab come, and was the ride all right?",
             "medicine": "has it come, and is it the right medicine and strength?"}.get(kind, "has it come all right?")
        return (f"[Order follow-up] The {label} {'ride' if kind == 'ride' else 'order'} for person {owner}{for_whom}: {_numbered(task)}"
                f"{total}, placed at {_hm(placed)}; the store said it would come by about {_hm(promised)}. {store_says(task)} "
                f"Ask person {owner} one short question in their language with send_message: {q} When they answer, call "
                f"order_feedback with task_id {task.id} (arrived yes / no / partly, and any problem in their words). Reply none.")
    if kind not in FEEDBACK_AFTER:
        return None
    # they did not answer "has it come?": once, together with how it was
    first = next_question(task, fb, await bought_before(session, task)) or "how it was."
    return (f"[Order follow-up] The {label} order for person {owner}{for_whom} ({_numbered(task)}), placed at {_hm(placed)}: they "
            f"did not answer whether it came. {store_says(task)} Send person {owner} one short, warm message with send_message "
            f"asking whether it came all right and {first} When they answer, save it with order_feedback (task_id {task.id}; "
            "arrived; items by their number) and ask only what its result suggests. Reply none.")


def _sentence(name: str, s: dict, label: str) -> str:
    bits = [{"yes": f"Likes {name}", "no": f"Did not like {name}", "mixed": f"Mixed about {name}"}.get(s.get("liked") or "", f"About {name}")]
    if s.get("usual") == "usual":
        bits.append("usually has it")
    elif s.get("usual") == "new":
        bits.append(f"first tried it in {clock.ist().strftime('%b %Y')} ({label})")
    if s.get("again") == "yes":
        bits.append("get it again")
    elif s.get("again") == "no":
        bits.append("do not order it again")
    return "; ".join(bits) + (f" ({s['note']})" if s.get("note") else "")


async def _remember(session: AsyncSession, task: Task, item: dict, s: dict, *, by: str, source_kind: str, confidence: float,
                    ref: str | None) -> str:
    """One product's feedback into the care memory: a preference (and never-order when they do not want it again)."""
    from app.tasks.skills import SKILLS

    label = SKILLS[task.service]["label"]
    text = _sentence(item["name"], s, label)
    value = {"item": item["product"], **{k: s[k] for k in ("liked", "usual", "again", "note") if s.get(k)}, "store": label,
             "restaurant": item.get("restaurant"), "task_id": str(task.id), "said_on": clock.ist_day()}
    await store.write_fact(session, family_id=task.family_id, subject_id=task.subject_id, domain="preference",
                           key=fact_key("preference", item["product"]), value={k: v for k, v in value.items() if v},
                           text=text, source_kind=source_kind, source_ref=ref, stated_by=by, confidence=confidence)
    if s.get("again") == "no" and s.get("liked") in ("no", "mixed"):
        why = s.get("note") or "did not like it"
        # the store's own product name: the cart check stops this product (not the whole kind) from being bought again
        await store.write_fact(session, family_id=task.family_id, subject_id=task.subject_id, domain="no_order",
                               key=fact_key("no_order", item["product"]), value={"item": item["product"], "why": why},
                               text=f"Never order {item['product']} again: {why}", source_kind=source_kind, source_ref=ref,
                               stated_by=by, confidence=confidence)
    return text


async def record(session: AsyncSession, task: Task, *, by: str, source_kind: str, confidence: float, arrived: str | None = None,
                 problem: str | None = None, items: list[dict] | None = None, order_again: str | None = None,
                 note: str | None = None, done: bool = False, ref: str | None = None) -> dict:
    """What the person said about a placed order (the order_feedback tool): saved on the order and in the care memory.
    Returns what to tell them and what, if anything, to ask next."""
    from app.tasks.runtime import _carers
    from app.tasks.runtime import note as task_note
    from app.tasks.skills import SKILLS

    now, kind, label = clock.now(), kind_of(task), SKILLS[task.service]["label"]
    r = dict(task.result or {})
    fb = dict(r.get("feedback") or {})
    loop = await loop_for(session, task)
    d = dict(loop.detail or {}) if loop else {}
    _, promised, overdue = times(task, d)
    out: dict = {"saved": []}
    if arrived in ("yes", "no", "partly"):
        fb["arrived"] = arrived
        if arrived != "no":
            fb.setdefault("arrived_at", now.isoformat())
            if not problem:
                out["saved"].append("it came" if arrived == "yes" else "it came, partly")
    if problem:
        fb["problem"] = str(problem)[:300]

    # Has it come? Not yet: tell them what the store says; long overdue (or the store says delivered): the caregiver.
    caregiver = None
    if arrived == "no":
        if r.get("delivered"):
            caregiver = ("late", "the store says it was delivered, but they say it has not come")
        elif now >= overdue:
            caregiver = ("late", f"it has not come; it was due by {_hm(promised)}, {int((now - promised).total_seconds() // 60)} minutes ago")
        else:
            if loop:  # ask again when it would be late
                loop.detail = {**d, "stage": "arrival", "wakes": 0, "max_wakes": 1}
                loop.wake_at = max(overdue, now + timedelta(minutes=10))
            out["tell"] = (f"It is not late yet. {store_says(task)} It should come by about {_hm(promised)}. Tell them that plainly in "
                           f"one short line; you will check with them again at {_hm(max(overdue, now + timedelta(minutes=10)))} (already "
                           "set: open no other loop). Do not tell the caregiver yet.")
    elif arrived == "partly" or problem:
        caregiver = ("problem", f"it came with a problem: {problem or 'something was missing or wrong'}")
    if caregiver:
        ac = dict((task.details or {}).get("aftercare") or {})
        carers = [c for c in await _carers(session, task) if c not in (task.subject_id, by)]
        names = await _names(session, task.family_id)
        who = ", ".join(names.get(c) or "the family" for c in carers)
        if any(t.get("kind") == caregiver[0] for t in ac.get("told") or []) or (ac.get("caregiver_due") or {}).get("kind") == caregiver[0]:
            out["tell"] = f"{who or 'The family'} already knows about this. Tell them kindly you are on it."
        elif carers:
            ac["caregiver_due"] = {"kind": caregiver[0], "why": caregiver[1], "said_by": by, "at": now.isoformat()}
            task.details = {**(task.details or {}), "aftercare": ac}
            out["tell"] = (f"{who} will be told now (Saheli sends it; do not message them yourself). Tell them in one short line that "
                           f"you have told {who}" + (" and that you will keep checking." if caregiver[0] == "late" else "."))
        else:
            out["tell"] = (f"There is no one else in the family to tell. {store_says(task)} Suggest kindly that they check the {label} "
                           "app or call the store; you will check with them again later.")
        if caregiver[0] == "late" and loop:
            loop.detail = {**d, "stage": "arrival", "wakes": 0, "max_wakes": 1}
            loop.wake_at = now + timedelta(hours=1)
        out["saved"].append(f"not delivered: {caregiver[1]}" if caregiver[0] == "late" else f"problem: {problem or 'missing or wrong items'}")

    # How was it: each product's answer, saved on the order and in the care memory.
    rows = {i["n"]: i for i in items_of(task)}
    answered = False
    for it in items or []:
        n = it.get("item") if it.get("item") in rows else (1 if len(rows) == 1 else None)
        if n is None:
            continue
        cur = dict((fb.get("items") or {}).get(str(n)) or {})
        cur.update({k: it[k] for k in VALID if it.get(k) in VALID[k]})
        if it.get("note"):
            cur["note"] = str(it["note"])[:200]
        if not any(cur.get(k) for k in ("liked", "usual", "again", "note")):
            continue
        fb["items"] = {**(fb.get("items") or {}), str(n): cur}
        answered = True
        out["saved"].append(await _remember(session, task, rows[n], cur, by=by, source_kind=source_kind, confidence=confidence, ref=ref))
    if order_again in ("yes", "no") or note:
        fb["order"] = {**(fb.get("order") or {}), **({"again": order_again} if order_again in ("yes", "no") else {}),
                       **({"note": str(note)[:200]} if note else {})}
        answered = True
        restaurant = next((i["restaurant"] for i in rows.values() if i.get("restaurant")), None)
        if restaurant and order_again in ("yes", "no"):
            text = (f"Likes food from {restaurant} ({label})" if order_again == "yes" else
                    f"Does not want food from {restaurant} ({label}) again") + (f" ({note})" if note else "")
            await store.write_fact(session, family_id=task.family_id, subject_id=task.subject_id, domain="preference",
                                   key=fact_key("preference", f"restaurant {restaurant}"),
                                   value={"restaurant": restaurant, "store": label, "again": order_again, "note": note, "task_id": str(task.id)},
                                   text=text, source_kind=source_kind, source_ref=ref, stated_by=by, confidence=confidence)
            out["saved"].append(text)
        elif note:
            out["saved"].append(f"about the order: {note}")
    if answered:
        fb["answers"] = int(fb.get("answers") or 0) + 1
        fb.setdefault("arrived", "yes")  # they are talking about how it was: it came

    # Where the follow-up goes next. "How was it?" is asked now only once that part has begun (its question went out, or
    # they are already saying how it was); right after it arrives they have not tried it yet.
    wakes = int(d.get("wakes") or 0)
    talking = answered or (wakes >= 1 if d.get("stage") == "feedback" else wakes >= 2)
    if loop and fb.get("arrived") in ("yes", "partly") and d.get("stage") != "feedback":
        if kind in FEEDBACK_AFTER:
            _to_feedback(loop, task, now, chatting=talking)
        else:
            await store.close_loop(session, loop.id, note="it came")
            loop = None
    if kind in FEEDBACK_AFTER and fb.get("arrived") in ("yes", "partly"):
        if not talking:
            if not caregiver:
                out["next"] = ("Say you are glad in a few words. Do not ask how it was now: they have not tried it yet; you will ask "
                               "later by yourself.")
        else:
            q = None if done else next_question(task, fb, await bought_before(session, task))
            if q:
                out["next"] = (f"Ask this next, in one short question, only if they seem happy to talk (otherwise just thank them in a "
                               f"few words): {q} Whatever they add about it later also goes into order_feedback.")
            else:
                out["next"] = "That is all: thank them warmly in a few words; ask nothing more."
                if loop:
                    await store.close_loop(session, loop.id, note="feedback done")
    elif fb.get("arrived") in ("yes", "partly") and not caregiver:
        out["next"] = "Say you are glad in a few words; ask nothing more."

    r["feedback"] = fb
    task.result = r
    if out["saved"]:
        task_note(task, "feedback: " + "; ".join(out["saved"])[:250])
        await store.record_event(session, family_id=task.family_id, subject_id=task.subject_id, kind="order_feedback",
                                 summary=f"{label}: " + "; ".join(out["saved"])[:300], actor_id=by,
                                 payload={"task_id": str(task.id), "feedback": fb})
    return out


async def sweep(sessions: async_sessionmaker, notify, *, limit: int = 10) -> int:
    """Send the caregiver messages order_feedback asked for: an order long overdue, or one that came wrong."""
    from app.tasks.runtime import _carers
    from app.tasks.runtime import note as task_note
    from app.tasks.skills import SKILLS

    now = clock.now()
    async with sessions() as session:
        rows = (await session.execute(select(Task).where(
            Task.kind.in_(("order", "ride")), Task.status == "done", Task.updated_at >= now - KEEP_ARRIVAL - timedelta(days=1))
            .order_by(Task.updated_at.desc()).limit(300))).scalars()
        due = [t.id for t in rows if ((t.details or {}).get("aftercare") or {}).get("caregiver_due")][:limit]
    sent = 0
    for task_id in due:
        sends: list[tuple[str, str]] = []
        async with sessions() as session:
            task = (await session.execute(select(Task).where(Task.id == task_id).with_for_update(skip_locked=True))).scalar_one_or_none()
            ac = dict(((task.details or {}).get("aftercare") or {}) if task else {})
            want = ac.pop("caregiver_due", None)
            if not task or not want:
                continue
            carers = [c for c in await _carers(session, task) if c not in (task.subject_id, want.get("said_by"))]
            label, r = SKILLS[task.service]["label"], task.result or {}
            placed, _, _ = times(task)
            order = f"{_numbered(task)}" + (f", {r['total']}" if r.get("total") else "") + ", cash on delivery" + (
                f", order id {r['order_id']}" if r.get("order_id") else "")
            for c in carers:
                sends.append((c, (
                    f"[Order needs you] The {label} order for person {task.subject_id} ({order}), placed at {_hm(placed)}: {want['why']}. "
                    f"Person {want.get('said_by')} told Saheli at {_hm(datetime.fromisoformat(want['at']))}. {store_says(task)} Tell person "
                    f"{c} in one or two short lines, with what was ordered, and suggest they check the {label} app or call the store; "
                    f"Saheli keeps checking with person {task.subject_id}.")))
            ac["told"] = [*(ac.get("told") or []), {"kind": want["kind"], "why": want["why"], "to": carers, "at": now.isoformat()}]
            task.details = {**(task.details or {}), "aftercare": ac}
            task_note(task, f"caregiver told: {want['why']}"[:200])
            await store.record_event(session, family_id=task.family_id, subject_id=task.subject_id, kind="order_caregiver_told",
                                     summary=f"{label}: {want['why']}"[:300], payload={"task_id": str(task.id), "to": carers})
            await session.commit()
        for person, text in sends:
            try:
                await notify(task.family_id, person, text)
                sent += 1
            except Exception:  # noqa: BLE001 — one failed message must not stop the rest
                logger.exception("aftercare caregiver message failed task=%s", task_id)
    return sent
