"""Task runtime: carries each order or ride through (browse →) prepare → confirm → place → done, durably.

A browser order first looks without logging in (browse): is it there, at what price, does it reach the family's
place. Only after the person's go-ahead does it log in (a code may come to their phone) and build the cart.

Every minute a tick advances running tasks: it polls the browser agent, reads its structured report,
and moves the task on. Anything the family must know or decide (an OTP, the cart to confirm, the
order placed, a failure) becomes a system turn for the brain, which tells the person in its own words.
Nothing is placed without an explicit confirm, payment is cash on delivery only, and a cancel is
honoured at every step, including after placing (a cancel run on the service).
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.core import clock, trace
from app.specialists import channels, guard, metrics
from app.specialists.agents import BEFORE_CONFIRM_RULE, specialist_for
from app.specialists.contract import CONTRACT_VERSION, Limits
from app.tasks.browser_use import AgentRun, BrowserAgent
from app.tasks import fastpath, later, practice, sandbox, tracking
from app.tasks.models import SkillNote, Task, TaskTold
from app.tasks.skills import SKILLS

logger = logging.getLogger(__name__)

LIVE = ("queued", "running", "needs_input", "awaiting_confirm")
TASK_LIFETIME = timedelta(minutes=45)
RUN_TIMEOUT = timedelta(minutes=12)
CHECK_TIMEOUT = timedelta(minutes=5)  # reading the store's own list is short (live 2026-10-13: one ran 11 minutes)

Notify = Callable[[str, str, str], Awaitable[None]]  # (family_id, requested_by, prompt)


rupees = guard.rupees
IDLE_CLOSE = timedelta(minutes=5)  # a browser waiting on a person longer than this is stopped (login stays in the profile)


def browse_first(task: Task) -> bool:
    """Look before logging in: browser orders that have not looked yet (TASK_BROWSE_FIRST=off turns it off; a
    comparison always looks first, it never logs in on every store)."""
    return ((os.getenv("TASK_BROWSE_FIRST", "on") == "on" or compare_group(task)) and task.kind in ("order", "ride")
            and task.phase == "prepare" and not (task.details or {}).get("browsed"))


# No store named: look on the usual stores for that kind of thing at once and offer every option (founder 2026-10-09:
# "it should go for other options also like Instamart and Zepto, and give available options as we had earlier").
COMPARE_STORES = {"grocery": ("blinkit", "instamart", "zepto"), "medicine": ("apollo", "1mg", "pharmeasy"), "food": ("swiggy", "zomato"),
                  # a ride with no app named, or "the cheapest": every ride app's fares (founder 2026-10-11: fares are the options)
                  "ride": ("uber", "ola", "rapido")}
# After the first store answers, a slower one gets this long; then every store is reported, a store still looking as too
# slow (founder 2026-10-11: never wait silently on a slow store; report the results from every store they named, so the
# stores they named themselves get longer; the "still looking" line goes out meanwhile).
COMPARE_WAIT = timedelta(seconds=int(os.getenv("COMPARE_WAIT_S", "90")))
COMPARE_WAIT_NAMED = timedelta(seconds=int(os.getenv("COMPARE_WAIT_NAMED_S", "180")))
# Founder 2026-10-11 (night): options at all times. What a look-up found is offered for the person to pick, never picked
# for them: per asked item, this many products from one store, or per store in a comparison.
OPTIONS_PER_ITEM, OPTIONS_PER_STORE = 4, 3
# A comparison nobody has heard about this long after it started: one "still looking" line (live 2026-10-11: 6 minutes of
# silence; the one-line rule below skipped comparisons until they were decided).
COMPARE_STILL_WORKING = timedelta(seconds=int(os.getenv("COMPARE_STILL_WORKING_S", "120")))


def compare_group(task: Task) -> str | None:
    return (task.details or {}).get("compare")


async def siblings(session: AsyncSession, task: Task) -> list[Task]:
    """Every task of the same comparison, this one included, as the database has them now (a turn's session may hold
    copies loaded before a tick changed them; pending changes are flushed first)."""
    gid = compare_group(task)
    if not gid:
        return [task]
    rows = (await session.execute(
        select(Task).where(Task.family_id == task.family_id, Task.created_at >= task.created_at - timedelta(minutes=5),
                           Task.created_at <= task.created_at + timedelta(minutes=5)).execution_options(populate_existing=True)
    )).scalars()
    return [t for t in rows if compare_group(t) == gid]


def _drop_others(group: list[Task], keep: Task, why: str, *, standby: bool = False) -> None:
    """Stop the other stores of a comparison. standby: the person picked `keep` from the options, so the others keep their
    look-up and are offered again if `keep` cannot build the cart (_reoffer_others)."""
    for t in group:
        if t.id != keep.id and t.status in LIVE:
            if standby and t.status in ("awaiting_confirm", "needs_input") and t.input_needed in ("go", "swap"):
                t.details = {**(t.details or {}), "dropped_for": str(keep.id), "dropped_input": t.input_needed}
            t.status, t.cancel_requested, t.input_needed = "cancelled", True, None
            note(t, why)


def _code_to(task: Task) -> str:
    login = (task.details or {}).get("login")
    return f"the phone ending {login}" if login and login != "none" else "the family phone"


def _code_line(task: Task) -> str:
    """What to say about a login code before one is needed: from the family's saved login (founder 2026-10-12: a code was
    promised while the login was still alive); a code is asked for only when the store actually sends one."""
    label = SKILLS[task.service]["label"]
    if (task.details or {}).get("login_live"):
        return f"the family's {label} login is still active, so no login code should be needed"
    return f"if {label} sends a login code, it goes to {_code_to(task)} and is asked for only then"


def _found(task: Task) -> list[dict]:
    return [i for i in ((task.result or {}).get("items") or []) if i.get("name") and i.get("available") is not False]


def _label(item: dict) -> str:
    """'Diet Coke Can (300 ml)': the name with its pack when the store gave it separately."""
    name, pack = str(item.get("name") or ""), str(item.get("pack") or "").strip()
    return f"{name} ({pack})" if pack and pack.lower() not in name.lower() else name


def _option(item: dict, asked: str = "") -> str:
    similar = ("" if item.get("exact_match") is not False else
               f" [similar: not exactly {asked}]" if asked else " [similar, not exactly what was asked]")
    where = f" from {item['restaurant']}" + (f" ({item['eta']})" if item.get("eta") else "") if item.get("restaurant") else ""
    rx = " [prescription needed]" if item.get("rx_required") else ""
    return f"{_label(item)} {item.get('price') or ''}".strip() + where + rx + similar


def _per_item(task: Task, *, compare: bool) -> int:
    """How many products are offered per asked item: fewer when several items (or several stores) share one message."""
    n = len((task.details or {}).get("items") or []) or 1
    return max(1, (OPTIONS_PER_STORE if compare else OPTIONS_PER_ITEM) - (n - 1))


def _option_groups(task: Task, limit: int, *, listing: bool = False) -> list[tuple[str, list[dict]]]:
    """(asked item, the products to offer for it) on this store, best first: Saheli's suggestion, the AI's exact listings,
    then the closest other variant (marked as not exactly what was asked). A medicine is offered only when it is exactly
    the one asked. listing (or no AI ranking): the store's own order. Products the store refused are left out."""
    d = task.details or {}
    found = _found(task)
    _number(found)
    matched = d.get("matched")
    medicine = d.get("agent") == "pharmacy"
    refused = set(d.get("refused") or [])
    suggested = {_pkey(c) for c in d.get("chosen") or []}
    out = []
    kept = d.get("kept") or {}
    for asked, g in _groups(task, found):
        if asked in kept:
            continue  # picked before and unchanged: not offered again
        g = [i for i in g if _pkey(i) not in refused]
        m = (matched or {}).get(asked) or {}
        exact = set(m.get("exact") or [])
        if listing or matched is None:
            ranked = [i if matched is None else {**i, "exact_match": _pkey(i) in exact} for i in g]
        else:
            byk = {_pkey(i): i for i in g}
            ranked = [{**byk[k], "exact_match": True} for k in m.get("exact") or [] if k in byk]
            if not medicine and m.get("closest") in byk and m["closest"] not in exact:
                ranked.append({**byk[m["closest"]], "exact_match": False})
        ranked.sort(key=lambda i: 0 if _pkey(i) in suggested else 1)
        out.append((asked, ranked[:limit]))
    return out


def _bundles(task: Task, limit: int) -> list[dict]:
    """Food with several dishes: one offer per restaurant (a cart holds one restaurant), the restaurants with an exact dish
    for the most items first; a dish it has no exact listing for is looked for in its own menu."""
    d = task.details or {}
    found = _found(task)
    _number(found)
    refused = set(d.get("refused") or [])
    groups = [(a, [i for i in g if _pkey(i) not in refused]) for a, g in _groups(task, found)]
    suggested = {_pkey(c) for c in d.get("chosen") or []}
    out = []
    for _, picks in _restaurants(groups, d.get("matched") or {})[:limit]:
        dishes = [{**picks[a], "exact_match": True} for a, _ in groups if picks.get(a)]
        out.append({"oids": [p["oid"] for p in dishes], "item": "", "items": dishes, "find": [a for a, _ in groups if not picks.get(a)],
                    "suggested": bool(dishes) and all(_pkey(p) in suggested for p in dishes)})
    return out


def _ride_opts(task: Task) -> list[dict]:
    return [o for o in ((task.result or {}).get("options") or []) if isinstance(o, dict) and o.get("type")]


def _end_name(task: Task, end: str) -> str:
    """A ride's pickup or drop in words: the saved place meant (its name and address), else their words; and where the app
    set it when it showed it (founder 2026-10-12: never "your exact address" from memory)."""
    d, r = task.details or {}, task.result or {}
    place = d.get(f"{end}_place") or {}
    meant = (f"{place['nickname']} ({place['full']})" if place.get("nickname") and place.get("full") else
             place.get("full") or d.get(end) or f"the {end}")
    used = str(r.get(f"{end}_used") or "").strip()
    return meant + (f" [the app set: {used}]" if used else "")


def _trip(task: Task) -> str:
    return f"from {_end_name(task, 'pickup')} to {_end_name(task, 'drop')}"


def _ride_offers(task: Task) -> list[dict]:
    """A ride app's fares as options: the one they picked before (after the login), else the type they named, else the one
    they took on this app last time, else the cheapest, suggested (founder 2026-10-12: show the types with fares and ask
    which one; the person always picks)."""
    opts = _ride_opts(task)
    if not opts:
        return []
    d = task.details or {}
    picked = (d.get("ride_pick") or {}).get("type")
    fares = [_rupee_value(o.get("fare")) or float("inf") for o in opts]
    wanted = next((m for m in (_match_option(opts, str(v)) for v in (d.get("vehicle"), d.get("usual_vehicle")) if v)
                   if isinstance(m, dict)), None)
    sugg = next((i for i, o in enumerate(opts) if picked and o.get("type") == picked),
                next((i for i, o in enumerate(opts) if wanted is o), min(range(len(opts)), key=fares.__getitem__)))
    return [{"oids": [str(i)], "item": "", "items": [], "find": [], "suggested": i == sugg,
             "text": f"{o['type']} {o.get('fare') or ''}".strip() + ((f" (pickup in {o['eta']})" if any(ch.isdigit() for ch in str(o["eta"]))
                                                                       else f" ({o['eta']})") if o.get("eta") else "")}
            for i, o in enumerate(opts)]


def _offers(task: Task, *, compare: bool = False, listing: bool = False, only: str | None = None) -> list[dict]:
    """What this store offers the person to pick from, best first. Each offer: oids (the store's products, one per asked
    item it covers), item (the asked item; '' for a whole meal), items (the products), find (dishes its restaurant's menu
    is searched for), suggested (Saheli's own pick). A ride's offers are its fares."""
    d = task.details or {}
    if task.kind == "ride":
        return _ride_offers(task)
    if SKILLS[task.service].get("kind") == "food" and len(d.get("items") or []) > 1 and d.get("matched") and not listing:
        meals = _bundles(task, _per_item(task, compare=compare) + 1)
        if meals:
            return meals
    suggested = {_pkey(c) for c in d.get("chosen") or []}
    groups = _option_groups(task, 10 if only else _per_item(task, compare=compare), listing=listing)
    if only:
        groups = [(a, g) for a, g in groups if a == only] or groups
    return [{"oids": [i["oid"]], "item": asked, "items": [i], "find": [], "suggested": _pkey(i) in suggested}
            for asked, opts in groups for i in opts]


def _offer_text(o: dict, asked: list | None = None) -> str:
    if o.get("text"):
        return o["text"]
    if o["item"] or (len(o["items"]) == 1 and not o["find"]):
        i = o["items"][0]
        n = channels._qty_for(asked or [], {**i, "for_item": i.get("for_item") or o["item"]}) if asked else 1
        each = _rupee_value(i.get("price"))
        # the quantity they asked times this pack (live sim 2026-10-11: "4 bars" read as the 4-pack being the suggestion)
        return _option(i, o["item"]) + (f" each; {n} = ₹{round(each * n)}" if n > 1 and each else "")
    first = o["items"][0] if o["items"] else {}
    eta = f" ({first['eta']})" if first.get("eta") else ""
    dishes = " + ".join(f"{_label(i)} {i.get('price') or ''}".strip() for i in o["items"])
    find = f" + {', '.join(o['find'])} from its menu (price at the cart)" if o["find"] else ""
    return f"{first.get('restaurant') or 'one restaurant'}{eta}: {dishes}{find}"


def _offer_lines(task: Task, offers: list[dict], n: int, *, mark: bool, indent: str = "") -> tuple[list[str], int]:
    """The offers as numbered lines (numbers go on from n). The numbers are kept on the task, so the person's pick ("the
    second one") maps back to the store's own products."""
    lines, kept, last = [], {}, None
    multi = len((task.details or {}).get("items") or []) > 1
    for o in offers:
        if multi and o["item"] and o["item"] != last:
            lines.append(f"{indent}for {o['item']}:")
            last = o["item"]
        n += 1
        text, sug = _offer_text(o, (task.details or {}).get("items")), mark and o["suggested"]
        lines.append(f"{indent}  {n}. {text}" + (" ← suggested" if sug else ""))
        kept[str(n)] = {"oids": o["oids"], "text": text[:140], **({"s": True} if sug else {})}
    task.details = {**(task.details or {}), "offer": kept or None}
    return lines, n


def _pick_ask(where: str, multi: bool, ride: bool = False, across: bool = False) -> str:
    shown = "ride type, fare, pickup time" if ride else "name, pack, price; keep every 'similar: not exactly …' note next to its option"
    if across:  # several stores or apps: the person must see where each option is from (sim 2026-10-12: dishes listed without the app)
        shown += f"; the {'app' if ride else 'store'} of every option with it"
    nothing = ("Nothing is booked before they pick and then say yes to the fare" if ride else
               "Nothing is ordered before they pick and then say yes to the total")
    return (f"Show them these options in ONE message, one option per line, each line starting with its number as here ({shown}, "
            "in their language; say which one you suggest), and ask which one they want" + (" (one per item)" if multi else "")
            + f". When they pick: task_input {where}, "
            "kind go, value = the number(s) they picked, comma separated; 'any', 'you choose', 'the suggested one' or a plain yes "
            f"= value yes. {nothing}; if they want none of these: task_input go no.")


def _ride_options_message(task: Task, *, head: str = "") -> str:
    """One ride app's fares before the login, numbered, for the person to pick."""
    label = SKILLS[task.service]["label"]
    lines, _ = _offer_lines(task, _offers(task), 0, mark=True)
    return (f"{head}Fares on {label}, {_trip(task)} (before the login, they may change a little; nothing is booked):\n"
            + "\n".join(lines) + f"\nBooking needs a login on {label}: {_code_line(task)}. "
            + _pick_ask(f"on task {task.id}", False, ride=True))


def _ride_choice_message(task: Task, extra: str = "") -> str:
    """The fresh fares after the login: the one confirm of the type they picked, with its fare now; else the fares to pick
    from (a pick here, with its fare, is their yes)."""
    d, r = task.details or {}, task.result or {}
    label = SKILLS[task.service]["label"]
    lines, _ = _offer_lines(task, _offers(task), 0, mark=True)
    offer = (task.details or {}).get("offer") or {}
    pick = d.get("ride_pick") or {}
    n = next((k for k, o in offer.items() if o.get("s")), None)
    same = next((o for o in _ride_opts(task) if pick and o.get("type") == pick.get("type")), None)
    tail = (" SURGE pricing is on now: say so." if r.get("surge") else "") + (f" {extra}" if extra else "")
    listed = f"Fares on {label} now, {_trip(task)} (nothing is booked):\n" + "\n".join(lines) + "\n"
    if same and n:
        return (listed + f"They picked {pick['type']} ({pick.get('fare') or 'its fare'} earlier); now it is {same.get('fare')}. Ask ONE "
                f"short question: book {pick['type']} for {same.get('fare')}, {_trip(task)}, pay cash? Yes → task_input choice value {n}; "
                "another option → task_input choice with its number; no → task_input choice no." + tail)
    gone = f"The {pick['type']} they picked is not offered now. " if pick else ""
    return (listed + gone + "Show them these fares with the same numbers and ask which one to book (say which one you suggest); "
            "their pick, with its fare, is their yes: task_input choice with its number; none → task_input choice no." + tail)


def _options_message(task: Task, out: dict, *, head: str = "", listing: bool = False, only: str | None = None) -> str:
    """One store's options for the person to pick from (founder 2026-10-11: options at all times, what was found, so the
    person picks), numbered, with Saheli's suggestion marked."""
    d = task.details or {}
    label = SKILLS[task.service]["label"]
    place = (d.get("limits") or {}).get("place") or {}
    where = place.get("nickname") or place.get("pincode") or "their place"
    eta = f", delivery in about {out['eta']}" if out.get("eta") else ""
    lines, _ = _offer_lines(task, _offers(task, listing=listing, only=only), 0, mark=True)
    missing = [m for m in d.get("missing_items") or [] if m]
    kept = [f"{_label(c)} {c.get('price') or ''}".strip() for c in (d.get("kept") or {}).values()]
    return (f"{head}Found on {label} (delivers to {where}{eta}); nothing is ordered yet:\n" + "\n".join(lines) + "\n"
            + (f"Kept as picked before: {'; '.join(kept)}.\n" if kept else "")
            + (f"Not on {label} at all: {', '.join(missing)}.\n" if missing else "")
            + _pick_ask(f"on task {task.id}", len(d.get("items") or []) > 1))


def _store_reason(task: Task) -> str:
    """Why a store of a comparison has nothing to offer, in a few words."""
    d = task.details or {}
    if d.get("too_slow") or task.status in ("queued", "running"):
        return "its website did not answer in time, so nothing from it"
    if d.get("store_down"):
        return "its website did not work just now, so nothing from it"
    return d.get("compare_reason") or re.split(r"; nothing was ordered|\. Offer|; looking on", str(d.get("compare_note") or "no result"))[0]


def _fallback_head(group: list[Task]) -> str:
    """The store they named, when the others were looked at because of it."""
    fb = next(((g.details or {}).get("fallback_from") for g in group if (g.details or {}).get("fallback_from")), None)
    if fb not in SKILLS:
        return ""
    label = SKILLS[fb]["label"]
    others = "apps" if group and group[0].kind == "ride" else "stores"
    if any(g.service == fb for g in group):
        return f"They asked for {label}: it has only similar products, not exactly this, so the other {others} were looked at too. "
    why = next(((g.details or {}).get("fallback_why") for g in group if (g.details or {}).get("fallback_why")), "")
    return f"They asked for {label}: {why or 'it does not have it or does not deliver there'}. So the other {others} were looked at. "


def _compare_summary(group: list[Task], head: str = "") -> str:
    """Every store of a comparison in ONE update (founder 2026-10-11: options at all times; report the results from every
    store the person named): what each store has, numbered across the stores with Saheli's suggestion marked, and why a
    store has nothing. The person picks; nothing is picked for them."""
    best = best_store(group)
    ride = any(t.kind == "ride" for t in group)

    def has(t: Task) -> bool:
        return t.status == "awaiting_confirm" and t.input_needed == "go" and bool(_ride_opts(t) if t.kind == "ride" else _found(t))

    def order(t: Task):
        scored = has(t) and ((t.details or {}).get("chosen") or t.kind == "ride")
        return (0 if t is best else 1 if has(t) else 2, _store_score(t) if scored else (), SKILLS[t.service]["label"])

    lines, offered, n = [], [], 0
    for t in sorted(group, key=order):
        label, r, d = SKILLS[t.service]["label"], t.result or {}, t.details or {}
        linked = " (linked account: no login code)" if d.get("channel") == "connector" else ""
        eta = f", delivers in about {r['eta']}" if r.get("eta") and t.kind != "ride" else ""
        if has(t):
            offers = _offers(t, compare=True)
            if not offers:
                lines.append(f"- {label}: nothing exactly this" + (" (a medicine is never swapped for another)" if d.get("agent") == "pharmacy" else ""))
                continue
            lines.append(f"- {label} [task {t.id}]{linked}{eta}:")
            more, n = _offer_lines(t, offers, n, mark=t is best, indent="  ")
            lines += more + [f"    not on {label}: {m}" for m in d.get("missing_items") or [] if m]
            offered.append(t)
        elif t.status == "awaiting_confirm" and t.input_needed == "go":
            lines.append(f"- {label} [task {t.id}]: shows its {'fares' if t.kind == 'ride' else 'products and prices'} only after a "
                         f"login ({_code_line(t)}); to look there: task_input go yes on this task")
            offered.append(t)
        elif t.status == "needs_input" and t.input_needed == "swap":
            alts = "; ".join(f"{a['name']}{(' ' + str(a['price'])) if a.get('price') else ''}" for a in (d.get("alternatives") or []))
            lines.append(f"- {label} [task {t.id}]: does not have it; it has instead: {alts} (to get one: task_input swap "
                         "with its name on this task)")
            offered.append(t)
        else:
            lines.append(f"- {label}: {_store_reason(t)}")
    stores = ", ".join(sorted(SKILLS[t.service]["label"] for t in group))
    trip = f", {_trip(group[0])}" if ride else ""
    top = (f"{head}Looked on {stores}{trip} without logging in; nothing is {'booked' if ride else 'ordered'}.\n" + "\n".join(lines) + "\n")
    if not offered:
        return top + ("No ride came up on any of them. Tell the person plainly; they can try again in a few minutes or book in an "
                      "app." if ride else "It is not available on any of them. Tell the person plainly and offer another name, or the "
                      "family can order in the app.")
    multi = any(len((t.details or {}).get("items") or []) > 1 for t in group)
    return top + (f"Say in one short line which {'apps' if ride else 'stores'} have nothing and why. " + _pick_ask(
        "on THAT option's store task (the [task …] above it)", multi, ride=ride, across=len(offered) > 1) + (" All of an order's items come from one store." if multi else "")
        + (f" A plain yes (the suggested one) is task_input go yes on task {best.id}." if best else ""))


async def _compare_update(session: AsyncSession, task: Task, own: str) -> tuple[str | None, bool]:
    """A store of a comparison finished looking: its result is kept (no message); choose_stores sends every store's
    options once every store has answered, or COMPARE_WAIT after the first did."""
    task.details = {**(task.details or {}), "compare_seen": clock.now().isoformat(), "compare_note": own[:240]}
    group = await siblings(session, task)
    if any((t.details or {}).get("compare_announced") for t in group if t.id != task.id) and task.status in LIVE:
        task.status, task.cancel_requested, task.input_needed = "cancelled", True, None
        note(task, "another store was already picked")
    return None, False


def _plain_amount(v) -> bool:
    """A total as a checkout shows it: a number, or "₹1,234" / "Rs 320.50", nothing else. An agent's words around a number
    ("₹320 (estimated …)") mean it did not see the bill. A data check on the store's report, not on anyone's words."""
    if isinstance(v, (int, float)):
        return v > 0
    return bool(re.fullmatch(r"\s*(?:₹|rs\.?|inr)?\s*\d[\d,]*(?:\.\d{1,2})?\s*", str(v or ""), re.I))


def _rupee_value(v) -> float:
    m = re.search(r"\d+(?:\.\d+)?", str(v or "").replace(",", ""))
    return float(m.group(0)) if m else 0.0


def _eta_minutes(v) -> int:
    m = re.search(r"(\d+)", str(v or ""))
    return int(m.group(1)) if m else 999


def best_store(group: list[Task]) -> Task | None:
    """The store Saheli suggests among the options (the person picks): the one that has the most of the items, then the
    most exact matches, then the lowest price, then no login code needed, then the soonest delivery. Rides: the cheapest
    fare."""
    cands = [t for t in group if t.status == "awaiting_confirm" and t.input_needed == "go"
             and ((t.details or {}).get("chosen") or (t.kind == "ride" and _ride_opts(t)))]
    return min(cands, key=_store_score) if cands else None


def _store_score(t: Task):
    """Lower is better: most items, then most exact matches, then lowest price, then no login code, then soonest."""
    d, r = t.details or {}, t.result or {}
    if t.kind == "ride":
        opts = _ride_opts(t)
        fares = [_rupee_value(o.get("fare")) for o in opts if _rupee_value(o.get("fare"))]
        return (0, 0, min(fares) if fares else float("inf"), 0 if r.get("logged_in") else 1,
                min((_eta_minutes(o.get("eta")) for o in opts), default=999))
    chosen = d.get("chosen") or []
    asked = d.get("items") or []
    price = sum(_rupee_value(c.get("price")) * channels._qty_for(asked, c) for c in chosen) if asked else sum(_rupee_value(c.get("price")) for c in chosen)
    no_code = d.get("channel") == "connector" or bool(r.get("logged_in"))
    return (-len(chosen), -sum(1 for c in chosen if c.get("exact_match")), price, 0 if no_code else 1, _eta_minutes(r.get("eta")))


async def _reoffer_others(session: AsyncSession, task: Task, why: str) -> str | None:
    """The store the person picked could not build the cart (not sold at its serving shop, the site failed): the other
    stores' options go back to them to pick from (founder 2026-10-11: options at all times; nothing is picked for them).
    Not when the reason holds for every store (no number to log in with, a family rule), and never after a confirm.
    Returns the update to send."""
    d = task.details or {}
    if (not compare_group(task) or task.kind not in ("order", "ride") or d.get("confirmed_by") or (task.result or {}).get("placed")
            or d.get("login_missing") or d.get("rule_block") or d.get("unauthorized_place") or d.get("practice_accident")):
        return None
    back = [t for t in await siblings(session, task) if t.id != task.id and t.status == "cancelled"
            and (t.details or {}).get("dropped_for") == str(task.id)]
    if not back:
        return None
    label = SKILLS[task.service]["label"]
    for t in back:
        td = t.details or {}
        t.status, t.input_needed, t.cancel_requested = "awaiting_confirm", td.get("dropped_input") or "go", False
        t.details = {**td, "dropped_for": None, "dropped_input": None, "revived_after": str(task.id), "told_working": None,
                     "active_from": clock.now().isoformat()}
        t.deadline_at = clock.now() + TASK_LIFETIME
        note(t, f"{label} could not build the cart; offered again")
    note(task, "could not build the cart; the other stores' options are offered again")
    why = re.sub(r"\s+", " ", why or "").strip().rstrip(".")[:160]
    return _compare_summary(back, head=f"{label}, which they picked, could not build the cart" + (f" ({why})" if why else "")
                            + "; nothing was ordered there. Offer the other stores' options again. ")


async def _fall_back_to_other_stores(session: AsyncSession, task: Task, why: str = "") -> list[str]:
    """Start the same order on the other usual stores of its kind, as one comparison. Returns the stores started. The store
    they named joins that comparison when it has close products to offer (similar_only), so every store's options go out
    together."""
    cat = next((c for c, stores in COMPARE_STORES.items() if task.service in stores), None)
    if not cat:
        return []
    d = task.details or {}
    busy = {t.service for t in await live_tasks(session, task.family_id) if t.kind == task.kind}
    gid, started = uuid.uuid4().hex[:10], []
    why = re.split(r"; nothing was ordered|\. Offer|; looking on", re.sub(r"\s+", " ", why or "").strip())[0][:200]
    for service in COMPARE_STORES[cat]:
        if service == task.service or service in busy:
            continue
        try:
            await create(session, family_id=task.family_id, subject_id=task.subject_id, requested_by=task.requested_by, service=service,
                         kind=task.kind, goal=task.goal, details={**{k: d[k] for k in ("items", "area", "pickup", "drop", "vehicle", "pickup_place", "drop_place",
                                                                                         "later") if d.get(k)},
                                                                 "compare": gid, "fallback_from": task.service, "fallback_why": why or None},
                         limits=Limits.from_dict(d.get("limits")))
            started.append(SKILLS[service]["label"])
        except TaskRefused:
            continue
    task.details = {**(task.details or {}), "fell_back": started or ["none"]}
    if started:
        note(task, "not here exactly; looking on " + ", ".join(started) + " too")
        if task.status == "awaiting_confirm":
            task.details = {**task.details, "compare": gid, "compare_seen": clock.now().isoformat(), "compare_note": why or None}
    return started


async def choose_stores(sessions: async_sessionmaker, agent: BrowserAgent | None = None) -> list[tuple[Task, str, list[Task]]]:
    """Comparisons where every store has looked (or COMPARE_WAIT passed since the first did): ONE update with what every
    store found, numbered, Saheli's suggestion marked, and why a store has nothing (founder 2026-10-11: options at all
    times; report the results from every store the person named). A store still looking then is stopped and reported as
    too slow. Nothing goes on to a cart before the person picks."""
    out: list[tuple[Task, str, list[Task]]] = []
    async with sessions() as session:
        rows = list((await session.execute(
            select(Task).where(Task.status.in_(("awaiting_confirm", "needs_input", "failed")), Task.created_at >= clock.now() - timedelta(hours=1))
        )).scalars())
        done: set[str] = set()
        for t in rows:
            gid, d = compare_group(t), t.details or {}
            if not gid or gid in done or not d.get("compare_seen") or d.get("compare_announced"):
                continue
            done.add(gid)
            ids = [g.id for g in await siblings(session, t)]
            group = list((await session.execute(
                select(Task).where(Task.id.in_(ids)).with_for_update(skip_locked=True).execution_options(populate_existing=True)
            )).scalars())
            if len(group) < len(ids) or any((g.details or {}).get("compare_announced") for g in group):
                continue  # another tick holds one of them, or it was decided already
            first = min(datetime.fromisoformat(g.details["compare_seen"]) for g in group if (g.details or {}).get("compare_seen"))
            wait = COMPARE_WAIT_NAMED if any((g.details or {}).get("named") for g in group) else COMPARE_WAIT
            if any(g.status in ("queued", "running") for g in group) and clock.now() - first < wait:
                continue
            for g in group:
                g.details = {**(g.details or {}), "compare_announced": True}
                if g.status in ("queued", "running"):
                    if g.status == "running" and g.agent_task and agent is not None:
                        try:
                            await agent.stop(g.agent_task, end_session=True)
                        except Exception:  # noqa: BLE001 — the session is released with the task
                            logger.warning("stop failed task=%s", g.id)
                    g.status, g.cancel_requested, g.input_needed = "cancelled", True, None
                    g.details = {**g.details, "too_slow": True}
                    note(g, "too slow: the other stores' options went out without it")
            best = best_store(group)
            anchor = best or next((g for g in group if g.status in ("awaiting_confirm", "needs_input")), t)
            out.append((anchor, _compare_summary(group, head=_fallback_head(group)), group))
        await session.commit()
    return out


def clean_item(i: dict) -> dict:
    """An item as the brain gave it (structured: it read the person's words): product words, quantity, the traits that
    must match, a price limit, cheapest. Kept as data; nothing here reads language."""
    out = {"name": str(i.get("name") or "").strip()[:120], "qty": max(1, min(int(i.get("qty") or 1), 50))}
    if isinstance(i.get("must_match"), list):
        out["must_match"] = [str(x).strip()[:60] for x in i["must_match"] if str(x).strip()][:5]
    try:
        if i.get("max_price"):
            out["max_price"] = float(i["max_price"])
    except (TypeError, ValueError):
        pass
    if i.get("cheapest"):
        out["cheapest"] = True
    if i.get("restaurant"):
        out["restaurant"] = str(i["restaurant"])[:80]
    return out


def _is_yes(value: str) -> bool:
    """The brain turns the person's words (any language) into "yes" or "no"; code never reads their words."""
    return (value or "").strip().lower() == "yes"


def _is_no(value: str) -> bool:
    return (value or "").strip().lower() == "no"


def flash_phases() -> set[str]:
    """Phases run in Browser Use flash mode (TASK_FLASH_PHASES, e.g. 'browse,place'); off until tested per store."""
    return {p.strip() for p in os.getenv("TASK_FLASH_PHASES", "").split(",") if p.strip()}


def task_max_cost() -> float:
    """Browser cost after which an order pauses and the caregiver is asked (keep trying or cancel)."""
    return float(os.getenv("TASK_MAX_COST_INR", "200"))  # a look-up, login, cart and placing (~80 steps) fit


# Founder 2026-10-10: an order not placed after 20 minutes → ask the caregiver; no answer in 5 minutes → email them too
# (if they have an email); still none 5 minutes later → cancel it properly. Any answer starts the 20 minutes again.
LADDER_ASK, LADDER_EMAIL, LADDER_CANCEL = timedelta(minutes=20), timedelta(minutes=5), timedelta(minutes=5)
STILL_WORKING = timedelta(minutes=5)  # one "still working" line after this long with nothing to answer


MAX_RETRIES = 1  # automatic retries of a browser run that crashed without a report (prepare only, never place)
MAX_CODE_RESENDS = 2  # new login codes asked for one order when the person says the code did not come
MAX_SKILL_NOTES = 12


def note(task: Task, text: str) -> None:
    task.history = [*(task.history or []), {"at": clock.now().isoformat(), "phase": task.phase, "status": task.status, "note": text[:300]}]
    task.updated_at = clock.now()


# ── one true status (founder 2026-10-11) ──────────────────────────────────────
# An update is written for a task as it is at that moment and reaches the person only after it gets the family's turn
# (the brain may be answering them right then). Live 2026-10-11: "cancelled", then "cancelling is in progress", then
# "placed, arriving in 14 minutes" for one order. An update whose task moved on before it is delivered is dropped there.
_UPDATE_FOR: contextvars.ContextVar[dict | None] = contextvars.ContextVar("kavach_task_update_for", default=None)


def _stage(task: Task) -> str:
    """What an update about this task tells the person: what it waits for (with the cart, fares or options shown), that it
    is working, placing or booking, or how it ended; and a cancel asked after it was placed."""
    d = task.details or {}
    ask = d.get("cancel_ask") or {}
    if task.status in ("queued", "running"):
        s = {"place": "placing", "cancel": "cancelling"}.get(task.phase, "working")
    elif task.status in ("needs_input", "awaiting_confirm"):
        s = f"asking:{task.input_needed}"
        if task.input_needed == "otp":
            # each new code the store sends is news, and so is a code that did not go in (live sim 2026-10-13: "send the
            # newest code again" was skipped as already told)
            s += f":{int(d.get('codes_sent') or 0)}:{int(d.get('code_fails') or 0)}"
        if task.input_needed in ("confirm", "choice", "approve"):
            s += ":" + str(d.get("cart_fp") or "")[:8]
        if task.input_needed in ("go", "choice"):
            s += ":" + hashlib.sha1(json.dumps(d.get("offer") or {}, sort_keys=True).encode()).hexdigest()[:8]
    else:
        s = task.status
    return f"{s}|{ask.get('state') or ''}|{ask.get('told_by') or ''}"


async def _notify_about(notify: Notify, tasks: list[Task], family_id: str, person: str, prompt: str) -> None:
    """Send an update written for these tasks as they are now; if one of them moves on before the update is delivered (the
    person answered, changed or cancelled it meanwhile), the host drops it at delivery (delivery_check)."""
    token = _UPDATE_FOR.set({str(t.id): _stage(t) for t in tasks})
    try:
        await notify(family_id, person, prompt)
    finally:
        _UPDATE_FOR.reset(token)


class TaskUpdate:
    """An update about tasks, as they were when it was written. The turn that delivers it (wake.system_turn):
    - calls it once the family's turn is held: a task that moved on meanwhile drops it (founder 2026-10-11: one true
      status);
    - skips it when this person was already told this exact state by an update (already_told);
    - otherwise it must reach them (founder 2026-10-12: the login-code ask and the booked ride never reached him because
      the brain answered "none"): only when the person wrote after it was written, and was answered, may the brain decide
      they already heard it (wrote_since: that answer saw the task as it is now);
    - records who was told which state (told), and as the last resort sends the news written from the task data (news)."""

    def __init__(self, expect: dict[str, str]):
        self.expect = dict(expect)
        self.written_at = clock.now()

    async def _tasks(self, session: AsyncSession) -> list[Task]:
        return list((await session.execute(select(Task).where(Task.id.in_([uuid.UUID(k) for k in self.expect]))
                                           .execution_options(populate_existing=True))).scalars())

    async def __call__(self, session: AsyncSession) -> bool:
        now = {str(t.id): _stage(t) for t in await self._tasks(session)}
        moved = {k: f"{v} → {now.get(k)}" for k, v in self.expect.items() if now.get(k) != v}
        if moved:
            logger.info("task update dropped, the task moved on before delivery: %s", moved)
        return not moved

    async def already_told(self, session: AsyncSession, person: str) -> bool:
        rows = (await session.execute(select(TaskTold.task_id, TaskTold.stage).where(
            TaskTold.task_id.in_([uuid.UUID(k) for k in self.expect]), TaskTold.person == person))).all()
        told = {(str(t), st) for t, st in rows}
        return all((k, v[:240]) in told for k, v in self.expect.items())

    async def wrote_since(self, session: AsyncSession, family_id: str, person: str) -> bool:
        from app.care.models import Turn

        return bool(await session.scalar(select(func.count()).select_from(Turn).where(
            Turn.family_id == family_id, Turn.thread_id == person, Turn.role == "user", Turn.at >= self.written_at)))

    async def told(self, session: AsyncSession, person: str, how: str) -> None:
        for k, v in self.expect.items():
            await session.execute(pg_insert(TaskTold).values(task_id=uuid.UUID(k), person=person, stage=v[:240], how=how, at=clock.now())
                                  .on_conflict_do_nothing(index_elements=["task_id", "person", "stage"]))
        await session.commit()

    async def booked(self, session: AsyncSession) -> bool:
        """Every task of this update was just placed or booked, plainly (no warning, no wrong pickup, no cancel): its message
        is written from the task data, so no detail can be left out (sim 2026-10-12: a booked ride's message came without
        the driver, car or PIN; founder: "I never received the booking details")."""
        tasks = await self._tasks(session)
        return bool(tasks) and all(
            t.status == "done" and ((t.result or {}).get("placed") or (t.result or {}).get("booked"))
            and self.expect.get(str(t.id), "").startswith("done|")
            and not any((t.details or {}).get(k) for k in ("placed_warn", "ends_wrong", "cancel_ask", "practice", "place_unclear"))
            for t in tasks)

    async def news(self, session: AsyncSession, person: str, names: dict[str, str] | None = None) -> tuple[str, tuple[str, str] | None] | None:
        tasks = await self._tasks(session)
        tasks.sort(key=lambda t: list(self.expect).index(str(t.id)))
        return plain_news(tasks, person, names or {}) if tasks else None


def delivery_check() -> TaskUpdate | None:
    """For the host's notify: the update being sent, as its tasks were when it was written (TaskUpdate). None when the
    update is not about a task."""
    expect = _UPDATE_FOR.get()
    return TaskUpdate(expect) if expect else None


def _end_plain(task: Task, end: str) -> str:
    """A ride's end for the person: where the app set it, else the place they meant ("the location you shared" for a bare
    map point, never its numbers)."""
    d, r = task.details or {}, task.result or {}
    place = d.get(f"{end}_place") or {}
    out = str(r.get(f"{end}_used") or place.get("full") or d.get(end) or f"the {end}").strip()
    return out if any(ch.isalpha() for ch in out) else "the location you shared"


def ride_details(task: Task) -> list[str]:
    """What a booked ride's person needs (founder 2026-10-12: the booking details never reached him): the ride id, the
    driver, the car and its plate, when the driver comes, and the ride PIN the app shows for starting the trip."""
    r = task.result or {}
    car = " ".join(str(r.get(k) or "").strip() for k in ("vehicle", "plate")).strip()
    return [x for x in (
        f"ride id {r['ride_id']}" if r.get("ride_id") else "",
        f"driver {r['driver']}" if r.get("driver") else "",
        f"car {car}" if car else "",
        f"pickup in about {r['pickup_eta']}" if r.get("pickup_eta") else "",
        f"ride PIN {r['ride_pin']} (tell the driver at pickup)" if r.get("ride_pin") else "",
    ) if x]


def order_details(task: Task) -> list[str]:
    r, d = task.result or {}, task.details or {}
    total = r.get("total") or d.get("confirmed_total")
    return [x for x in (
        f"order id {r['order_id']}" if r.get("order_id") else "",
        f"total {total}, cash on delivery" if total else "",
        f"delivery in about {r['eta']}" if r.get("eta") else "",
    ) if x]


def plain_news(tasks: list[Task], person: str, names: dict[str, str]) -> tuple[str, tuple[str, str] | None]:
    """An update written by code from the tasks as they are now: the last resort when the brain sent nothing twice (founder
    2026-10-12). Plain English; the deliverer puts it in the person's own words when a model answers. Returns (text,
    buttons as (kind, key) or None). Never an id of a person: names, else "your family"."""
    who = lambda pid: names.get(pid) or "your family"  # noqa: E731
    lead = tasks[0]
    d, r = lead.details or {}, lead.result or {}
    label = SKILLS[lead.service]["label"]
    done, thing, ordered = _words(lead)
    ride = lead.kind == "ride"
    trip = f"from {_end_plain(lead, 'pickup')} to {_end_plain(lead, 'drop')}" if ride else ""
    many = len(tasks) > 1
    offered = [t for t in tasks if t.status == "awaiting_confirm" and t.input_needed in ("go", "choice") and (t.details or {}).get("offer")]
    if offered:
        rows = sorted(((int(n), o, t) for t in offered for n, o in ((t.details or {}).get("offer") or {}).items() if str(n).isdigit()),
                      key=lambda x: x[0])
        lines = [f"{n}. {o.get('text')}" + (f" ({SKILLS[t.service]['label']})" if many else "") + (" - my suggestion" if o.get("s") else "")
                 for n, o, t in rows]
        login = [f"{SKILLS[t.service]['label']}: shows its {'fares' if ride else 'prices'} only after a login" for t in tasks
                 if t not in offered and t.status == "awaiting_confirm" and t.input_needed == "go"]
        none = [f"{SKILLS[t.service]['label']}: {_store_reason(t)}" for t in tasks if t.status not in ("awaiting_confirm", "needs_input")]
        head = ("Fares" if ride else "What I found") + ("" if many else f" on {label}") + (f", {trip}" if ride else "") + f" (nothing is {done} yet):"
        return "\n".join([head, *lines, *login, *none, f"Which one shall I {'book' if ride else 'get'}? Reply with its number."]), None
    ask = d.get("cancel_ask") or {}
    lt = later.of(lead)
    if lt:
        at, book_at = later.parse_at(lt.get("at")), later.parse_at(lt.get("book_at"))
        if lead.status == "awaiting_confirm" and lead.input_needed == "later":
            amount = _later_amount(lead)
            sug = later.suggest_max(amount)
            what = (d.get("ride_pick") or {}).get("type") or "the ride" if ride else "; ".join(_label(c) for c in d.get("chosen") or []) or lead.goal
            return ((f"For later: {what} on {label}" + (f", {trip}" if ride else "") + f", for {later.hm(at)}, cash. Today it is about "
                     f"₹{amount:.0f}; then it may differ. Shall I {'book' if ride else 'order'} it then, paying up to ₹{sug}? (If it is above "
                     f"about ₹{later.cap(sug):.0f} I will ask you first.) You can also tell me your own limit.") if amount and sug else (
                     f"For later: {what} on {label}, for {later.hm(at)}, cash. Shall I {'book' if ride else 'order'} it then? Tell me the most "
                     "you want to pay."), ("later", f"{lead.id}:{sug or ''}"))
        if lead.status == "needs_input" and lead.input_needed == "fare":
            asked = lt.get("asked") or {}
            return (f"For your {'ride' if ride else 'order'} on {later.hm(book_at)}: it costs {asked.get('amount')} on {label} now, above "
                    f"the ₹{lt.get('max') or 0:.0f} you set. Shall I {'book' if ride else 'order'} it at {asked.get('amount')}?"), None
        if lead.status == "scheduled":
            return (f"Set for later: {label} will be {'booked' if ride else 'ordered'} on {later.hm(book_at)}"
                    + (f", up to ₹{lt['max']:.0f}" if lt.get("max") else "") + "; I will send every detail then."), None
    if lead.status == "needs_input" and lead.input_needed == "otp":
        mine = "your phone" if d.get("code_from") == person else _code_to(lead)
        return (f"{label} has sent a login code to {mine} for the {thing} ({lead.goal}). Please send me that code; nothing is "
                f"{ordered} yet."), None
    if lead.status == "awaiting_confirm" and lead.input_needed == "go":
        return (f"{label} shows its {'fares' if ride else 'prices'} only after a login ({_code_line(lead)}). "
                f"{'Shall I look?' if (lead.details or {}).get('login_live') else 'Shall I log in and look?'} Nothing is {ordered} yet."), None
    if lead.status == "awaiting_confirm" and lead.input_needed == "confirm":
        where = ((d.get("limits") or {}).get("place") or {}).get("nickname") or "your saved place"
        eta = f", delivery in about {r['eta']}" if r.get("eta") else ""
        return (f"{label} cart: {bought(lead)}. Total {r.get('total')} with fees, to {where}, cash on delivery{eta}. Shall I place "
                "the order?"), ("order", f"{lead.id}:{str(d.get('cart_fp') or '')[:8]}")
    if lead.status == "awaiting_confirm" and lead.input_needed == "approve":
        ap = d.get("approval_needed") or {}
        what = (ap.get("choice") or {}).get("choice") if ride else r.get("total")
        return (f"{who(lead.requested_by)} asked for this {thing} on {label}: {bought(lead)}" + (f" ({what})" if what else "")
                + ". It needs your approval: approve or decline?"), None
    if lead.status == "needs_input" and lead.input_needed == "fee":
        return (f"{label} will cancel the {thing} ({bought(lead)}), but cancelling costs {d.get('cancel_fee')}. Shall I cancel it with "
                "that fee?"), None
    if lead.status == "needs_input" and lead.input_needed == "swap":
        alts = "; ".join(f"{a.get('name')} {a.get('price') or ''}".strip() for a in d.get("alternatives") or [] if isinstance(a, dict))
        return f"{label} does not have it; it has instead: {alts}. Shall I get one of these?", None
    if ask.get("elder") and person in (ask.get("carers") or []) and ask.get("state") in ("told_carers", "unclear"):
        maybe = "may already be" if ask["state"] == "unclear" else "is already"
        return (f"{who(ask.get('by'))} asked to cancel the {label} {thing} ({_placed_line(lead).split(': ', 1)[-1]}), but it {maybe} "
                f"{done}. Shall I ask {label} to cancel it? You can also cancel it in the {label} app."), None
    if lead.status == "cancelled" or ask.get("state") == "cancelled":
        if ask.get("state") == "cancelled":
            return f"Cancelled on {label}: {bought(lead)}.", None
        return f"Stopped: nothing was {ordered} on {label}.", None
    if ask.get("state") == "refused":
        return (f"{label} would not cancel it ({ask.get('store_said') or 'too late'}); it stays {done}. You can cancel it in the "
                f"{label} app."), None
    if ask.get("state") == "told_carers" and person == ask.get("by"):
        carers = " and ".join(who(c) for c in ask.get("carers") or []) or "your family"
        return f"It is already {done} on {label}, so I cannot cancel it from here; I have told {carers}.", None
    if lead.status == "done" and (r.get("placed") or r.get("booked")):
        cap = lambda x: x[:1].upper() + x[1:]  # noqa: E731
        if ride:
            return ". ".join(cap(x) for x in [f"Booked on {label}: {d.get('choice') or 'your ride'}, {trip}", *ride_details(lead)]) + ".", None
        return ". ".join(cap(x) for x in [f"Ordered on {label}: {bought(lead)}", *order_details(lead)]) + ".", None
    if lead.status == "failed" and d.get("place_unclear"):
        return (f"It is not clear whether the {thing} went through on {label}. Please check the {label} app before "
                f"{'booking' if ride else 'ordering'} again."), None
    if lead.status == "failed":
        why = re.sub(r"\s+", " ", str(d.get("compare_reason") or r.get("problem") or "")).strip().rstrip(".")[:140]
        return f"Nothing was {ordered} on {label}" + (f" ({why})" if why else "") + ".", None
    if lead.phase == "place" and lead.status in ("queued", "running"):
        return f"{'Booking' if ride else 'Placing'} it on {label} now; I will tell you as soon as it is done.", None
    if lead.phase == "cancel" and lead.status in ("queued", "running"):
        return f"Asking {label} to cancel it now; I will tell you what it says.", None
    stores = ", ".join(SKILLS[t.service]["label"] for t in tasks)
    return f"Still working on it on {stores}; I will tell you as soon as there is news. Nothing is {ordered} yet.", None


class TaskRefused(ValueError):
    """The request breaks a rule (guard.check_request); nothing was started."""


async def create(
    session: AsyncSession, *, family_id: str, subject_id: str, requested_by: str, service: str, kind: str, goal: str, details: dict,
    limits: Limits | None = None,
) -> Task:
    if service not in SKILLS:
        raise ValueError(f"unknown service {service}")
    agent = specialist_for(service)
    # One live task per service per family: a retry or a repeated ask never starts a second order.
    for t in await live_tasks(session, family_id):
        if t.service == service and t.kind == kind:
            return t
    limits = limits or Limits()
    if kind == "order" and details.get("items"):
        details = {**details, "items": [clean_item(i) for i in details["items"] if isinstance(i, dict)]}
    verdict = guard.check_request(kind, agent.name, details.get("items") or [], limits, pickup=details.get("pickup"), drop=details.get("drop"))
    if not verdict.ok:
        raise TaskRefused("; ".join(verdict.block))
    if "code_from" not in details:
        from app.care import boundaries

        # the person whose number the store logs in with, so a login code reaches them (the family's setting)
        details = {**details, "code_from": boundaries.code_person(await boundaries.get(session, family_id), subject_id, requested_by)}
    if practice.family_listed(family_id):
        details = {**details, "practice": True}  # goes up to the store's last step and stops (app/tasks/practice.py)
    now = clock.now()
    task = Task(
        id=uuid.uuid4(), family_id=family_id, subject_id=subject_id, requested_by=requested_by, service=service, kind=kind,
        goal=goal, details={**details, "contract": CONTRACT_VERSION, "agent": agent.name, "limits": limits.to_dict()},
        status="queued", phase="prepare", created_at=now, updated_at=now, deadline_at=now + TASK_LIFETIME, history=[], result={},
        trace_id=trace.current(),  # the message that asked for it: everything the task does is filed under it (D-13)
    )
    metrics.on_create(task, agent.name)
    session.add(task)
    await session.flush()
    await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind="task_started", summary=f"{SKILLS[service]['label']}: {goal}",
        payload={"task_id": str(task.id), "agent": agent.name}, actor_id=requested_by,
    )
    return task


async def shown_tasks(session: AsyncSession, family_id: str) -> list[Task]:
    """What the brain sees as the family's tasks: the live ones and the ones set for later."""
    q = select(Task).where(Task.family_id == family_id, Task.status.in_((*LIVE, "scheduled"))).order_by(Task.created_at)
    return list((await session.execute(q)).scalars())


async def live_tasks(session: AsyncSession, family_id: str) -> list[Task]:
    q = select(Task).where(Task.family_id == family_id, Task.status.in_(LIVE)).order_by(Task.created_at)
    return list((await session.execute(q)).scalars())


def bought(task: Task) -> str:
    """What was actually bought (the cart as the store showed it, else what was picked), and when it is not exactly what
    was asked, which ask it stands for (founder 2026-10-12: confirms, outcomes, tracking and after-order questions name what
    was bought, never the words of the request; live, a 1mg order was announced as "Magnesium Citrate tablets" though the
    cart held "Calcium and Vitamin D3 … Magnesium & Alfalfa")."""
    r, d = task.result or {}, task.details or {}
    if task.kind == "ride":
        return f"{d.get('choice') or 'a ride'} {_trip(task)}"
    lines = [i for i in r.get("items") or [] if isinstance(i, dict) and i.get("name")] if (d.get("cart_fp") or r.get("placed")) else []
    asked = d.get("items") or []
    rows = (["{} x {}".format(i.get("qty", 1), _label(i)) for i in lines[:12]] or
            ["{} x {}".format(channels._qty_for(asked, c), _label(c)) for c in (d.get("chosen") or [])[:12] if isinstance(c, dict)])
    one = str(asked[0].get("name") or "") if len(asked) == 1 and isinstance(asked[0], dict) else ""
    near = [f"{_label(c)} is not exactly '{c.get('for_item') or one or 'what was asked'}'" for c in d.get("chosen") or []
            if isinstance(c, dict) and c.get("exact_match") is False]
    return "; ".join(rows) + (f" (NOT exactly what was asked: {'; '.join(near)})" if near else "") if rows else task.goal


def describe(task: Task) -> str:
    r = task.result or {}
    d = task.details or {}
    label = SKILLS[task.service]["label"]
    placed = bool(r.get("placed") or r.get("booked"))
    if task.kind == "order" and (placed or (task.phase != "browse" and (d.get("cart_fp") or r.get("total")))):
        # Name what was bought (or is in the cart), never the words of the request (founder 2026-10-12)
        head = (f"[{task.id}] {label} order: {'BOUGHT' if placed else 'in the cart'}: {bought(task)} (their ask, in their words: "
                f"{task.goal}; name the products above, never the ask)")
    else:
        head = f"[{task.id}] {label} {task.kind}: {task.goal}" + (f", booked: {d.get('choice')}" if task.kind == "ride" and placed and d.get("choice") else "")
        if task.kind == "ride":
            # where the app set the ends, if it showed them (nothing else may be claimed about the address)
            head += f"; {_trip(task)}" + ("" if (r.get("pickup_used") or not placed) else " [the app has not shown the pickup it set]")
    # whom it is for (founder 2026-10-12): never tell or ask the care recipient about someone else's own order
    whose = (f"for person {task.subject_id}, asked by person {task.requested_by}" if task.subject_id != task.requested_by
             else f"person {task.subject_id}'s own")
    bits = [head, whose, f"status {task.status}/{task.phase}"]
    if d.get("channel"):
        bits.append(f"via {d['channel']}")
    if task.input_needed:
        bits.append(f"waiting for {task.input_needed}")
    if d.get("items") and task.kind == "order":
        bits.append("asked: " + "; ".join(f"{n}. {i.get('name')} x{i.get('qty', 1)}" for n, i in enumerate(d["items"], 1) if isinstance(i, dict)))
    in_cart = task.phase != "browse" and bool(d.get("cart_fp") or r.get("total"))
    offered = task.input_needed in ("go", "choice") and d.get("offer") and (task.kind == "ride" or task.phase == "browse")
    if offered:
        # The numbered options the person was sent: their pick ("the second one") is task_input go / choice with that number.
        bits.append(f"options sent, waiting for their pick (task_input {task.input_needed}, value = the number): " + "; ".join(
            f"{n}. {o.get('text')}" + (" (suggested)" if o.get("s") else "") for n, o in list(d["offer"].items())[:12]))
    elif d.get("chosen") and not in_cart and task.kind == "order" and task.phase != "browse":
        # What they picked, and only that (live 2026-10-11: with every product found listed here, the brain read them out
        # as options again). The one confirm names it with the total; they can change it there.
        asked = d.get("items") or []
        bits.append("picked (not in a cart yet; the one confirm with the total comes next): " + "; ".join(
            f"{channels._qty_for(asked, c)} x {_label(c)} {c.get('price') or ''}".strip() for c in d["chosen"][:8]))
    elif r.get("items") and not (task.phase == "browse" and compare_group(task)):
        bits.append(("cart: " if in_cart else "found (not in a cart yet): ") + "; ".join(f"{i.get('qty', 1)} x {i.get('name')} {i.get('price', '')}".strip() for i in r["items"][:12]))
    if r.get("alternatives"):
        bits.append("alternatives: " + "; ".join(map(str, r["alternatives"][:5])))
    if r.get("options") and not offered:
        bits.append("ride options: " + "; ".join(f"{o.get('type')} {o.get('fare')}" + (f" (pickup in {o['eta']})" if o.get("eta") else "") for o in r["options"][:6]))
    if r.get("surge"):
        bits.append("SURGE pricing")
    if r.get("total"):
        bits.append(f"total {r['total']}" + (f" incl. fees {r['fees']}" if r.get("fees") else ""))
    if r.get("eta") and task.status == "done":
        bits.append(f"eta {r['eta']}")
    if r.get("order_id") or r.get("ride_id"):
        bits.append(f"id {r.get('order_id') or r.get('ride_id')}")
    lt = d.get("later") or {}
    if lt:
        at, book_at, login_at = later.parse_at(lt.get("at")), later.parse_at(lt.get("book_at")), later.parse_at(lt.get("login_at"))
        most = (f"up to ₹{lt['max']:.0f} without asking (asks above about ₹{later.cap(lt['max']):.0f})" if lt.get("max") else
                "no price limit (they said any amount)" if lt.get("no_max") else "the most they will pay not set yet")
        bits.append(f"SET FOR LATER: for {later.hm(at)}, {'booked' if task.kind == 'ride' else 'ordered'} at {later.hm(book_at)}; {most}; "
                    + {"waiting": f"login code to be asked on {later.hm(login_at)}" if login_at else "waiting for its time",
                       "logged_in": "logged in; waiting for its time", "login": "logging in now (a code may be asked)",
                       "check": "looking at the fare now", "ready": "fare agreed; it books at its time",
                       "fare_ask": "asked them about the new fare", "booking": "booking it now", "missed": "the time passed; not booked"}
                    .get(lt.get("state"), "waiting for their yes to all of it (task_input later)"))
    ask = d.get("cancel_ask") or {}
    if ask:
        # One true status: what is true about a cancel they asked for after it was placed (founder 2026-10-11 and -12).
        label = SKILLS[task.service]["label"]
        app = "the caregiver is told to cancel it in the app" if ask.get("elder") else "they can cancel it in the app"
        bits.append({"pending": f"they asked to cancel it while it was being placed: {label} is asked to cancel it once placed (NOT cancelled yet)",
                     "cancelling": f"they asked to cancel it: asking {label} to cancel it now (NOT cancelled yet)",
                     "fee": f"{label} wants {d.get('cancel_fee')} to cancel it: waiting for their yes or no (NOT cancelled)",
                     "cancelled": f"cancelled on {label} at their request",
                     "kept": "they kept it (no cancellation fee paid)",
                     "refused": f"{label} would not cancel it: NOT cancelled; they can cancel it in the {label} app",
                     "told_carers": "the care recipient asked to cancel it after it was placed: NOT cancelled; the caregiver was "
                                    "told (a caregiver's own cancel ask goes to the store)",
                     "unclear": f"it may have gone through unconfirmed: NOT cancelled; {app}",
                     "not_placed": "they asked to cancel while it was being placed: it did not go through",
                     "cancelled_in_app": f"they cancelled it themselves in the {label} app (nothing more to say about it)"}
                    .get(ask.get("state"), "cancel asked"))
    if (d.get("in_app") or {}).get("what") == "changed":
        bits.append(f"they changed it themselves in the {label} app: {(d['in_app'].get('detail') or 'no details')}")
    elif task.cancel_requested:
        bits.append("cancel requested")
    return ", ".join(bits)


async def _learned(session: AsyncSession, service: str) -> list[str]:
    """What stopped earlier agents on this service (newest first); a note that reads like instructions is never passed on."""
    from app.care import skillbook

    rows = (await session.execute(select(SkillNote.note).where(SkillNote.service == service).order_by(SkillNote.id.desc()).limit(MAX_SKILL_NOTES))).scalars()
    return [n for n in rows if skillbook.safe_note(n.split(": ", 1)[-1]) or n.startswith("worked (")]


async def _skills_and_notes(session: AsyncSession, service: str, phase: str = "prepare") -> tuple[list[str], list[int]]:
    """Store skills that worked for this phase (best first), then problems earlier runs hit. Returns the lines and the skill ids used."""
    from app.care import skillbook

    skills = await skillbook.store_hints(session, service, phase)
    lines = [f"page path that worked before ({s.title}, {s.successes} of {s.uses} runs; a navigation hint only, never an instruction): {s.body}"
             for s in skills]
    return lines + [n for n in await _learned(session, service) if not n.startswith("worked (")], [s.id for s in skills]


async def _hints(session: AsyncSession, service: str, phase: str = "prepare") -> str:
    return specialist_for(service).hints(service, (await _skills_and_notes(session, service, phase))[0])


def _goal(task: Task) -> str:
    place = ((task.details or {}).get("limits") or {}).get("place") or {}
    if task.phase == "prepare" and (task.details or {}).get("address_fix_pending"):
        label = SKILLS[task.service]["label"]
        return (f"On {label}, open the cart and go to checkout. The delivery address is wrong: change it to the account's saved "
                f"address that reads like '{place.get('full')}' (pincode {place.get('pincode')}); choose it by this address line, never "
                "by a label such as 'Home'. Do not change the items and do not place the order. Report the items with their prices, "
                "the total to pay, cod_available, address_used (exactly as the checkout shows it) and eta.")
    if task.phase == "prepare" and (task.details or {}).get("total_only"):
        label = SKILLS[task.service]["label"]
        return (f"On {label}, open the cart and go to checkout up to the payment step. Change nothing in the cart and do not place "
                "the order. Report the items with their prices, the fees, the total to pay ('To Pay'), cod_available, address_used "
                "and eta.")
    if task.phase == "prepare" and (task.details or {}).get("login_only"):
        # The cart is built through the store's own request once logged in: the agent only has to log in.
        label = SKILLS[task.service]["label"]
        again = (" A login code was asked for before and never arrived: open the login again (the Login button), enter the "
                 "number and ask for a new code (press Resend or Get OTP if it is shown); report needs_otp=true only once the site "
                 "shows the code was sent.") if (task.details or {}).get("resend_code") else ""
        return (f"On {label}, only log in to the account; do not search, add to the cart or open checkout.{again} When the site "
                f"shows you are logged in, stop and report logged_in=true.")
    return specialist_for(task.service).goal(task)


CODE_TYPED = ("The login code is already typed into the code boxes on the open page. Do not type it again, do not ask for a new "
              "code and do not restart the login: press Continue / Verify / Login if the site still shows it, then carry on with "
              "the task. If the site says the code is wrong or has expired, stop and report needs_otp=false and "
              "problem='code not accepted'.")
CODE_BY_HAND = ("Enter this login code where the site asks for it: {code}. Click the first code box and type the digits one key at a "
                "time (the site moves to the next box itself); never paste it or set it with a script. Then carry on with the "
                "task. Never ask for a new code and never restart the login yourself: if the boxes do not take it, stop and report "
                "needs_otp=false and problem='code not accepted'.")
CODE_TRIES = 2  # login codes that did not go in before it stops and says so (each one is a code the person had to send)


async def _type_code(agent: BrowserAgent, task: Task, code: str) -> bool:
    """The code typed into the open login page of this task's browser (app.tasks.codeentry); False lets the agent type it."""
    from app.tasks import codeentry
    from app.tasks.browser_use import DOMAINS

    if not task.agent_session or not hasattr(agent, "cdp_url") or os.getenv("TYPE_LOGIN_CODES", "on") != "on":
        return False
    try:
        cdp = await agent.cdp_url(task.agent_session)
    except Exception:  # noqa: BLE001
        return False
    hosts = tuple(d.removeprefix("*.") for d in DOMAINS.get(task.service, ()))
    return bool(cdp) and await codeentry.type_code(cdp, code, hosts=hosts)


def _guest_run(task: Task) -> bool:
    """The task's browser is a look-up's own, with no login: nothing can be booked or placed in it."""
    d = task.details or {}
    return bool(task.agent_session) and d.get("guest_session") == task.agent_session


async def _drop_session(agent: BrowserAgent, task: Task, why: str) -> None:
    sid, task.agent_session = task.agent_session, None
    note(task, why)
    if sid and hasattr(agent, "stop_session"):
        try:
            await agent.stop_session(sid)
        except Exception:  # noqa: BLE001 — the sweeper stops a browser nobody uses
            logger.warning("stop session failed task=%s", task.id)


async def _start_run(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for: Callable[[Task], Awaitable[str | None]], extra: str = "") -> None:
    spec = specialist_for(task.service)
    goal = _goal(task) + (f"\n{extra}" if extra else "")
    d0 = task.details or {}
    if task.phase in guard.BEFORE_CONFIRM:
        goal += "\n" + BEFORE_CONFIRM_RULE + (f"\n{d0['guard_extra']}" if d0.get("guard_extra") else "")
    learned, skill_ids = await _skills_and_notes(session, task.service, task.phase)
    task.details = {**(task.details or {}), "skills_used": skill_ids}
    profile_id = None
    got = None
    guest = task.phase == "browse"
    if guest and task.agent_session and not _guest_run(task):
        # A look-up never runs where the family is logged in (founder 2026-10-12: a look-up with a live login can book):
        # the fast look-up's browser is on the family's profile, so the agent looks in a browser of its own.
        await _drop_session(agent, task, "the look-up runs in a browser without the family's login")
    elif not guest and _guest_run(task):
        await _drop_session(agent, task, "the look-up's browser had no login: this step opens the family's own")
    if guest:
        pass  # no profile: not logged in anywhere, so nothing can be booked or placed in this run
    else:
        task.details = {**(task.details or {}), "login_at_start": await sandbox.login_live(session, task.family_id, task.service)}
    if guest:
        pass
    elif not task.agent_session:
        got = await profile_for(task)
        # The host answers {profileId, loginPhone}; older callers (bench, tests) give just the profile id.
        profile_id = got.get("profileId") if isinstance(got, dict) else got
        await sandbox.bind_profile(session, task.family_id, task.service, profile_id)  # raises if it belongs to someone else
    def login_line(got: object) -> str:
        if not isinstance(got, dict):
            return ""
        phone = "".join(c for c in str(got.get("loginPhone") or "") if c.isdigit())[-10:]
        # Only the last 4 digits are kept on the task (for "the code went to the phone ending 1234").
        task.details = {**(task.details or {}), "login": phone[-4:] if len(phone) == 10 else "none"}
        return "\n" + (
            f"If the site asks you to log in: enter the mobile number {phone} (India, +91), ask for the code, then stop and report "
            f"needs_otp=true and otp_sent_to='phone ending {phone[-4:]}'." if len(phone) == 10 else
            "There is no mobile number for this account: if the site asks you to log in, do not try. Stop and report "
            "logged_in=false, needs_otp=false and problem='login needed'."
        )

    if task.phase == "prepare" and not extra:  # with a code to enter (extra), it must not ask for a new one
        if got is None:
            got = await profile_for(task)
        goal += login_line(got)
    if (task.phase == "prepare" and (task.details or {}).get("browsed") and task.kind == "order" and not (task.details or {}).get("login_only")
            and not (task.details or {}).get("total_only") and not (task.details or {}).get("address_fix_pending")):
        # The products picked at the go-ahead (one per item asked); older tasks: everything the look-up found.
        seen = (task.details or {}).get("chosen") or [i for i in ((task.result or {}).get("items") or []) if i.get("name")]
        if seen and SKILLS[task.service].get("kind") == "food":
            # One restaurant per cart: the agent opens that restaurant and adds these dishes with their quantities.
            asked = (task.details or {}).get("items") or []
            rest = next((str(i["restaurant"]) for i in seen if i.get("restaurant")), "")
            goal += ((f"\nOrder everything from ONE restaurant: {rest}. Open that restaurant's menu (search its name) and add exactly: " if rest
                      else "\nAdd exactly: ")
                     + "; ".join(f"{channels._qty_for(asked, i)} x {_label(i)} {i.get('price') or ''}".strip() for i in seen[:8]))
            find = [a for a in (task.details or {}).get("menu_find") or [] if a]
            if find:
                goal += (". Also find in the same restaurant's menu and add: " + "; ".join(
                    f"{next((int(x.get('qty') or 1) for x in asked if str(x.get('name')) == a), 1)} x {a}" for a in find)
                         + " (the plainest everyday version; if the menu has none, add nothing for it and report it missing)")
            goal += ". Nothing else; never items from another restaurant (if the cart has another restaurant's items, replace them)."
        elif seen:
            goal += ("\nThe person chose these exact products: " + "; ".join(f"{_label(i)} {i.get('price') or ''}".strip() for i in seen[:8])
                     + ". Put exactly these in the cart (same pack size), nothing else.")
            pages = [product_page(task.service, i) for i in seen[:8]]
            if all(pages):
                # Straight to each product's own page (founder 2026-10-10: the fallback took 5-6 minutes searching).
                goal += " Open each product's own page and add it there (no searching): " + "; ".join(
                    f"{_label(i)} → {u}" for i, u in zip(seen[:8], pages)) + "."
    if task.phase != "prepare" and isinstance(got, dict) and got.get("loginPhone"):
        phone = "".join(c for c in str(got.get("loginPhone")) if c.isdigit())[-10:]
        task.details = {**(task.details or {}), "login": phone[-4:] if len(phone) == 10 else "none"}
    async def start(session_id: str | None, profile: str | None) -> AgentRun:
        return await agent.run(
            goal=goal,
            hints=spec.hints(task.service, learned),
            schema=spec.schema(task),
            session_id=session_id,
            profile_id=profile,
            # a browser opened for a fast look-up has no store tab of its own: the agent starts on the store's page
            start_url=None if session_id and not (task.details or {}).get("fast_session") else SKILLS[task.service]["start_url"],
            # a practice place also opens the payment step and empties the cart afterwards
            max_steps=spec.steps.get(task.phase, 40) + (15 if task.phase == "place" and practice.on(task) else 0),
            metadata={"app": "kavach", "task": str(task.id), "phase": task.phase, "service": task.service, "agent": spec.name},
            llm=spec.model,
            flash=task.phase in flash_phases(),
        )

    try:
        run = await start(task.agent_session, profile_id)
    except RuntimeError as exc:
        # The browser of the last step already closed (live 2026-10-09: the look-up's browser was gone when the go-ahead
        # came, and the login step failed three times). Start a fresh one on the family's profile, which keeps the login.
        if not task.agent_session or "session is stopped" not in str(exc).lower():
            raise
        note(task, "the earlier browser had closed; starting a fresh one")
        task.agent_session = None
        if not guest:
            got = got if got is not None else await profile_for(task)
            profile_id = got.get("profileId") if isinstance(got, dict) else got
            await sandbox.bind_profile(session, task.family_id, task.service, profile_id)
        if extra:
            # The code was for a login page that is gone: typing it into a new one fails (live 2026-10-09 the agent typed
            # the code as the phone number). Log in again; the store sends a new code and the person is told why.
            goal = _goal(task) + login_line(got)
            task.details = {**(task.details or {}), "code_lost": True, "otp_entered": False}
        run = await start(None, profile_id)
    # A browser now exists and bills: record it (and that the task is running on it) and commit before anything else,
    # so a failure later in this tick cannot roll back the only record of it (the sweeper stops what the ledger knows).
    task.agent_session, task.agent_task = run.session_id, run.task_id
    task.status, task.input_needed, task.runs = "running", None, task.runs + 1
    task.details = {**(task.details or {}), "run_started": clock.now().isoformat(), "channel": "browser",
                    **({"guest_session": run.session_id} if guest else
                       {"login_runs": int((task.details or {}).get("login_runs") or 0) + 1} if task.phase in guard.BEFORE_CONFIRM else {})}
    await sandbox.started(session, session_id=run.session_id, task_id=str(task.id), family_id=task.family_id, service=task.service,
                          profile_id=profile_id)
    await session.commit()
    note(task, f"started {task.phase} run ({spec.name} agent, browser)")




async def _save_usual_vehicle(session: AsyncSession, task: Task, vehicle: str | None, *, by: str, by_is_elder: bool) -> None:
    """The ride type they picked on this app, saved for the rider: next time it is the suggestion among the fares."""
    if not vehicle or task.kind != "ride":
        return
    label = SKILLS[task.service]["label"]
    try:
        await store.write_fact(session, family_id=task.family_id, subject_id=task.subject_id, domain="preference",
                               key=f"ride_vehicle:{task.service}", value={"service": label, "vehicle": str(vehicle)[:40], "task_id": str(task.id)},
                               text=f"Takes {vehicle} on {label}", source_kind="elder_said" if by_is_elder else "caregiver_said",
                               stated_by=by, confidence=0.9)
    except Exception:  # noqa: BLE001 — the ride goes on; only next time's suggestion is lost
        logger.warning("ride preference not saved task=%s", task.id)


def _ride_fp(cart_fp: str | None, choice: str) -> str:
    return f"{cart_fp or ''}:{re.sub(r'[^a-z0-9]+', '', (choice or '').lower())}"


def _match_option(options: list[dict], value: str) -> dict | str | None:
    """The option the person named: an exact name first; a partial name only if it fits exactly one option.
    Returns "ambiguous" when it could be more than one."""
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", str(x or "").lower()).strip()  # noqa: E731
    v = norm(value)
    if len(v) < 2:
        return None
    exact = [o for o in options if norm(o.get("type")) == v or norm(f"{o.get('type')} {o.get('fare') or ''}") == v]
    if len(exact) == 1:
        return exact[0]
    partial = [o for o in options if norm(o.get("type")) and (norm(o.get("type")) in v or v in norm(o.get("type")))]
    if len(partial) == 1:
        return partial[0]
    return "ambiguous" if partial or len(exact) > 1 else None


def _groups(task: Task, found: list[dict]) -> list[tuple[str, list[dict]]]:
    """(asked item, its products): one group per asked item when the look-up tagged them (for_item), else one group."""
    asked = [str(i.get("name") or "") for i in ((task.details or {}).get("items") or [])]
    if not any(i.get("for_item") for i in found):
        return [(asked[0] if len(asked) == 1 else "", found)] if found else []
    keyed: dict[str, list[dict]] = {}
    for i in found:
        keyed.setdefault(str(i.get("for_item") or ""), []).append(i)
    return list(keyed.items())


def _picked(task: Task, pick: list[dict] | None) -> tuple[list[dict], list[str]]:
    """The cart's products after the person's pick: per asked item what they picked, else Saheli's suggestion for it, else
    the AI's first exact listing (food: only from the restaurant they picked). Also the asked items left without a product
    (not in the cart; the one confirm says so, or that restaurant's menu is searched for them)."""
    d = task.details or {}
    matched = d.get("matched")
    sugg = d.get("chosen") or []
    refused = set(d.get("refused") or [])
    food = SKILLS[task.service].get("kind") == "food"
    rests = {str(p.get("restaurantId") or p.get("restaurant") or "") for p in pick or []}
    out, left = [], []
    for asked, g in _groups(task, _found(task)):
        keys = {_pkey(i) for i in g}
        exact = set(((matched or {}).get(asked) or {}).get("exact") or [])
        mine = [p for p in pick or [] if _pkey(p) in keys]
        if mine:
            out += [p if matched is None else {**p, "exact_match": _pkey(p) in exact} for p in mine]
            continue
        s = [c for c in sugg if _pkey(c) in keys]
        if not s:  # no suggestion for this item: the AI's first exact listing (one the store has not refused)
            s = [{**i, "exact_match": True} for k in ((matched or {}).get(asked) or {}).get("exact") or [] for i in g
                 if _pkey(i) == k and k not in refused][:1]
        if food and pick:
            s = [c for c in s if str(c.get("restaurantId") or c.get("restaurant") or "") in rests]
        if s:
            out += s
        elif asked:
            left.append(asked)
    return out, left


async def _option_elsewhere(session: AsyncSession, task: Task, tokens: list[str]) -> str | None:
    """A number the person picked that belongs to another store of the same comparison: which task to call."""
    if not compare_group(task) or not tokens:
        return None
    for t in await siblings(session, task):
        offer = (t.details or {}).get("offer") or {}
        if t.id != task.id and t.status in LIVE and all(x in offer for x in tokens):
            return (f"option {', '.join(tokens)} is on {SKILLS[t.service]['label']}'s list: call task_input go on task {t.id} with "
                    f"value {', '.join(tokens)}")
    return None


def _offer_instead(task: Task, alts: list[dict], why: str) -> str:
    """The cart could not take what was picked (out of stock): the store's other options for it and what it has instead
    (marked similar) go back to the person, numbered; they pick again."""
    d = task.details or {}
    gone = {_pkey(c) for c in d.get("chosen") or []}
    names = "; ".join(_label(c) for c in d.get("chosen") or []) or "what they picked"
    one = [str(i.get("name")) for i in d.get("items") or [] if isinstance(i, dict)]
    extra = [{"name": a["name"], "price": a.get("price"), "available": True, "exact_match": False,
              **({"for_item": one[0]} if len(one) == 1 else {})} for a in alts]
    found = [i for i in d.get("found") or [] if _pkey(i) not in gone] + extra
    task.result = {**(task.result or {}), "items": found, "total": None, "fees": None, "alternatives": None}
    task.details = {**d, "refused": sorted(gone | set(d.get("refused") or [])), "chosen": None, "kept": None, "offer": None,
                    "cart_fp": None, "confirm_token": None, "fast_cart_tried": False, "menu_find": None, "missing_items": None,
                    "found": found[:40], "auto_picked": None, "browsed": True}  # the pick then goes to the cart as picked
    task.phase, task.input_needed = "browse", "go"
    note(task, f"{names} could not go in the cart ({why}); the other options and what the store has instead go back to them")
    return _options_message(task, task.result, listing=True, head=(
        f"{names} could not go in the cart on {SKILLS[task.service]['label']} ({why}); nothing is ordered. Ask them to pick "
        "again; the ones marked similar are what the store has instead. "))


def _reoffer_self(task: Task, gone: list[str], why: str) -> str | None:
    """A product the person picked themselves turned out not to be available at the store: nothing is swapped for them;
    the store's other options go back to them to pick from. None when it was Saheli's own suggestion (the next exact one
    is tried, _repick_without) or nothing else is left."""
    d = task.details or {}
    if d.get("auto_picked") or not d.get("found") or task.kind != "order":
        return None
    refused = sorted(set(map(str, gone)) | set(d.get("refused") or []))
    names = [_label(c) for c in d.get("chosen") or [] if _pkey(c) in set(map(str, gone))] or ["the product they picked"]
    rest = [c for c in d.get("chosen") or [] if _pkey(c) not in set(refused)]
    saved = (task.result, task.details, task.phase, task.input_needed)
    task.result = {**(task.result or {}), "items": d["found"], "total": None, "fees": None, "alternatives": None}
    # the items whose product is still fine keep it; only the refused one is offered again
    kept = {asked: c for asked, g in _groups(task, _found(task)) if asked for c in rest if _pkey(c) in {_pkey(i) for i in g}}
    task.details = {**d, "refused": refused, "chosen": rest or None, "kept": kept or None, "cart_fp": None, "confirm_token": None,
                    "fast_cart_tried": False, "menu_find": None, "missing_items": None, "offer": None}
    task.phase, task.input_needed = "browse", "go"
    if not _offers(task):
        task.result, task.details, task.phase, task.input_needed = saved
        return None
    note(task, f"{'; '.join(names)} not available ({why}); its other options go back to them")
    return _options_message(task, task.result, head=f"{'; '.join(names)}, which they picked, turned out to be {why} on "
                                                    f"{SKILLS[task.service]['label']}; nothing is ordered. Ask them to pick another. ")


EDIT_KINDS = ("more", "change", "add", "remove")
async def _relook(session: AsyncSession, task: Task, d: dict, new_items: list[dict], kind: str, value: str, by: str) -> str:
    v = guard.check_request("order", d.get("agent") or "shopping", new_items, Limits.from_dict(d.get("limits")))
    if not v.ok:
        return "cannot do that: " + "; ".join(v.block)
    task.details = {**d, **RELOOK, "items": [clean_item(i) for i in new_items], "kept": _kept(d, new_items), "more": None, "alternatives": None,
                    "connector_card": None, "channel": None}  # a linked store's connector is tried again for the new items
    task.result = {}
    task.phase, task.status, task.input_needed = "prepare", "queued", None
    task.deadline_at = clock.now() + TASK_LIFETIME
    note(task, f"{kind} '{value}' by {by}")
    names = ", ".join(str(i.get("name")) for i in new_items)
    return (f"getting it now: {names} (NOT ordered); the new cart and total come back in a task update for one confirm. "
            "Say only that you are updating it")
LADDER_RESET = {"ladder_asked_at": None, "ladder_emailed_at": None, "ladder_carers": None, "ladder_due": None}
# What a new look-up clears (the products, the cart and anything confirmed for it).
RELOOK = {"browsed": False, "chosen": None, "cart_fp": None, "connector_card": None, "fast_cart_tried": False, "total_relook": False,
          "cart_cleaned": False,
          "total_only": False, "menu_find": None,
          "fast_place_check": None, "confirm_token": None, "missing_items": None, "auto_picked": None, "matched": None,
          "refused": None, "repicks": None, "found": None, "kept": None, "offer": None}


def _kept(d: dict, new_items: list[dict], skip: str | None = None) -> dict | None:
    """A change at the confirm: the products already picked for the items that stay the same (by item name) are kept and
    not offered again; only what changed is."""
    cur = [str(i.get("name") or "") for i in d.get("items") or [] if isinstance(i, dict)]
    chosen = [c for c in d.get("chosen") or [] if isinstance(c, dict)]
    by = ({cur[0]: chosen[0]} if len(cur) == 1 and len(chosen) == 1 else
          {str(c["for_item"]): c for c in chosen if c.get("for_item")})
    names = {str(i.get("name") or "") for i in new_items}
    return {n: c for n, c in by.items() if n and n in names and n != skip} or None


APPROVAL_WAIT = timedelta(minutes=90)
MORE_LIMIT = 40  # products looked up for an item they asked to see more of (24 otherwise)


def _eta_wait(task: Task) -> timedelta:
    """When to ask whether it arrived: the store's ETA plus a margin, else a sensible default per kind."""
    eta = str((task.result or {}).get("eta") or "").lower()
    m = re.search(r"(\d+)\s*(?:-|to)?\s*(\d+)?\s*(min|mins|minute|minutes|hr|hrs|hour|hours)", eta)
    if m:
        n = int(m.group(2) or m.group(1))
        unit = timedelta(hours=1) if m.group(3).startswith("h") else timedelta(minutes=1)
        return n * unit + (timedelta(minutes=10) if task.kind == "ride" else timedelta(minutes=20))
    if task.kind == "ride":
        return timedelta(minutes=20)
    if specialist_for(task.service).name == "pharmacy" or re.search(r"(mon|tue|wed|thu|fri|sat|sun|tomorrow|\d{1,2} \w{3})", eta):
        return timedelta(hours=26)
    return timedelta(minutes=60)


async def delivery_followup(session: AsyncSession, task: Task) -> None:
    """After an order or ride is placed: the follow-up that asks whether it came, and later how it was (app.tasks.aftercare)."""
    from app.tasks import aftercare

    await aftercare.open_followup(session, task)


async def _approval_gate(session: AsyncSession, task: Task, amount: float | None, by: str, choice: dict | None = None) -> str | None:
    """None when this may be placed now; otherwise the task waits for an approver (the family's boundaries)."""
    from app.care import boundaries

    d = task.details or {}
    policy = await boundaries.get(session, task.family_id)
    roster = await store.roster(session, task.family_id)
    allowed = boundaries.approvers(policy, roster)
    if not allowed:
        return None  # nobody but the person themselves (self care): nobody to ask, so nothing waits for an approval
    elder = ((roster.elder or {}).get("id") if roster else None) or task.subject_id
    why = await boundaries.reasons(session, family_id=task.family_id, service=task.service, kind=task.kind, amount=amount,
                                   requested_by=task.requested_by, elder_id=elder, policy=policy)
    # what the cart check found that needs a caregiver (the old fixed-budget line is now the family's own limit, above)
    why += [r for r in d.get("approval") or [] if "limit" not in r and r not in why]
    if not why or boundaries.may_approve(allowed, by, elder):
        return None
    names = {m.get("id"): (m.get("name") or "").split(" ")[0] for m in (roster.members if roster else [])}
    task.details = {**d, "approval_needed": {"reasons": why, "approvers": allowed, "asked_by": by, "amount": amount, "choice": choice,
                                             "asked_at": clock.now().isoformat(), "notified": []}}
    task.input_needed = "approve"
    task.deadline_at = clock.now() + APPROVAL_WAIT
    note(task, "waiting for approval: " + "; ".join(why))
    who = ", ".join(names.get(a) or a for a in allowed) or "the caregiver"
    return (f"needs approval first: {'; '.join(why)}. The request goes to {who} now; nothing is placed until they say yes. "
            "Tell the person that in one short line.")


def _later_amount(task: Task) -> float | None:
    """Today's fare (the type picked) or the picked items' prices times quantities (or the cart's total once built)."""
    d, r = task.details or {}, task.result or {}
    if task.kind == "ride":
        return rupees((d.get("ride_pick") or {}).get("fare"))
    if task.phase != "browse" and d.get("cart_fp") and rupees(r.get("total")):
        return rupees(r.get("total"))
    asked = d.get("items") or []
    total = sum((rupees(c.get("price")) or 0) * channels._qty_for(asked, c) for c in d.get("chosen") or [])
    return total or None


def _later_ask(task: Task) -> str:
    """The ONE question for a booking set for later (later.ask_text), and the task waits for it."""
    d = task.details or {}
    label = SKILLS[task.service]["label"]
    task.phase, task.status, task.input_needed = "browse" if task.phase == "browse" else task.phase, "awaiting_confirm", "later"
    task.deadline_at = clock.now() + TASK_LIFETIME
    if task.kind == "ride":
        pick = d.get("ride_pick") or {}
        what, where = f"{pick.get('type') or 'the ride'}", _trip(task)
    else:
        asked = d.get("items") or []
        what = "; ".join(f"{channels._qty_for(asked, c)} x {_label(c)} {c.get('price') or ''}".strip() for c in d.get("chosen") or []) or bought(task)
        where = "to " + (((d.get("limits") or {}).get("place") or {}).get("nickname") or "their saved place")
    return later.ask_text(task, label=label, what=what, amount=_later_amount(task), where=where)


async def _schedule(session: AsyncSession, task: Task) -> str:
    """Confirmed for later: the login is planned by how long the app keeps one, and the task waits (later.due moves it)."""
    lt = later.of(task)
    label = SKILLS[task.service]["label"]
    book_at = later.parse_at(lt.get("book_at"))
    now = clock.now()
    life = await later.login_life(session, task.service)
    since = await later.logged_in_since(session, task.family_id, task.service)
    login_at = later.plan_login(now, book_at, life, since)
    lt.update(login_at=login_at.isoformat() if login_at else None, life_h=round(life.total_seconds() / 3600, 1),
              state="waiting" if login_at else "logged_in")
    task.details = {**(task.details or {}), "later": lt, "offer": None}
    task.deadline_at = book_at + later.GIVE_UP_AFTER
    if compare_group(task):
        _drop_others(await siblings(session, task), task, f"set for later on {label}")
    if login_at and login_at <= now + timedelta(minutes=1):
        task.details = {**task.details, "later": {**lt, "state": "login", "login_started": now.isoformat()}, "login_only": True,
                        **LADDER_RESET, "ladder_from": now.isoformat()}
        task.phase, task.status, task.input_needed = "prepare", "queued", None
    else:
        task.status, task.input_needed = "scheduled", None
    note(task, f"set for {later.hm(book_at)}; login " + (f"at {later.hm(login_at)}" if login_at else "not needed") + f" (logins last ~{life})")
    return later.scheduled_text(task, label=label, login_at=login_at, logged=since is not None)


def _later_ready(task: Task, *, choice: str | None, fare, by: str) -> str:
    """The fare (or cart) for a later booking is agreed: it books at its time, or now if that time has come."""
    lt = later.of(task)
    d = task.details or {}
    book_at = later.parse_at(lt.get("book_at"))
    lt.update(state="ready", asked=None)
    total = fare if task.kind == "ride" else ((task.result or {}).get("total") or fare)
    task.details = {**d, "later": lt, **({"choice": choice} if choice else {}), "confirmed_total": total}
    if book_at and clock.now() >= book_at:
        lt.update(state="booking")
        task.details = {**task.details, "later": lt}
        return (_start_booking(task, lt.get("by") or by, choice or d.get("choice") or "", total) if task.kind == "ride"
                else _start_placing(task, lt.get("by") or by))
    task.status, task.input_needed = "scheduled", None
    note(task, f"ready for {later.hm(book_at)} at {total}")
    return f"agreed: it will be {'booked' if task.kind == 'ride' else 'ordered'} at {later.hm(book_at)} for about {total} (NOT yet). Say only that."


async def _later_step(session: AsyncSession, task: Task, status: str, message: str | None) -> tuple[str, str | None]:
    """A step of a task set for later has ended: the login (back to waiting), the fare check (ready, or ONE question with the
    new amount), or the question with everything when the fares or cart came only after a login."""
    lt = later.of(task)
    if not lt:
        return status, message
    d = task.details or {}
    label = SKILLS[task.service]["label"]
    book_at = later.parse_at(lt.get("book_at"))
    state = lt.get("state")
    if not lt.get("by") and status == "awaiting_confirm" and task.input_needed in ("confirm", "choice"):
        if task.kind == "ride" and not d.get("ride_pick"):
            return status, message  # the fares to pick from first; the pick goes into the one question
        return "awaiting_confirm", _later_ask(task)
    if state == "login" and status == "queued" and d.get("login_done") and not d.get("login_only"):
        lt.update(state="logged_in", login_done_at=clock.now().isoformat())
        task.details = {**d, "later": lt}
        return "scheduled", (f"Logged in on {label} for the booking on {later.hm(book_at)}; nothing more is needed now. Tell them in one "
                             "short line that it is all set.")
    if state == "login" and status == "needs_input" and task.input_needed == "otp":
        return status, (f"For the {label} booking set for {later.hm(book_at)}: logging in now, ahead of it, so it can be booked then "
                        "without asking anything. Say that in a few words with the code ask. " + (message or ""))
    if state in ("check", "booking") and status == "needs_input" and task.input_needed == "otp":
        since = lt.get("login_done_at") or lt.get("confirmed_at")
        return status, (f"For the booking on {later.hm(book_at)}: the {label} login ran out" + (f" (it was logged in on {later.hm(later.parse_at(since))})" if since else "")
                        + f", so {label} sent a new login code. Say why in a few words and ask for the code now. " + (message or ""))
    if state != "check" or status != "awaiting_confirm":
        return status, message
    mx = lt.get("max")
    limit = later.cap(mx)
    if task.kind == "ride":
        pick = d.get("ride_pick") or {}
        same = next((o for o in _ride_opts(task) if o.get("type") == pick.get("type")), None)
        if not same:
            return "awaiting_confirm", (f"For the booking on {later.hm(book_at)}: {pick.get('type')} is not offered on {label} now. "
                                        + _ride_choice_message(task))
        fare = rupees(same.get("fare"))
        choice = f"{same.get('type')} {same.get('fare') or ''}".strip()
        eta = re.search(r"\d+", str(same.get("eta") or ""))
        if eta and lt.get("means") != "book":
            # the app's own pickup time: booked that long before the time they asked for
            lt["book_at"] = max(later.book_time(later.parse_at(lt["at"]), "arrive", timedelta(minutes=int(eta.group()))), clock.now()).isoformat()
            book_at = later.parse_at(lt["book_at"])
    else:
        fare = rupees((task.result or {}).get("total"))
        choice = None
    if lt.get("no_max") and fare is not None:
        task.details = {**d, "later": lt}
        gate = await _approval_gate(session, task, fare, lt.get("by") or task.requested_by,
                                    choice={"choice": choice, "fare": same.get("fare")} if task.kind == "ride" else None)
        if gate:
            lt.update(state="approval")
            task.details = {**task.details, "later": lt}
            task.deadline_at = book_at + later.GIVE_UP_AFTER  # the approver's yes counts until its time, not later
            return "awaiting_confirm", (f"For the booking on {later.hm(book_at)}: it is {'₹%.0f' % fare} now; {gate}")
    if lt.get("no_max") or (fare is not None and limit is not None and fare <= limit):
        task.details = {**d, "later": lt}
        task.input_needed = None
        _later_ready(task, choice=choice, fare=same.get("fare") if task.kind == "ride" else (task.result or {}).get("total"),
                     by=lt.get("by") or task.requested_by)
        return task.status, None  # nothing to say: it books at its time
    shown = f"₹{fare:.0f}" if fare else "not shown"
    lt.update(state="fare_ask", asked={"choice": choice, "amount": shown})
    task.details = {**d, "later": lt}
    task.input_needed = "fare"
    return "needs_input", (f"For the booking on {later.hm(book_at)}: the {'fare for ' + choice.split(' ', 1)[0] if choice else 'total'} on {label} is "
                           f"{shown} now, above the ₹{mx:.0f} they set. Ask ONE short question at once: book it at {shown}? Yes → "
                           "task_input fare yes; no → task_input fare no (nothing is booked).")


def _start_placing(task: Task, by: str) -> str:
    d = task.details or {}
    fp = d.get("cart_fp")
    task.details = {**d, "confirmed_by": by, "confirmed_total": (task.result or {}).get("total"), "confirmed_fp": fp,
                    "confirmed_items_fp": guard.items_fingerprint(task.result or {}),
                    "confirm_token": guard.confirm_token(str(task.id), fp, by), "place_started": None}
    task.phase, task.status, task.input_needed = "place", "queued", None
    task.deadline_at = clock.now() + TASK_LIFETIME
    note(task, f"confirmed by {by}")
    return "confirmed; placing it now (NOT placed yet; you will get a task update when it is placed)"


def _start_booking(task: Task, by: str, choice: str, fare) -> str:
    d = task.details or {}
    fp = _ride_fp(d.get("cart_fp"), choice)
    task.details = {**d, "choice": choice, "confirmed_by": by, "confirmed_total": fare, "confirmed_fp": fp,
                    "confirm_token": guard.confirm_token(str(task.id), fp, by), "place_started": None}
    task.phase, task.status, task.input_needed = "place", "queued", None
    task.deadline_at = clock.now() + TASK_LIFETIME
    note(task, f"choice {choice} by {by}")
    return f"booking {choice} now (NOT booked yet; you will get a task update)"


async def ask_approvers(sessions: async_sessionmaker, notify) -> int:
    """Send each waiting approval request to the family's approvers (once each)."""
    async with sessions() as session:
        # locked: the scheduler's run and the background loop can overlap, and an approver is asked once
        rows = list((await session.execute(select(Task).where(Task.status == "awaiting_confirm", Task.input_needed == "approve")
                                           .with_for_update(skip_locked=True))).scalars())
        out = []
        for t in rows:
            ap = (t.details or {}).get("approval_needed") or {}
            todo = [a for a in ap.get("approvers") or [] if a != "*" and a not in (ap.get("notified") or [])]
            if not todo:
                continue
            t.details = {**t.details, "approval_needed": {**ap, "notified": list(ap.get("notified") or []) + todo}}
            amount = f" ₹{ap['amount']:.0f}" if isinstance(ap.get("amount"), (int, float)) else ""
            what = (ap.get("choice") or {}).get("choice") if t.kind == "ride" else (t.result or {}).get("total")
            for a in todo:
                out.append((t, a, (
                    f"[Approval needed] [{t.id}] Person {t.requested_by} asked for: {describe(t)}{amount}"
                    f"{f' ({what})' if what else ''}. It needs approval because: {'; '.join(ap.get('reasons') or [])}. Ask person {a} "
                    f"with send_message, short, in their language: approve or decline? When they answer, call task_input with "
                    f"task_id {t.id}, kind approve, value yes or no. Then reply none.")))
        await session.commit()
    for t, person, text in out:
        await _notify_about(notify, [t], t.family_id, person, text)
    return len(out)


async def provide_input(session: AsyncSession, task: Task, *, kind: str, value: str = "", by: str, by_is_elder: bool,
                        items: list[dict] | None = None, target: int | str | None = None, max_amount: float | None = None,
                        no_max: bool = False) -> str:
    """provide_input under the task's own trace (the turn that answered is kept as `via`), with an audit event for a
    confirm or an approval: who, when and for what total (D-13 why-trail, F4 audit of confirm / place / cancel)."""
    with trace.for_task(task):
        d0 = dict(task.details or {})
        out = await _provide_input(session, task, kind=kind, value=value, by=by, by_is_elder=by_is_elder, items=items, target=target,
                                   max_amount=max_amount, no_max=no_max)
        d = task.details or {}
        label = SKILLS[task.service]["label"]
        total = d.get("confirmed_total") or (d.get("approval_needed") or {}).get("amount") or (task.result or {}).get("total")

        async def audit(kind_: str, summary: str, **extra) -> None:
            await store.record_event(session, family_id=task.family_id, subject_id=task.subject_id, kind=kind_, actor_id=by,
                                     summary=summary[:300], payload={"task_id": str(task.id), "by": by, "total": total, "via": trace.via(),
                                                                     "input": kind, **extra})

        # an approval comes first: it also starts placing (a new confirm token), and the approver did not confirm the cart
        if d.get("approved_by") and d.get("approved_by") != d0.get("approved_by"):
            await audit("task_approved", f"{label}: approved by {by}" + (f" for {total}" if total else ""))
        elif d.get("approval_needed") and not d0.get("approval_needed"):
            # the person's yes that hit the family's approval rule: recorded now, placed only after the approver's yes
            await audit("task_confirmed", f"{label}: confirmed by {by}, waiting for approval" + (f" ({total})" if total else ""),
                        awaiting_approval=True)
        elif d.get("confirm_token") and d.get("confirm_token") != d0.get("confirm_token"):
            await audit("task_confirmed", f"{label}: confirmed by {by}" + (f" for {total}" if total else ""))
        return out


async def _provide_input(session: AsyncSession, task: Task, *, kind: str, value: str = "", by: str, by_is_elder: bool,
                         items: list[dict] | None = None, target: int | str | None = None, max_amount: float | None = None,
                         no_max: bool = False) -> str:
    """The person answered: an OTP, a confirm, a fee approval, a ride choice or a swap. Returns what happens next.

    The answer must be the one the task is waiting for (task.input_needed); anything else is refused, so a
    stray "yes" or a ride name can never place something nobody was shown."""
    if kind == "keep":
        # A caregiver asked about a slow order says keep trying: the clock starts again, and a cost pause lifts.
        if task.status not in LIVE:
            return f"nothing to keep going (status {task.status})"
        d = task.details or {}
        spent = float((d.get("metrics") or {}).get("cost_inr", 0))
        task.details = {**d, **LADDER_RESET, "ladder_from": clock.now().isoformat(), "cost_ok_until": spent + task_max_cost()}
        task.deadline_at = max(task.deadline_at, clock.now() + TASK_LIFETIME)
        note(task, f"keep trying, said {by}")
        return "keeping at it; the next update comes when it needs someone or is placed"
    if kind == "only":
        # They named one store of a running comparison (live 2026-10-11: "Zepto" — Saheli said "checking only on Zepto"
        # and all three stores went on). That store goes on alone, as an order on the store they named.
        d = task.details or {}
        if not compare_group(task) or task.status not in LIVE or task.phase not in ("prepare", "browse"):
            return "this is not a store of a running comparison; for another store, cancel_task it and start_task with that service"
        label = SKILLS[task.service]["label"]
        _drop_others(await siblings(session, task), task, f"{by} wants {label} only")
        task.details = {**d, "compare": None, "compare_seen": None, "compare_note": None, "only_store": True}
        note(task, f"only {label}, said {by}; the other stores are dropped")
        if task.status == "awaiting_confirm" and task.input_needed == "go":
            if (_ride_opts(task) if task.kind == "ride" else _found(task)) and _offers(task):
                # it has looked already: its own options, for them to pick from
                return f"only {label} now; the other stores are dropped. " + (
                    _ride_options_message(task) if task.kind == "ride" else _options_message(task, task.result or {}))
            task.details = {**task.details, "browsed": False}  # nothing to offer from its look-up: it looks again on its own
            task.phase, task.status, task.input_needed = "prepare", "queued", None
            task.deadline_at = clock.now() + TASK_LIFETIME
        return (f"only {label} now; the other stores are dropped. What {label} has comes in a task update for them to pick; "
                "say only that")
    if task.status not in ("needs_input", "awaiting_confirm"):
        return f"not waiting for input (status {task.status})"
    from app.care.boundaries import may_approve

    value = (value or "").strip()
    task.details = {**(task.details or {}), "active_from": clock.now().isoformat(), "told_working": None}
    if (task.details or {}).get("ladder_asked_at"):
        # someone answered: the order is moving again, so the 20 minutes start again
        task.details = {**(task.details or {}), **LADDER_RESET, "ladder_from": clock.now().isoformat()}
    d = task.details or {}
    placed = bool((task.result or {}).get("placed") or (task.result or {}).get("booked"))
    if kind == "resend":
        # The login code did not come. Live 2026-10-11: Saheli promised to ask for a new one, nothing could, and the order
        # sat until it was cancelled. The store is asked again with a fresh login run (at most twice).
        if task.input_needed != "otp":
            return "not waiting for a login code right now"
        n = int(d.get("code_resends") or 0)
        if n >= MAX_CODE_RESENDS:
            return (f"a new code was already asked for {n} times; ask whether the number ending {d.get('login') or '?'} is right "
                    "and whether to keep trying or cancel")
        task.details = {**d, "code_resends": n + 1, "login_only": True, "resend_code": True, "otp_entered": False, "code_lost": False}
        task.phase, task.status, task.input_needed = "prepare", "queued", None
        task.deadline_at = clock.now() + TASK_LIFETIME
        note(task, f"login code did not come (said {by}): asking the store for a new one")
        return ("asking the store for a new login code now (NOT ordered); a task update comes when it is sent. Say only that a "
                "new code is coming to the same phone in a minute")
    if kind in ("confirm", "choice", "swap", "go", "approve", "later", "fare") and _is_no(value) and not placed:
        if kind == "approve" and not may_approve((d.get("approval_needed") or {}).get("approvers") or [], by,
                                                 await _care_recipient(session, task.family_id) or task.subject_id):
            return "only the family's approver can decline this"
        task.status, task.cancel_requested = "cancelled", True
        note(task, f"declined by {by}" + (" (approver)" if kind == "approve" else ""))
        metrics.on_milestone(task, "cancelled")
        # In the day's ledger, so the brain sees the earlier price was dropped (it re-offered one live, 2026-10-09).
        await store.record_event(session, family_id=task.family_id, subject_id=task.subject_id, kind="task_cancelled",
                                 summary=f"{SKILLS[task.service]['label']}: dropped, nothing was ordered (that price is no longer valid)",
                                 payload={"task_id": str(task.id)}, actor_id=by)
        if compare_group(task) and task.phase == "browse":
            _drop_others(await siblings(session, task), task, f"declined by {by}")
            return "declined on every store; nothing was ordered"
        _, _, ordered = _words(task)
        if kind == "approve":
            return f"declined; nothing was {ordered}. Tell person {task.requested_by} kindly with send_message, in one line."
        return f"declined; nothing was {ordered}"
    may = may_approve((d.get("approval_needed") or {}).get("approvers") or [], by,
                      await _care_recipient(session, task.family_id) or task.subject_id)
    if task.input_needed == "approve" and kind == "confirm" and may:
        kind = "approve"  # the approver saying "yes, confirm" is the approval
    if task.input_needed and kind != task.input_needed and not (kind in EDIT_KINDS and task.input_needed in ("go", "confirm", "swap")):
        if task.input_needed == "approve":
            return "this is waiting for the family approver's answer (already asked); nothing is placed until they say yes"
        return f"the task is waiting for {task.input_needed}, not {kind}; ask for that"
    if kind == "go" and task.kind == "ride":
        if task.phase != "browse" or task.status != "awaiting_confirm":
            return "nothing to go ahead with right now"
        opts, offer = _ride_opts(task), d.get("offer") or {}
        pick = None
        if opts and offer:
            v = value.strip().lower().lstrip("#")
            o = (next((x for x in offer.values() if x.get("s")), None) or next(iter(offer.values()))) if _is_yes(value) else offer.get(v)
            if o is None:
                elsewhere = await _option_elsewhere(session, task, [v] if v else [])
                if elsewhere:
                    return elsewhere
                return "pass the number of the fare they picked, as listed on this task: " + "; ".join(f"{k}. {x.get('text')}" for k, x in offer.items())
            i = int((o.get("oids") or ["-1"])[0])
            pick = {"type": opts[i].get("type"), "fare": opts[i].get("fare")} if 0 <= i < len(opts) else None
        label = SKILLS[task.service]["label"]
        task.details = {**d, "ride_pick": pick, "offer": None}
        if pick:
            await _save_usual_vehicle(session, task, pick.get("type"), by=by, by_is_elder=by_is_elder)
        if later.of(task) and pick:
            # For later (founder 2026-10-12): the one question now, with everything and the most they will pay; nothing
            # logs in yet (the login is planned by how long the app keeps it).
            note(task, f"picked for later by {by}: {pick['type']} {pick.get('fare') or ''}".rstrip())
            if compare_group(task):
                _drop_others(await siblings(session, task), task, f"{by} picked {label} for later")
            return _later_ask(task)
        task.phase, task.status, task.input_needed = "prepare", "queued", None
        task.deadline_at = clock.now() + TASK_LIFETIME
        note(task, f"go-ahead from {by}" + (f" for {pick['type']} {pick.get('fare') or ''}".rstrip() if pick else ""))
        dropped = ""
        if compare_group(task):
            group = await siblings(session, task)
            _drop_others(group, task, f"{by} picked {label}", standby=True)
            if len(group) > 1:
                dropped = ("; the other apps wait without anything to do (do not cancel them): their fares come back only if this one "
                           "cannot book")
        return (f"going ahead on {label}" + (f" with {pick['type']} (about {pick.get('fare')} before the login)" if pick else "")
                + f"{dropped}: getting the fresh fare (NOT booked; a login code may be asked). Say only that you are getting the fresh "
                "fare; the one confirm with the fare comes next")
    if kind == "go":
        if task.phase != "browse" or task.status != "awaiting_confirm":
            return "nothing to go ahead with right now"
        pick = None
        found = _found(task)
        _number(found)
        if not _is_yes(value):
            # The person picked from the numbered options: the brain passes the numbers as listed (or a product's own id,
            # p3), never a name to match.
            byid = {str(i.get("oid")).lower(): i for i in found}
            offer = d.get("offer") or {}
            oids: list[str] | None = []
            tokens = [x.strip().lower().lstrip("#") for x in re.split(r"[,\s]+", value) if x.strip()]
            for x in tokens:
                if x in offer:
                    oids += [str(o).lower() for o in offer[x].get("oids") or []]
                elif x in byid:
                    oids.append(x)
                else:
                    oids = None
                    break
            if not tokens or not oids or any(o not in byid for o in oids):
                elsewhere = await _option_elsewhere(session, task, tokens)
                if elsewhere:
                    return elsewhere
                listed = ("; ".join(f"{n}. {o.get('text')}" for n, o in offer.items()) if offer else
                          "; ".join(f"[{i['oid']}] {_label(i)} {i.get('price') or ''}".strip() for i in found[:12]))
                return f"pass the number(s) of the option(s) they picked, as listed on this task: {listed}"
            pick = [byid[o] for o in dict.fromkeys(oids)]
            if SKILLS[task.service].get("kind") == "food" and len({str(p.get("restaurantId") or p.get("restaurant") or "") for p in pick}) > 1:
                return ("one food order is from one restaurant: ask them to pick dishes from one restaurant (one numbered option, or "
                        "dishes of the same place)")
        chosen, left = _picked(task, pick)
        food = SKILLS[task.service].get("kind") == "food"
        if pick:
            # what they picked is checked against the care record like what they asked (an allergy, the never-order list)
            asked = d.get("items") or []
            v = guard.check_request("order", d.get("agent") or "shopping", [{"name": c.get("name"), "qty": channels._qty_for(asked, c)}
                                                                           for c in chosen], Limits.from_dict(d.get("limits")))
            if not v.ok:
                return "cannot get that: " + "; ".join(v.block) + ". Tell them kindly; they can pick another option or none"
        if not found:
            # It shows its products only after a login: it logs in, then looks again, and the options come back to pick
            # from (founder 2026-10-11: options at all times; nothing goes in a cart they did not pick).
            d = {**d, "login_only": True, "relook_after_login": True}
        task.details = ({**d, "chosen": chosen, "more": None, "auto_picked": not pick, "offer": None,
                         "menu_find": (left or None) if food else None,
                         "missing_items": [*(d.get("missing_items") or []), *([] if food else left)] or None,
                         "found": [{k: v for k, v in i.items() if k != "image_url"} for i in found][:40]}
                        if chosen else {**d, "more": None, "offer": None})
        if later.of(task) and chosen and found:
            note(task, f"picked for later by {by}: " + "; ".join(_label(c) for c in chosen))
            if compare_group(task):
                _drop_others(await siblings(session, task), task, f"{by} picked {SKILLS[task.service]['label']} for later")
            return _later_ask(task)
        task.phase, task.status, task.input_needed = "prepare", "queued", None
        task.deadline_at = clock.now() + TASK_LIFETIME
        note(task, f"go-ahead from {by}" + (f" for {'; '.join(_label(p) for p in pick)}" if pick else ""))
        dropped = ""
        if compare_group(task):
            group = await siblings(session, task)
            # the others wait: if this store cannot build the cart, their options go back to the person
            _drop_others(group, task, f"{by} picked {SKILLS[task.service]['label']}", standby=True)
            if len(group) > 1:
                dropped = ("; the other stores wait without anything to do (do not cancel them): their options come back only if "
                           "this one cannot build the cart")
        what = "; ".join(f"{_label(i)} {i.get('price') or ''}".strip() for i in chosen)
        how = ("building the cart on the linked account (no login code; NOT ordered; you will get a task update)"
               if (task.details or {}).get("channel") == "connector" else
               "building the cart (NOT ordered). Say only that you are getting the cart ready; do not promise a login code: if one "
               "is needed you will get a task update to ask for it")
        return f"going ahead on {SKILLS[task.service]['label']}" + (f" with {what}" if what else "") + f"{dropped}: {how}"
    if kind in EDIT_KINDS:
        # At the confirm (or while choosing): the brain says, as data, which item (target: its number in the list) and the
        # new item(s); it is looked up again, the AI picks, and the new cart and total come back for the one confirm.
        if task.kind != "order" or task.status not in ("awaiting_confirm", "needs_input"):
            return "nothing to change right now"
        cur = list(d.get("items") or [])
        new = [clean_item(i) for i in (items or []) if isinstance(i, dict) and str(i.get("name") or "").strip()]
        idx = (int(target) - 1) if isinstance(target, int) or (isinstance(target, str) and target.isdigit()) else None
        if idx is not None and not 0 <= idx < len(cur):
            return "target must be the number of an asked item: " + "; ".join(f"{n}. {i.get('name')}" for n, i in enumerate(cur, 1))
        if kind == "add":
            if not new:
                return "pass the item(s) to add in items"
            new_items = cur + new
        elif kind == "remove":
            if idx is None:
                return "pass target, the number of the item to remove: " + "; ".join(f"{n}. {i.get('name')}" for n, i in enumerate(cur, 1))
            new_items = cur[:idx] + cur[idx + 1:]
            if not new_items:
                return "that is the only item; to drop the whole order use task_input confirm no"
        elif kind == "change":
            if not new:
                return "pass the replacement item(s) in items (and target, the item they replace)"
            if idx is None and len(cur) > 1:
                return "pass target, the number of the item to change: " + "; ".join(f"{n}. {i.get('name')}" for n, i in enumerate(cur, 1))
            idx = idx or 0
            new_items = cur[:idx] + new + cur[idx + 1:]
        else:  # more: that item's options are listed (a refined ask may come in items)
            if idx is None and len(cur) > 1:
                return "pass target, the number of the item to see more of: " + "; ".join(f"{n}. {i.get('name')}" for n, i in enumerate(cur, 1))
            idx = idx or 0
            new_items = cur[:idx] + (new[:1] or [cur[idx]]) + cur[idx + 1:]
            v = guard.check_request("order", d.get("agent") or "shopping", new_items, Limits.from_dict(d.get("limits")))
            if not v.ok:
                return "cannot look for that: " + "; ".join(v.block)
            name = str(new_items[idx].get("name"))
            task.details = {**d, **RELOOK, "items": [clean_item(i) for i in new_items], "more": name, "kept": _kept(d, new_items, skip=name)}
            task.result = {**(task.result or {}), "alternatives": None}
            task.phase, task.status, task.input_needed = "prepare", "queued", None
            task.deadline_at = clock.now() + TASK_LIFETIME
            note(task, f"more options for {name} asked by {by}")
            return (f"looking for more {name} on {SKILLS[task.service]['label']}; the list comes back in a task update (nothing "
                    "ordered). Say only that you are looking for more options")
        return await _relook(session, task, d, new_items, kind, value or ", ".join(i["name"] for i in new), by)
    if kind == "confirm":
        if task.kind != "order" or task.status != "awaiting_confirm":
            return "nothing to confirm right now"
        if not _is_yes(value):
            return "pass value yes to place it (or no to drop it), after the person clearly agreed"
        fp = d.get("cart_fp")
        if not fp:
            return "there is no cart to confirm yet; wait for the task update"
        # The family's boundaries decide whether this may go ahead now or waits for an approver (checked in code).
        gate = await _approval_gate(session, task, rupees((task.result or {}).get("total")), by)
        if gate:
            return gate
        return _start_placing(task, by)
    if kind == "otp":
        if not re.fullmatch(r"\d{4,8}", value):
            return "that is not a code; ask for the digits only"
        task.phase, task.status = "otp", "queued"
        task.details = {**d, "otp": value}
        note(task, "otp received")
        return "code received; continuing"
    if kind == "fee":
        if _is_yes(value):
            task.details = {**d, "fee_ok": True, "cancel_ask": {**(d.get("cancel_ask") or {}), "state": "cancelling"}}
            task.phase, task.status, task.input_needed = "cancel", "queued", None
            task.deadline_at = clock.now() + TASK_LIFETIME
            note(task, f"cancel fee {d.get('cancel_fee') or ''} agreed by {by}")
            return (f"asking {SKILLS[task.service]['label']} to cancel it with the fee of {d.get('cancel_fee') or 'shown'} (NOT cancelled "
                    "yet); the result comes in a task update. Say only that")
        task.status, task.input_needed = "done", None
        task.details = {**d, "cancel_ask": {**(d.get("cancel_ask") or {}), "state": "kept"}}
        note(task, f"cancel fee declined by {by}; it stays")
        return f"kept: not cancelled; it stays {_words(task)[0]} (no fee paid)"
    if kind == "swap":
        if d.get("agent") == "pharmacy":
            return "medicines are never swapped for another; ask the caregiver or doctor which medicine to order"
        alts = guard.alternatives(d.get("alternatives"))
        low = value.lower()
        pick = next((a for a in alts if len(low) >= 3 and (low in a["name"].lower() or a["name"].lower() in low)), None)
        if not pick:
            return "that is not one of the alternatives; read them again: " + "; ".join(a["name"] for a in alts)
        items = list(d.get("items") or [])
        # Replace the requested item the alternative is closest to (by its main words); keep the rest.
        words = set(re.findall(r"[a-z]{4,}", pick["name"].lower()))
        best = max(range(len(items)), key=lambda i: len(words & set(re.findall(r"[a-z]{4,}", str(items[i].get("name", "")).lower()))), default=None)
        if best is not None and (len(items) == 1 or words & set(re.findall(r"[a-z]{4,}", str(items[best].get("name", "")).lower()))):
            new_items = items[:best] + [{"name": pick["name"], "qty": guard.qty(items[best].get("qty"))}] + items[best + 1:]
        else:
            new_items = items + [{"name": pick["name"], "qty": 1}]
        v = guard.check_request("order", d.get("agent") or "shopping", new_items, Limits.from_dict(d.get("limits")))
        if not v.ok:
            return "cannot get that: " + "; ".join(v.block)
        task.details = {**d, "items": new_items, "alternatives": None, "channel": None, "connector_card": None, "cart_fp": None,
                        "browsed": d.get("browsed") or task.phase == "browse"}
        task.phase, task.status, task.input_needed = "prepare", "queued", None
        task.deadline_at = clock.now() + TASK_LIFETIME
        note(task, f"swapped to {pick['name']} by {by}")
        if compare_group(task):
            _drop_others(await siblings(session, task), task, f"{by} picked {SKILLS[task.service]['label']}")
        return f"getting {pick['name']} instead; the new cart comes back for a confirm"
    if kind == "choice":
        if task.kind != "ride" or task.status != "awaiting_confirm":
            return "nothing to choose right now"
        options = (task.result or {}).get("options") or []
        offer = d.get("offer") or {}
        v = value.strip().lower().lstrip("#")
        o = (next((x for x in offer.values() if x.get("s")), None) if _is_yes(value) else offer.get(v))
        i = int((o.get("oids") or ["-1"])[0]) if o else -1
        # the number as listed; an older update named the option instead
        opt = options[i] if 0 <= i < len(options) else _match_option(options, value)
        listed = "; ".join(f"{k}. {x.get('text')}" for k, x in offer.items()) or "; ".join(
            f"{o_.get('type')} {o_.get('fare', '')}".strip() for o_ in options)
        if opt == "ambiguous":
            return "that could be more than one option; pass its number as listed: " + listed
        if not opt:
            return "that is not one of the options; pass its number as listed: " + listed
        task.details = d = {**d, "ride_pick": {"type": opt.get("type"), "fare": opt.get("fare")}}
        await _save_usual_vehicle(session, task, opt.get("type"), by=by, by_is_elder=by_is_elder)
        choice = f"{opt.get('type')} {opt.get('fare') or ''}".strip()
        lt = later.of(task)
        if lt and not lt.get("by"):
            return _later_ask(task)  # not confirmed for later yet: this pick goes into the one question
        if lt:
            # they picked this fare themselves at the check: that is their yes to it
            return _later_ready(task, choice=choice, fare=opt.get("fare"), by=by)
        gate = await _approval_gate(session, task, rupees(opt.get("fare")), by, choice={"choice": choice, "fare": opt.get("fare")})
        if gate:
            return gate
        return _start_booking(task, by, choice, opt.get("fare"))
    if kind == "later":
        lt = later.of(task)
        if not lt or task.input_needed != "later":
            return "nothing is waiting for an answer about booking it for later"
        if not _is_yes(value):
            return "pass value yes (with max_amount, or no_max true) once they agreed, or value no"
        amount = _later_amount(task)
        mx = None if no_max else (float(max_amount) if max_amount else later.suggest_max(amount))
        lt.update(max=mx, no_max=mx is None, by=by, confirmed_at=clock.now().isoformat())
        task.details = d = {**d, "later": lt}
        note(task, f"set for later by {by}: " + (f"up to ₹{mx:.0f}" if mx else "no limit"))
        # the family's limits against the most it may cost, now: an approver's yes comes up front (founder 2026-10-12)
        gate = await _approval_gate(session, task, later.cap(mx) or amount, by,
                                    choice={"choice": f"{(d.get('ride_pick') or {}).get('type') or ''} up to ₹{mx:.0f}".strip() if mx else None,
                                            "fare": mx} if task.kind == "ride" else None)
        if gate:
            return gate
        return await _schedule(session, task)
    if kind == "fare":
        lt = later.of(task)
        if task.input_needed != "fare" or not lt.get("asked"):
            return "nothing is waiting for a yes to a new fare or total"
        if not _is_yes(value):
            return "pass value yes to book it at the new amount, or no"
        a = lt["asked"]
        lt.update(max=rupees(a.get("amount")) or lt.get("max"))
        task.details = {**d, "later": lt}
        return _later_ready(task, choice=a.get("choice"), fare=a.get("amount"), by=by)
    if kind == "approve":
        ap = d.get("approval_needed") or {}
        if not ap:
            return "nothing is waiting for approval"
        if not may_approve(ap.get("approvers") or [], by, await _care_recipient(session, task.family_id) or task.subject_id):
            return "only the family's approver can approve this; it was sent to them"
        if not _is_yes(value):
            return "pass value yes to approve or no to decline"
        task.details = {**d, "approval_needed": None, "approved_by": by, "approved_at": clock.now().isoformat()}
        note(task, f"approved by {by}")
        if later.of(task).get("by") and later.of(task).get("state") in (None, "new"):
            out = await _schedule(session, task)
            return f"approved. {out} Tell person {task.requested_by} in one line with send_message that it was approved."
        if later.of(task).get("state") == "approval":
            c = ap.get("choice") or {}
            out = _later_ready(task, choice=c.get("choice"), fare=c.get("fare") or (task.result or {}).get("total"), by=by)
            return f"approved. {out} Tell person {task.requested_by} in one line with send_message that it was approved."
        if task.kind == "ride":
            c = ap.get("choice") or {}
            out = _start_booking(task, by, c.get("choice") or "", c.get("fare"))
        else:
            if not (task.details or {}).get("cart_fp"):
                return "the cart is gone; nothing was placed. Ask them to order again"
            out = _start_placing(task, by)
        return f"approved. {out}. Tell person {task.requested_by} in one line with send_message that it was approved."
    return f"unknown input {kind}"


def _words(task: Task) -> tuple[str, str, str]:
    """(placed | booked, order | ride, ordered | booked): how an order and a ride are spoken of."""
    return ("booked", "ride", "booked") if task.kind == "ride" else ("placed", "order", "ordered")


def _placed_line(task: Task) -> str:
    r = task.result or {}
    label = SKILLS[task.service]["label"]
    if task.kind == "ride":
        return (f"BOOKED on {label}" + (f" (ride {r['ride_id']})" if r.get("ride_id") else "")
                + (f", driver {r['driver']}" if r.get("driver") else "") + (f", {r['status']}" if r.get("status") else ""))
    return (f"PLACED on {label}: {bought(task)}" + (f" (order {r['order_id']})" if r.get("order_id") else "")
            + (f", arriving in about {r['eta']}" if r.get("eta") else ""))


def _cancel_tell(task: Task, ask: dict, why: str = "") -> str:
    """What the person who asked to cancel hears when it cannot be cancelled from here (the store would not, or it may have
    gone through without a confirmation): the founder's words, 2026-10-11."""
    label = SKILLS[task.service]["label"]
    done, thing, _ = _words(task)
    because = f"{label} would not cancel it ({why})" if why else f"it is already {done}, so it cannot be cancelled from here"
    if ask.get("elder"):
        who = " and ".join(f"person {c}" for c in ask.get("carers") or [])
        return (f"Tell them kindly in one or two short lines: {because}; you are telling {who} right now so they can get it "
                f"cancelled. Do not message {who} yourself: the {thing} system tells them now, with the {thing}. Never say "
                "cancelled or that cancelling is in process.")
    where = ("its rides page; the app may charge a cancellation fee" if task.kind == "ride" else
             "its orders page, while the store still allows it")
    return (f"Tell them plainly in one or two short lines: {because}; they can cancel it in the {label} app ({where}). Never "
            "say cancelled or that cancelling is in process.")


def _start_store_cancel(task: Task) -> None:
    """Ask the store (or the ride app) to cancel a placed order or a booked ride: a cancel run. A cancel that costs
    nothing goes through without asking again; a real fee is asked once, with its amount (_outcome, phase cancel)."""
    task.phase, task.status, task.input_needed = "cancel", "queued", None
    task.deadline_at = clock.now() + TASK_LIFETIME
    task.details = {**(task.details or {}), "fee_ok": None, "cancel_fee": None}


async def _cancel_after_placed(session: AsyncSession, task: Task, *, by: str, reason: str, placing: bool, unclear: bool = False) -> str:
    """A cancel asked for an order that is placed (or a ride that is booked), or being placed or booked right now
    (founder 2026-10-12, 02:35 IST, replacing part of the 2026-10-11 rule):
    - a caregiver, or someone ordering for themselves: Saheli asks the store to cancel it; a free cancel goes through
      without asking again, a real fee is asked once with its amount, the result comes in ONE update; only if the store
      will not cancel it are they told so plainly and sent to the store's app;
    - the care recipient (with caregivers in the family): it cannot be cancelled from here; they hear that the caregiver is
      told right away, and every caregiver is told at once (tell_carers), who can then ask Saheli to cancel it on the store.
    A second ask gets the same truth."""
    d = task.details or {}
    done, thing, _ = _words(task)
    label = SKILLS[task.service]["label"]
    ask = dict(d.get("cancel_ask") or {})
    carers = [c for c in await _carers(session, task) if c != by]
    elder = by == (await _care_recipient(session, task.family_id) or task.subject_id) and bool(carers)
    if ask.get("state") in ("cancelling", "fee") or (ask.get("state") == "pending" and not ask.get("elder")):
        return (f"NOT cancelled yet: {label} is " + ("asked to cancel it the moment it is placed" if ask.get("state") == "pending" else
                "already being asked to cancel it") + (" (it is waiting for their yes or no to the cancellation fee of "
                f"{d.get('cancel_fee')}: task_input fee)" if ask.get("state") == "fee" else "") + ". The result comes in a task update; "
                "tell them only that, in one short line.")
    if ask.get("state") == "pending" and elder:
        return (f"NOT cancelled yet: it is being {done} on {label} right now and that cannot be stopped; if it goes through, the "
                "caregiver is told at once. Tell them only that, in one short line.")
    if ask.get("state") == "refused":
        return (f"NOT cancelled: {label} would not cancel it ({ask.get('store_said') or 'too late'}); it stays {done}. "
                + _cancel_tell(task, ask, why=ask.get("store_said") or "too late"))
    if ask.get("state") in ("told_carers", "unclear") and elder:
        return (f"NOT cancelled: it is already {_placed_line(task)}, so it cannot be cancelled from here; the caregiver was told. "
                + _cancel_tell(task, ask))
    first = dict(ask) if ask else {}
    ask = {"by": by, "at": clock.now().isoformat(), "reason": reason[:160], "elder": elder, "carers": carers if elder else [],
           **({"first_by": first.get("by")} if first.get("by") and first.get("by") != by else {}),
           **({"carers_told": first["carers_told"]} if first.get("carers_told") else {})}
    if placing:
        ask["state"] = "pending"
        task.details = {**d, "cancel_ask": ask}
        note(task, f"cancel asked by {by} while it is being {done}")
        after = ("the caregiver is told at once" if elder else f"{label} is asked to cancel it, and ONE task update comes with the result")
        return (f"NOT cancelled yet: the {thing} is being {done} on {label} right now and that cannot be stopped; if it goes "
                f"through, {after}. Tell them only that, in one short line; never say cancelled.")
    if unclear:
        ask.update(state="unclear", told_by="reply")
        task.details = {**d, "cancel_ask": ask}
        note(task, f"cancel asked by {by}; it may have gone through unconfirmed: " + ("caregivers told" if elder else "the app"))
        return (f"NOT cancelled: it is not clear whether it went through on {label} (its confirmation did not come), so it cannot be "
                "cancelled from here. " + _cancel_tell(task, ask).replace(f"it is already {done}", f"it may already be {done}"))
    if elder:
        ask.update(state="told_carers", told_by="reply")
        task.details = {**d, "cancel_ask": ask}
        note(task, f"cancel asked by {by} (the care recipient) after it was {done}: not cancelled; caregivers told")
        return f"NOT cancelled: it is ALREADY {_placed_line(task)}, so it cannot be cancelled from here. " + _cancel_tell(task, ask)
    ask["state"] = "cancelling"
    task.details = {**d, "cancel_ask": ask}
    _start_store_cancel(task)
    note(task, f"cancel asked by {by} after it was {done}: asking {label} to cancel it")
    return (f"NOT cancelled yet: it is already {_placed_line(task)}; asking {label} to cancel it now. A free cancel goes through "
            "without asking them again; if the store wants a fee they are asked once, with the amount; if it will not cancel, they "
            "hear what to do instead. ONE task update comes with the result in a minute or two. Tell them only that, in one "
            "short line; never say it is cancelled before that update.")


def _placed_by_mistake(task: Task, *, ref: str | None, how: str, already_cancelled: bool = False) -> tuple[str, str]:
    """A run before the confirm booked or placed it (founder 2026-10-12, 21:52 IST: a fare look-up on Ola booked an auto,
    the driver came to his door, and he heard "nothing was booked"). Recorded as it is, said at once and owned, and the
    store is asked to cancel it under the cancel rule (free: no second ask; a fee: asked once with the amount; refused:
    said plainly, use its app). Already cancelled on the store: recorded and said so."""
    d = task.details or {}
    done, thing, _ = _words(task)
    label = SKILLS[task.service]["label"]
    idk = "ride_id" if task.kind == "ride" else "order_id"
    r = {**(task.result or {}), ("booked" if task.kind == "ride" else "placed"): True, "by_mistake": True}
    if ref and not r.get(idk):
        r[idk] = ref
    task.result = r
    ref = r.get(idk) or ref
    now = clock.now().isoformat()
    ask = {"by": task.requested_by, "at": now, "reason": f"{done} by mistake before anyone said yes", "mistake": True}
    task.details = {**d, "unauthorized_place": True, "confirm_token": None, "check": None, "guard_extra": None,
                    "mistake": {"at": now, "how": how[:160], "ref": ref, "phase": task.phase}}
    what = f"the {thing} on {label}" + (f" ({ref})" if ref else "")
    if already_cancelled:
        task.details = {**task.details, "cancel_ask": {**ask, "state": "cancelled", "told_by": "update"}}
        note(task, f"{done} by mistake before anyone said yes ({how}); {label} shows it cancelled already")
        metrics.on_milestone(task, "cancelled")
        return "cancelled", (f"MISTAKE, yours: while only looking it up, {what} was {done} before anyone said yes; {label} now shows it "
                             f"cancelled. Tell them at once, plainly, in one or two short lines: it was {done} by mistake and it is "
                             f"cancelled; say sorry once. Never say nothing was {done}.")
    task.details = {**task.details, "cancel_ask": {**ask, "state": "cancelling"}}
    _start_store_cancel(task)
    note(task, f"{done} by mistake before anyone said yes ({how}): asking {label} to cancel it")
    return task.status, (f"MISTAKE, yours: while only looking it up, {what} was {done} before anyone said yes. Tell them at once, "
                         f"plainly, in one or two short lines: it was {done} by mistake, you are cancelling it on {label} right now, "
                         f"and the next update brings the result (if {label} wants a fee to cancel, they are asked first). Say sorry "
                         f"once. Never say nothing was {done}.")


def _start_check(task: Task, *, then: str, why: str, held: dict | None = None) -> tuple[str, None]:
    """Look at the store's own list (read only) before anyone hears "nothing was booked": then = what follows a clean
    check ("cancelled": they asked to stop; "retry": the step once more, told to keep away from that page; "fail": stop
    and say so; "outcome": the run's own report goes on)."""
    d = task.details or {}
    done, _, _ = _words(task)
    task.details = {**d, "check": {"then": then, "why": why[:160], "from": task.phase, "since": d.get("run_started") or clock.now().isoformat(),
                                   "run": task.agent_task, **({"held": held} if held is not None else {})}}
    task.phase, task.status, task.input_needed = "check", "queued", None
    note(task, f"checking {SKILLS[task.service]['label']} that nothing was {done} ({why})")
    return "queued", None


GUARD_LINE = ("The last run of this step went to {page}, a page this step must never open (it carries the final book / place "
              "button, or only comes after it). Never open it: what this step needs is on the pages before it{wait}.")


def _check_outcome(task: Task, out: dict) -> tuple[str, str | None]:
    """The store's own list after a run before the confirm went where it must not, or was stopped. The agent judges the
    list (this ride or order, cancelled or not); the code acts on its fields."""
    d = task.details or {}
    chk = dict(d.get("check") or {})
    done, thing, _ = _words(task)
    label = SKILLS[task.service]["label"]
    if out.get("this_task"):
        ref = out.get("ride_id") if task.kind == "ride" else out.get("order_id")
        task.phase = chk.get("from") or "prepare"
        return _placed_by_mistake(task, ref=ref, how=f"{label}'s own list after: {chk.get('why') or 'a stop'}",
                                  already_cancelled=bool(out.get("cancelled")))
    if out.get("logged_in") is False or out.get("blocked") or out.get("problem") and out.get("this_task") is None:
        task.details = {**d, "check": None, "check_failed": True}
        task.phase = chk.get("from") or "prepare"
        note(task, f"could not read {label}'s own list ({out.get('problem') or 'not logged in'})")
        return "failed", (f"It is NOT known whether anything was {done}: {label}'s own list could not be read. Tell them plainly in "
                          f"one short line to look in the {label} app for a {thing} made just now (and cancel it there if it is "
                          f"there), and say sorry. Never say nothing was {done}.")
    then = chk.get("then")
    task.phase = chk.get("from") or "prepare"
    task.details = {**d, "check": None}
    if then == "cancelled":
        note(task, f"stopped before anything was {done} (checked on {label}: nothing there)")
        metrics.on_milestone(task, "cancelled")
        return "cancelled", (f"Checked on {label}: nothing was {done}; it is stopped. Tell them in one short line.")
    if then == "retry":
        page = (chk.get("why") or "").split("page ", 1)[-1] or "that page"
        wait = "; fares and prices that show as grey placeholders are still loading: wait for them" if task.kind == "ride" else ""
        task.details = {**task.details, "guard_retried": True, "guard_extra": GUARD_LINE.format(page=page, wait=wait)}
        note(task, f"checked on {label}: nothing was {done}; this step runs once more, kept away from {page}")
        return "queued", None
    note(task, f"stopped at the page with the final button twice; checked on {label}: nothing was {done}")
    what = "fares" if task.kind == "ride" else "cart"
    return "failed", (f"Could not get the {what} on {label} without opening its {'booking' if task.kind == 'ride' else 'payment'} "
                      f"page, so it was stopped there; checked on {label}: nothing was {done}. Tell them plainly in one short line "
                      "and offer another app or store.")


def _after_check(task: Task, out: dict) -> tuple[tuple[str, str | None] | None, dict]:
    """A finished check: (status, message) when it decides the step (owned and cancelled, stopped, run again, failed, or
    not readable), else (None, the step's own held report, which goes on as it would have)."""
    chk = dict((task.details or {}).get("check") or {})
    if chk.get("then") == "outcome" and chk.get("held") is not None and not out.get("this_task") and out.get("logged_in") is not False \
            and not out.get("blocked") and not (out.get("problem") and out.get("this_task") is None):
        task.phase = chk.get("from") or "prepare"
        task.details = {**(task.details or {}), "check": None, "checked_run": task.agent_task}  # its report goes on, not checked again
        note(task, f"checked on {SKILLS[task.service]['label']}: nothing was {_words(task)[0]}")
        return None, chk["held"]
    return _check_outcome(task, out), out


def _needs_store_check(task: Task, out: dict) -> bool:
    """After a run with the family's login before the confirm: the store's own list is read before its result goes to the
    person (founder 2026-10-12: "after every run that must not place, check the store's live state"). Not for a run that
    had no login (it asked for a code, or saw a login wall), nor twice for the same run."""
    d = task.details or {}
    return (task.phase in ("prepare", "otp", "browse") and not _guest_run(task) and bool(task.agent_session)
            and d.get("checked_run") != task.agent_task and not out.get("needs_otp") and out.get("logged_in") is not False
            and not out.get("blocked") and os.getenv("STORE_CHECK_AFTER_RUNS", "on") == "on")


def _before_confirm_guard(task: Task, path: list[str], out: dict) -> tuple[str, str | None] | None:
    """A run before the confirm, at its end (or stopped by the guard): booked or placed on the store's own after-order
    page or in its own report → owned and cancelled; on the page with the final button, or a page shaped like an
    after-order one → the store's own list is checked first. None: nothing wrong."""
    if task.phase not in guard.BEFORE_CONFIRM or task.phase == "check" or _guest_run(task):
        return None
    d = task.details or {}
    hit = guard.past_confirm(task.service, path)
    said = out.get("placed") or out.get("booked") or out.get("order_id") or out.get("ride_id") or out.get("pressed_place")
    if said or (hit and hit[0] == "after"):
        ref = out.get("order_id") or out.get("ride_id") or (guard.ref_in(hit[1]) if hit else None)
        return _placed_by_mistake(task, ref=str(ref) if ref else None, how="its own report" if said else f"the page {hit[1]}")
    if not hit:
        return None
    if hit[0] == "suspect" and out.get("problem") != "stopped" and not out.get("blocked"):
        return _start_check(task, then="outcome", why=f"{hit[0]} page {hit[1]}", held=out)
    return _start_check(task, then="retry" if not d.get("guard_retried") else "fail", why=f"{hit[0]} page {hit[1]}")


def _after_cancel_ask(task: Task, status: str, message: str | None) -> str | None:
    """The place or booking step finished after they asked to cancel it. Placed or booked: a caregiver's (or a self-care
    person's) ask goes to the store now (its one result comes after that); the care recipient's ask: ONE message, it cannot
    be cancelled from here and the caregiver is told. Not placed: one message, nothing was ordered."""
    ask = dict((task.details or {}).get("cancel_ask") or {})
    if ask.get("state") != "pending":
        return message
    done, thing, ordered = _words(task)
    if status == "done" and not ask.get("elder"):
        ask.update(state="cancelling")
        task.details = {**(task.details or {}), "cancel_ask": ask}
        _start_store_cancel(task)
        note(task, f"{done} just after the cancel was asked: asking {SKILLS[task.service]['label']} to cancel it")
        return None
    if status == "done":
        ask.update(state="told_carers", told_by="update")
        task.details = {**(task.details or {}), "cancel_ask": ask}
        return (f"They had asked to cancel while it was being {done}, and it went through: it is {_placed_line(task)}, so it cannot "
                "be cancelled from here. " + _cancel_tell(task, ask) + f" Give the {thing} id.")
    if (task.details or {}).get("place_unclear"):
        ask.update(state="unclear", told_by="update")
        task.details = {**(task.details or {}), "cancel_ask": ask}
        return ((message or f"It is not clear whether the {thing} went through.") + " They had asked to cancel it: if it did go "
                f"through, it can be cancelled only in the {SKILLS[task.service]['label']} app"
                + (" and the caregiver is being told now" if ask.get("elder") else "") + ". Never say cancelled.")
    ask.update(state="not_placed", told_by="update")
    task.details = {**(task.details or {}), "cancel_ask": ask}
    return f"They had asked to cancel it: it did NOT go through, so nothing was {ordered}. Tell them in one short line."


async def tell_carers(sessions: async_sessionmaker, notify: Notify) -> int:
    """The care recipient asked to cancel an order that was already placed (or a ride already booked): every caregiver is
    told at once, with the order and the store, to cancel it in that app (founder 2026-10-11: tell the caregiver, and do)."""
    sends: list[tuple[Task, str, str]] = []
    async with sessions() as session:
        rows = list((await session.execute(
            select(Task).where(Task.kind.in_(("order", "ride")), Task.status.in_(("done", "failed")), Task.updated_at >= clock.now() - timedelta(hours=6))
            .with_for_update(skip_locked=True)
        )).scalars())
        for t in rows:
            ask = (t.details or {}).get("cancel_ask") or {}
            if not ask.get("elder") or ask.get("carers_told") or ask.get("state") not in ("told_carers", "unclear"):
                continue
            t.details = {**(t.details or {}), "cancel_ask": {**ask, "carers_told": clock.now().isoformat()}}
            note(t, "caregivers told to cancel it in the store's app: " + ", ".join(ask.get("carers") or []))
            label = SKILLS[t.service]["label"]
            done, thing, ordered = _words(t)
            maybe = "may have gone through (its confirmation did not come)" if ask["state"] == "unclear" else f"is already {done}"
            soon = ("rides can be cancelled in the app, maybe with a fee" if t.kind == "ride" else
                    "stores allow that only for a short while")
            for c in ask.get("carers") or []:
                sends.append((t, c, (
                    f"[Order needs you] {describe(t)}. Person {ask.get('by')} asked to cancel this {thing} ({ask.get('reason') or 'no reason given'}), "
                    f"but it {maybe} on {label}. Tell person {c} now, in one or two short lines with what was {ordered}, the "
                    f"{thing} id and the app, and ask whether to cancel it: if they say cancel, call cancel_task on task {t.id} "
                    f"(the store is then asked to cancel it; you get the result), or they can cancel it in the {label} app "
                    f"themselves ({soon}). Never say it is cancelled.")))
        await session.commit()
    for t, person, text in sends:
        try:
            await _notify_about(notify, [t], t.family_id, person, text)
        except Exception:  # noqa: BLE001 — one failed message must not stop the rest
            logger.exception("telling a caregiver failed family=%s", t.family_id)
    return len(sends)


async def done_in_app(session: AsyncSession, task: Task, *, what: str, by: str, detail: str = "") -> str:
    """They cancelled or changed an order or ride themselves in the store's or ride's app (founder 2026-10-12: he cancelled
    a booked ride in the Ola app himself; the task still read booked, counted in the month's spend, and the "did the cab
    come?" question was due in the morning). Recorded on the task; nothing is run on the store. A cancel takes it out of
    the spend total and stops tracking and the after-order questions; nothing more is said about it."""
    from app.tasks import aftercare

    label = SKILLS[task.service]["label"]
    done, thing, ordered = _words(task)
    d = task.details or {}
    r = task.result or {}
    if not (r.get("placed") or r.get("booked")) and task.status not in ("done",):
        return (f"Saheli did not {('book' if task.kind == 'ride' else 'place')} this {thing} (it is {task.status}): nothing to record. If it "
                "is still running and they want it stopped, call cancel_task.")
    now = clock.now().isoformat()
    record = {"what": what, "by": by, "at": now, **({"detail": detail[:200]} if detail else {})}
    if what == "changed":
        task.details = {**d, "in_app": record}
        note(task, f"{by} changed it themselves in the {label} app: {detail[:120] or 'no details'}")
        return (f"Recorded on the {thing}: they changed it themselves in the {label} app ({detail or 'no details given'}). Nothing is "
                "run on the store. Thank them in one short line; say nothing else about it.")
    loop = await aftercare.loop_for(session, task)
    if loop is not None:
        await store.close_loop(session, loop.id, status="cancelled", note=f"they cancelled it themselves in the {label} app")
    ask = dict(d.get("cancel_ask") or {})
    task.status, task.cancel_requested, task.input_needed = "cancelled", True, None
    task.details = {**d, "in_app": record, "cancel_ask": {**ask, "by": ask.get("by") or by, "state": "cancelled_in_app", "told_by": "reply"},
                    **({"tracking": {**d["tracking"], "done": True}} if d.get("tracking") else {})}
    note(task, f"{by} cancelled it themselves in the {label} app")
    metrics.on_milestone(task, "cancelled")
    return (f"Recorded: they cancelled the {label} {thing} themselves in the {label} app. It is no longer counted as {ordered}; its "
            "tracking and after-order questions stop. Thank them in one short line; say nothing else about it.")


async def request_cancel(session: AsyncSession, agent: BrowserAgent, task: Task, *, by: str, reason: str) -> str:
    """request_cancel under the task's own trace, with an audit event: who cancelled, when and why (D-13, F4)."""
    with trace.for_task(task):
        before = (task.status, task.phase, task.cancel_requested)
        out = await _request_cancel(session, agent, task, by=by, reason=reason)
        if (task.status, task.phase, task.cancel_requested) != before:
            # cancelled now, or asked while placing / after placing (never cancelled by Saheli then: the caregiver is told,
            # or they cancel it in the store's app)
            now_cancelled = task.status == "cancelled"
            await store.record_event(session, family_id=task.family_id, subject_id=task.subject_id,
                                     kind="task_cancelled" if now_cancelled else "task_cancel_requested", actor_id=by,
                                     summary=f"{SKILLS[task.service]['label']}: {'cancelled' if now_cancelled else 'cancel asked'} by {by}: {reason}"[:300],
                                     payload={"task_id": str(task.id), "by": by, "via": trace.via(), "reason": reason[:200], "outcome": out[:200]})
        return out


async def _request_cancel(session: AsyncSession, agent: BrowserAgent, task: Task, *, by: str, reason: str) -> str:
    d = task.details or {}
    _, _, ordered = _words(task)
    # an order the agent may have placed although nobody heard "placed": never "nothing to cancel"
    maybe_placed = bool(d.get("place_unclear") or d.get("unauthorized_place") or d.get("practice_accident"))
    if ((d.get("cancel_ask") or {}).get("state") == "cancelled" or (task.status == "cancelled" and task.phase == "cancel")
            and ((task.result or {}).get("placed") or (task.result or {}).get("booked"))):
        return f"already cancelled on {SKILLS[task.service]['label']}; nothing more to do"
    if task.status in ("failed", "cancelled") and not maybe_placed:
        return f"nothing to cancel: it had already stopped ({task.status}) and nothing was {ordered}"
    if task.status == "scheduled":
        task.status, task.cancel_requested = "cancelled", True
        note(task, f"the booking set for later was cancelled by {by}: {reason}")
        return f"stopped: the {'ride' if task.kind == 'ride' else 'order'} set for later is cancelled; nothing was {ordered}"
    if task.phase == "cancel" and task.status in LIVE:
        return await _cancel_after_placed(session, task, by=by, reason=reason, placing=False)
    task.cancel_requested = True
    note(task, f"cancel requested by {by}: {reason}")
    placed = bool((task.result or {}).get("placed") or (task.result or {}).get("booked"))
    unclear = task.status == "failed" and maybe_placed
    if placed or unclear or (task.phase == "place" and (task.status == "running" or d.get("place_started"))):
        # One true status (live 2026-10-11: "cancelled", then "cancelling is in progress", then "placed, 14 minutes", then
        # a failed cancel run): the store is asked to cancel it and the one result comes in an update (founder 2026-10-12).
        return await _cancel_after_placed(session, task, by=by, reason=reason, placing=not placed and not unclear, unclear=unclear)
    if compare_group(task) and not placed and task.phase in ("prepare", "browse"):
        for t in await siblings(session, task):  # the whole look-up stops, not just one store of it
            if t.id != task.id and t.status in LIVE and t.phase in ("prepare", "browse"):
                if await _stop_before_confirm(agent, t) is None:
                    t.status, t.cancel_requested, t.input_needed = "cancelled", True, None
                    note(t, f"cancelled with the rest of the look-up by {by}")
                else:
                    t.cancel_requested = True
    checked = await _stop_before_confirm(agent, task)
    if checked is not None:
        # A run with the family's login has run: the store's own list is read before anyone hears "nothing was booked"
        # (founder 2026-10-12: "nothing was booked" was said while an auto he never asked for was at his door).
        done_, label_ = _words(task)[0], SKILLS[task.service]["label"]
        if task.agent_session:
            sandbox.audit(task, phase=task.phase, status="stopped", steps=0, path=[], cost=0,
                          error=f"cancelled by the family: {reason}; checking the store"[:120])
        if checked[1]:  # its pages already show it booked or placed: owned, and the store is asked to cancel it
            ref_ = (task.result or {}).get("ride_id") or (task.result or {}).get("order_id")
            return (f"It WAS {done_} by mistake before anyone said yes{f' ({ref_})' if ref_ else ''}: asking {label_} to cancel it "
                    f"now; ONE update comes with the result (a fee, if {label_} wants one, is asked first). Tell them that plainly, "
                    f"own the mistake and say sorry once; never say nothing was {done_}.")
        return (f"STOPPED; checking on {SKILLS[task.service]['label']} now that nothing was {_words(task)[0]} by mistake (the result "
                f"comes in an update in a minute). Tell them only that, in one short line; never say nothing was {_words(task)[0]} "
                "before that update.")
    task.status = "cancelled"
    note(task, "stopped before placing" + (" (looked up without the family's login: nothing can be booked or placed there)"
                                          if _guest_run(task) else ""))
    if task.agent_session:
        sandbox.audit(task, phase=task.phase, status="stopped", steps=0, path=[], cost=0, error=f"cancelled by the family: {reason}"[:120])
    return "stopped; nothing was placed" if task.kind != "ride" else "stopped; nothing was booked"


async def _stop_before_confirm(agent: BrowserAgent, task: Task) -> tuple[str, str | None] | None:
    """Stop a task before its confirm. Nothing ran with the family's login (a look-up runs without it): its run is stopped
    and None comes back (it can be called cancelled). Otherwise the browser stays: the run's own pages so far decide
    (booked or placed → owned and cancelled on the store), else the store's own list is checked; (status, message)."""
    d = task.details or {}
    if task.phase == "check":  # already reading the store's list: it goes on, and a clean list now means stopped
        task.details = {**d, "check": {**(d.get("check") or {}), "then": "cancelled"}}
        return "queued", None
    # only an account that was logged in can have something on it (live 2026-10-13: a check ran for an account whose login
    # code never went in): a run saw it logged in, or its login was alive when a run began
    logged = (int(d.get("login_runs") or 0) > 0 and task.phase in guard.BEFORE_CONFIRM and not _guest_run(task)
              and bool(d.get("seen_logged_in") or d.get("login_at_start")))
    path: list[str] = []
    if task.status == "running" and task.agent_task:
        if logged:
            try:
                path = (await agent.poll(task.agent_task)).path or []
            except Exception:  # noqa: BLE001 — the check reads the store's list instead
                logger.warning("poll before stop failed task=%s", task.id)
        try:
            await agent.stop(task.agent_task, end_session=not logged)
        except Exception:  # noqa: BLE001
            logger.warning("stop failed task=%s", task.id)
    if not logged:
        return None
    hit = guard.past_confirm(task.service, path)
    if hit and hit[0] == "after":
        return _placed_by_mistake(task, ref=guard.ref_in(hit[1]), how=f"the page {hit[1]}")
    return _start_check(task, then="cancelled", why="they asked to stop it")


def _outcome(task: Task, out: dict) -> tuple[str, str]:
    """(new status, what the family should hear) from an agent report for the current phase. Every cart and
    every placement passes the guard: a rule broken in the report stops it here, whatever the agent said."""
    d = task.details or {}
    limits = Limits.from_dict(d.get("limits"))
    spec = specialist_for(task.service)
    if task.phase == "check":
        return _check_outcome(task, out)
    if task.phase == "browse":
        if out.get("blocked"):
            return "failed", f"{SKILLS[task.service]['label']} blocked the request ({out.get('problem') or 'site wall'}). Offer another service."
        return _browse_outcome(task, out)
    login = d.get("login")
    if out.get("needs_otp") and login == "none":
        # It had no number to log in with, so no code can have been sent (live 2026-10-08: "sent to phone", field empty).
        out = {**out, "needs_otp": False, "problem": "login needed"}
    if d.get("otp_entered") and not out.get("logged_in") and (out.get("needs_otp") or str(out.get("problem") or "").lower().startswith("code not accepted")):
        # The code they sent did not go in (PharmEasy, live 2026-10-13: asked six times). Once more with the newest code,
        # then it stops and says so; the agent never asks the store for new codes by itself.
        fails = int(d.get("code_fails") or 0) + 1
        task.details = d = {**d, "code_fails": fails}
        label = SKILLS[task.service]["label"]
        if fails >= CODE_TRIES:
            return "failed", (f"{label}'s login did not take the login code ({fails} codes tried), so nothing was ordered and no more "
                              f"codes are asked for. Say so plainly in one short line, sorry for the codes, and offer another store or "
                              f"ordering in the {label} app.")
        task.input_needed = "otp"
        return "needs_input", (f"The login code did not go in on {label} (it may have been mistyped or expired). Ask them in one short "
                               f"line to send the newest code from the SMS again; if no new one came, task_input kind resend asks {label} "
                               "for a new one.")
    if out.get("needs_otp"):
        task.input_needed = "otp"
        task.details = d = {**d, "codes_sent": int(d.get("codes_sent") or 0) + 1}
        to = f"the phone ending {login}" if login else (out.get("otp_sent_to") or "the family phone")
        if d.get("code_lost"):
            task.details = {**(task.details or {}), "code_lost": False}
            return "needs_input", (f"The store's login page closed before the code could be used, so it sent a new code to {to}. "
                                   "Say sorry briefly and ask for the new code (the earlier one will not work).")
        if d.get("resend_code"):
            return "needs_input", (f"The store sent a NEW login code to {to} (the earlier one never came); ask the person who has that "
                                   "phone for it. If this one does not come either, task_input kind resend asks once more.")
        return "needs_input", (f"The service sent a login code to {to}; ask the person who has that phone for it. If it does not "
                               "come, task_input kind resend asks the store for a new one.")
    if task.phase in ("prepare", "otp") and d.get("login_only") and out.get("logged_in") and not out.get("blocked"):
        # Logged in: back to the fast cart (once more), then the cart comes back for a confirm; or, for a store that showed
        # nothing before the login, a fresh look-up whose options go to the person.
        task.details = {**d, "login_only": False, "login_done": True, "fast_cart_tried": False, "resend_code": False,
                        **({"browsed": False, "relook_after_login": None} if d.get("relook_after_login") else {})}
        return "queued", ""
    if (out.get("problem") or "").lower().startswith("login needed") and not out.get("blocked"):
        task.details = {**(task.details or {}), "login_missing": True}  # true on every store: no next store is tried
        return "failed", (f"{SKILLS[task.service]['label']} needs a login and there is no phone number to log in with, so nothing was "
                          "ordered and no code was sent. Offer another service, or the family can order in the app.")
    if out.get("blocked"):
        return "failed", f"{SKILLS[task.service]['label']} blocked the request ({out.get('problem') or 'site wall'}). Offer another service."
    if task.phase in ("prepare", "otp") and (out.get("placed") or out.get("booked") or out.get("order_id") or out.get("ride_id")):
        # The agent went past the cart without anyone's yes (a bad hint, a misread page). Never place again on top of it:
        # it is owned and the store is asked to cancel it (founder 2026-10-12).
        ref = out.get("order_id") or out.get("ride_id")
        return _placed_by_mistake(task, ref=str(ref) if ref else None, how="its own report")
    if task.phase in ("prepare", "otp"):
        if task.kind == "ride":
            if not out.get("options"):
                return "failed", f"No ride options came up ({out.get('problem') or 'unknown'})."
            v = guard.check_cart("ride", spec.name, [], out, limits)
            task.details = {**d, "approval": v.approval, "confirm_token": None,
                            "cart_fp": guard.cart_fingerprint({"items": [{"name": f"{o.get('type')} {o.get('fare')}"} for o in out["options"]], "total": None})}
            task.result = {**(task.result or {}), "options": out["options"]}
            task.input_needed = "choice"
            metrics.on_milestone(task, "ready")
            return "awaiting_confirm", _ride_choice_message(task, " ".join(v.warn) + _approval_note(v))
        if out.get("needs_prescription") and not guard.rx_covers(d.get("items") or [], limits.rx_on_file, cart=out.get("items")):
            return "failed", "This medicine needs a prescription upload; ask the family to upload the prescription on the dashboard, then try again."
        dead = [i for i in out.get("items") or [] if i.get("available") is False or str(i.get("qty", 1)).strip() in ("0", "0.0")]
        if dead:
            # Live 2026-10-10: the agent reported "0 x Prolicious Khakhra" (not sold here) as a cart and Maa was asked to pick
            # from four ₹246 packs. Saheli picks the next best product herself and builds the cart again.
            names = [str(i.get("name") or "").lower() for i in dead]
            ids = [str(c.get("store_id")) for c in d.get("chosen") or [] if c.get("store_id")
                   and (len(d.get("chosen") or []) == 1 or any(str(c.get("name") or "").lower()[:25] in n or n[:25] in str(c.get("name") or "").lower() for n in names))]
            tried = _repick_without(task, ids)
            if tried:
                task.phase, task.input_needed = "prepare", None
                task.result = {**(task.result or {}), "alternatives": None, "total": None, "fees": None}
                note(task, f"not available here; trying {tried}")
                return "queued", ""
            again = _reoffer_self(task, ids, "not available at the store right now") if ids else None
            if again:
                return "awaiting_confirm", again
            out = {**out, "items": [i for i in out.get("items") or [] if i not in dead]}
        chosen_ = [c for c in d.get("chosen") or [] if c.get("name")]
        wanted = [{"name": c["name"]} for c in chosen_] + [{"name": a} for a in d.get("menu_find") or []]
        extra = [str(i.get("name")) for i in out.get("items") or [] if i.get("name") and guard.missing_items([{"name": i["name"]}], wanted)
                 and guard.missing_items(wanted, [i]) == [w["name"] for w in wanted]] if chosen_ and spec.name != "pharmacy" else []
        if extra and not d.get("cart_cleaned"):
            # Items nobody asked for in this order (live run 2026-10-11: rotis left in the cart from earlier runs were offered):
            # the cart is emptied and built again once, before anyone is asked.
            task.details = {**d, "cart_cleaned": True}
            task.phase, task.input_needed = "prepare", None
            task.result = {**(task.result or {}), "total": None, "fees": None}
            note(task, f"the cart had items nobody asked for ({', '.join(extra)[:120]}); emptying it and building it again")
            return "queued", ""
        if chosen_ and d.get("auto_picked") and spec.name != "pharmacy" and out.get("items"):
            # A picked product the agent left out of the cart (live run 2026-10-11: Indian Bawarchi's paneer failed to go in three
            # times and Maa was offered a roti-only cart): the next best one is tried, before anyone is asked.
            left_out = guard.missing_items([{"name": c["name"]} for c in chosen_], out.get("items") or [])
            ids = [str(c.get("store_id")) for c in chosen_ if c.get("store_id") and str(c.get("name")) in left_out]
            tried = (_repick_food(task, ids) if SKILLS[task.service].get("kind") == "food" else _repick_without(task, ids)) if ids else None
            if tried:
                task.phase, task.input_needed = "prepare", None
                task.result = {**(task.result or {}), "alternatives": None, "total": None, "fees": None}
                note(task, f"left out of the cart: {', '.join(left_out)}; trying {tried}")
                return "queued", ""
        if not out.get("items"):
            alts = guard.alternatives(out.get("alternatives"))[:4]
            if alts and out.get("cod_available") is not False and spec.name != "pharmacy":
                # Out of stock at the cart: what the store has instead, and its other options, go back to the person as
                # numbered options (marked similar); nothing is swapped without their pick.
                return "awaiting_confirm", _offer_instead(task, alts, out.get("problem") or "out of stock")
            return "failed", f"Could not build the cart ({out.get('problem') or 'unknown'})."
        asked = d.get("items") or []
        if d.get("chosen") and spec.name != "pharmacy":
            # The person picked the products: check the cart against those, not the words asked (order lab 2026-10-10:
            # "biscuits, munchies" were reported "not in the cart" next to Dark Fantasy and Potazos). Medicines keep the
            # asked names: never substituted.
            asked = [{"name": c.get("name"), "qty": channels._qty_for(asked, c) if asked else 1} for c in d["chosen"] if c.get("name")]
        v = guard.check_cart("order", spec.name, asked, out, limits)
        if not v.ok:
            if v.family_rule:  # the care record or a prescription: the next store would be stopped the same way
                task.details = {**(task.details or {}), "rule_block": True}
            return "failed", "Stopped before placing: " + "; ".join(v.block) + ". Nothing was placed."
        if not _plain_amount(out.get("total")) and not d.get("total_relook"):
            # Live 2026-10-11: the cart came back without a total and Maa was asked to confirm with no amount; later the same
            # day a Blinkit agent reported "₹320 (estimated based on ₹80 per item; full bill summary hidden)". A total must be
            # an amount read off the checkout: else one short look at the checkout for it, changing nothing (founder: one
            # confirm WITH the amount).
            task.details = {**d, "total_relook": True, "total_only": True}
            note(task, f"cart reported without a checkout total ({str(out.get('total') or 'none')[:60]}): looking at the checkout for it")
            return "queued", ""
        d = {**d, "total_only": False}
        task.details = {**d, "approval": v.approval, "cart_fp": guard.cart_fingerprint(out), "confirm_token": None}
        if not out.get("alternatives"):
            task.result = {**(task.result or {}), "alternatives": None}  # an earlier report's, not this cart's
        prev = d.get("replace_confirmed") or {}
        same = (task.details["cart_fp"] == prev.get("fp") or bool(prev.get("items_fp")) and guard.items_fingerprint(out) == prev["items_fp"])
        if (prev.get("fp") and prev.get("by") and same and not v.approval
                and int(d.get("auto_replaced") or 0) < 2
                and guard.within_leeway(rupees(out.get("total")), rupees(prev.get("total")))):
            # Rebuilt after a place step could not send (lab: a changed site): the same products at a total within the
            # leeway of the one they agreed (founder 2026-10-12: ₹900 may be ₹920) go on the yes they already gave.
            task.details = {**task.details, "replace_confirmed": None, "auto_replaced": int(d.get("auto_replaced") or 0) + 1}
            note(task, "same cart and total as confirmed: placing on that yes")
            _start_placing(task, prev["by"])
            return "queued", ""
        task.details = {**task.details, "replace_confirmed": None}
        task.input_needed = "confirm"
        metrics.on_milestone(task, "ready")
        extra = " ".join(v.warn) + _approval_note(v)
        return "awaiting_confirm", _confirm_ask(task, out) + (f" {extra.strip()}" if extra.strip() else "")
    if task.phase == "place":
        if practice.on(task) and (out.get("placed") or out.get("booked") or out.get("order_id")):
            # A practice must never place: whatever the agent did, the family checks the app at once.
            task.details = {**(task.details or {}), "practice_accident": True}
            logger.error("PRACTICE ORDER REPORTED PLACED task=%s order=%s", task.id, out.get("order_id"))
            return "failed", (f"This was a practice order, but {SKILLS[task.service]['label']} may have placed a real order"
                              + (f" (order id {out.get('order_id')})" if out.get("order_id") else "") + ". Tell the caregiver right "
                              "away to check the app and cancel it there; do not order again.")
        if out.get("price_changed") and (d.get("later") or {}).get("by"):
            lt = dict(d["later"])
            shown = out.get("new_total") or "a higher amount"
            lt.update(state="fare_ask", asked={"choice": f"{((d.get('ride_pick') or {}).get('type') or '').strip()} {shown}".strip()
                                               if task.kind == "ride" else None, "amount": shown})
            task.details = {**d, "later": lt, "confirm_token": None, "place_started": None}
            task.phase, task.input_needed = "prepare", "fare"
            return "needs_input", (f"At the booking time the {'fare' if task.kind == 'ride' else 'total'} on {SKILLS[task.service]['label']} is "
                                   f"{shown}, above the ₹{lt.get('max') or 0:.0f} they set; nothing was {'booked' if task.kind == 'ride' else 'ordered'}. "
                                   f"Ask ONE short question at once: {'book' if task.kind == 'ride' else 'order'} it at {shown}? Yes → task_input fare "
                                   "yes; no → task_input fare no.")
        if out.get("price_changed") and task.kind == "ride":
            # Fares moved at booking: fetch fresh fares and let the person choose again.
            task.phase, task.input_needed = "prepare", None
            task.details = {**d, "confirm_token": None, "place_started": None, "choice": None, "cart_fp": None}
            return "queued", f"The fare changed at booking ({out.get('new_total') or 'new fare'}); nothing was booked. Getting fresh fares to choose again."
        if out.get("price_changed"):
            task.result = {**(task.result or {}), "total": out.get("new_total") or (task.result or {}).get("total")}
            task.details = {**d, "cart_fp": guard.cart_fingerprint(task.result), "confirm_token": None}
            task.input_needed = "confirm"
            return "awaiting_confirm", f"The total changed to {out.get('new_total')} since it was confirmed; nothing was placed. Read the new total and ask again."
        if practice.on(task):
            return "cancelled", _practice_message(task, out)
        if out.get("placed") or out.get("booked"):
            v = guard.check_placed(task.kind, out, guard.rupees(d.get("confirmed_total")))
            metrics.on_milestone(task, "placed")
            if v.warn:
                task.details = {**d, "placed_warn": v.warn}  # said by the brain (it tells the caregiver), not by code
                return "done", "Placed, but: " + "; ".join(v.warn)
            details = "; ".join(ride_details(task) if task.kind == "ride" else order_details(task))
            if task.kind == "ride" and d.get("ends_wrong"):
                return "done", (f"Booked, BUT {SKILLS[task.service]['label']} set {d['ends_wrong']}: the driver will go there. Tell "
                                f"them this first, plainly: {bought(task)}; {details}. They can call the driver with the right "
                                "address, or ask to cancel it. Never say it is from the place they meant.")
            if task.kind == "ride":
                # founder 2026-10-12: the booking details never reached him
                return "done", (f"Booked: {bought(task)}; {details}. Tell the person all of these details as given, in one message; "
                                "never add a pickup time, PIN or detail the app did not give.")
            what = (f"Placed: {bought(task)}; {details}. Tell the person what was bought by these names (never the words they asked "
                    "with; say so plainly when it is not exactly what they asked), the order id, the total and the delivery time")
            if str(out.get("eta") or "").strip():
                return "done", f"{what}."
            # Live 2026-10-10: connector and Blinkit orders came back without a time and the brain was told to give one.
            return "done", f"{what}. The store gave no delivery time: do not guess one; say you will check on it."
        if out.get("unclear") or (task.kind == "order" and out.get("pressed_place") is not False):
            # Live 2026-10-11: the agent pressed Pay Now (cash), ran out of steps on the confirmation page, reported "not
            # placed", and the person heard "nothing was charged" while the order was being packed. Only a report that the
            # final button was never pressed is "not placed"; anything else is unclear and the store's order list is read
            # next (_order_on_store_list) before anyone hears anything.
            task.details = {**(task.details or {}), "place_unclear": True}
            return "failed", (f"It is not clear whether the order went through ({out.get('problem') or 'no confirmation came'}). Do not "
                              "order again and do not say it failed; tell them to check the store's app, and the caregiver too.")
        return "failed", f"Placing did not go through ({out.get('problem') or 'unknown'}); the final button was never pressed. Nothing was charged."
    if task.phase == "cancel":
        label = SKILLS[task.service]["label"]
        done, thing, _ = _words(task)
        ask = dict(d.get("cancel_ask") or {})
        fee = out.get("cancel_fee")
        if fee in (None, "", False) and d.get("fee_ok"):
            fee = d.get("cancel_fee")  # the fee they agreed to (or the free one) on the run before
        amount = rupees(fee) if fee not in (None, "", False) else None
        if out.get("cancelled"):
            metrics.on_milestone(task, "cancelled")
            ask.update(state="cancelled", told_by="update")
            task.details = {**d, "cancel_ask": ask}
            paid = f" (a cancellation fee of {fee}, as they agreed)" if amount else ""
            also = (f" Also tell person {ask['first_by']} in one short line with send_message that it is cancelled (they had asked "
                    "for it first)." if ask.get("first_by") else "")
            return "cancelled", (f"Cancelled on {label}: {label} confirmed the cancellation of the {thing}{paid}. Tell them in one short "
                                 "line that it is cancelled" + ("; nothing is to be paid" if not amount else "") + "." + also)
        if out.get("cancel_fee") not in (None, "", False) and not d.get("fee_ok"):
            if amount == 0:
                # Free: no second ask (founder 2026-10-12: "after cancellation it asked me to confirm" for a ₹0 fee).
                task.details = {**d, "fee_ok": True, "cancel_fee": fee}
                note(task, f"{label} shows a cancellation fee of {fee}: it is free, so the cancel goes ahead without asking again")
                return "queued", ""
            task.input_needed = "fee"
            ask.update(state="fee")
            task.details = {**d, "cancel_ask": ask, "cancel_fee": fee}
            return "needs_input", (f"{label} will cancel it, but cancelling costs {fee}. Ask ONE short question with that amount: cancel "
                                   f"it anyway for {fee}? Yes → task_input fee yes; no → task_input fee no (it stays {done}). Nothing is "
                                   "cancelled before they say yes.")
        # The store would not cancel it (too late, or its page did not let it): it stays placed, and what to do instead.
        why = str(out.get("problem") or "it did not allow cancelling now")[:160]
        ask.update(state="refused", told_by="update", store_said=why)
        task.details = {**d, "cancel_ask": ask}
        return "done", (f"NOT cancelled: {label} would not cancel it ({why}); it stays {_placed_line(task)}. "
                        + _cancel_tell(task, ask, why=why))
    return "failed", "Unexpected state."


async def _ride_ends_check(task: Task, out: dict) -> tuple[str, str] | None:
    """A ride's pickup and drop as the app set them must be the places the person meant (an AI check; live 2026-10-12: a
    saved home became the city in the app, the ride was booked from there, and Saheli said it was their
    address). Another place before booking: that step is done again once, told what was wrong; still another: nothing is
    booked and they hear it plainly. Booked from another place: they hear it first (_outcome). None: the report goes on."""
    d = task.details or {}
    if task.kind != "ride" or task.phase not in ("browse", "prepare", "place") or out.get("blocked") or out.get("needs_otp"):
        return None
    from app.specialists.agents import ride_end
    from app.tasks import matcher

    # a shared map point with no name or address has nothing to compare words with (the agent sets it by its point)
    ends = [e for e in ("pickup", "drop") if str(out.get(f"{e}_used") or "").strip() and ride_end(d, e)
            and not ((d.get(f"{e}_place") or {}).get("pin") and not any(ch.isalpha() for ch in ride_end(d, e)))]
    verdicts = await asyncio.gather(*(matcher.same_ride_end(ride_end(d, e), str(out[f"{e}_used"]).strip(),
                                                            saved=bool((d.get(f"{e}_place") or {}).get("full"))) for e in ends))
    wrong = {e: str(out[f"{e}_used"]).strip()[:120] for e, same in zip(ends, verdicts) if same is False}
    booked = bool(out.get("booked") or out.get("ride_id"))
    if out.get("pickup_mismatch") and not booked:
        wrong.setdefault("pickup", str(out.get("pickup_used") or "another place").strip()[:120])
    if not wrong:
        if d.get("ends_fix"):
            task.details = {**d, "ends_fix": None}
        return None
    label = SKILLS[task.service]["label"]
    said = " and ".join(f"the {e} to '{used}' instead of '{ride_end(d, e)[:120]}'" for e, used in wrong.items())
    if booked:
        task.details = {**d, "ends_wrong": said}
        return None
    task.result = {**(task.result or {}), "options": None}  # fares for another route: never offered
    if d.get("ends_retry") != task.phase:
        task.details = {**d, "ends_retry": task.phase, "ends_fix": said, "place_started": None}
        task.input_needed = None
        note(task, f"{label} set {said}; doing this step again")
        return "queued", ""
    task.details = {**d, "ends_fix": None, "compare_reason": f"it kept setting {said}, so nothing from it"}
    return "failed", (f"Nothing was booked: {label} kept setting {said}. Tell them plainly in one or two short lines; they can "
                      "give a landmark near it, or book it in the app themselves.")


async def _check_address(task: Task, out: dict, status: str, message: str | None) -> tuple[str, str | None]:
    """A browser cart about to be confirmed: the checkout's address must be the family's saved place (an AI check). Another
    address → the agent switches it once; still another → stopped, nothing ordered."""
    d = task.details or {}
    place = (d.get("limits") or {}).get("place") or {}
    if d.get("address_fix_pending"):
        task.details = d = {**d, "address_fix_pending": False}
    if not (task.phase == "prepare" and status == "awaiting_confirm" and task.kind == "order" and out.get("address_used") and place.get("full")):
        return status, message
    from app.tasks import matcher

    same = await matcher.same_place(str(place["full"]), str(place.get("pincode") or ""), str(out["address_used"]))
    if same is not False:
        return status, message
    task.input_needed = None
    if not d.get("address_fix"):
        task.details = {**d, "address_fix": 1, "address_fix_pending": True, "cart_fp": None, "confirm_token": None}
        note(task, f"the cart is on another address ('{str(out['address_used'])[:80]}'); switching it to {place.get('nickname')}")
        return "queued", None
    label = SKILLS[task.service]["label"]
    return "failed", (f"Stopped before the confirm: {label} keeps using another delivery address ('{out['address_used']}') instead of "
                      f"{place.get('nickname')} ({place['full']}); nothing was ordered. Tell them plainly and tell the caregiver to check "
                      "the saved addresses in the app.")


def _practice_message(task: Task, out: dict) -> str:
    """What the family hears when a practice order stopped (app/tasks/practice.py)."""
    d = task.details or {}
    where = ((d.get("limits") or {}).get("place") or {}).get("nickname") or "their saved place"
    items = "; ".join(_label(c) + (f" x{c['qty']}" if str(c.get("qty") or 1) != "1" else "") for c in d.get("chosen") or []) or task.goal
    total = out.get("total") or d.get("confirmed_total") or (task.result or {}).get("total")
    if out.get("reached_final_step"):
        button = f" ('{out['final_button']}')" if out.get("final_button") else ""
        return (f"Practice order (the family turned on practice orders for testing): everything was done up to the store's final "
                f"place-order button{button}: {items}, total {total}, cash on delivery, to {where}. It was NOT placed: nothing will "
                "come and nothing is to be paid. Tell them in one short line that it was a practice order and nothing was ordered.")
    return (f"Practice order (the family turned on practice orders for testing): it stopped before the last step "
            f"({out.get('problem') or 'unknown'}); nothing was placed. Tell them in one short line that it was a practice order and "
            "nothing was ordered.")


def _approval_note(v) -> str:
    """The cart or fare needs a caregiver's OK as well: the person who asked is still asked first (sim 2026-10-12: told
    "I have asked Mohan to approve" with no question and no buttons, before anyone was asked)."""
    return (f" It also needs an approver's OK ({'; '.join(v.approval)}): still ask THEM this question now, with the buttons, and "
            "add that after their yes the approver is asked; nobody has been asked yet." if v.approval else "")


def _confirm_ask(task: Task, out: dict) -> str:
    """The one question of an order (founder 2026-10-10: what, where, then confirm with the amount)."""
    d = task.details or {}
    place = (d.get("limits") or {}).get("place") or {}
    where = place.get("nickname") or "their saved place"
    near = [_label(c) for c in d.get("chosen") or [] if c.get("exact_match") is False]
    missing = [m for m in d.get("missing_items") or [] if m]
    eta = f", delivery in about {out['eta']}" if out.get("eta") else ""
    return (f"The cart is ready on {SKILLS[task.service]['label']}. Ask ONE short question: the items (name, pack, quantity), the "
            f"total {out.get('total') or 'shown'} with fees, to {where}{eta}, cash on delivery, shall I order? Nothing else to ask. "
            f"Attach buttons {{kind: order, key: {task.id}:{str(d.get('cart_fp') or '')[:8]}}} to that message (yes / change / cancel)."
            + (f" Say these are the closest match to what they asked: {'; '.join(near)}." if near else "")
            + (f" Not on {SKILLS[task.service]['label']} (only other kinds), so NOT in the cart: {', '.join(missing)}. Say this plainly; "
               "never call another kind the same thing. They can add something else or get it from another store." if missing else "")
            + " If they want another product or size for an item, more of an item, or one item added or removed: task_input "
              "change / add / remove (or more to see that item's options); the new total comes back for the same one confirm.")


def _price(i: dict) -> float:
    try:
        return float(re.sub(r"[^\d.]", "", str(i.get("price") or "")) or 0)
    except ValueError:
        return 0.0


def _restaurants(groups: list[tuple[str, list[dict]]], matched: dict) -> list[tuple[str, dict[str, dict]]]:
    """Food: restaurants with an exact dish for the asked items, best first: the most asked items, the first (the meal)
    among them, then the lowest total. Each with its best exact dish per item (the AI's order)."""
    by_rest: dict[str, dict[str, dict]] = {}
    for asked, g in groups:
        byk = {_pkey(i): i for i in g}
        for k in (matched.get(asked) or {}).get("exact") or []:
            i = byk.get(k)
            if i:
                by_rest.setdefault(str(i.get("restaurantId") or i.get("restaurant") or ""), {}).setdefault(asked, i)
    first = groups[0][0] if groups else ""  # what they asked for first is the meal; a side dish alone never picks the restaurant
    return sorted(by_rest.items(), key=lambda r: (-len(r[1]), first not in r[1], sum(_price(i) for i in r[1].values())))


def _one_restaurant(groups: list[tuple[str, list[dict]]], matched: dict) -> tuple[dict, list[str]]:
    """Food: the restaurant with an exact dish for the most asked items (then the lowest total) and each item's best exact
    dish there (the AI's order). Items it has no exact listing for are looked for in its own menu by the agent. Returns
    (picks by asked item, items to find in that restaurant's menu)."""
    ranked = _restaurants(groups, matched)
    if not ranked:
        return {}, []
    picks = {asked: ranked[0][1].get(asked) for asked, _ in groups}
    return {a: p for a, p in picks.items() if p}, [a for a, _ in groups if not picks.get(a)]


def _closed_note(out: dict) -> str:
    """Restaurants in a food look-up that have it but are closed now, with when they open (the store's own words)."""
    seen, bits = set(), []
    for i in out.get("items") or []:
        r = str(i.get("restaurant") or "")
        if i.get("closed") and r and r not in seen:
            seen.add(r)
            bits.append(r + (f" ({i['opens']})" if i.get("opens") else ""))
    return "; ".join(bits[:4])


def _browse_outcome(task: Task, out: dict) -> tuple[str, str]:
    """What the look-up found (no login yet): not delivered there, not found, or the options for the person to pick."""
    d = task.details or {}
    label = SKILLS[task.service]["label"]
    place = (d.get("limits") or {}).get("place") or {}
    where = place.get("nickname") or place.get("pincode") or "their place"
    task.details = {**d, "browsed": True}
    if out.get("deliverable") is False:
        return "failed", (f"{label} does not deliver to {where} ({out.get('location_set') or out.get('problem') or 'not serviceable'}); "
                          "nothing was ordered. Offer another service.")
    if task.kind == "ride":
        opts = [o for o in (out.get("options") or []) if o.get("type")]
        if opts and (not out.get("logged_in") or compare_group(task)):
            # Founder 2026-10-11: the fares are the options, from every app asked; the person picks, then the one confirm
            # with the fare after the login.
            task.input_needed = "go"
            task.result = {**(task.result or {}), "options": opts}
            task.details = {**task.details, "offer": None}
            if compare_group(task):
                return "awaiting_confirm", f"{label} fares: " + "; ".join(f"{o.get('type')} {o.get('fare') or ''}".strip() for o in opts[:6])
            return "awaiting_confirm", _ride_options_message(task)
        if opts or out.get("logged_in"):
            task.phase = "prepare"  # logged in already: the fresh fares come back to pick from (their pick is the yes)
            return "queued", ""
    found = [i for i in (out.get("items") or []) if i.get("name") and i.get("available") is not False]
    if task.kind == "ride" and not out.get("login_required"):
        return "failed", f"No ride options came up on {label} ({out.get('problem') or 'unknown'}); nothing was booked. Offer another service."
    if not found and out.get("login_required") and task.kind == "ride":
        task.input_needed = "go"
        return "awaiting_confirm", (f"{label} shows fares only after a login; nothing is booked. To see the fares there: {_code_line(task)}. "
                                    "Ask whether to go ahead (task_input go); the fares come back for them to pick, nothing is booked "
                                    "before their pick and yes.")
    if not found and out.get("login_required"):
        if compare_group(task):
            task.input_needed = "go"
            return "awaiting_confirm", f"{label} shows prices only after a login"
        if d.get("login_done"):
            return "failed", f"{label} shows nothing for this even after logging in ({out.get('problem') or 'nothing listed'}); nothing was ordered."
        # The site shows nothing without a login: it logs in (the code is asked if one comes), then looks again, and the
        # options go to the person (founder 2026-10-11: options at all times).
        task.details = {**task.details, "login_only": True, "relook_after_login": True}
        task.phase = "prepare"
        return "queued", ""
    closed = _closed_note(out)
    if not found and closed:
        # Live 2026-10-11 night: every place that has it is shut; say so, with when it opens (not "not found").
        return "failed", (f"On {label} the places that have it are closed now ({closed}); nothing was ordered. Tell them plainly, "
                          "with when it opens, and offer to order it then or something from a place that is open now.")
    instead = False
    if not found:
        alts = guard.alternatives(out.get("alternatives"))[:4]
        if not alts or d.get("agent") == "pharmacy":
            if d.get("store_down"):
                return "failed", f"{label}'s website did not work just now ({out.get('problem') or 'no answer'}); nothing was ordered."
            return "failed", f"Could not find it on {label} ({out.get('problem') or 'not found'}); nothing was ordered. Offer another service."
        # Not there; the store has these instead: offered as options marked similar, and the other stores look too.
        found = [{"name": a["name"], "price": a.get("price"), "available": True, "exact_match": False} for a in alts]
        out = {**out, "items": found}
        instead = True
    more = d.get("more")
    medicine = d.get("agent") == "pharmacy"
    food = SKILLS[task.service].get("kind") == "food"
    if (task.result or {}).get("items") is not out.get("items"):
        task.result = {**(task.result or {}), "items": out.get("items") or []}  # the options are read from the task's result
    if instead:
        task.details = {**task.details, "matched": None}  # what the store has instead: none of it is what was asked
    _number(found)
    groups = _groups(task, found)
    matched = None if instead else d.get("matched")
    missing = [m.removeprefix("not found: ") for m in [str(out.get("problem") or "")] if m.startswith("not found: ")]
    missing = missing[0].split(", ") if missing else []
    # Founder 2026-10-11: options at all times. What was found goes to the person, numbered with Saheli's suggestion
    # marked; they pick, then the one confirm with the amount. The look-up's products are kept (a cart report replaces
    # the result's items, and a product the store refuses later needs the others).
    task.details = {**task.details, "found": [{k: v for k, v in i.items() if k != "image_url"} for i in found][:40],
                    "chosen": None, "offer": None, "menu_find": None, "missing_items": missing or None, "similar_only": None,
                    "auto_picked": None, "refused": None}
    task.input_needed = "go"
    picks: dict[str, dict | None] = {}
    if matched is not None:
        used: set[str] = set()
        # items with the fewest exact choices pick first, so no item takes the only product another one has
        for asked, g in sorted(groups, key=lambda x: len((matched.get(x[0]) or {}).get("exact") or [])):
            keys = (matched.get(asked) or {}).get("exact") or []
            byk = {_pkey(i): i for i in g}
            # the AI's exact listings, best first; each item its own product
            picks[asked] = next((byk[k] for k in keys if k in byk and k not in used), None)
            if picks[asked]:
                used.add(_pkey(picks[asked]))
    menu_find: list[str] = []
    if matched is not None and food and len(groups) > 1:
        # A food cart holds one restaurant (live 2026-10-11: paneer from one place, roti from another; only the roti went in).
        picks, menu_find = _one_restaurant(groups, matched)
    kept: dict[str, dict] = {}
    if not food or len(groups) == 1:
        # A change at the confirm: the products already picked for the items that stayed the same are kept, if still listed.
        for asked, g in groups:
            k = (d.get("kept") or {}).get(asked)
            same = next((i for i in g if k and _pkey(i) == _pkey(k)), None)
            if same is not None:
                kept[asked] = {**same, "exact_match": k.get("exact_match", True)}
        picks.update(kept)
    task.details = {**task.details, "kept": kept or None}
    if kept and all(a in kept for a, _ in groups) and not more:
        task.details = {**task.details, "chosen": [kept[a] for a, _ in groups], "auto_picked": False, "kept": None}
        task.phase = "prepare"  # nothing new to pick: the new cart comes back for the one confirm
        return "queued", ""
    if more:
        # They asked to see more of one item: its listings in the store's order (the other items keep their pick).
        task.details = {**task.details, "chosen": [p if p.get("exact_match") is False else {**p, "exact_match": True}
                                                   for p in picks.values() if p] or None,
                        "auto_picked": not kept, "menu_find": menu_find or None}
        return "awaiting_confirm", _options_message(task, out, listing=True, only=more)
    named = [str(i["restaurant"]) for i in d.get("items") or [] if isinstance(i, dict) and i.get("restaurant")]
    if matched is not None and groups and not any(picks.values()) and named and food and not compare_group(task):
        # The restaurant they named does not have it here (or is shut): other places on this same app do. Their options,
        # before going to another app (live run 2026-10-11: "Haldiram se" → Zomato, though other Swiggy places had the dish).
        return "awaiting_confirm", _options_message(task, out, listing=True, head=(
            f"{', '.join(named)} on {label} does not have it now" + (f" (closed now: {closed})" if closed else "")
            + f". Other places on {label} have it. Tell them plainly and ask whether to get it from one of these places instead, "
            "or wait for the one they named. "))
    exact_any = any(picks.values())
    if not medicine and (not food or len(groups) == 1) and (compare_group(task) or matched is None or not exact_any):
        # The suggestion for an item with no exact listing here is its closest one, marked as such: in a comparison (so the
        # store with exact matches is suggested), when nothing here is exact, or when the AI could not choose (the store's
        # own top listing).
        taken = {_pkey(p) for p in picks.values() if p}
        for asked, g in groups:
            if picks.get(asked) or asked in menu_find or not g:
                continue
            k = (matched.get(asked) or {}).get("closest") if matched is not None else _pkey(g[0])
            c = next((i for i in g if _pkey(i) == k), None) if k else None
            if c is not None and _pkey(c) not in taken:
                picks[asked] = {**c, "exact_match": False}
                taken.add(_pkey(c))
    # an exact pick says so (the store comparison prefers exact over closest; a fast look-up's listing carries no flag)
    chosen = [picks[a] if picks[a].get("exact_match") is False else {**picks[a], "exact_match": True} for a, _ in groups if picks.get(a)]
    task.details = {**task.details, "chosen": chosen or None, "auto_picked": True if chosen else None, "menu_find": menu_find or None}
    if instead:
        task.details = {**task.details, "similar_only": True}
        if compare_group(task):
            return "awaiting_confirm", f"{label} does not have it; it has instead: " + "; ".join(_option(i) for i in found[:4])
        return "awaiting_confirm", _options_message(task, out, listing=True, head=(
            f"Not on {label} ({out.get('problem') or 'not found'}); it has these instead, none exactly what was asked. Say so "
            "plainly. "))
    if compare_group(task):
        # Every store's options go out together once every store has looked (choose_stores).
        return "awaiting_confirm", (f"{label} has: " + "; ".join(_option(i) for i in chosen) if chosen else
                                    f"{label} has nothing exactly this")
    if not chosen and not medicine and matched is not None:
        # Nothing here is what they asked for, not even close: the other usual stores look.
        return "failed", (f"Not on {label} ({'; '.join(a for a, _ in groups)})"
                          + (f". Closed now on {label}: {closed}" if closed else "") + "; looking on the other stores.")
    if not chosen and medicine:
        return "awaiting_confirm", _options_message(task, out, listing=True, head=(
            f"Nothing on {label} is exactly this medicine (the same name and strength), and medicines are never swapped for a "
            "near match: say so plainly; if none of these is exactly the one on the prescription, suggest checking the exact "
            "name with the caregiver or doctor. "))
    if matched is not None and not exact_any:
        # Only close variants here: they are offered, and the other usual stores look too (_fall_back_to_other_stores).
        task.details = {**task.details, "similar_only": True}
        return "awaiting_confirm", _options_message(task, out, head=(
            f"Nothing on {label} is exactly what they asked; these are the closest (marked similar). Say so plainly. "))
    return "awaiting_confirm", _options_message(task, out, listing=matched is None)


def product_page(service: str, item: dict) -> str | None:
    """The store's own page for a product, when its id is known (checked live 2026-10-10: Blinkit opens /prn/x/prid/<id>)."""
    ref = item.get("cart_ref") or {}
    if service == "blinkit" and (ref.get("product_id") or item.get("store_id")):
        return f"https://blinkit.com/prn/x/prid/{ref.get('product_id') or item.get('store_id')}"
    return None


def _pkey(p: dict) -> str:
    """A store product's own id (what the AI's choice and a refused cart refer to)."""
    ref = p.get("ref") or p.get("cart_ref") or {}
    return str(p.get("store_id") or ref.get("spinId") or ref.get("product_id") or p.get("item_id") or p.get("name") or "")


def _number(found: list[dict]) -> None:
    """Short ids for the options shown to the person (the brain passes them back; nothing is matched by name)."""
    for n, i in enumerate(found, 1):
        i.setdefault("oid", f"p{n}")


def _canon_for_item(names: list[str], found: list[dict]) -> None:
    """A look-up's for_item tags as the asked items' own names. A browser agent echoes the goal ("4 x RiteBite Choco Berry
    protein bar") or writes the item's number; live 2026-10-11 the picker then got no listings for "RiteBite Choco Berry
    protein bar", chose nothing, and the person was sent a list. Our own item names only; never the person's words."""
    if not names:
        return
    for i in found:
        tag = str(i.get("for_item") or "").strip()
        if not tag or tag in names:
            continue
        if tag.isdigit() and 1 <= int(tag) <= len(names):
            i["for_item"] = names[int(tag) - 1]
            continue
        hits = [n for n in names if n and n.lower() in tag.lower()]
        if len(names) == 1 or len(hits) == 1:
            i["for_item"] = names[0] if len(names) == 1 else hits[0]
        else:
            i.pop("for_item", None)  # cannot tell which item: the picker chooses it for whichever item it fits


async def family_tastes(session: AsyncSession, task: Task) -> list[str]:
    """What the AI matcher should know about this family's shopping: products they bought before (placed orders, last 90
    days) and their likes, diet and never-order notes from memory. Data only; the AI decides what fits."""
    out: list[str] = []
    rows = (await session.execute(
        select(Task).where(Task.family_id == task.family_id, Task.kind == "order", Task.status == "done",
                           Task.created_at >= clock.now() - timedelta(days=90)).order_by(Task.created_at.desc()).limit(30)
    )).scalars()
    bought: dict[tuple[str, str], int] = {}
    for t in rows:
        if not (t.result or {}).get("placed"):
            continue
        for c in (t.details or {}).get("chosen") or []:
            name = f"{_label(c)}".strip()
            if name:
                bought[(SKILLS[t.service]["label"], name)] = bought.get((SKILLS[t.service]["label"], name), 0) + 1
    # what they said about it after it came (liked, a first try, not again) is in the preference and never-order facts below
    for (label, name), n in sorted(bought.items(), key=lambda x: -x[1]):
        out.append(f"bought before on {label}: {name}" + (f" ({n} times)" if n > 1 else ""))
    try:
        facts = await store.facts(session, task.family_id, task.subject_id, domains=["preference", "diet", "no_order", "allergy", "dish"],
                                  statuses=("active",))
        out += [f"{f.domain}: {f.text}" for f in facts if f.text][:20]
    except Exception:  # noqa: BLE001 — tastes are a hint, never a blocker
        logger.warning("family tastes unavailable task=%s", task.id)
    return out[:40]


async def _ai_pick(task: Task, report: dict, session: AsyncSession | None = None) -> None:
    """The AI chooses the product for each asked item from the look-up (founder 2026-10-10: the AI decides, not word
    rules, since people order in many languages). Saved as details.matched {asked name: {exact: [ids], closest, why}};
    None when it could not answer (the options are then shown to the person)."""
    d = task.details or {}
    found = [i for i in report.get("items") or [] if i.get("name") and i.get("available") is not False]
    items = [i for i in d.get("items") or [] if isinstance(i, dict)]
    if task.kind != "order" or not found or not items:
        return
    _number(found)
    names = [str(i.get("name") or "") for i in items]
    _canon_for_item(names, report.get("items") or [])
    tagged = any(i.get("for_item") for i in found)
    if tagged:
        for i in found:
            i.setdefault("for_item", names[0])  # an untagged product of a tagged report: the first item's (as before)
        lists = [[i for i in found if i.get("for_item") == n] for n in names]
    else:  # one look-up for everything (a browser agent's report): every item is chosen from all of it
        lists = [found for _ in names]
    from app.tasks import matcher

    tastes = await family_tastes(session, task) if session is not None else []
    res = await matcher.choose(items, lists, medicine=d.get("agent") == "pharmacy", family=tastes or None)
    if res is None:  # a model hiccup: once more before anything falls back
        res = await matcher.choose(items, lists, medicine=d.get("agent") == "pharmacy", family=tastes or None)
    if res is None:
        task.details = {**d, "matched": None}
        note(task, "the AI product choice failed twice; the store's own top listing is taken as the closest match")
        return
    if not tagged and len(items) > 1:
        # tag each listing with the item it was chosen for, so the items are kept apart from here on
        for n, ls, r in zip(names, lists, res):
            for j in r["exact"] + ([r["closest"]] if r.get("closest") is not None else []):
                ls[j].setdefault("for_item", n)
        for i in found:
            i.setdefault("for_item", names[0])
    key = (lambda n: n) if (tagged or len(items) > 1) else (lambda n: _groups(task, found)[0][0])
    matched = {key(n): {"exact": [_pkey(ls[i]) for i in r["exact"]],
                        "closest": _pkey(ls[r["closest"]]) if r.get("closest") is not None else None, "why": r.get("why")}
               for n, ls, r in zip(names, lists, res)}
    task.details = {**d, "matched": matched}
    note(task, "AI choice: " + "; ".join(f"{n or 'item'}: {('exact ' + str(len(m['exact']))) if m['exact'] else 'none'}"
                                         for n, m in matched.items())[:280])


async def _connector_lookup(session: AsyncSession, host, task: Task) -> dict | None:
    """Browse through the linked store's connector. Returns a look-up report, or None to look in the browser instead."""
    metrics.on_connector_call(task)
    asked_items = (task.details or {}).get("items") or [{}]
    try:
        if SKILLS[task.service].get("kind") == "food" and len(asked_items) > 1:
            # One restaurant per food cart: the first dish is looked up anywhere, the others inside the restaurant that has it
            # (live run 2026-10-11: paneer from one place and roti from another cannot go in one cart).
            first = await channels.connector_search(host, task, asked_items[0])
            anchor = next(((i.get("ref") or {}).get("restaurantName") for i in first.get("items") or [] if (i.get("ref") or {}).get("restaurantName")), None)
            rest = await asyncio.gather(*(channels.connector_search(host, task, {**it, "restaurant": it.get("restaurant") or anchor})
                                          for it in asked_items[1:]))
            results = [first, *rest]
        else:
            # One look-up per asked item, together (several items go in one cart).
            results = await asyncio.gather(*(channels.connector_search(host, task, it) for it in asked_items))
    except Exception as exc:  # noqa: BLE001 — a broken connector falls back to the browser
        await channels.record(session, task.service, "connector", False, str(exc)[:200])
        return None
    bad = next((r for r in results if not r.get("ok") and r.get("kind") != "unserviceable"), None)
    if bad is not None:
        if not channels.family_link_problem(bad):  # this family's expired or missing link is not the connector's fault
            await channels.record(session, task.service, "connector", False, bad.get("detail"))
        note(task, f"connector look-up: {str(bad.get('kind') or 'error')} ({str(bad.get('detail') or '')[:100]})")
        return None
    await channels.record(session, task.service, "connector", True)
    task.phase = "browse"
    if any(not r.get("ok") for r in results):
        r = next(r for r in results if not r.get("ok"))
        return {"items": [], "deliverable": False, "problem": r.get("detail") or "not serviceable here", "logged_in": True}
    many = len(asked_items) > 1
    items = [{"name": i.get("name"), "price": i.get("price"), "available": True, "ref": i.get("ref"),
              # food: which restaurant (the picker keeps one restaurant per cart)
              **({"restaurant": (i.get("ref") or {}).get("restaurantName"), "restaurantId": str((i.get("ref") or {}).get("restaurantId"))}
                 if (i.get("ref") or {}).get("restaurantId") else {}),
              **({"for_item": it.get("name")} if many else {})}
             for it, r in zip(asked_items, results) for i in (r.get("items") or []) if i.get("name")]
    missing = [it.get("name") for it, r in zip(asked_items, results) if not [i for i in (r.get("items") or []) if i.get("name")]]
    return {"items": items, "deliverable": True, "logged_in": True,
            "problem": "" if items and not missing else f"not found: {', '.join(str(m) for m in missing)}" if items else "not found"}


def fastpath_on() -> bool:
    return os.getenv("TASK_FASTPATH", "on") == "on"


async def _ride_ends(host, task: Task) -> tuple[dict, dict] | None:
    d = task.details or {}
    if host is None or not d.get("pickup") or not d.get("drop"):
        return None
    ends = []
    from app.brain.tools import saved_place_args

    fam = task.family_id.removeprefix("shadow:")
    for end, words in (("pickup", d["pickup"]), ("drop", d["drop"])):
        place = d.get(f"{end}_place") or {}
        if place.get("lat") is not None and place.get("lng") is not None:  # the saved place meant, found at the start
            ends.append({"lat": place["lat"], "lng": place["lng"], "label": f"{place.get('nickname') or words}, {place.get('full') or ''}".strip(", ")})
            continue
        # a saved place by its exact name when the words mean one (an AI decision), else the words to look up on the map
        args = await saved_place_args(host, words, family_id=fam, subject_id=task.subject_id, actor_id=task.requested_by)
        got = await host.call("ride_place", args, family_id=fam, subject_id=task.subject_id, actor_id=task.requested_by)
        if not got or got.get("lat") is None or got.get("lng") is None:
            return None
        ends.append({"lat": got["lat"], "lng": got["lng"], "label": got.get("label") or words})
    return ends[0], ends[1]


async def _fast_lookup(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for, host=None) -> dict | None:
    """The look-up through the store's own web request in the task's cloud browser (seconds, not minutes). The browser
    stays open for the login and cart runs. None (and a note) when it cannot run, so the browser agent looks instead."""
    place = ((task.details or {}).get("limits") or {}).get("place") or {}
    lat, lng, pincode = place.get("lat"), place.get("lng"), place.get("pincode")
    site = fastpath.SITES[task.service]
    ride = site.kind == "ride"
    if not hasattr(agent, "open_session"):
        return None
    if not ride and site.needs_point and not fastpath.has_point(lat, lng):
        # Live 2026-10-11 (a Bangalore family, pincode only): Blinkit and Zepto answered 400, Instamart "not found". The
        # place is given a point (backend); until then the browser agent looks with the written address.
        try:
            got = await host.call("delivery_place", {"words": place.get("nickname") or ""}, family_id=task.family_id.removeprefix("shadow:"),
                                  subject_id=task.subject_id, actor_id=task.requested_by) if host is not None and place.get("addressId") else None
        except Exception:  # noqa: BLE001 — no point: the agent looks
            got = None
        if got and got.get("addressId") == place.get("addressId") and fastpath.has_point(got.get("lat"), got.get("lng")):
            place = {**place, "lat": got["lat"], "lng": got["lng"], "geoPrecision": got.get("geoPrecision")}
            lat, lng = place["lat"], place["lng"]
            d = task.details or {}
            task.details = {**d, "limits": {**(d.get("limits") or {}), "place": place}}
        else:
            note(task, f"no map point for {place.get('nickname') or 'the place'}: the browser agent looks with the address instead")
            return None
    if not ride and not site.needs_point and not pincode and not fastpath.has_point(lat, lng):
        return None
    ends = None
    if ride:
        try:
            ends = await _ride_ends(host, task)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ride places failed task=%s: %s", task.id, exc)
        if not ends:
            return None
    try:
        got = await profile_for(task)
        profile_id = got.get("profileId") if isinstance(got, dict) else got
        await sandbox.bind_profile(session, task.family_id, task.service, profile_id)
        if not task.agent_session:
            sid = await agent.open_session(profile_id)
            task.agent_session = sid
            await sandbox.started(session, session_id=sid, task_id=str(task.id), family_id=task.family_id, service=task.service,
                                  profile_id=profile_id)
            _claim(task)
            await session.commit()  # the browser bills: record it before anything else can fail
        cdp = await agent.cdp_url(task.agent_session)
        if not cdp:
            raise fastpath.FastPathError("no CDP address for the browser")
        if ride:
            got = await fastpath.fares(task.service, cdp, pickup=ends[0], drop=ends[1])
            task.phase = "browse"
            task.details = {**(task.details or {}), "fast_session": True}
            note(task, f"fast fares on {SKILLS[task.service]['label']}: {len(got['options'])} options")
            return {"options": got["options"], "logged_in": False, "login_required": got["login_required"], "problem": ""}
        # one look-up per asked item, together (order lab 2026-10-10: "banana + apple" searched only banana)
        asked = [str(i.get("name") or "") for i in ((task.details or {}).get("items") or [])][:channels.MAX_ITEMS] or [""]
        more = (task.details or {}).get("more")  # the item they asked to see more of gets a longer list
        results = await asyncio.gather(*(fastpath.search(task.service, cdp, q, lat=lat, lon=lng, pincode=pincode,
                                                         **({"limit": MORE_LIMIT} if more and q == more else {})) for q in asked))
        many = len(asked) > 1
        found = {"deliverable": all(r["deliverable"] for r in results), "eta": next((r.get("eta") for r in results if r.get("eta")), None),
                 "logged_in": all(r.get("logged_in") for r in results),
                 "items": [{**i, **({"for_item": q} if many else {})} for q, r in zip(asked, results) for i in r["items"]],
                 "missing": [q for q, r in zip(asked, results) if r["deliverable"] and not r["items"]]}
    except fastpath.NoPoint as exc:  # not a broken step: no alert
        note(task, f"fast look-up skipped ({str(exc)[:120]}); the browser agent looks instead")
        return None
    except Exception as exc:  # noqa: BLE001 — a changed site or a slow browser: the agent looks instead
        note(task, f"fast look-up did not work ({str(exc)[:120]}); the browser agent looks instead")
        logger.warning("fast look-up failed task=%s: %s", task.id, exc)
        await _ops_alert(host, task, "look-up", str(exc))
        return None
    if not found["deliverable"] and site.needs_point and place.get("geoPrecision") in ("area", "pincode"):
        # The point is the area's or pincode's centre, not the door: "does not deliver here" may be about the centre. The
        # browser agent checks with the written address before anyone hears "not delivered".
        note(task, f"{SKILLS[task.service]['label']} says it does not deliver at the area's centre; the browser agent checks the address")
        task.details = {**(task.details or {}), "fast_session": True}  # this browser has no store tab: the agent opens the store
        return None
    task.phase = "browse"
    task.details = {**(task.details or {}), "fast_session": True}
    items = found["items"]
    note(task, f"fast look-up on {SKILLS[task.service]['label']}: " + (f"{len(items)} matching products" if found["deliverable"] else "does not deliver there"))
    return {"items": items, "deliverable": found["deliverable"], "eta": found.get("eta"),
            # Instamart tells whether this profile is logged in already: then no code is needed (the cart is built straight away)
            "logged_in": bool(found.get("logged_in")),
            "problem": ("" if not found.get("missing") else f"not found: {', '.join(found['missing'])}") if items else
                       ("not serviceable" if not found["deliverable"] else "not found")}


async def _fast_cart(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for, host=None) -> dict | None:
    """Build the cart through the store's own request when the family's profile is already logged in (seconds instead of
    minutes). Tried once per task; None (not logged in, an item gone, a changed site) lets the browser agent log in and
    build it as before."""
    d = task.details or {}
    chosen = [i for i in (d.get("chosen") or []) if i.get("store_id") or i.get("cart_ref")]
    if not chosen or not hasattr(agent, "open_session"):
        return None
    task.details = {**d, "fast_cart_tried": True, "fast_place_check": None}
    place = (d.get("limits") or {}).get("place") or {}
    try:
        if not task.agent_session:
            got = await profile_for(task)
            profile_id = got.get("profileId") if isinstance(got, dict) else got
            await sandbox.bind_profile(session, task.family_id, task.service, profile_id)
            task.agent_session = await agent.open_session(profile_id)
            await sandbox.started(session, session_id=task.agent_session, task_id=str(task.id), family_id=task.family_id,
                                  service=task.service, profile_id=profile_id)
            _claim(task)
            await session.commit()
        cdp = await agent.cdp_url(task.agent_session)
        if not cdp:
            raise fastpath.FastPathError("no CDP address for the browser")
        asked = d.get("items") or []
        cart = await fastpath.cart(task.service, cdp, [{**c, "qty": c.get("qty") or channels._qty_for(asked, c)} for c in chosen],
                                   place=place, lat=place.get("lat"), lon=place.get("lng"))
    except fastpath.ItemsUnavailable as exc:
        # Not sold at the serving store or out of stock (not a broken step): the next best product, fast cart again; the
        # person's own pick is never swapped: the store's other options go back to them.
        repick = _repick_without(task, exc.ids)
        if repick:
            return {"requeue": True, "requeue_note": f"not available here: {', '.join(exc.ids)}; trying {repick}"}
        again = _reoffer_self(task, exc.ids, "not available at the store right now")
        if again:
            return {"reoffer": again}
        note(task, "no other product of that kind is available here; the browser agent looks")
        return None
    except Exception as exc:  # noqa: BLE001 — not logged in yet, or the site changed: the agent does it
        if "not logged in" in str(exc) and not (task.details or {}).get("login_done"):
            task.details = {**(task.details or {}), "login_only": True}
            note(task, "not logged in yet: the browser agent logs in, then the cart is built fast")
        else:
            note(task, f"fast cart not possible ({str(exc)[:100]}); the browser agent builds it")
            if not any(w in str(exc) for w in ("address_missing", "not_serviceable", "out_of_stock", "no request to copy headers")):
                # (a page too slow to make its own requests twice is not a broken step: no alert; live 2026-10-11)
                await _ops_alert(host, task, "cart", str(exc))
        return None
    note(task, f"fast cart on {SKILLS[task.service]['label']}: total {cart.get('total')}")
    check = cart.pop("place_check", None)
    if check:
        # the place step re-reads the cart against this before sending the order
        task.details = {**(task.details or {}), "fast_place_check": check}
    return cart


def _repick_food(task: Task, ids: list[str]) -> str | None:
    """Food: a dish that would not go in the cart is dropped and the restaurant is chosen again from what is left (one
    restaurant per cart, at most 3 times). Returns what is tried now, or None."""
    d = task.details or {}
    if not d.get("auto_picked") or int(d.get("repicks") or 0) >= 3 or not ids or not d.get("matched"):
        return None
    gone = set(map(str, ids)) | set(d.get("refused") or [])
    found = [i for i in d.get("found") or [] if _pkey(i) not in gone]
    matched = {a: {**(v or {}), "exact": [k for k in (v or {}).get("exact") or [] if k not in gone]} for a, v in d["matched"].items()}
    groups = _groups(task, found)
    picks, find = _one_restaurant(groups, matched) if groups else ({}, [])
    if not picks:
        return None
    task.details = {**d, "chosen": [picks[a] for a, _ in groups if picks.get(a)], "menu_find": find or None, "refused": sorted(gone),
                    "fast_cart_tried": False, "repicks": int(d.get("repicks") or 0) + 1}
    return "; ".join(f"{_label(p)} from {p.get('restaurant')}" for p in picks.values())


def _repick_without(task: Task, ids: list[str]) -> str | None:
    """A product the store refused (not sold at the serving store, out of stock): the AI's next exact listing for that
    item (at most 3 times). Returns what is tried now, or None (it was the person's own pick, a medicine, or nothing
    exact is left: the cart is then built without it and the confirm says so)."""
    d = task.details or {}
    if not d.get("auto_picked") or d.get("agent") == "pharmacy" or int(d.get("repicks") or 0) >= 3 or not ids:
        return None
    gone = set(map(str, ids)) | set(d.get("refused") or [])
    found = d.get("found") or []
    byk = {_pkey(i): i for i in found}
    chosen, missing, tried = [], list(d.get("missing_items") or []), []
    for c in d.get("chosen") or []:
        if _pkey(c) not in gone:
            chosen.append(c)
            continue
        asked = str(c.get("for_item") or "") or next(iter((d.get("matched") or {}).keys()), "")
        nxt = next((byk[k] for k in ((d.get("matched") or {}).get(asked) or {}).get("exact") or []
                    if k in byk and k not in gone and k not in {_pkey(x) for x in chosen}), None)
        if nxt:
            chosen.append(nxt)
            tried.append(f"{_label(nxt)} {nxt.get('price') or ''}".strip())
        else:
            missing.append(asked or _label(c))
    if not chosen or not tried and len(chosen) == len(d.get("chosen") or []):
        return None
    task.details = {**d, "chosen": chosen, "missing_items": missing or None, "refused": sorted(gone), "fast_cart_tried": False,
                    "repicks": int(d.get("repicks") or 0) + 1}
    return "; ".join(tried) or "the rest without the unavailable item"


async def _fast_place(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for, host=None) -> dict | None:
    """Place the confirmed cart with the store's own requests (order lab 2026-10-10: Instamart, cash). The cart is read
    again first: any change → nothing is sent and the cart is built again for a fresh confirm. A sent order without a
    clear answer is reported unclear and never retried. None only when the browser could not be reached (agent places)."""
    d = task.details or {}
    try:
        cdp = await agent.cdp_url(task.agent_session) if task.agent_session else None
        if not cdp:
            # the kept-alive browser ended while the family decided: a new one on the same profile (the login is saved there)
            got = await profile_for(task)
            profile_id = got.get("profileId") if isinstance(got, dict) else got
            task.agent_session = await agent.open_session(profile_id)
            await sandbox.started(session, session_id=task.agent_session, task_id=str(task.id), family_id=task.family_id,
                                  service=task.service, profile_id=profile_id)
            _claim(task)
            await session.commit()
            cdp = await agent.cdp_url(task.agent_session)
        if not cdp:
            raise fastpath.FastPathError("no CDP address for the browser")
    except Exception as exc:  # noqa: BLE001 — nothing was sent: the agent places as before
        note(task, f"fast place not possible ({str(exc)[:100]}); the browser agent places it")
        task.details = {**d, "fast_place_check": None}
        return None
    if practice.on(task):
        # Practice: Blinkit's saved steps go to checkout and choose cash, then stop before Pay Now; others are not run.
        total = _rupees_text(d["fast_place_check"]["total"]) if (d.get("fast_place_check") or {}).get("total") else None
        if task.service != "blinkit":
            return {"practice": True, "reached_final_step": True, "final_button": "place order (saved step not run)", "total": total}
        try:
            got = await fastpath.blinkit_place(cdp, d["fast_place_check"], dry=True)
        except fastpath.FastPathError as exc:
            return {"practice": True, "reached_final_step": False, "problem": str(exc)[:160]}
        return {"practice": True, "reached_final_step": bool(got.get("ready")), "final_button": "Pay Now", "total": total,
                "payment_method": "Cash on Delivery", "problem": got.get("problem")}
    try:
        out = await fastpath.place(task.service, cdp, d["fast_place_check"])
    except fastpath.FastPathError as exc:
        # Nothing was sent: the cart, total or address changed, or the site changed. The cart is built again (the browser
        # agent if the saved steps fail); if it is the same cart at no higher total, it is placed on the same yes, without
        # asking again (founder 2026-10-10: keep going until it is placed). The next place is the agent's.
        changed = any(w in str(exc) for w in ("cart_changed", "total_changed", "address_changed", "out_of_stock", "unavailable"))
        if not changed:
            await _ops_alert(host, task, "place", str(exc))
        task.phase = "prepare"
        task.details = {**d, "fast_place_check": None, "fast_cart_tried": False, "cart_fp": None, "confirm_token": None, "place_started": None,
                        "fast_place_failures": int(d.get("fast_place_failures") or 0) + (0 if changed else 1),
                        "replace_confirmed": {"fp": d.get("confirmed_fp"), "total": d.get("confirmed_total"), "by": d.get("confirmed_by"),
                                              "items_fp": d.get("confirmed_items_fp")}}
        return {"requeue": True, "requeue_note": f"not placed, nothing sent ({str(exc)[:120]}); building the cart again"}
    if out.get("unclear") and task.service == "instamart" and host is not None and (task.details or {}).get("fast_place_check", {}).get("total"):
        # Sent but no clear answer: the linked Swiggy account's order list says whether it went through (never placed twice).
        try:
            sent = datetime.fromisoformat(str((task.details or {}).get("place_started") or clock.now().isoformat()))
            got = await host.call("connector_recent_order", {"store": "instamart", "total": d["fast_place_check"]["total"],
                                                            "since_ms": int(sent.timestamp() * 1000)},
                                  family_id=task.family_id.removeprefix("shadow:"), subject_id=task.subject_id, actor_id=task.requested_by)
            if got and got.get("orderId"):
                out = {"placed": True, "order_id": got["orderId"], "total": _rupees_text(d["fast_place_check"]["total"]),
                       "payment_method": "Cash on Delivery", "eta": got.get("eta")}
        except Exception:  # noqa: BLE001 — stays unclear: never retried
            logger.warning("instamart order list check failed task=%s", task.id)
    note(task, f"fast place on {SKILLS[task.service]['label']}: {'placed ' + str(out.get('order_id')) if out.get('placed') else 'unclear'}")
    return out


def _rupees_text(v) -> str:
    return f"₹{round(float(v))}"


_ALERTED: dict[str, datetime] = {}


async def _ops_alert(host, task: Task, step: str, error: str) -> None:
    """A saved step broke (the store changed its site): tell the founder once per store and step every 6 hours, so it gets
    fixed; the order itself carries on with the browser agent."""
    key = f"{task.service}:{step}"
    if host is None or task.family_id.startswith("shadow:") or (key in _ALERTED and clock.now() - _ALERTED[key] < timedelta(hours=6)):
        return
    _ALERTED[key] = clock.now()
    try:
        await host.call("ops_alert", {"subject": f"Saved {step} step broke on {SKILLS[task.service]['label']}",
                                      "text": f"Task {task.id}: {error[:600]}\nThe order continues with the browser agent; fix the saved step."},
                        family_id=task.family_id, subject_id=task.subject_id, actor_id=task.requested_by)
    except Exception:  # noqa: BLE001 — an alert must never break the order
        logger.warning("ops alert failed for %s", key)


async def _order_on_store_list(task: Task, agent: BrowserAgent, host=None) -> dict | None:
    """After a place run with no clear answer: the order as the store's own order list shows it (logged-in profile), else
    None. Never places anything; an order found here is reported placed, so nobody orders twice."""
    d, r = task.details or {}, task.result or {}
    total = d.get("confirmed_total") or r.get("total")
    names = [str(c.get("name")) for c in d.get("chosen") or [] if c.get("name")]
    try:
        if task.service == "blinkit" and task.agent_session:
            cdp = await agent.cdp_url(task.agent_session)
            found = await fastpath.blinkit_recent_order(cdp, total, names=names) if cdp else None
            if found:
                return {"placed": True, "order_id": found, "payment_method": "Cash on Delivery"}
        if task.service == "instamart" and host is not None and rupees(total):
            sent = datetime.fromisoformat(str(d.get("run_started") or clock.now().isoformat()))
            got = await host.call("connector_recent_order", {"store": "instamart", "total": rupees(total),
                                                            "since_ms": int(sent.timestamp() * 1000)},
                                  family_id=task.family_id.removeprefix("shadow:"), subject_id=task.subject_id, actor_id=task.requested_by)
            if got and got.get("orderId"):
                return {"placed": True, "order_id": got["orderId"], "payment_method": "Cash on Delivery", "eta": got.get("eta")}
    except Exception as exc:  # noqa: BLE001 — stays unclear: the person checks the app, nothing is placed again
        logger.warning("order list check failed task=%s: %s", task.id, exc)
    return None


def _confirm_still_valid(task: Task) -> bool:
    """The confirm still matches what is in the cart (or the fares shown) and its token is genuine."""
    d = task.details or {}
    expected = d.get("cart_fp") if task.kind == "order" else _ride_fp(d.get("cart_fp"), d.get("choice") or "")
    return bool(expected) and d.get("confirmed_fp") == expected and guard.token_valid(
        str(task.id), d.get("confirmed_fp") or "", d.get("confirmed_by") or "", d.get("confirm_token"))


async def _connector_step(session: AsyncSession, host, task: Task) -> dict | None:
    """Run prepare/place on the store's connector. Returns an agent-style report, or None to fall back to the browser."""
    metrics.on_connector_call(task)
    d = task.details or {}
    try:
        if task.phase == "place" and practice.on(task):
            # Practice: the exact cart is built and checked again on the store, then emptied; the place call is never made.
            r = await channels.connector_prepare(host, task)
            if not r.get("ok"):
                return {"practice": True, "reached_final_step": False, "problem": r.get("detail") or r.get("kind") or "the cart could not be built again"}
            confirmed, now_total = guard.rupees(d.get("confirmed_total")), guard.rupees(r.get("total"))
            if confirmed and now_total and not guard.within_leeway(now_total, confirmed):
                task.details = {**d, "connector_card": r.get("card")}
                return {"price_changed": True, "new_total": r.get("total")}
            return {"practice": True, "reached_final_step": True, "final_button": "the store's place-order call (not made)",
                    "total": r.get("total"), "payment_method": "Cash on Delivery", "cart_emptied": True}
        if task.phase == "place":
            r = await channels.connector_place(host, task)
            if r.get("shadow"):
                return {"placed": False, "problem": "shadow mode: nothing is placed"}
            st = r.get("status")
            if st == "placed":
                await channels.record(session, task.service, "connector", True)
                return {"placed": True, "order_id": r.get("orderId"), "total": r.get("total"), "payment_method": "Cash on Delivery", "eta": r.get("eta")}
            if st == "refused" and r.get("newCard"):
                task.details = {**d, "connector_card": r["newCard"]}
                return {"price_changed": True, "new_total": r.get("newTotal")}
            if st in ("unknown", "duplicate"):
                # The store may have taken it: never retry on another channel (no double order).
                return {"placed": False, "unclear": True, "problem": "the store did not answer after the order was sent; check the app before trying again"}
            if st == "expired":
                task.phase = "prepare"  # build a fresh card on the connector; the family confirms it again
                task.details = {**(task.details or {}), "cart_fp": None, "confirm_token": None, "place_started": None}
                return {"requeue": True}
            await channels.record(session, task.service, "connector", False, r.get("detail"))
            return None
        r = await channels.connector_prepare(host, task)
        if r.get("ok"):
            await channels.record(session, task.service, "connector", True)
            task.details = {**(task.details or {}), "connector_card": r.get("card")}
            return {"items": r.get("items") or [], "total": r.get("total"), "fees": r.get("fees"), "cod_available": r.get("cod_available", True),
                    "address_used": r.get("address_used"), "eta": r.get("eta"), "logged_in": True}
        if r.get("kind") in ("cod_unavailable", "not_found"):
            await channels.record(session, task.service, "connector", True)
            return {"items": [], "cod_available": r.get("kind") != "cod_unavailable", "problem": r.get("detail"), "alternatives": r.get("alternatives") or []}
        if not channels.family_link_problem(r):
            await channels.record(session, task.service, "connector", False, r.get("detail"))
        return None
    except Exception as exc:  # noqa: BLE001 — a broken connector falls back to the browser
        await channels.record(session, task.service, "connector", False, str(exc)[:200])
        return None


def route(urls: list[str], limit: int = 8) -> str:
    """'https://x.com/search?q=atta', … -> '/ → /search → /cart → /checkout' (paths only, repeats removed)."""
    from urllib.parse import urlparse

    out: list[str] = []
    for u in urls:
        p = urlparse(u).path or "/"
        p = re.sub(r"/\d{3,}|/[0-9a-f]{8,}(-[0-9a-f]{4,})*", "/…", p)[:40]
        if not out or out[-1] != p:
            out.append(p)
    return " → ".join(out[:limit]) + (" → …" if len(out) > limit else "")


async def release_sessions(sessions: async_sessionmaker, agent: BrowserAgent, limit: int = 50) -> int:
    """Stop cloud browsers nobody is using: tasks that ended, or that wait on a person for more than a few minutes.
    (A finished Browser Use run keeps its browser running and billing; the family's profile keeps the login.)"""
    idle_before = clock.now() - IDLE_CLOSE
    async with sessions() as session:
        rows = list((await session.execute(
            select(Task).where(Task.agent_session.is_not(None), or_(
                Task.status.in_(("done", "failed", "cancelled")),
                (Task.status.in_(("awaiting_confirm", "needs_input"))) & (Task.updated_at < idle_before),
            )).limit(limit).with_for_update(skip_locked=True)
        )).scalars())
        released = 0
        for t in rows:
            if t.status == "needs_input" and t.input_needed == "otp":
                continue  # the login page is waiting for the code in this browser
            try:
                await agent.stop_session(t.agent_session)
                await sandbox.stopped(session, t.agent_session, f"task {t.status}")
            except Exception:  # noqa: BLE001 — already gone, or the service is down: the sweeper retries from the ledger
                logger.warning("stop session failed task=%s", t.id)
            t.agent_session = None
            released += 1
        await session.commit()
    return released


def _waiting_for(task: Task) -> str:
    d, r = task.details or {}, task.result or {}
    if task.status in ("queued", "running") and d.get("ladder_due"):
        return f"a go-ahead to keep trying: it has used ₹{float((d.get('metrics') or {}).get('cost_inr', 0)):.0f} of browser time"
    if task.status in ("queued", "running"):
        return "nobody: the store's website is slow or changed and it is still trying"
    if task.input_needed == "otp":
        login = d.get("login")
        return "the store's login code" + (f" (sent to the phone ending {login})" if login and login != "none" else "")
    if task.input_needed == "confirm":
        return f"a yes to the cart (total {r.get('total') or 'shown'})"
    if task.input_needed == "approve":
        return "the family approver's yes"
    if task.input_needed == "go":
        return "their pick from the options found"
    if task.input_needed == "fee":
        return f"their yes or no to the cancellation fee of {d.get('cancel_fee')}"
    return f"an answer ({task.input_needed or 'from the family'})"


async def _care_recipient(session: AsyncSession, family_id: str) -> str | None:
    """The family's care recipient. An order's subject is the person it is for, who may be a caregiver (founder 2026-10-12)."""
    roster = await store.roster(session, family_id)
    return ((roster.elder or {}).get("id") if roster else None) or None


async def _carers(session: AsyncSession, task: Task) -> list[str]:
    """Who hears about trouble with an order (stuck, late, came wrong, the care recipient asked to cancel it): for an order of
    the care recipient (for or by them), the family's approvers (caregivers); for anyone else's own order, the person who
    asked (and the person it is for), never the care recipient about it (founder 2026-10-12). Self care: the person."""
    from app.care import boundaries

    roster = await store.roster(session, task.family_id)
    elder = ((roster.elder or {}).get("id") if roster else None) or task.subject_id
    if elder in (task.subject_id, task.requested_by):
        people = [a for a in boundaries.approvers(await boundaries.get(session, task.family_id), roster)
                  if a != boundaries.ANYONE and a != elder]
        return people or [task.requested_by]
    return list(dict.fromkeys(p for p in (task.requested_by, task.subject_id) if p and p != elder))


async def cancel_properly(session: AsyncSession, agent: BrowserAgent, task: Task, why: str) -> None:
    """Stop an order that is not placed: the browser run, its cloud browser (the login stays in the profile) and the task."""
    if task.status == "running" and task.agent_task:
        try:
            await agent.stop(task.agent_task, end_session=True)
        except Exception:  # noqa: BLE001 — the sweeper stops what the ledger knows
            logger.warning("stop failed task=%s", task.id)
    if task.agent_session:
        try:
            await agent.stop_session(task.agent_session)
            await sandbox.stopped(session, task.agent_session, "order cancelled")
        except Exception:  # noqa: BLE001
            logger.warning("stop session failed task=%s", task.id)
        task.agent_session = None
    task.status, task.cancel_requested, task.input_needed = "cancelled", True, None
    note(task, why)
    metrics.on_milestone(task, "cancelled")
    await store.record_event(session, family_id=task.family_id, subject_id=task.subject_id, kind="task_cancelled",
                             summary=f"{SKILLS[task.service]['label']}: {why}; nothing was ordered", payload={"task_id": str(task.id)})


async def escalate(sessions: async_sessionmaker, agent: BrowserAgent, notify: Notify, host_for=None) -> int:
    """The order ladder (founder 2026-10-10): 20 minutes without being placed → ask the caregiver (keep trying, confirm,
    or cancel); 5 more minutes without an answer → email them too; 5 more → cancel it properly and tell everyone."""
    now = clock.now()
    sends: list[tuple] = []
    async with sessions() as session:
        rows = list((await session.execute(
            select(Task).where(Task.kind.in_(("order", "ride")), Task.status.in_(LIVE),
                               or_(Task.created_at >= now - timedelta(days=1), Task.updated_at >= now - timedelta(days=1)))
            .with_for_update(skip_locked=True)
        )).scalars())
        picking: set[str] = set()
        rows.sort(key=lambda t: (not (t.details or {}).get("ladder_asked_at"), str(t.id)))  # the same task carries a comparison's ladder
        for t in rows:
            d = t.details or {}
            if t.family_id.startswith("shadow:") or t.phase == "cancel" or (t.phase == "place" and d.get("place_started")):
                continue  # a test turn, a cancel in progress, or an order the store may be taking right now
            if (d.get("later") or {}).get("state") in ("login", "new") or ((d.get("later") or {}).get("state") is None and d.get("later")):
                continue  # a login made hours ahead of a booking, or the question for later: they have until it is due
            gid = compare_group(t)
            if gid and d.get("compare_announced") and t.input_needed == "go":
                if gid in picking:
                    continue  # the options of every store wait for one pick: one ask for the whole comparison
                picking.add(gid)
            if compare_group(t) and not d.get("compare_announced"):
                # Still looking on several stores. Live 2026-10-11 nobody heard anything for 6 minutes: one line for the
                # whole comparison once it has taken COMPARE_STILL_WORKING, no question.
                if t.status in ("queued", "running") and now - t.created_at >= COMPARE_STILL_WORKING and not d.get("told_working"):
                    group = await siblings(session, t)
                    if not any((g.details or {}).get("told_working") for g in group):
                        t.details = d = {**d, "told_working": True}
                        slow = sorted({SKILLS[g.service]["label"] for g in group if g.status in ("queued", "running")})
                        sends.append(("notify", t.family_id, t.requested_by, group, (
                            f"[Task update] Still looking on {', '.join(slow)} (the website is slow); nothing is "
                            f"{'booked' if t.kind == 'ride' else 'ordered'} and there is nothing to answer. Tell them in one short line "
                            "that it is taking a minute or two more and that you will come back with what each one has. Do not ask "
                            "anything.")))
                continue
            if t.kind != "order" or t.phase == "check" or t.cancel_requested:
                # the 20/25/30-minute ladder is for orders; a ride waiting on a person ends at its deadline; a store check, or
                # an order they asked to stop, is not "not placed yet" (live 2026-10-13: asked, emailed and cancelled while
                # she had said cancel four times)
                continue
            start = datetime.fromisoformat(d["ladder_from"]) if d.get("ladder_from") else t.created_at
            busy_since = datetime.fromisoformat(d["active_from"]) if d.get("active_from") else t.created_at
            if (t.status in ("queued", "running") and not d.get("told_working") and now - busy_since >= STILL_WORKING
                    and not d.get("ladder_asked_at")):
                # Live 2026-10-10: Maa heard nothing for minutes while the browser agent worked. One short line, no question.
                t.details = d = {**d, "told_working": True}
                sends.append(("notify", t.family_id, t.requested_by, [t], (
                    f"[Task update] {describe(t)}. Still working on it on {SKILLS[t.service]['label']} (the website is slow); nothing "
                    "to answer. Tell them in one short line that it is taking a few more minutes and you will message when it is "
                    "ready. Do not ask anything.")))
            asked = datetime.fromisoformat(d["ladder_asked_at"]) if d.get("ladder_asked_at") else None
            emailed = datetime.fromisoformat(d["ladder_emailed_at"]) if d.get("ladder_emailed_at") else None
            label = SKILLS[t.service]["label"]
            if asked is None:
                if now - start < LADDER_ASK and not d.get("ladder_due"):
                    continue
                carers = await _carers(session, t)
                t.details = {**d, "ladder_asked_at": now.isoformat(), "ladder_carers": carers}
                note(t, "not placed after 20 minutes: asked " + ", ".join(carers))
                minutes = int((now - start).total_seconds() // 60)
                for c in carers:
                    sends.append(("notify", t.family_id, c, [t], (
                        f"[Order needs you] {describe(t)}. Asked by person {t.requested_by} for person {t.subject_id} {minutes} minutes ago "
                        f"and not placed yet; it is waiting for {_waiting_for(t)}. Ask person {c} in one or two short lines, with what "
                        f"is being ordered: keep trying or cancel?" + (" They can also say yes to the cart." if t.input_needed == "confirm" else "")
                        + f" Their answer on task {t.id}: keep trying → task_input keep; yes to the cart → task_input confirm yes; "
                        f"the code → task_input otp; cancel → cancel_task, then tell person {t.requested_by} kindly who cancelled.")))
            elif emailed is None:
                if now - asked < LADDER_EMAIL:
                    continue
                t.details = {**d, "ladder_emailed_at": now.isoformat()}
                note(t, "no answer in 5 minutes: emailing too")
                for c in d.get("ladder_carers") or []:
                    sends.append(("email", t.family_id, c, t.subject_id, f"Saheli: {label} order waiting for you",
                                  f"An order for your family on {label} ({bought(t) if (t.details or {}).get('cart_fp') else t.goal}) is not placed yet. It is waiting for {_waiting_for(t)}. "
                                  "Please reply to Saheli on WhatsApp: keep trying, or cancel. If nobody answers in 5 minutes it will be "
                                  "cancelled and nothing will be ordered."))
            elif now - emailed >= LADDER_CANCEL:
                was_picking = t.input_needed == "go"
                await cancel_properly(session, agent, t, "cancelled: not placed after 30 minutes and nobody answered")
                if gid and was_picking:
                    for g in await siblings(session, t):
                        if g.id != t.id and g.status in LIVE and g.input_needed == "go":
                            g.status, g.cancel_requested, g.input_needed = "cancelled", True, None
                            note(g, "cancelled with the rest of the options: nobody picked one in 30 minutes")
                for person in dict.fromkeys([t.requested_by, *(d.get("ladder_carers") or [])]):
                    sends.append(("notify", t.family_id, person, [t], (
                        f"[Task update] {describe(t)}. It could not be finished in 30 minutes and nobody answered, so it was cancelled "
                        "properly; nothing was ordered. Tell them in one kind line; they can ask again any time.")))
        await session.commit()
    for x in sends:
        try:
            if x[0] == "notify":
                await _notify_about(notify, x[3], x[1], x[2], x[4])
            elif host_for is not None:
                await host_for(x[1]).call("email_member", {"to": x[2], "subject": x[4], "text": x[5]},
                                          family_id=x[1], subject_id=x[3], actor_id=x[2])
        except Exception:  # noqa: BLE001 — one failed message must not stop the rest
            logger.exception("ladder message failed task family=%s", x[1])
    return len(sends)


MAX_TICK_ERRORS = 3
# A runner that commits in the middle of its work (a fast step records the browser it opened) releases the task's row
# lock; for this long no other runner takes the still-queued task (live 2026-10-11: the scheduler's tick and the one a
# message kicked off both started a browser run on the same browser, which refused the second: "Agent session already
# has a running task"). A runner that crashed leaves the claim; it simply runs out.
CLAIM_FOR = timedelta(minutes=3)


def _claim(task: Task) -> None:
    task.details = {**(task.details or {}), "claim": clock.now().isoformat()}


def _claimed(task: Task) -> bool:
    at = (task.details or {}).get("claim")
    try:
        return bool(at) and task.status == "queued" and clock.now() - datetime.fromisoformat(at) < CLAIM_FOR
    except ValueError:
        return False


async def _count_tick_error(session: AsyncSession, task_id, final: str | None = None) -> tuple[Task, str] | None:
    """A task whose tick keeps crashing (a report the code cannot read) is stopped after a few tries and the
    family is told, instead of being stuck 'running' and blocking that service for good."""
    try:
        task = (await session.execute(select(Task).where(Task.id == task_id).with_for_update())).scalar_one_or_none()
        if not task or task.status in ("done", "failed", "cancelled"):
            return None
        n = int((task.details or {}).get("tick_errors", 0)) + 1
        task.details = {**(task.details or {}), "tick_errors": n}
        message = None
        if final and task.phase not in ("place", "cancel"):
            n = MAX_TICK_ERRORS
        if n >= MAX_TICK_ERRORS:
            task.status = "failed"
            sandbox.audit(task, phase=task.phase, status="failed", steps=0, path=[], cost=0, error=f"stopped after {n} errors in Kavach")
            message = ("Something went wrong while placing it. It is not clear whether it went through: do not order again; "
                       "tell the caregiver to check the app.") if task.phase in ("place", "cancel") else (
                final or "Something went wrong while working on it, so it was stopped; nothing was placed.")
            note(task, f"stopped after {n} errors")
            metrics.on_milestone(task, "failed")
            await store.record_event(
                session, family_id=task.family_id, subject_id=task.subject_id, kind="task_failed",
                summary=f"{SKILLS[task.service]['label']}: {message}", payload={"task_id": str(task.id)},
            )
        await session.commit()
        return (task, message) if message else None
    except Exception:  # noqa: BLE001
        logger.exception("could not count tick error task=%s", task_id)
        await session.rollback()
        return None


async def tick(
    sessions: async_sessionmaker,
    agent: BrowserAgent,
    *,
    profile_for: Callable[[Task], Awaitable[str | None]],
    notify: Notify,
    limit: int = 20,
    host_for: Callable[[str], object] | None = None,
    light: bool = False,
) -> dict:
    """light: the few-seconds loop between the minute's runs; it moves the tasks and their messages and leaves the
    housekeeping (idle browsers, delivery tracking, aftercare) to the full tick."""
    stats = {"started": 0, "polled": 0, "finished": 0, "expired": 0}
    try:
        stats["later"] = await later.due(sessions, notify)  # bookings set for later whose login, fare check or time has come
    except Exception:  # noqa: BLE001
        logger.exception("later bookings failed")
    try:
        if not light:
            stats["released"] = await release_sessions(sessions, agent)
    except Exception:  # noqa: BLE001 — cleanup must never stop the tick
        logger.exception("release sessions failed")
    async with sessions() as session:
        ids = list(
            (
                await session.execute(
                    select(Task.id).where(or_(Task.status.in_(("queued", "running")), (Task.status.in_(("needs_input", "awaiting_confirm"))) & (Task.deadline_at < clock.now())))
                    .order_by(Task.updated_at).limit(limit)
                )
            ).scalars()
        )
    async def process(task_id):
        with trace.scope():  # the task's trace is bound below, and undone when it is done
            return await _process(task_id)

    async def _process(task_id):
        message, summary = None, False
        async with sessions() as session:
            task = (await session.execute(select(Task).where(Task.id == task_id).with_for_update(skip_locked=True))).scalar_one_or_none()
            if not task or _claimed(task):
                return None  # locked, or another runner is in the middle of it
            trace.bind_task(task)  # its model calls, events and outcome are filed under the trace that started it (D-13)
            was_browse, expired = task.phase == "browse", False
            try:
                if task.deadline_at < clock.now() and task.status != "running" and later.of(task).get("state") == "login":
                    lt = later.of(task)
                    task.status, task.phase, task.input_needed = "scheduled", "prepare", None
                    task.details = {**(task.details or {}), "login_only": False, "later": {**lt, "state": "waiting", "login_at": None}}
                    task.deadline_at = later.parse_at(lt["book_at"]) + later.GIVE_UP_AFTER
                    note(task, "nobody gave the login code ahead; it is asked again at the fare check")
                    message = None
                elif task.deadline_at < clock.now() and task.status != "running":
                    task.status, expired = "failed", True
                    note(task, "timed out waiting")
                    message = "Nobody answered in time, so the task was stopped; nothing was placed." if task.phase != "cancel" else "Cancel was not finished; tell the caregiver."
                    from app.brain.wake import in_quiet_hours

                    lt_ = later.of(task)
                    if lt_.get("by"):
                        # they counted on it at that time: never left unsaid (not even in the early morning)
                        message = (f"NOT {'booked' if task.kind == 'ride' else 'ordered'}: the {SKILLS[task.service]['label']} "
                                   f"{'ride' if task.kind == 'ride' else 'order'} set for {later.hm(later.parse_at(lt_.get('at')))} was "
                                   f"waiting for {_waiting_for(task)} and nobody answered in time. Tell them plainly in one short line.")
                    elif task.phase != "cancel" and in_quiet_hours():
                        message = None  # live 2026-10-08 22:21: "nobody answered, stopped" woke her at night; nothing was placed
                    if compare_group(task) and was_browse:
                        # One "nobody answered" for the whole comparison, not one per store.
                        group = await siblings(session, task)
                        message = None if any((t.details or {}).get("compare_expired") for t in group if t.id != task.id) else (
                            "Nobody picked one of the options in time, so the look-up on every store was stopped; nothing was ordered.")
                        task.details = {**(task.details or {}), "compare_expired": True}
                    metrics.on_milestone(task, "failed")
                    stats["expired"] += 1
                elif task.status == "queued" and task.phase == "place" and not _confirm_still_valid(task):
                    task.status, task.input_needed = "awaiting_confirm", "confirm" if task.kind == "order" else "choice"
                    task.details = {**(task.details or {}), "confirm_token": None}
                    message = "The cart changed after it was confirmed, so nothing was placed. Read it again and ask the person to confirm."
                    note(task, "confirm no longer matches the cart")
                elif task.status == "queued" and task.phase == "place" and (task.details or {}).get("place_started"):
                    # A place step started before and never recorded its result (a crash after the store call).
                    # It may have gone through: never place again.
                    task.status = "failed"
                    task.details = {**(task.details or {}), "place_unclear": True}
                    message = _after_cancel_ask(task, "failed", (
                        "It is not clear whether it was placed (the last attempt was interrupted). Do not order again; tell the "
                        "caregiver to check the app."))
                    note(task, "place interrupted; not retried")
                    metrics.on_milestone(task, "failed")
                elif task.status == "queued":
                    if task.phase == "place":
                        # Mark before calling the store, and commit, so a crash after the call cannot place twice.
                        marker = clock.now().isoformat()
                        task.details = {**(task.details or {}), "place_started": marker}
                        await session.commit()
                        # Re-read from the database (not the identity map) under the lock: another tick or a cancel
                        # may have changed it between the commit and here.
                        task = (await session.execute(
                            select(Task).where(Task.id == task_id).with_for_update().execution_options(populate_existing=True)
                        )).scalar_one()
                        if task.status == "queued" and task.cancel_requested and (task.details or {}).get("place_started") == marker:
                            # A cancel came in after the mark and before the store was called: nothing was sent.
                            task.status, task.details = "cancelled", {**(task.details or {}), "place_started": None}
                            note(task, "cancelled just before placing; nothing was sent")
                            metrics.on_milestone(task, "cancelled")
                            await session.commit()
                            return (task, "They asked to cancel just before it was placed: it was stopped in time and NOTHING was "
                                          "ordered. Tell them in one short line.", False)
                        if task.status != "queued" or task.cancel_requested or (task.details or {}).get("place_started") != marker:
                            return None
                    report = None
                    if task.phase in ("prepare", "place") and (task.details or {}).get("channel") != "browser":
                        host = host_for(task.family_id) if host_for else None
                        if task.phase == "prepare":
                            channel, why = await channels.pick_channel(session, host, task)
                            metrics.on_channel(task, channel, why)
                            task.details = {**(task.details or {}), "channel": channel}
                        if (task.details or {}).get("channel") == "connector" and task.phase == "prepare" and browse_first(task):
                            # A linked store looks it up through its connector: seconds, no browser, no login code.
                            report = await _connector_lookup(session, host, task)
                            if report is None:
                                metrics.on_channel(task, "browser", "connector look-up failed; falling back")
                                task.details = {**(task.details or {}), "channel": "browser"}
                                note(task, "connector look-up failed, looking it up in the browser")
                            else:
                                was_browse = True
                        elif (task.details or {}).get("channel") == "connector":
                            report = await _connector_step(session, host, task)
                            if report is None:
                                metrics.on_channel(task, "browser", "connector failed; falling back")
                                task.details = {**(task.details or {}), "channel": "browser"}
                                if task.phase == "place":
                                    # the browser builds its own cart; the same cart at no higher total goes on the same yes
                                    d_ = task.details or {}
                                    task.phase = "prepare"
                                    task.details = {**d_, "replace_confirmed": {"fp": d_.get("confirmed_fp"), "total": d_.get("confirmed_total"),
                                                                                "items_fp": d_.get("confirmed_items_fp"),
                                                                                "by": d_.get("confirmed_by")}, "place_started": None}
                                note(task, "connector failed, falling back to the browser")
                    if (report is None and task.phase == "prepare" and browse_first(task) and task.service in fastpath.FAST_SERVICES
                            and fastpath_on()):
                        report = await _fast_lookup(session, agent, task, profile_for, host_for(task.family_id) if host_for else None)
                        if report is not None:
                            was_browse = True
                    elif (report is None and task.phase == "prepare" and task.kind == "order" and task.service in fastpath.CART_SITES
                            and fastpath_on() and not (task.details or {}).get("fast_cart_tried")):
                        report = await _fast_cart(session, agent, task, profile_for, host_for(task.family_id) if host_for else None)
                    elif (report is None and task.phase == "place" and task.kind == "order" and task.service in fastpath.PLACE_SITES
                            and fastpath_on() and (task.details or {}).get("fast_place_check")
                            and not (task.details or {}).get("fast_place_failures")):
                        report = await _fast_place(session, agent, task, profile_for, host_for(task.family_id) if host_for else None)
                    if report is not None and report.get("requeue"):
                        note(task, report.get("requeue_note") or "connector card expired; building it again")
                    elif report is not None and report.get("reoffer"):
                        # the product they picked is not sold here after all: the store's other options go back to them
                        task.status, message, summary = "awaiting_confirm", report["reoffer"], True
                        stats["finished"] += 1
                    elif report is not None:
                        task.result = {**(task.result or {}), **{k: v for k, v in report.items() if v not in (None, "", [])}}
                        if task.phase == "browse":
                            await _ai_pick(task, task.result, session)
                        status, message = _outcome(task, task.result if task.phase == "browse" else report)
                        status, message = await _later_step(session, task, status, message)
                        task.status = status
                        via = ("connector" if (task.details or {}).get("channel") == "connector" else
                               "fast look-up" if task.phase == "browse" else "saved steps")
                        note(task, f"{task.phase} → {status} ({via}): {message}")
                        message = _after_cancel_ask(task, status, message)
                        if status == "failed":
                            metrics.on_milestone(task, "failed")
                        stats["finished"] += 1
                    elif (float(((task.details or {}).get("metrics") or {}).get("cost_inr", 0))
                          >= float((task.details or {}).get("cost_ok_until") or task_max_cost())):
                        # Many browser steps already: pause (no new run) and ask the caregiver whether to keep trying; the
                        # ladder cancels it properly if nobody answers.
                        if not (task.details or {}).get("ladder_due"):
                            task.details = {**(task.details or {}), "ladder_due": True}
                            note(task, f"paused at ₹{task_max_cost():.0f} of browser time; asking the caregiver")
                    else:
                        if browse_first(task):
                            task.phase = "browse"
                            note(task, "looking it up before any login")
                        extra = ""
                        if task.phase == "otp":
                            # Typed key by key into the store's own code boxes by code (PharmEasy, live 2026-10-13: the
                            # agent put 3 of 4 digits in, restarted the login and the number was blocked); else the agent
                            # types it, told how. The code is handed over once and not kept on the task.
                            code = str((task.details or {}).get("otp") or "")
                            typed = await _type_code(agent, task, code)
                            extra = CODE_TYPED if typed else CODE_BY_HAND.format(code=code)
                            task.phase = "prepare"
                            task.details = {**{k: v for k, v in (task.details or {}).items() if k != "otp"}, "otp_entered": True, "code_typed": typed}
                            note(task, "the login code was typed into the code boxes" if typed else "the agent types the login code")
                        await _start_run(session, agent, task, profile_for, extra)
                        stats["started"] += 1
                elif task.status == "running":
                    run: AgentRun = await agent.poll(task.agent_task)
                    stats["polled"] += 1
                    hit = (guard.past_confirm(task.service, run.path)
                           if task.phase in guard.BEFORE_CONFIRM and not _guest_run(task) and run.status not in ("finished", "failed", "stopped")
                           else None)
                    if hit and (task.phase != "check" or hit[0] == "final"):
                        # On every look, a run before the confirm on a page it must never be on is stopped at once (founder
                        # 2026-10-12: the look-up pressed "Confirm & Book"); the browser stays for the check or the cancel.
                        try:
                            await agent.stop(task.agent_task)
                        except Exception:  # noqa: BLE001 — the run's own end comes next look; the guard acts on its pages
                            logger.warning("guard stop failed task=%s", task.id)
                        note(task, f"stopped by the guard on {hit[1]} ({hit[0]})")
                        run = dataclasses.replace(run, status="stopped")
                    if run.status in ("finished", "failed", "stopped"):
                        before = float(((task.details or {}).get("metrics") or {}).get("cost_inr", 0))
                        metrics.on_browser_run(task, run.steps)
                        sandbox.audit(task, phase=task.phase, status=run.status, steps=run.steps, path=run.path,
                                      cost=float(((task.details or {}).get("metrics") or {}).get("cost_inr", 0)) - before, error=run.error)
                        await sandbox.note_login(session, task.family_id, task.service, run.output or {},
                                                 fresh=bool((task.details or {}).get("otp_entered")), phase=task.phase)
                        await channels.record(session, task.service, "browser", run.status == "finished" and not (run.output or {}).get("blocked"), run.error)
                        out = run.output or {}
                        if (task.details or {}).get("otp_entered") and out.get("problem"):
                            out = {**out, "problem": re.sub(r"\b\d{4,8}\b", "••••", str(out["problem"]))}  # never store a login code
                        retries = int((task.details or {}).get("retries", 0))
                        if run.status == "failed" and not out and retries < MAX_RETRIES and task.phase in ("browse", "prepare", "otp", "check"):
                            # A crash or timeout of the browser run, not an answer from the store: try once more.
                            task.details = {**(task.details or {}), "retries": retries + 1}
                            task.status = "queued"
                            note(task, f"run failed ({run.error or 'no report'}); retrying once")
                            await session.commit()
                            return None
                        if run.status != "finished" and not out:
                            out = {"blocked": False, "problem": run.error or run.status}
                            if task.phase == "place":
                                out["unclear"] = True  # the run may have clicked place before it died
                            if task.phase == "browse":
                                # the store's website did not work (twice): said as that, never as "not found"
                                task.details = {**(task.details or {}), "store_down": True}
                        if task.phase == "place" and practice.on(task) and task.service == "swiggy" and task.agent_session:
                            # Practice: the account's own order list must show nothing new (the agent might have pressed it).
                            try:
                                since = datetime.fromisoformat(task.details["run_started"]) - timedelta(minutes=2)
                                new = await fastpath.swiggy_orders_since(await agent.cdp_url(task.agent_session), since)
                                if new:
                                    out = {**out, "placed": True, "order_id": new[0].get("id") or "unknown"}
                                note(task, "practice check of the order list: " + ("unreadable" if new is None else f"{len(new)} new orders"))
                            except Exception as exc:  # noqa: BLE001 — the agent's own report still stands
                                logger.warning("practice order-list check failed task=%s: %s", task.id, exc)
                        if task.phase == "browse":  # how the login is spoken of comes from the family's saved login
                            task.details = {**(task.details or {}), "login_live": await sandbox.login_live(session, task.family_id, task.service)}
                        if out.get("logged_in") is True:
                            task.details = {**(task.details or {}), "seen_logged_in": True}
                        guarded = _before_confirm_guard(task, run.path or [], out)
                        if not guarded and task.phase == "check":
                            # the store's own list: owned and cancelled, or the step's own report goes on as it would have
                            guarded, out = _after_check(task, out)
                        if not guarded and _needs_store_check(task, out):
                            guarded = _start_check(task, then="outcome", why="every run with the family's login is checked", held=out)
                        if not guarded:  # a check's report is about the store's list, not this task's result
                            task.result = {**(task.result or {}), **{k: v for k, v in out.items() if v not in (None, "", [])}}
                        if guarded:
                            status, message = guarded
                        else:
                            if task.phase == "browse":
                                await _ai_pick(task, task.result, session)
                                out = task.result
                            status, message = await _ride_ends_check(task, out) or _outcome(task, out)
                            status, message = await _check_address(task, out, status, message)
                            status, message = await _later_step(session, task, status, message)
                        if task.phase == "place" and status == "failed" and (task.details or {}).get("place_unclear") and not practice.on(task):
                            found = await _order_on_store_list(task, agent, host_for(task.family_id) if host_for else None)
                            if found:
                                task.result = {**(task.result or {}), **found}
                                task.details = {**(task.details or {}), "place_unclear": False}
                                note(task, f"the store's order list shows it placed: {found.get('order_id')}")
                                metrics.on_milestone(task, "placed")
                                status, message = "done", (f"Placed: {bought(task)} (the store's order list shows it; its confirmation page was slow). Tell the "
                                                           "person the order id; the store gave no delivery time here: say you will check on it.")
                        task.status = status
                        note(task, f"{task.phase} → {status}: {message}")
                        if status == "failed":
                            metrics.on_milestone(task, "failed")
                        stats["finished"] += 1
                        message = _after_cancel_ask(task, status, message)
                        from app.care import skillbook

                        used = list((task.details or {}).get("skills_used") or [])
                        if status in ("awaiting_confirm", "done") and task.phase != "cancel":
                            # The path that worked becomes (or reinforces) a store skill the next agent starts with.
                            await skillbook.record_store_success(
                                session, task.service, task.phase, route(run.path or [], limit=10).split(" → ") if run.path else [],
                                task_id=str(task.id), n_steps=run.steps, used=used,
                            )
                        elif status == "failed" and not out.get("blocked") and not out.get("needs_prescription"):
                            await skillbook.record_store_use(session, used, ok=False)
                        if (task.details or {}).get("unauthorized_place"):
                            await skillbook.block_store_skills(session, used, "placed without a yes")
                        d_ = task.details or {}
                        note_ = (skillbook.safe_note(str(out.get("problem") or ""))
                                 if status == "failed" and not d_.get("otp_entered") and not d_.get("code_lost") else None)
                        task.details = {**d_, "otp_entered": False}
                        if note_:
                            # The next agent on this service reads what stopped this one (once per distinct problem).
                            text_ = f"{task.phase} failed: {note_}"
                            if text_ not in await _learned(session, task.service):
                                session.add(SkillNote(service=task.service, note=text_, created_at=clock.now()))
                    elif clock.now() - datetime.fromisoformat(task.details["run_started"]) > (CHECK_TIMEOUT if task.phase == "check" else RUN_TIMEOUT):
                        await agent.stop(task.agent_task)
                        sandbox.audit(task, phase=task.phase, status="timeout", steps=run.steps, path=run.path, cost=0)
                        task.status = "failed"
                        message = {
                            "check": (f"Checking {SKILLS[task.service]['label']} took too long, so it is NOT known whether anything was "
                                      f"{_words(task)[0]} by mistake. Tell them plainly in one short line to look in the app for one made "
                                      "just now; never say nothing was booked or placed."),
                            "cancel": "Cancel took too long; tell the caregiver.",
                            "place": "Placing took too long and was stopped. It is not clear whether it went through: do not order again; tell the caregiver to check the app.",
                        }.get(task.phase, "The service took too long and the task was stopped; nothing was placed.")
                        note(task, "run timed out")
                if (task.details or {}).get("practice_accident") and not (task.details or {}).get("practice_alerted"):
                    task.details = {**(task.details or {}), "practice_alerted": True}
                    await _ops_alert(host_for(task.family_id) if host_for else None, task, "practice",
                                     f"A practice order reported placed: {(task.result or {}).get('order_id') or 'no id'}. Check and cancel it.")
                d_ = task.details or {}
                if (task.kind in ("order", "ride") and was_browse and not expired and not compare_group(task) and not d_.get("fell_back")
                        and not d_.get("only_store") and (task.status == "failed" or (task.status == "awaiting_confirm" and d_.get("similar_only")))):
                    # The store named does not deliver there or does not have it (or only close variants: they join the
                    # comparison): the other usual stores look, and every store's options go out together.
                    if await _fall_back_to_other_stores(session, task, why=message or ""):
                        message = None
                again = None
                if (task.status == "failed" and task.kind in ("order", "ride") and not was_browse and not expired and compare_group(task)
                        and task.phase in ("prepare", "otp")):
                    # The store they picked could not build the cart (live 2026-10-11: Zepto failed after the others were
                    # dropped): the other stores' options go back to them to pick from.
                    again = await _reoffer_others(session, task, message or "")
                own = message
                if own and was_browse and not expired and compare_group(task) and task.status in ("awaiting_confirm", "needs_input", "failed"):
                    # A store of a comparison: its news waits for the others, then all options go out in one update.
                    message, summary = await _compare_update(session, task, own)
                if own:
                    await store.record_event(
                        session, family_id=task.family_id, subject_id=task.subject_id, kind=f"task_{task.status}",
                        summary=f"{SKILLS[task.service]['label']}: {own}"[:600], payload={"task_id": str(task.id), "result": task.result},
                    )
                if again:
                    message, summary = again, True
                if task.status == "done" and ((task.result or {}).get("placed") or (task.result or {}).get("booked")):
                    await delivery_followup(session, task)
                    tracking.start(task)
                elif task.status == "cancelled" and ((task.result or {}).get("placed") or (task.result or {}).get("booked")):
                    # cancelled on the store after it was placed: its "has it come?" is closed now (live 2026-10-12: a
                    # cancelled order's follow-up stayed open, in the care recipient's name, until its next wake)
                    from app.tasks import aftercare

                    loop = await aftercare.loop_for(session, task)
                    if loop:
                        await store.close_loop(session, loop.id, note="the order was cancelled")
                if (task.details or {}).get("claim"):
                    task.details = {**task.details, "claim": None}  # this runner is done with it
                await session.commit()
            except Exception as exc:  # noqa: BLE001 — one task's failure must not stop the others
                logger.exception("task tick failed task=%s", task_id)
                await session.rollback()
                if " 402" in str(exc) and "credit" in str(exc).lower():
                    # The browser service account is out of credit: retrying cannot help (live 2026-10-09).
                    logger.error("BROWSER SERVICE OUT OF CREDIT: top up Browser Use (task=%s)", task_id)
                    got = await _count_tick_error(session, task_id, final="Ordering online is paused for a short while on "
                                                  "our side, so nothing was placed. Tell them kindly; they can try again later or "
                                                  "order in the app.")
                else:
                    got = await _count_tick_error(session, task_id)
                if not got:
                    return None
                task, message = got
        return (task, message, summary) if message else None

    async def notify_one(r) -> None:
        if r:
            with trace.for_task(r[0]):  # the update Saheli sends about it belongs to the task's trail too
                await _notify_one(r)

    async def _notify_one(r) -> None:
        if r:
            task, message, summary = r
            code_from = (task.details or {}).get("code_from")
            if task.status == "needs_input" and task.input_needed == "otp" and code_from and code_from != task.requested_by:
                # The family set someone else to give login codes for this person's orders: ask them, with the order.
                login = (task.details or {}).get("login")
                await _notify_about(notify, [task], task.family_id, code_from, (
                    f"[Login code needed] {describe(task)}. The order is for person {task.subject_id}, asked by person "
                    f"{task.requested_by}. The store sent a login code to the phone of person {code_from}"
                    + (f" (ending {login})" if login and login != "none" else "") + ". Ask them for it in one short line with what the "
                    f"order is and from which store; when they send it, task_input otp on task {task.id}. If they say cancel, cancel_task "
                    f"it and tell person {task.requested_by} kindly in one line that person {code_from} asked to cancel it."))
                return
            ask = (task.details or {}).get("cancel_ask") or {}
            person = ask.get("by") if task.phase == "cancel" and ask.get("by") else task.requested_by  # a cancel's result: whoever asked for it
            await _notify_about(notify, [task], task.family_id, person,
                                f"[Task update] {message}" if summary else f"[Task update] {describe(task)}. {message}")

    try:
        # An approval someone is waiting for goes out before anything else is done.
        stats["approvals_asked"] = await ask_approvers(sessions, notify)
    except Exception:  # noqa: BLE001
        logger.exception("approval requests failed")
    # Every task moves at once, and its news (a login code, a cart, a choice) goes out the moment its own step is done,
    # never after the other tasks' steps (founder 2026-10-11: an ask that waits on a person is never queued behind other
    # work; live a login code was asked for 33 s after the store sent it).
    parallel = max(1, int(os.getenv("TASK_PARALLEL", "6")))  # tests share one connection: 1
    gate = asyncio.Semaphore(parallel)

    async def one(task_id):
        async with gate:
            r = await process(task_id)
            if parallel == 1:
                await notify_one(r)
                return
        await notify_one(r)  # outside the gate: a slow brain turn never holds up another task's step

    for r in await asyncio.gather(*(one(i) for i in ids), return_exceptions=True):
        if isinstance(r, BaseException):
            logger.error("task step or its update failed: %r", r)
    try:
        for t, text, group in await choose_stores(sessions, agent):
            await _notify_about(notify, group, t.family_id, t.requested_by, f"[Task update] {text}")
    except Exception:  # noqa: BLE001
        logger.exception("store choice failed")
    try:
        stats["carers_told"] = await tell_carers(sessions, notify)
    except Exception:  # noqa: BLE001
        logger.exception("telling caregivers about a placed order failed")
    try:
        stats["approvals_asked"] = stats.get("approvals_asked", 0) + await ask_approvers(sessions, notify)
    except Exception:  # noqa: BLE001
        logger.exception("approval requests failed")
    try:
        stats["escalated"] = await escalate(sessions, agent, notify, host_for)
    except Exception:  # noqa: BLE001
        logger.exception("order escalation failed")
    if light:
        return stats
    try:
        stats["tracked"] = await tracking.sweep(sessions, agent, notify, profile_for=profile_for, host_for=host_for)
    except Exception:  # noqa: BLE001
        logger.exception("delivery tracking failed")
    try:
        from app.tasks import aftercare

        stats["caregivers_told"] = await aftercare.sweep(sessions, notify)
    except Exception:  # noqa: BLE001
        logger.exception("order aftercare failed")
    return stats
