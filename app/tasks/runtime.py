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
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import store
from app.core import clock
from app.specialists import channels, guard, metrics
from app.specialists.agents import specialist_for
from app.specialists.contract import CONTRACT_VERSION, Limits
from app.tasks.browser_use import AgentRun, BrowserAgent
from app.tasks import fastpath, sandbox
from app.tasks.models import SkillNote, Task
from app.tasks.skills import SKILLS

logger = logging.getLogger(__name__)

LIVE = ("queued", "running", "needs_input", "awaiting_confirm")
TASK_LIFETIME = timedelta(minutes=45)
RUN_TIMEOUT = timedelta(minutes=12)

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
COMPARE_STORES = {"grocery": ("blinkit", "instamart", "zepto"), "medicine": ("apollo", "1mg", "pharmeasy"), "food": ("swiggy", "zomato")}
COMPARE_WAIT = timedelta(minutes=4)  # after the first store answers, the others get this long before the options go out


def compare_group(task: Task) -> str | None:
    return (task.details or {}).get("compare")


async def siblings(session: AsyncSession, task: Task) -> list[Task]:
    """Every task of the same comparison, this one included."""
    gid = compare_group(task)
    if not gid:
        return [task]
    rows = (await session.execute(
        select(Task).where(Task.family_id == task.family_id, Task.created_at >= task.created_at - timedelta(minutes=5),
                           Task.created_at <= task.created_at + timedelta(minutes=5))
    )).scalars()
    return [t for t in rows if compare_group(t) == gid]


def _drop_others(group: list[Task], keep: Task, why: str) -> None:
    for t in group:
        if t.id != keep.id and t.status in LIVE:
            t.status, t.cancel_requested, t.input_needed = "cancelled", True, None
            note(t, why)


def _code_to(task: Task) -> str:
    login = (task.details or {}).get("login")
    return f"the phone ending {login}" if login and login != "none" else "the family phone"


def _found(task: Task) -> list[dict]:
    return [i for i in ((task.result or {}).get("items") or []) if i.get("name") and i.get("available") is not False]


def _label(item: dict) -> str:
    """'Diet Coke Can (300 ml)': the name with its pack when the store gave it separately."""
    name, pack = str(item.get("name") or ""), str(item.get("pack") or "").strip()
    return f"{name} ({pack})" if pack and pack.lower() not in name.lower() else name


def _option(item: dict) -> str:
    similar = "" if item.get("exact_match") is not False else " [similar, not exactly what was asked]"
    where = f" from {item['restaurant']}" + (f" ({item['eta']})" if item.get("eta") else "") if item.get("restaurant") else ""
    rx = " [prescription needed]" if item.get("rx_required") else ""
    return f"{_label(item)} {item.get('price') or ''}".strip() + where + rx + similar


def _compare_summary(group: list[Task]) -> str:
    """One update with what every store of the comparison found, for the brain to read out as numbered options."""
    lines, open_ = [], []
    for t in sorted(group, key=lambda t: (not _found(t), SKILLS[t.service]["label"])):
        label, r, d = SKILLS[t.service]["label"], t.result or {}, t.details or {}
        if t.status in ("queued", "running"):
            lines.append(f"- {label}: still looking (its result comes as a separate update)")
        elif t.status == "awaiting_confirm" and t.input_needed == "go":
            found = _found(t)
            eta = f"; delivers in about {r['eta']}" if r.get("eta") else ""
            what = "; ".join(_option(i) for i in found[:4]) if found else "shows prices only after a login"
            linked = " (linked account: no login code)" if d.get("channel") == "connector" else ""
            lines.append(f"- {label} [task {t.id}]{linked}: {what}{eta}")
            open_.append(t)
        elif t.status == "needs_input" and t.input_needed == "swap":
            alts = "; ".join(f"{a['name']}{(' ' + str(a['price'])) if a.get('price') else ''}" for a in (d.get("alternatives") or []))
            lines.append(f"- {label} [task {t.id}]: not there; it has instead: {alts}")
            open_.append(t)
        else:
            lines.append(f"- {label}: {d.get('compare_note') or 'no result'}")
    stores = ", ".join(sorted(SKILLS[t.service]["label"] for t in group))
    head = f"Looked on {stores} without logging in; nothing is ordered.\n" + "\n".join(lines) + "\n"
    if not open_:
        return head + "It is not available on any of them. Tell the person plainly and offer another name, or the family can order in the app."
    return head + (
        "Tell the person the options as a short numbered list: exact matches first (cheapest first), then the similar ones marked "
        "as not exactly what they asked, from every store above that has any (do not leave a store out); in one short line, say "
        "which stores do not deliver there or had nothing. Each option with product, pack size, price, store and delivery time. "
        "Ask which one they want. "
        "When they pick, call task_input on that option's task: kind go with value = the product with its pack and price exactly "
        "as listed above (e.g. 'Diet Coke Can (300 ml) ₹40'); for a 'has instead' item: kind swap. The other stores are dropped "
        f"by themselves. A store marked linked account needs no login code; for the others a login code then comes to "
        f"{_code_to(open_[0])}. If they want none, task_input go no on any one of them.")


async def _compare_update(session: AsyncSession, task: Task, own: str) -> tuple[str | None, bool]:
    """A store of a comparison finished looking: hold its news until every store answered (or COMPARE_WAIT passed since
    the first did), then one update with all the options. A store that answers after that is told on its own.
    Returns (what to tell, whether it is the summary of every store)."""
    group = await siblings(session, task)
    task.details = {**(task.details or {}), "compare_seen": clock.now().isoformat(), "compare_note": own[:240]}
    if any((t.details or {}).get("compare_announced") for t in group if t.id != task.id):
        return (f"Another store answered after the options were sent: {own}" if task.status != "failed" else None), False
    pending = [t for t in group if t.id != task.id and t.status in ("queued", "running")]
    seen = [datetime.fromisoformat(t.details["compare_seen"]) for t in group if (t.details or {}).get("compare_seen")]
    if pending and clock.now() - min(seen) < COMPARE_WAIT:
        return None, False
    task.details = {**task.details, "compare_announced": True}
    return _compare_summary(group), True


async def _announce_overdue(sessions: async_sessionmaker) -> list[tuple[Task, str]]:
    """Comparisons where some stores answered long enough ago while another is still looking: send what is there."""
    out: list[tuple[Task, str]] = []
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
            group = await siblings(session, t)
            if any((s.details or {}).get("compare_announced") for s in group):
                continue
            first = min(datetime.fromisoformat(s.details["compare_seen"]) for s in group if (s.details or {}).get("compare_seen"))
            pending = any(s.status in ("queued", "running") for s in group)
            if pending and clock.now() - first < COMPARE_WAIT:
                continue  # (stores answering at the same time may each have seen the others still looking)
            locked = (await session.execute(select(Task).where(Task.id == t.id).with_for_update(skip_locked=True))).scalar_one_or_none()
            if not locked:
                continue
            locked.details = {**(locked.details or {}), "compare_announced": True}
            out.append((locked, _compare_summary(group)))
        await session.commit()
    return out


YES_WORDS = ("yes", "confirm", "true", "haan", "ha", "han", "ok", "okay", "theek hai", "go", "go ahead", "kar do", "karo")


def flash_phases() -> set[str]:
    """Phases run in Browser Use flash mode (TASK_FLASH_PHASES, e.g. 'browse,place'); off until tested per store."""
    return {p.strip() for p in os.getenv("TASK_FLASH_PHASES", "").split(",") if p.strip()}


def task_max_cost() -> float:
    return float(os.getenv("TASK_MAX_COST_INR", "120"))  # a look-up, login, cart and placing (~80 steps) fit


MAX_RETRIES = 1  # automatic retries of a browser run that crashed without a report (prepare only, never place)
MAX_SKILL_NOTES = 12


def note(task: Task, text: str) -> None:
    task.history = [*(task.history or []), {"at": clock.now().isoformat(), "phase": task.phase, "status": task.status, "note": text[:300]}]
    task.updated_at = clock.now()


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
    verdict = guard.check_request(kind, agent.name, details.get("items") or [], limits, pickup=details.get("pickup"), drop=details.get("drop"))
    if not verdict.ok:
        raise TaskRefused("; ".join(verdict.block))
    now = clock.now()
    task = Task(
        id=uuid.uuid4(), family_id=family_id, subject_id=subject_id, requested_by=requested_by, service=service, kind=kind,
        goal=goal, details={**details, "contract": CONTRACT_VERSION, "agent": agent.name, "limits": limits.to_dict()},
        status="queued", phase="prepare", created_at=now, updated_at=now, deadline_at=now + TASK_LIFETIME, history=[], result={},
    )
    metrics.on_create(task, agent.name)
    session.add(task)
    await session.flush()
    await store.record_event(
        session, family_id=family_id, subject_id=subject_id, kind="task_started", summary=f"{SKILLS[service]['label']}: {goal}",
        payload={"task_id": str(task.id), "agent": agent.name}, actor_id=requested_by,
    )
    return task


async def live_tasks(session: AsyncSession, family_id: str) -> list[Task]:
    q = select(Task).where(Task.family_id == family_id, Task.status.in_(LIVE)).order_by(Task.created_at)
    return list((await session.execute(q)).scalars())


def describe(task: Task) -> str:
    r = task.result or {}
    d = task.details or {}
    bits = [f"[{task.id}] {SKILLS[task.service]['label']} {task.kind}: {task.goal}", f"status {task.status}/{task.phase}"]
    if d.get("channel"):
        bits.append(f"via {d['channel']}")
    if task.input_needed:
        bits.append(f"waiting for {task.input_needed}")
    if r.get("items"):
        in_cart = task.phase != "browse" and bool(d.get("cart_fp") or r.get("total"))
        bits.append(("cart: " if in_cart else "found (not in a cart yet): ") + "; ".join(f"{i.get('qty', 1)} x {i.get('name')} {i.get('price', '')}".strip() for i in r["items"][:12]))
    if r.get("alternatives"):
        bits.append("alternatives: " + "; ".join(map(str, r["alternatives"][:5])))
    if r.get("options"):
        bits.append("ride options: " + "; ".join(f"{o.get('type')} {o.get('fare')}" + (f" (pickup in {o['eta']})" if o.get("eta") else "") for o in r["options"][:6]))
    if r.get("surge"):
        bits.append("SURGE pricing")
    if r.get("total"):
        bits.append(f"total {r['total']}" + (f" incl. fees {r['fees']}" if r.get("fees") else ""))
    if r.get("eta") and task.status == "done":
        bits.append(f"eta {r['eta']}")
    if r.get("order_id") or r.get("ride_id"):
        bits.append(f"id {r.get('order_id') or r.get('ride_id')}")
    if task.cancel_requested:
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
    if task.phase == "prepare" and (task.details or {}).get("login_only"):
        # The cart is built through the store's own request once logged in: the agent only has to log in.
        label = SKILLS[task.service]["label"]
        return (f"On {label}, only log in to the account; do not search, add to the cart or open checkout. When the site shows "
                f"you are logged in, stop and report logged_in=true.")
    return specialist_for(task.service).goal(task)


async def _start_run(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for: Callable[[Task], Awaitable[str | None]], extra: str = "") -> None:
    spec = specialist_for(task.service)
    goal = _goal(task) + (f"\n{extra}" if extra else "")
    learned, skill_ids = await _skills_and_notes(session, task.service, task.phase)
    task.details = {**(task.details or {}), "skills_used": skill_ids}
    profile_id = None
    got = None
    if not task.agent_session:
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
    if task.phase == "prepare" and (task.details or {}).get("browsed") and task.kind == "order" and not (task.details or {}).get("login_only"):
        # The products picked at the go-ahead (one per item asked); older tasks: everything the look-up found.
        seen = (task.details or {}).get("chosen") or [i for i in ((task.result or {}).get("items") or []) if i.get("name")]
        if seen:
            goal += ("\nThe person chose these exact products: " + "; ".join(f"{_label(i)} {i.get('price') or ''}".strip() for i in seen[:8])
                     + ". Put exactly these in the cart (same pack size), nothing else.")
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
            max_steps=spec.steps.get(task.phase, 40),
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
    task.details = {**(task.details or {}), "run_started": clock.now().isoformat(), "channel": "browser"}
    await sandbox.started(session, session_id=run.session_id, task_id=str(task.id), family_id=task.family_id, service=task.service,
                          profile_id=profile_id)
    await session.commit()
    note(task, f"started {task.phase} run ({spec.name} agent, browser)")


NO_WORDS = ("no", "nahi", "nahin", "na", "cancel", "drop", "none", "stop", "mat")


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


def _match_product(found: list[dict], value: str) -> dict | str | None:
    """The product the person picked from a look-up: its exact name, else a part that fits exactly one, else the one
    sharing clearly the most words (two at least). "ambiguous" when it could be more than one."""
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", str(x or "").lower()).strip()  # noqa: E731
    v = norm(value)
    if len(v) < 3:
        return None
    # Two packs can share a name (live 2026-10-09: "Pepsi Zero Sugar Soft Drink" at ₹20 and ₹40, and Saheli asked
    # "₹20 or ₹40?" again and again). The name with its pack and price as listed is unique.
    full = [i for i in found if norm(_label(i)) == v or norm(f"{_label(i)} {i.get('price') or ''}") == v
            or norm(f"{i.get('name')} {i.get('price') or ''}") == v]
    if len(full) == 1:
        return full[0]
    exact = [i for i in found if norm(i.get("name")) == v]
    if len(exact) == 1:
        return exact[0]
    price = re.search(r"(?:₹|rs\.?\s*)\s*(\d+)", str(value), re.I)
    if exact and price:
        by_price = [i for i in exact if re.sub(r"\D", "", str(i.get("price") or "")) == price.group(1)]
        if len(by_price) == 1:
            return by_price[0]
    part = [i for i in found if v in norm(i.get("name")) or norm(i.get("name")) in v]
    if len(part) == 1:
        return part[0]
    words = set(v.split())
    scored = sorted(((len(words & set(norm(i.get("name")).split())), n) for n, i in enumerate(found)), reverse=True)
    if scored and scored[0][0] >= 2 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return found[scored[0][1]]
    return "ambiguous" if len(part) > 1 or len(exact) > 1 else None


def _groups(task: Task, found: list[dict]) -> list[tuple[str, list[dict]]]:
    """(asked item, its products): one group per asked item when the look-up tagged them (for_item), else one group."""
    asked = [str(i.get("name") or "") for i in ((task.details or {}).get("items") or [])]
    if not any(i.get("for_item") for i in found):
        return [(asked[0] if len(asked) == 1 else "", found)] if found else []
    keyed: dict[str, list[dict]] = {}
    for i in found:
        keyed.setdefault(str(i.get("for_item") or ""), []).append(i)
    return list(keyed.items())


def _named_in(value: str, groups: list[tuple[str, list[dict]]]) -> list[dict] | None:
    """The product named for each item when the answer contains whole product names; the longest name wins inside a
    group ("71.5 g x 2" over "71.5 g" when both appear). None unless every item got exactly one."""
    norm = lambda x: re.sub(r"[^a-z0-9]+", "", str(x or "").lower())  # noqa: E731
    v, picks = norm(value), []
    for _, g in groups:
        hits = sorted((i for i in g if norm(i.get("name")) and norm(i.get("name")) in v), key=lambda i: len(norm(i.get("name"))), reverse=True)
        if not hits:
            return None
        priced = [i for i in hits if len(norm(i.get("name"))) == len(norm(hits[0].get("name")))]
        if len(priced) > 1:
            priced = [i for i in priced if norm(f"{i.get('name')} {i.get('price') or ''}") in v] or priced
        if len(priced) != 1:
            return None
        picks.append(priced[0])
    return picks


def _chosen(task: Task, pick: dict | list | None) -> list[dict]:
    """One product per requested item for the cart: the one picked, else the best match the look-up listed first."""
    if isinstance(pick, list):
        picks = pick
        return [next((p for p in picks if p in g), None) or best_exact(asked, g) or g[0] for asked, g in _groups(task, _found(task))]
    found = _found(task)
    asked = [str(i.get("name") or "") for i in ((task.details or {}).get("items") or [])]
    if not any(i.get("for_item") for i in found):
        # Older reports (one best match per item, no for_item): one item asked → one group; several → as listed.
        groups = [found] if len(asked) <= 1 else [[i] for i in found]
    else:
        keyed: dict[str, list[dict]] = {}
        for i in found:
            keyed.setdefault(str(i.get("for_item") or i["name"]).lower(), []).append(i)
        groups = list(keyed.values())
    return [pick if pick in g else g[0] for g in groups if g]


APPROVAL_WAIT = timedelta(minutes=90)
MORE_LIMIT = 12  # products listed for an item they asked to see more of (6 otherwise)


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
    """After an order or ride is placed: one follow-up to check it arrived (the task is not finished until it has)."""
    label = SKILLS[task.service]["label"]
    what = "the cab came" if task.kind == "ride" else "the order arrived"
    await store.open_loop(
        session, family_id=task.family_id, subject_id=task.subject_id, kind="delivery",
        title=f"Check {what}: {label} {task.goal}"[:200], owner_id=task.requested_by,
        wake_at=clock.now() + _eta_wait(task), alert_rule="dashboard", dedupe_key=f"delivery:{task.id}",
        detail={"next_action": f"confirm {what}", "task_id": str(task.id), "max_wakes": 1,
                "order_id": (task.result or {}).get("order_id") or (task.result or {}).get("ride_id")},
    )


async def _approval_gate(session: AsyncSession, task: Task, amount: float | None, by: str, choice: dict | None = None) -> str | None:
    """None when this may be placed now; otherwise the task waits for an approver (the family's boundaries)."""
    from app.care import boundaries

    d = task.details or {}
    policy = await boundaries.get(session, task.family_id)
    roster = await store.roster(session, task.family_id)
    allowed = boundaries.approvers(policy, roster)
    why = await boundaries.reasons(session, family_id=task.family_id, service=task.service, kind=task.kind, amount=amount,
                                   requested_by=task.requested_by, elder_id=task.subject_id, policy=policy)
    # what the cart check found that needs a caregiver (the old fixed-budget line is now the family's own limit, above)
    why += [r for r in d.get("approval") or [] if "limit" not in r and r not in why]
    if not why or boundaries.may_approve(allowed, by, task.subject_id):
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


def _start_placing(task: Task, by: str) -> str:
    d = task.details or {}
    fp = d.get("cart_fp")
    task.details = {**d, "confirmed_by": by, "confirmed_total": (task.result or {}).get("total"), "confirmed_fp": fp,
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
        rows = list((await session.execute(select(Task).where(Task.status == "awaiting_confirm", Task.input_needed == "approve"))).scalars())
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
                out.append((t.family_id, a, (
                    f"[Approval needed] [{t.id}] Person {t.requested_by} asked for: {describe(t)}{amount}"
                    f"{f' ({what})' if what else ''}. It needs approval because: {'; '.join(ap.get('reasons') or [])}. Ask person {a} "
                    f"with send_message, short, in their language: approve or decline? When they answer, call task_input with "
                    f"task_id {t.id}, kind approve, value yes or no. Then reply none.")))
        await session.commit()
    for family_id, person, text in out:
        await notify(family_id, person, text)
    return len(out)


async def provide_input(session: AsyncSession, task: Task, *, kind: str, value: str, by: str, by_is_elder: bool) -> str:
    """The person answered: an OTP, a confirm, a fee approval, a ride choice or a swap. Returns what happens next.

    The answer must be the one the task is waiting for (task.input_needed); anything else is refused, so a
    stray "yes" or a ride name can never place something nobody was shown."""
    if task.status not in ("needs_input", "awaiting_confirm"):
        return f"not waiting for input (status {task.status})"
    from app.care.boundaries import may_approve

    value = (value or "").strip()
    d = task.details or {}
    placed = bool((task.result or {}).get("placed") or (task.result or {}).get("booked"))
    if kind in ("confirm", "choice", "swap", "go", "approve") and value.lower() in NO_WORDS and not placed:
        if kind == "approve" and not may_approve((d.get("approval_needed") or {}).get("approvers") or [], by, task.subject_id):
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
        if kind == "approve":
            return f"declined; nothing was placed. Tell person {task.requested_by} kindly with send_message, in one line."
        return "declined; nothing was placed"
    may = may_approve((d.get("approval_needed") or {}).get("approvers") or [], by, task.subject_id)
    if task.input_needed == "approve" and kind == "confirm" and may:
        kind = "approve"  # the approver saying "yes, confirm" is the approval
    if task.input_needed and kind != task.input_needed and not (kind == "more" and task.input_needed == "go"):
        if task.input_needed == "approve":
            return "this is waiting for the family approver's answer (already asked); nothing is placed until they say yes"
        return f"the task is waiting for {task.input_needed}, not {kind}; ask for that"
    if kind == "go":
        if task.phase != "browse" or task.status != "awaiting_confirm":
            return "nothing to go ahead with right now"
        pick = None
        if value.lower() not in YES_WORDS:
            found = _found(task)
            groups = _groups(task, found)
            if len(groups) > 1:
                # One name per item (several items in one cart). Full product names first: a name can itself contain
                # " | " (Instamart: "Britannia 5050 Potazos … Crisps | 71.5 g | Potato and Biscuit …"); then "a | b" parts.
                picks = _named_in(value, groups)
                if picks is None:
                    picks = []
                    for part in [p.strip() for p in re.split(r"\s*\|\s*|\n", value) if p.strip()]:
                        hit = next((m for _, g in groups if isinstance(m := _match_product(g, part), dict)), None)
                        if not hit:
                            return f"'{part}' is not one of the products found; ask again, one name per item separated by ' | '"
                        picks.append(hit)
                pick = picks
            else:
                pick = _match_product(found, value)
                listed = "; ".join(f"{_label(i)} {i.get('price') or ''}".strip() for i in found)
                if pick == "ambiguous":
                    return f"that could be more than one product; ask which exactly: {listed}"
                if not pick:
                    return (f"say yes to log in and build the cart, name one of the products found ({listed}), or no to drop it"
                            if found else "say yes to log in and build the cart, or no to drop it")
        chosen = _chosen(task, pick)
        task.details = {**d, "chosen": chosen} if chosen else d
        task.phase, task.status, task.input_needed = "prepare", "queued", None
        task.deadline_at = clock.now() + TASK_LIFETIME
        note(task, f"go-ahead from {by}" + (f" for {'; '.join(_label(p) for p in pick)}" if isinstance(pick, list) else f" for {_label(pick)}" if pick else ""))
        dropped = ""
        if compare_group(task):
            group = await siblings(session, task)
            _drop_others(group, task, f"{by} picked {SKILLS[task.service]['label']}")
            if len(group) > 1:
                dropped = "; the other stores are dropped"
        what = "; ".join(f"{_label(i)} {i.get('price') or ''}".strip() for i in chosen)
        how = ("building the cart on the linked account (no login code; NOT ordered; you will get a task update)"
               if (task.details or {}).get("channel") == "connector" else
               "building the cart (NOT ordered). Say only that you are getting the cart ready; do not promise a login code: if one "
               "is needed you will get a task update to ask for it")
        return f"going ahead on {SKILLS[task.service]['label']}" + (f" with {what}" if what else "") + f"{dropped}: {how}"
    if kind == "more":
        # During the look-up: more options, or another pack or size, of one item (order lab 2026-10-10: "Maggi more" →
        # the 840 g pack was not among the six listed). That item is looked up again with the words given, longer list.
        if task.kind != "order" or task.phase != "browse" or task.status != "awaiting_confirm":
            return "nothing is being looked up right now"
        words, items = value.strip(), list(d.get("items") or [])
        if not words or not items:
            return "say which item to look for again, e.g. 'maggi' or 'maggi 840 g'"
        want = _stems(words)
        groups = dict(_groups(task, _found(task)))

        def score(i: int) -> int:
            asked = str(items[i].get("name") or "")
            seen = _stems(asked) | {w for p in groups.get(asked, []) for w in _stems(str(p.get("name") or ""))}
            return len(want & seen)

        best = max(range(len(items)), key=score)
        if score(best) == 0 and len(items) > 1:
            return "which item is that for? say its name with what to look for, e.g. 'maggi 840 g'"
        asked = str(items[best].get("name") or "")
        name = words if want & _stems(asked) or not asked else f"{asked} {words}"
        new_items = items[:best] + [{**items[best], "name": name}] + items[best + 1:]
        v = guard.check_request("order", d.get("agent") or "shopping", new_items, Limits.from_dict(d.get("limits")))
        if not v.ok:
            return "cannot look for that: " + "; ".join(v.block)
        task.details = {**d, "items": new_items, "more": name, "browsed": False, "chosen": None}
        task.phase, task.status, task.input_needed = "prepare", "queued", None
        task.deadline_at = clock.now() + TASK_LIFETIME
        note(task, f"more options for {name} asked by {by}")
        return (f"looking for more {name} on {SKILLS[task.service]['label']}; the list comes back in a task update (nothing ordered). "
                "Say only that you are looking for more options")
    if kind == "confirm":
        if task.kind != "order" or task.status != "awaiting_confirm":
            return "nothing to confirm right now"
        if value.lower() not in YES_WORDS:
            return "say yes to place it, or no to drop it"
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
        if value.lower() in ("yes", "approve", "true", "haan", "ok"):
            task.details = {**d, "fee_ok": True}
            task.phase, task.status = "cancel", "queued"
            task.deadline_at = clock.now() + TASK_LIFETIME
            note(task, f"cancel fee approved by {by}")
            return "cancelling with the fee"
        task.status = "done"
        note(task, f"cancel fee declined by {by}; order stays")
        return "kept the order; not cancelled"
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
        opt = _match_option(options, value)
        if opt == "ambiguous":
            return "that could be more than one option; ask which exactly: " + "; ".join(f"{o.get('type')} {o.get('fare', '')}".strip() for o in options)
        if not opt:
            return "that is not one of the options; read them again: " + "; ".join(f"{o.get('type')} {o.get('fare', '')}".strip() for o in options)
        choice = f"{opt.get('type')} {opt.get('fare') or ''}".strip()
        gate = await _approval_gate(session, task, rupees(opt.get("fare")), by, choice={"choice": choice, "fare": opt.get("fare")})
        if gate:
            return gate
        return _start_booking(task, by, choice, opt.get("fare"))
    if kind == "approve":
        ap = d.get("approval_needed") or {}
        if not ap:
            return "nothing is waiting for approval"
        if not may_approve(ap.get("approvers") or [], by, task.subject_id):
            return "only the family's approver can approve this; it was sent to them"
        if value.lower() not in YES_WORDS:
            return "say yes to approve or no to decline"
        task.details = {**d, "approval_needed": None, "approved_by": by, "approved_at": clock.now().isoformat()}
        note(task, f"approved by {by}")
        if task.kind == "ride":
            c = ap.get("choice") or {}
            out = _start_booking(task, by, c.get("choice") or "", c.get("fare"))
        else:
            if not (task.details or {}).get("cart_fp"):
                return "the cart is gone; nothing was placed. Ask them to order again"
            out = _start_placing(task, by)
        return f"approved. {out}. Tell person {task.requested_by} in one line with send_message that it was approved."
    return f"unknown input {kind}"


async def request_cancel(session: AsyncSession, agent: BrowserAgent, task: Task, *, by: str, reason: str) -> str:
    if task.status in ("failed", "cancelled"):
        return f"already {task.status}"
    if task.phase == "cancel" and task.status in LIVE:
        return "cancelling is already in progress; you will get a task update"
    task.cancel_requested = True
    note(task, f"cancel requested by {by}: {reason}")
    placed = bool((task.result or {}).get("placed") or (task.result or {}).get("booked"))
    if task.status == "done" and placed:
        task.phase, task.status = "cancel", "queued"
        task.deadline_at = clock.now() + TASK_LIFETIME
        return "already placed; cancelling it on the service now"
    if task.phase == "place" and task.status == "queued" and (task.details or {}).get("place_started"):
        return "placing has started; it will be cancelled right after if it went through"
    if task.status == "running" and task.phase == "place":
        return "placing is in progress; it will be cancelled right after if it went through"
    if task.status == "running" and task.agent_task:
        try:
            await agent.stop(task.agent_task, end_session=True)
        except Exception:  # noqa: BLE001
            logger.warning("stop failed task=%s", task.id)
    task.status = "cancelled"
    note(task, "stopped before placing")
    if task.agent_session:
        sandbox.audit(task, phase=task.phase, status="stopped", steps=0, path=[], cost=0, error=f"cancelled by the family: {reason}"[:120])
    return "stopped; nothing was placed"


def _outcome(task: Task, out: dict) -> tuple[str, str]:
    """(new status, what the family should hear) from an agent report for the current phase. Every cart and
    every placement passes the guard: a rule broken in the report stops it here, whatever the agent said."""
    d = task.details or {}
    limits = Limits.from_dict(d.get("limits"))
    spec = specialist_for(task.service)
    if task.phase == "browse":
        if out.get("blocked"):
            return "failed", f"{SKILLS[task.service]['label']} blocked the request ({out.get('problem') or 'site wall'}). Offer another service."
        return _browse_outcome(task, out)
    login = d.get("login")
    if out.get("needs_otp") and login == "none":
        # It had no number to log in with, so no code can have been sent (live 2026-10-08: "sent to phone", field empty).
        out = {**out, "needs_otp": False, "problem": "login needed"}
    if out.get("needs_otp"):
        task.input_needed = "otp"
        to = f"the phone ending {login}" if login else (out.get("otp_sent_to") or "the family phone")
        if d.get("code_lost"):
            task.details = {**(task.details or {}), "code_lost": False}
            return "needs_input", (f"The store's login page closed before the code could be used, so it sent a new code to {to}. "
                                   "Say sorry briefly and ask for the new code (the earlier one will not work).")
        return "needs_input", f"The service sent a login code to {to}; ask the person who has that phone for it."
    if task.phase in ("prepare", "otp") and d.get("login_only") and out.get("logged_in") and not out.get("blocked"):
        # Logged in: back to the fast cart (once more), then the cart comes back for a confirm.
        task.details = {**d, "login_only": False, "login_done": True, "fast_cart_tried": False}
        return "queued", ""
    if (out.get("problem") or "").lower().startswith("login needed") and not out.get("blocked"):
        return "failed", (f"{SKILLS[task.service]['label']} needs a login and there is no phone number to log in with, so nothing was "
                          "ordered and no code was sent. Offer another service, or the family can order in the app.")
    if out.get("blocked"):
        return "failed", f"{SKILLS[task.service]['label']} blocked the request ({out.get('problem') or 'site wall'}). Offer another service."
    if task.phase in ("prepare", "otp") and (out.get("placed") or out.get("booked") or out.get("order_id") or out.get("ride_id")):
        # The agent went past the cart without anyone's yes (a bad hint, a misread page). Never place again on top of it.
        task.details = {**d, "unauthorized_place": True, "confirm_token": None}
        ref = out.get("order_id") or out.get("ride_id") or "no id shown"
        return "failed", (f"The agent reports it {'booked' if task.kind == 'ride' else 'placed'} this on {SKILLS[task.service]['label']} before anyone "
                          f"said yes ({ref}). Nothing more will be placed. Tell the caregiver now to check the app and cancel it if it is not wanted.")
    if task.phase in ("prepare", "otp"):
        if task.kind == "ride":
            if not out.get("options"):
                return "failed", f"No ride options came up ({out.get('problem') or 'unknown'})."
            v = guard.check_cart("ride", spec.name, [], out, limits)
            task.details = {**d, "approval": v.approval, "confirm_token": None,
                            "cart_fp": guard.cart_fingerprint({"items": [{"name": f"{o.get('type')} {o.get('fare')}"} for o in out["options"]], "total": None})}
            task.input_needed = "choice"
            metrics.on_milestone(task, "ready")
            extra = " ".join(v.warn + v.approval)
            return "awaiting_confirm", "Ride options are ready; tell the person the fares (and any surge) and ask which one to book." + (f" {extra}" if extra else "")
        if out.get("needs_prescription") and not guard.rx_covers(d.get("items") or [], limits.rx_on_file, cart=out.get("items")):
            return "failed", "This medicine needs a prescription upload; ask the family to upload the prescription on the dashboard, then try again."
        if not out.get("items"):
            alts = guard.alternatives(out.get("alternatives"))[:4]
            if alts and out.get("cod_available") is not False:
                # Out of stock: offer what the store has instead; nothing is swapped without the person choosing.
                task.details = {**d, "alternatives": alts}
                task.input_needed = "swap"
                listed = "; ".join(f"{a['name']}{(' ' + str(a['price'])) if a.get('price') else ''}" for a in alts)
                return "needs_input", f"Not available ({out.get('problem') or 'out of stock'}). The store has: {listed}. Ask which one to get instead, or whether to drop it."
            return "failed", f"Could not build the cart ({out.get('problem') or 'unknown'})."
        asked = d.get("items") or []
        if d.get("chosen") and spec.name != "pharmacy":
            # The person picked the products: check the cart against those, not the words asked (order lab 2026-10-10:
            # "biscuits, munchies" were reported "not in the cart" next to Dark Fantasy and Potazos). Medicines keep the
            # asked names: never substituted.
            asked = [{"name": c.get("name"), "qty": channels._qty_for(asked, c) if asked else 1} for c in d["chosen"] if c.get("name")]
        v = guard.check_cart("order", spec.name, asked, out, limits)
        if not v.ok:
            return "failed", "Stopped before placing: " + "; ".join(v.block) + ". Nothing was placed."
        task.details = {**d, "approval": v.approval, "cart_fp": guard.cart_fingerprint(out), "confirm_token": None}
        task.input_needed = "confirm"
        metrics.on_milestone(task, "ready")
        extra = " ".join(v.warn + v.approval)
        return "awaiting_confirm", "The cart is ready; read the items and the total to the person and ask them to confirm." + (f" {extra}" if extra else "")
    if task.phase == "place":
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
        if out.get("placed") or out.get("booked"):
            v = guard.check_placed(task.kind, out, guard.rupees(d.get("confirmed_total")))
            metrics.on_milestone(task, "placed")
            if v.warn:
                return "done", "Placed, but: " + "; ".join(v.warn)
            eta = str(out.get("eta") or "").strip()
            if eta:
                return "done", f"Placed. Tell the person the order/ride id and the delivery time: {eta}."
            # Live 2026-10-10: connector and Blinkit orders came back without a time and the brain was told to give one.
            return "done", ("Placed. Tell the person the order/ride id. The store gave no delivery time: do not guess one; "
                            "say you will check on it.")
        if out.get("unclear"):
            return "failed", f"It is not clear whether the order went through ({out.get('problem')}). Do not order again; tell the caregiver to check the app."
        return "failed", f"Placing did not go through ({out.get('problem') or 'unknown'}). Nothing was charged."
    if task.phase == "cancel":
        if out.get("cancel_fee") and not d.get("fee_ok"):
            task.input_needed = "fee"
            return "needs_input", f"Cancelling now costs {out['cancel_fee']}; ask whether to cancel anyway."
        if out.get("cancelled"):
            metrics.on_milestone(task, "cancelled")
            return "cancelled", "Cancelled on the service. Tell the person."
        return "failed", f"Could not cancel ({out.get('problem') or 'unknown'}); tell the caregiver to cancel from the app."
    return "failed", "Unexpected state."


def _browse_outcome(task: Task, out: dict) -> tuple[str, str]:
    """What the look-up found (no login yet): not delivered there, not found, or found → go-ahead to log in."""
    d = task.details or {}
    label = SKILLS[task.service]["label"]
    place = (d.get("limits") or {}).get("place") or {}
    where = place.get("nickname") or place.get("pincode") or "their place"
    task.details = {**d, "browsed": True}
    login = d.get("login")
    code_to = f"the phone ending {login}" if login and login != "none" else "the family phone"
    if out.get("deliverable") is False:
        return "failed", (f"{label} does not deliver to {where} ({out.get('location_set') or out.get('problem') or 'not serviceable'}); "
                          "nothing was ordered. Offer another service.")
    if task.kind == "ride":
        opts = [o for o in (out.get("options") or []) if o.get("type")]
        if opts and not out.get("logged_in"):
            task.input_needed = "go"
            fares = "; ".join(f"{o.get('type')} {o.get('fare') or ''}".strip() for o in opts[:6])
            return "awaiting_confirm", (f"Fares on {label} (before login, may change a little): {fares}. Nothing is booked. To book, {label} "
                                        f"needs a login: a code will come to {code_to}. Ask which option they want and whether to go ahead "
                                        "(task_input go); after the login the fresh fares come back to choose from.")
        if opts or out.get("logged_in"):
            task.phase = "prepare"  # logged in already: get the fares to choose from straight away
            return "queued", ""
    found = [i for i in (out.get("items") or []) if i.get("name") and i.get("available") is not False]
    if task.kind == "ride" and not out.get("login_required"):
        return "failed", f"No ride options came up on {label} ({out.get('problem') or 'unknown'}); nothing was booked. Offer another service."
    if not found and out.get("login_required"):
        # The site shows nothing useful without a login: the go-ahead is to log in and look.
        task.input_needed = "go"
        return "awaiting_confirm", (f"{label} shows prices only after a login; nothing is ordered. To look and order, a login code will "
                                    f"come to {code_to}. Ask whether to go ahead (task_input go).")
    if not found:
        alts = guard.alternatives(out.get("alternatives"))[:4]
        if alts and d.get("agent") != "pharmacy":
            task.details = {**task.details, "alternatives": alts}
            task.input_needed = "swap"
            listed = "; ".join(f"{a['name']}{(' ' + str(a['price'])) if a.get('price') else ''}" for a in alts)
            return "needs_input", f"Not on {label} ({out.get('problem') or 'not found'}). It has: {listed}. Ask which one to get instead, or whether to drop it."
        return "failed", f"Could not find it on {label} ({out.get('problem') or 'not found'}); nothing was ordered. Offer another service."
    more = d.get("more")

    def shown(asked: str) -> int:  # the item they asked to see more of keeps its longer list
        return MORE_LIMIT if more and asked == more else 5

    seen = "; ".join(_option(i) for i in found[:MORE_LIMIT if more else 6])
    eta = f", about {out['eta']}" if out.get("eta") else ""
    if out.get("logged_in") and not compare_group(task):  # in a comparison the person picks the store first
        # Already logged in: no code needed. Build the cart straight away only for the one product that is clearly what
        # they asked for; otherwise they pick (lab 2026-10-10: "Parle-G 475 g" got the first hit, "1 kg x 2" ₹250).
        groups = _groups(task, found)
        picks = [best_exact(asked, g) for asked, g in groups]
        if groups and all(picks):
            task.details = {**task.details, "chosen": picks}
            task.phase = "prepare"
            return "queued", ""
        task.input_needed = "go"
        if len(groups) > 1:
            listed = " ".join(f"For {asked}: " + "; ".join(_option(i) for i in g[:shown(asked)]) + "." for asked, g in groups)
            return "awaiting_confirm", (f"Found on {label}: {listed} Delivers to {where}{eta}. Nothing is ordered yet and no login code is "
                                        "needed. Ask which one they want for each item and pass the names, separated by ' | ', as the "
                                        "value of task_input go (more options or another size of one item: task_input more).")
        return "awaiting_confirm", (f"Found on {label}: {seen}; delivers to {where}{eta}. Nothing is ordered yet and no login code is "
                                    "needed. Ask which one they want and pass its name as the value of task_input go (more options or another size of one item: task_input more).")
    task.input_needed = "go"
    groups = _groups(task, found)
    if len(groups) > 1:
        # several items: every item's options, not the first six of all (lab 2026-10-10: the paneer options were cut off)
        listed = " ".join(f"For {asked}: " + "; ".join(_option(i) for i in g[:shown(asked)]) + "." for asked, g in groups)
        return "awaiting_confirm", (f"Found on {label}: {listed} Delivers to {where}{eta}. Nothing is ordered yet. To order, {label} needs a "
                                    f"login: a code will come to {code_to}. Ask which one they want for each item and pass the names, "
                                    "separated by ' | ', as the value of task_input go (or yes for the first of each).")
    pick = (" If several are listed, ask which one and pass its name as the value of task_input go." if len(found) > 1 else "")
    return "awaiting_confirm", (f"Found on {label}: {seen}; delivers to {where}{eta}. Nothing is ordered yet. To order, {label} needs a "
                                f"login: a code will come to {code_to}. Ask whether to go ahead (task_input go).{pick}")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+|\d+", str(text or "").lower()) if len(w) > 2 or w.isdigit()}


def _stems(text: str) -> set[str]:
    """Words without a plural s, so "biscuit" matches "Biscuits" and "475 g" keeps its 475."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in _words(text)}


def asked_match(asked: str, name: str) -> bool | None:
    """Every word (and pack number) they asked for is in the product name; None when nothing was asked. A long word
    also matches when the name writes it apart ("RiteBite" in "Rite Bite Max Protein")."""
    want = _stems(asked)
    if not want:
        return None
    have, joined = _stems(name), re.sub(r"[^a-z0-9]", "", str(name or "").lower())
    return all(w in have or (len(w) >= 5 and w in joined) for w in want)


MULTIPACK = re.compile(r"\bx\s*\d+\b|\bpack of \d+|\b\d+\s*x\b", re.I)


def best_exact(asked: str, found: list[dict]) -> dict | None:
    """The one product that is clearly what was asked: a single listing, else the only full match, else (when the ask
    names no multipack) the only full match that is not a multipack ("240 g" over "240 g x 2")."""
    if len(found) == 1:
        return found[0]
    exact = [i for i in found if i.get("exact_match")]
    if len(exact) > 1 and not MULTIPACK.search(asked or ""):
        exact = [i for i in exact if not MULTIPACK.search(str(i.get("name") or ""))]
    return exact[0] if len(exact) == 1 else None


async def _connector_lookup(session: AsyncSession, host, task: Task) -> dict | None:
    """Browse through the linked store's connector. Returns a look-up report, or None to look in the browser instead."""
    metrics.on_connector_call(task)
    asked_items = (task.details or {}).get("items") or [{}]
    try:
        # One look-up per asked item, together (several items go in one cart).
        results = await asyncio.gather(*(channels.connector_search(host, task, it) for it in asked_items))
    except Exception as exc:  # noqa: BLE001 — a broken connector falls back to the browser
        await channels.record(session, task.service, "connector", False, str(exc)[:200])
        return None
    bad = next((r for r in results if not r.get("ok") and r.get("kind") != "unserviceable"), None)
    if bad is not None:
        await channels.record(session, task.service, "connector", False, bad.get("detail"))
        return None
    await channels.record(session, task.service, "connector", True)
    task.phase = "browse"
    if any(not r.get("ok") for r in results):
        r = next(r for r in results if not r.get("ok"))
        return {"items": [], "deliverable": False, "problem": r.get("detail") or "not serviceable here", "logged_in": True}
    many = len(asked_items) > 1
    items = [{"name": i.get("name"), "price": i.get("price"), "available": True, "ref": i.get("ref"),
              **({"for_item": it.get("name")} if many else {}),
              # the connector does not say; a product missing a word that was asked for (Coke Zero for Diet Coke) is only similar
              "exact_match": asked_match(it.get("name") or "", i.get("name"))}
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
    for words in (d["pickup"], d["drop"]):
        got = await host.call("ride_place", {"words": words}, family_id=task.family_id.removeprefix("shadow:"),
                              subject_id=task.subject_id, actor_id=task.requested_by)
        if not got or got.get("lat") is None or got.get("lng") is None:
            return None
        ends.append({"lat": got["lat"], "lng": got["lng"], "label": got.get("label") or words})
    return ends[0], ends[1]


async def _fast_lookup(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for, host=None) -> dict | None:
    """The look-up through the store's own web request in the task's cloud browser (seconds, not minutes). The browser
    stays open for the login and cart runs. None (and a note) when it cannot run, so the browser agent looks instead."""
    place = ((task.details or {}).get("limits") or {}).get("place") or {}
    lat, lng, pincode = place.get("lat"), place.get("lng"), place.get("pincode")
    ride = fastpath.SITES[task.service].kind == "ride"
    if not hasattr(agent, "open_session") or (not ride and (lat is None or lng is None) and not pincode):
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
    except Exception as exc:  # noqa: BLE001 — a changed site or a slow browser: the agent looks instead
        note(task, f"fast look-up did not work ({str(exc)[:120]}); the browser agent looks instead")
        logger.warning("fast look-up failed task=%s: %s", task.id, exc)
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


async def _fast_cart(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for) -> dict | None:
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
            await session.commit()
        cdp = await agent.cdp_url(task.agent_session)
        if not cdp:
            raise fastpath.FastPathError("no CDP address for the browser")
        asked = d.get("items") or []
        cart = await fastpath.cart(task.service, cdp, [{**c, "qty": c.get("qty") or channels._qty_for(asked, c)} for c in chosen],
                                   place=place, lat=place.get("lat"), lon=place.get("lng"))
    except Exception as exc:  # noqa: BLE001 — not logged in yet, or the site changed: the agent does it
        if "not logged in" in str(exc) and not (task.details or {}).get("login_done"):
            task.details = {**(task.details or {}), "login_only": True}
            note(task, "not logged in yet: the browser agent logs in, then the cart is built fast")
        else:
            note(task, f"fast cart not possible ({str(exc)[:100]}); the browser agent builds it")
        return None
    note(task, f"fast cart on {SKILLS[task.service]['label']}: total {cart.get('total')}")
    check = cart.pop("place_check", None)
    if check:
        # the place step re-reads the cart against this before sending the order
        task.details = {**(task.details or {}), "fast_place_check": check}
    return cart


async def _fast_place(session: AsyncSession, agent: BrowserAgent, task: Task, profile_for) -> dict | None:
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
            await session.commit()
            cdp = await agent.cdp_url(task.agent_session)
        if not cdp:
            raise fastpath.FastPathError("no CDP address for the browser")
    except Exception as exc:  # noqa: BLE001 — nothing was sent: the agent places as before
        note(task, f"fast place not possible ({str(exc)[:100]}); the browser agent places it")
        task.details = {**d, "fast_place_check": None}
        return None
    try:
        out = await fastpath.place(task.service, cdp, d["fast_place_check"])
    except fastpath.FastPathError as exc:
        # Nothing was sent (the cart, total or address changed): build the cart again and ask for a fresh confirm.
        task.phase = "prepare"
        task.details = {**d, "fast_place_check": None, "fast_cart_tried": False, "cart_fp": None, "confirm_token": None, "place_started": None}
        return {"requeue": True, "requeue_note": f"not placed, nothing sent ({str(exc)[:120]}); building the cart again for a fresh confirm"}
    note(task, f"fast place on {SKILLS[task.service]['label']}: {'placed ' + str(out.get('order_id')) if out.get('placed') else 'unclear'}")
    return out


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


MAX_TICK_ERRORS = 3


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
) -> dict:
    stats = {"started": 0, "polled": 0, "finished": 0, "expired": 0}
    try:
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
        message, summary = None, False
        async with sessions() as session:
            task = (await session.execute(select(Task).where(Task.id == task_id).with_for_update(skip_locked=True))).scalar_one_or_none()
            if not task:
                return None
            was_browse, expired = task.phase == "browse", False
            try:
                if task.deadline_at < clock.now() and task.status != "running":
                    task.status, expired = "failed", True
                    note(task, "timed out waiting")
                    message = "Nobody answered in time, so the task was stopped; nothing was placed." if task.phase != "cancel" else "Cancel was not finished; tell the caregiver."
                    from app.brain.wake import in_quiet_hours

                    if task.phase != "cancel" and in_quiet_hours():
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
                    message = ("It is not clear whether it was placed (the last attempt was interrupted). Do not order again; "
                               "tell the caregiver to check the app.")
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
                                    task.phase = "prepare"  # the browser builds its own cart; the family confirms it again
                                note(task, "connector failed, falling back to the browser")
                    if (report is None and task.phase == "prepare" and browse_first(task) and task.service in fastpath.FAST_SERVICES
                            and fastpath_on()):
                        report = await _fast_lookup(session, agent, task, profile_for, host_for(task.family_id) if host_for else None)
                        if report is not None:
                            was_browse = True
                    elif (report is None and task.phase == "prepare" and task.kind == "order" and task.service in fastpath.CART_SITES
                            and fastpath_on() and not (task.details or {}).get("fast_cart_tried")):
                        report = await _fast_cart(session, agent, task, profile_for)
                    elif (report is None and task.phase == "place" and task.kind == "order" and task.service in fastpath.PLACE_SITES
                            and fastpath_on() and (task.details or {}).get("fast_place_check")):
                        report = await _fast_place(session, agent, task, profile_for)
                    if report is not None and report.get("requeue"):
                        note(task, report.get("requeue_note") or "connector card expired; building it again")
                    elif report is not None:
                        task.result = {**(task.result or {}), **{k: v for k, v in report.items() if v not in (None, "", [])}}
                        status, message = _outcome(task, report)
                        task.status = status
                        via = ("connector" if (task.details or {}).get("channel") == "connector" else
                               "fast look-up" if task.phase == "browse" else "saved steps")
                        note(task, f"{task.phase} → {status} ({via}): {message}")
                        if status == "done" and task.cancel_requested:
                            task.phase, task.status = "cancel", "queued"
                            task.deadline_at = clock.now() + TASK_LIFETIME
                            message = "It was placed just before the cancel; cancelling it on the service now."
                        if status == "failed":
                            metrics.on_milestone(task, "failed")
                        stats["finished"] += 1
                    elif float(((task.details or {}).get("metrics") or {}).get("cost_inr", 0)) >= task_max_cost():
                        # Too many browser steps already: stop rather than keep paying for a task that is not working.
                        task.status = "failed"
                        message = ("This took too many tries on the website, so it was stopped to keep costs down; nothing was placed. "
                                   "Offer to try another store, or the family can order in the app.") if task.phase != "place" else (
                            "This took too many tries and was stopped. It is not clear whether it was placed: do not order again; check the app.")
                        note(task, f"cost ceiling ₹{task_max_cost():.0f} reached")
                        metrics.on_milestone(task, "failed")
                    else:
                        if browse_first(task):
                            task.phase = "browse"
                            note(task, "looking it up before any login")
                        extra = f"Enter this code where the login asks for it: {task.details.get('otp')}. Then continue with the task." if task.phase == "otp" else ""
                        if task.phase == "otp":
                            task.phase = "prepare"
                            # The code is handed to the agent once and not kept on the task.
                            task.details = {**{k: v for k, v in (task.details or {}).items() if k != "otp"}, "otp_entered": True}
                        await _start_run(session, agent, task, profile_for, extra)
                        stats["started"] += 1
                elif task.status == "running":
                    run: AgentRun = await agent.poll(task.agent_task)
                    stats["polled"] += 1
                    if run.status in ("finished", "failed", "stopped"):
                        before = float(((task.details or {}).get("metrics") or {}).get("cost_inr", 0))
                        metrics.on_browser_run(task, run.steps)
                        sandbox.audit(task, phase=task.phase, status=run.status, steps=run.steps, path=run.path,
                                      cost=float(((task.details or {}).get("metrics") or {}).get("cost_inr", 0)) - before, error=run.error)
                        await sandbox.note_login(session, task.family_id, task.service, run.output or {})
                        await channels.record(session, task.service, "browser", run.status == "finished" and not (run.output or {}).get("blocked"), run.error)
                        out = run.output or {}
                        if (task.details or {}).get("otp_entered") and out.get("problem"):
                            out = {**out, "problem": re.sub(r"\b\d{4,8}\b", "••••", str(out["problem"]))}  # never store a login code
                        retries = int((task.details or {}).get("retries", 0))
                        if run.status == "failed" and not out and retries < MAX_RETRIES and task.phase in ("browse", "prepare", "otp"):
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
                        task.result = {**(task.result or {}), **{k: v for k, v in out.items() if v not in (None, "", [])}}
                        status, message = _outcome(task, out)
                        task.status = status
                        note(task, f"{task.phase} → {status}: {message}")
                        if status == "failed":
                            metrics.on_milestone(task, "failed")
                        stats["finished"] += 1
                        if status == "done" and task.cancel_requested:
                            task.phase, task.status = "cancel", "queued"
                            task.deadline_at = clock.now() + TASK_LIFETIME
                            message = "It was placed just before the cancel; cancelling it on the service now."
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
                    elif clock.now() - datetime.fromisoformat(task.details["run_started"]) > RUN_TIMEOUT:
                        await agent.stop(task.agent_task)
                        sandbox.audit(task, phase=task.phase, status="timeout", steps=run.steps, path=run.path, cost=0)
                        task.status = "failed"
                        message = {
                            "cancel": "Cancel took too long; tell the caregiver.",
                            "place": "Placing took too long and was stopped. It is not clear whether it went through: do not order again; tell the caregiver to check the app.",
                        }.get(task.phase, "The service took too long and the task was stopped; nothing was placed.")
                        note(task, "run timed out")
                own = message
                if own and was_browse and not expired and compare_group(task) and task.status in ("awaiting_confirm", "needs_input", "failed"):
                    # A store of a comparison: its news waits for the others, then all options go out in one update.
                    message, summary = await _compare_update(session, task, own)
                if own:
                    await store.record_event(
                        session, family_id=task.family_id, subject_id=task.subject_id, kind=f"task_{task.status}",
                        summary=f"{SKILLS[task.service]['label']}: {own}", payload={"task_id": str(task.id), "result": task.result},
                    )
                if task.status == "done" and ((task.result or {}).get("placed") or (task.result or {}).get("booked")):
                    await delivery_followup(session, task)
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
            task, message, summary = r
            await notify(task.family_id, task.requested_by, f"[Task update] {message}" if summary else f"[Task update] {describe(task)}. {message}")

    # Look-ups (several stores of one comparison) run at the same time: each is a network wait of seconds.
    async with sessions() as session:
        rows = list((await session.execute(select(Task).where(Task.id.in_(ids), Task.status == "queued", Task.phase == "prepare"))).scalars()) if ids else []
        together = {t.id for t in rows if browse_first(t)}
    gate = asyncio.Semaphore(max(1, int(os.getenv("TASK_PARALLEL", "6"))))  # tests share one connection: 1

    async def guarded(task_id):
        async with gate:
            return await process(task_id)

    for r in await asyncio.gather(*(guarded(i) for i in ids if i in together)):
        await notify_one(r)
    for task_id in ids:
        if task_id not in together:
            await notify_one(await process(task_id))
    try:
        for t, text in await _announce_overdue(sessions):
            await notify(t.family_id, t.requested_by, f"[Task update] {text}")
    except Exception:  # noqa: BLE001
        logger.exception("compare announce failed")
    try:
        stats["approvals_asked"] = await ask_approvers(sessions, notify)
    except Exception:  # noqa: BLE001
        logger.exception("approval requests failed")
    return stats
