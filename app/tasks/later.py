"""Book at a time (founder 2026-10-12): "book a cab at 6 am Friday from home to the airport", "order milk at 7 tomorrow".

At the request Saheli looks now and the person picks from the numbered options, as for any order or ride. Then ONE
question confirms everything: what (the ride type, or the items), the route as the app set it (or where it goes), the
time, today's fare or prices, cash, and the most they will pay: a number suggested from today's amount that they accept
or replace, or "whatever it costs" (no max). The family's approval limits are checked against that max (plus the leeway)
right then, so an approver's yes comes up front, never at the booking time. Then the task waits (status "scheduled"):
- the login: needed only if the account's login will not last until the booking. Asked at once when a login made now
  lasts until then; else 2-4 hours before it would run out (founder: "if it keeps 10 hours, ask 6-8 hours earlier"),
  never at the last minute and not at night when another time works. How long each app keeps a login is measured
  (sandbox.StoreLogin); a safe default until then;
- CHECK_BEFORE the booking it looks at the fare (or builds the cart): above the max plus the leeway it asks ONE question
  with the new amount at once (founder, 2026-10-12: "ask earlier for a number till how max it can go ... 900 would not be
  strict, it can book for 902 or 920"); else it is ready;
- at the booking time it books (places) without asking, and every detail goes to the person.
A login that ran out anyway is asked for at once, saying why; the order ladder applies from then.

The apps' own advance booking (Uber Reserve, Ola Ride Later, Rapido's schedule) is not used: Uber Reserve may cost more
than an on-demand ride and does not take every type, Ola's is app-only, none promises a driver, and the stores have no
such thing for most items; booking on demand at the time works the same on every app and store."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core import clock
from app.specialists import guard

logger = logging.getLogger(__name__)

MIN_AHEAD = timedelta(minutes=20)  # a time sooner than this is booked now
CHECK_BEFORE = timedelta(minutes=25)  # the fare is looked at (the cart built) this long before booking
ARRIVE_LEAD = timedelta(minutes=10)  # "a cab at 6": booked this long before, unless the app's pickup time says otherwise
LEAD_RANGE = (timedelta(minutes=5), timedelta(minutes=20))
GIVE_UP_AFTER = timedelta(minutes=20)  # not booked this long after its time: stopped and said plainly
DEFAULT_LOGIN_LIFE = timedelta(hours=12)  # until measured: no app publishes how long a web login lasts
LOGIN_LIFE_RANGE = (timedelta(hours=2), timedelta(days=7))
ASK_BEFORE_END = timedelta(hours=3)  # the code is asked 2-4 h before the login would run out
MARGIN = timedelta(minutes=30)
LOGIN_WAIT = timedelta(minutes=60)  # a planned code ask nobody answered: asked again at the fare check
HEADROOM = 1.15  # the suggested max: today's amount + 15 %, rounded up to ₹10
QUIET = (21, 8)  # IST hours with no planned asks when another time works (founder: nothing at night)


def of(task) -> dict:
    return dict((task.details or {}).get("later") or {})


def parse_at(text: str | None) -> datetime | None:
    """The time the brain passed (ISO 8601; IST when it has no offset)."""
    if not text:
        return None
    try:
        at = datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=clock.IST)


def book_time(at: datetime, means: str, lead: timedelta = ARRIVE_LEAD) -> datetime:
    """When to book (place): the time itself, or a little before it when it is the time the cab should come or the order
    should arrive."""
    return at if means == "book" else at - min(max(lead, LEAD_RANGE[0]), LEAD_RANGE[1])


def suggest_max(amount: float | None) -> int | None:
    return int(math.ceil(amount * HEADROOM / 10.0) * 10) if amount else None


def cap(max_amount: float | None) -> float | None:
    """The most it may cost and still go ahead without a question: their max plus the leeway."""
    return max_amount + guard.leeway(max_amount) if max_amount else None


def hm(at: datetime) -> str:
    return clock.ist(at).strftime("%a %d %b, %H:%M")


def _quiet(at: datetime) -> bool:
    h = clock.ist(at).hour
    return h >= QUIET[0] or h < QUIET[1]


async def login_life(session: AsyncSession, service: str) -> timedelta:
    """How long a login on this app is known to last (measured across families): the shortest time after which a login
    was found gone, a little less; or, never seen gone, the longest it was seen alive, at least the default."""
    from app.tasks.sandbox import StoreLogin

    since = clock.now() - timedelta(days=90)
    rows = list((await session.execute(select(StoreLogin).where(StoreLogin.service == service, StoreLogin.login_at >= since))).scalars())
    died = [r.died_at - r.login_at for r in rows if r.died_at]
    alive = [r.last_ok_at - r.login_at for r in rows if r.last_ok_at]
    life = min(died) * 0.9 if died else max([DEFAULT_LOGIN_LIFE, *alive])
    return min(max(life, LOGIN_LIFE_RANGE[0]), LOGIN_LIFE_RANGE[1])


async def logged_in_since(session: AsyncSession, family_id: str, service: str) -> datetime | None:
    """When the account's current login began, if it is logged in now (as far as the last run saw)."""
    from app.tasks.sandbox import ServiceLogin, StoreLogin

    fam = family_id.removeprefix("shadow:")
    row = await session.get(ServiceLogin, (fam, service))
    if row is None or row.state != "ok":
        return None
    cur = (await session.execute(select(StoreLogin).where(StoreLogin.family_id == fam, StoreLogin.service == service,
                                                          StoreLogin.died_at.is_(None)).order_by(StoreLogin.login_at.desc()).limit(1))).scalar_one_or_none()
    return cur.login_at if cur else row.last_login_ok_at


def plan_login(now: datetime, book_at: datetime, life: timedelta, since: datetime | None) -> datetime | None:
    """When to log in (ask the code) for a booking at book_at: None when the current login lasts until then; now when a
    login made now does; else ASK_BEFORE_END before a login made then would run out, moved off the night when it can be."""
    if since and since + life >= book_at + MARGIN:
        return None
    if book_at + MARGIN - now <= life:
        return now
    t = book_at - (life - ASK_BEFORE_END)
    earliest, latest = book_at + MARGIN - life, book_at - CHECK_BEFORE - MARGIN
    if _quiet(t):
        local = clock.ist(t)
        evening = (local if local.hour >= QUIET[0] else local - timedelta(days=1)).replace(hour=QUIET[0] - 1, minute=30, second=0, microsecond=0)
        morning = (local if local.hour < QUIET[1] else local + timedelta(days=1)).replace(hour=QUIET[1], minute=0, second=0, microsecond=0)
        for alt in (evening, morning):
            if earliest <= alt <= latest:
                return max(alt, now)
    return max(min(max(t, earliest), latest), now)


def ask_text(task, *, label: str, what: str, amount: float | None, where: str) -> str:
    """The ONE question after the pick: everything, and the most they will pay."""
    lt = of(task)
    at, book_at = parse_at(lt.get("at")), parse_at(lt.get("book_at"))
    ride = task.kind == "ride"
    sug = suggest_max(amount)
    timing = (f"for {hm(at)} ({'booked' if ride else 'ordered'} at {clock.ist(book_at).strftime('%H:%M')} so it "
              f"{'reaches the pickup' if ride else 'arrives'} around then)" if lt.get("means") != "book" else f"{'booked' if ride else 'ordered'} at {hm(at)}")
    today = (f"Today's {'fare' if ride else 'price'} is about ₹{amount:.0f}" if amount else "Today's price is not known yet") + " and may differ then."
    ask = (f"Ask ONE short question with all of this, and the most they will pay: suggest ₹{sug} (say 'if it is above about "
           f"₹{cap(sug):.0f} I will ask you first'); they say yes, give their own number, or say any amount is fine. " if sug else
           "Ask ONE short question with all of this, and the most they will pay (their number, or any amount). ")
    return (f"For later, nothing is {'booked' if ride else 'ordered'} now: {what} on {label}, {where}, {timing}, cash. {today} {ask}"
            f"Attach buttons {{kind: later, key: {task.id}}} to that question (yes up to the suggested limit / any amount / no). "
            f"If they answer in words: task_input later on task {task.id}: value yes with max_amount = their number"
            + (f" (or {sug} when they just say yes)" if sug else "") + "; no_max true ONLY when they say any amount is fine; value no "
            "if they do not want it.")


def scheduled_text(task, *, label: str, login_at: datetime | None, logged: bool) -> str:
    lt = of(task)
    book_at = parse_at(lt.get("book_at"))
    ride = task.kind == "ride"
    most = f"up to ₹{lt['max']:.0f} without asking (above about ₹{cap(lt['max']):.0f} they are asked first)" if lt.get("max") else "at whatever it costs then"
    code = ("no login code is needed (the app is logged in until then)" if logged and login_at is None else
            "a login code is needed now: it comes in a task update in a minute" if login_at and login_at <= clock.now() + timedelta(minutes=1) else
            f"a login code will be asked on {hm(login_at)}" if login_at else "")
    return (f"Set for later (NOT {'booked' if ride else 'ordered'} yet): {label} will be {'booked' if ride else 'ordered'} on {hm(book_at)}, "
            f"{most}; then every detail is sent. {code[:1].upper() + code[1:]}. Tell them this in one or two short lines.")


async def due(sessions: async_sessionmaker, notify=None) -> int:
    """Scheduled tasks whose next step is due: the login, the fare check, the booking (or giving up, said plainly). Returns
    how many moved."""
    from app.tasks import runtime
    from app.tasks.models import Task
    from app.tasks.skills import SKILLS

    now = clock.now()
    moved = 0
    missed: list = []
    async with sessions() as session:
        rows = list((await session.execute(select(Task).where(Task.status == "scheduled").with_for_update(skip_locked=True))).scalars())
        for t in rows:
            lt = of(t)
            book_at = parse_at(lt.get("book_at"))
            login_at = parse_at(lt.get("login_at"))
            if not book_at:
                continue
            d = t.details or {}
            if now >= book_at + GIVE_UP_AFTER:  # a cab long after its time is no use (the engine was down): said plainly
                t.status = "failed"
                t.details = {**d, "later": {**lt, "state": "missed"}}
                runtime.note(t, "the booking time passed before it could be booked")
                missed.append(t)
                moved += 1
                continue
            if lt.get("state") == "ready" and now >= book_at:
                t.details = {**d, "later": {**lt, "state": "booking"}, **runtime.LADDER_RESET, "ladder_from": now.isoformat()}
                by = lt.get("by") or t.requested_by
                if t.kind == "ride":
                    runtime._start_booking(t, by, d.get("choice") or "", d.get("confirmed_total") or (d.get("ride_pick") or {}).get("fare"))
                else:
                    runtime._start_placing(t, by)
                t.deadline_at = now + runtime.TASK_LIFETIME
                runtime.note(t, "booking time: booking it on the yes given ahead")
                moved += 1
            elif now >= book_at - CHECK_BEFORE and lt.get("state") in ("waiting", "logged_in"):
                t.details = {**d, "later": {**lt, "state": "check"}, **runtime.LADDER_RESET, "ladder_from": now.isoformat(),
                             "active_from": now.isoformat(), "told_working": None, "cart_fp": None, "confirm_token": None}
                t.phase, t.status, t.input_needed = "prepare", "queued", None
                t.deadline_at = book_at + GIVE_UP_AFTER
                runtime.note(t, "looking at the fare (the cart) before the booking time")
                moved += 1
            elif lt.get("state") == "waiting" and login_at and now >= login_at:
                t.details = {**d, "later": {**lt, "state": "login", "login_started": now.isoformat()}, "login_only": True,
                             **runtime.LADDER_RESET, "ladder_from": now.isoformat(), "active_from": now.isoformat()}
                t.phase, t.status, t.input_needed = "prepare", "queued", None
                t.deadline_at = min(book_at - CHECK_BEFORE, now + LOGIN_WAIT)
                runtime.note(t, "logging in ahead of the booking")
                moved += 1
        await session.commit()
    for t in missed if notify else []:
        lt = of(t)
        await runtime._notify_about(notify, [t], t.family_id, t.requested_by, (
            f"[Task update] {runtime.describe(t)}. NOT {'booked' if t.kind == 'ride' else 'ordered'}: the {SKILLS[t.service]['label']} "
            f"{'ride' if t.kind == 'ride' else 'order'} set for {hm(parse_at(lt.get('at')))} could not be done by its time. Tell them "
            "plainly in one short line; they can ask again."))
    return moved
