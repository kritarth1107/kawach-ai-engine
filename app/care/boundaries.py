"""The family's boundaries: who may order, how much, and when someone must approve (the 4th ring of the delegate model).

Set by the family (dashboard or Saheli on WhatsApp, caregivers only), stored as one versioned fact (subject "family",
key boundary:policy) so every change has who/when/undo like the rest of the record. Enforced in code where an order or
a ride is confirmed: the brain can be asked to place something, but nothing outside these limits is placed until an
approver says yes. Memory cannot grant permission; only this policy and an approver's answer can.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.care import store
from app.core import clock

KEY = "boundary:policy"
SUBJECT = "family"
CATEGORIES = ("grocery", "food", "pharmacy", "ride")
CATEGORY_OF = {"blinkit": "grocery", "instamart": "grocery", "zepto": "grocery", "swiggy": "food", "zomato": "food",
               "apollo": "pharmacy", "1mg": "pharmacy", "pharmeasy": "pharmacy", "uber": "ride", "ola": "ride", "rapido": "ride"}
DEFAULTS: dict = {
    "elder_order_limit": 1500,   # ₹: an order the care recipient asks for above this needs approval
    "elder_ride_limit": 800,     # ₹: same for a ride
    "anyone_over": None,         # ₹: any order or ride above this needs approval, whoever asks (None: off)
    "monthly_cap": None,         # ₹: orders + rides placed this month; going past it needs approval (None: off)
    "approval_categories": [],   # categories that always need approval (grocery, food, pharmacy, ride)
    "approvers": [],             # member ids; empty: the primary caregivers
    "members": {},               # {member id: {"can_order": bool, "can_ride": bool}}; missing: allowed
    # Who gives the store login codes for a person's orders (founder 2026-10-10: some elders cannot): {person id: "self"
    # or a member id}. The store logs in with that person's number, so the code reaches them. Missing: whoever asked.
    "login_codes": {},
}


def _clean(policy: dict) -> dict:
    out = {**DEFAULTS, **{k: v for k, v in (policy or {}).items() if k in DEFAULTS}}
    for k in ("elder_order_limit", "elder_ride_limit", "anyone_over", "monthly_cap"):
        v = out[k]
        out[k] = None if v in (None, "", 0) and k in ("anyone_over", "monthly_cap") else (max(0, int(float(v))) if v not in (None, "") else DEFAULTS[k])
    out["approval_categories"] = sorted({c for c in out["approval_categories"] or [] if c in CATEGORIES})
    out["approvers"] = [str(a) for a in out["approvers"] or [] if a]
    out["members"] = {str(m): {"can_order": bool(v.get("can_order", True)), "can_ride": bool(v.get("can_ride", True))}
                      for m, v in (out["members"] or {}).items() if isinstance(v, dict)}
    out["login_codes"] = {str(k): str(v) for k, v in (out["login_codes"] or {}).items() if k and v} if isinstance(out["login_codes"], dict) else {}
    return out


def code_person(policy: dict, subject_id: str, requested_by: str) -> str:
    """Whose phone a store login code goes to for an order for subject_id (their number is the one the store logs in with)."""
    who = _clean(policy)["login_codes"].get(str(subject_id))
    if who == "self":
        return subject_id
    return who or requested_by


async def get(session: AsyncSession, family_id: str) -> dict:
    row = await store.active_fact(session, family_id, SUBJECT, KEY)
    return _clean(row.value if row else {})


def describe(policy: dict, names: dict[str, str]) -> str:
    """One paragraph for people (and for Saheli's prompt)."""
    p = _clean(policy)
    parts = [f"orders the care recipient asks for over ₹{p['elder_order_limit']} and rides over ₹{p['elder_ride_limit']} need approval"]
    if p["anyone_over"]:
        parts.append(f"anything over ₹{p['anyone_over']} needs approval whoever asks")
    if p["monthly_cap"]:
        parts.append(f"monthly limit ₹{p['monthly_cap']}")
    if p["approval_categories"]:
        parts.append("always ask first for: " + ", ".join(p["approval_categories"]))
    blocked = [f"{names.get(m, m)} cannot {' or '.join(w for w, ok in (('order', v['can_order']), ('book rides', v['can_ride'])) if not ok)}"
               for m, v in p["members"].items() if not (v["can_order"] and v["can_ride"])]
    parts += blocked
    parts += [f"login codes for {names.get(k, k)}'s orders go to {'them' if v in ('self', k) else names.get(v, v)}"
              for k, v in p["login_codes"].items()]
    who = ", ".join(names.get(a, a) for a in p["approvers"]) or "the primary caregiver"
    return "; ".join(parts) + f". Approver: {who}."


async def save(session: AsyncSession, family_id: str, changes: dict, *, by: str, source_kind: str = "caregiver_said") -> dict:
    """Merge changes into the policy and save a new version."""
    current = await get(session, family_id)
    members = {**current["members"], **{str(k): v for k, v in (changes.get("members") or {}).items()}}
    codes = {**current["login_codes"], **{str(k): str(v) for k, v in (changes.get("login_codes") or {}).items()}}
    new = _clean({**current, **{k: v for k, v in changes.items() if k in DEFAULTS and k not in ("members", "login_codes")},
                  "members": members, "login_codes": codes})
    await store.write_fact(session, family_id=family_id, subject_id=SUBJECT, domain="boundary", key=KEY, value=new,
                           text="Family boundaries: " + describe(new, {}), source_kind=source_kind, stated_by=by, replace=True)
    await store.record_event(session, family_id=family_id, subject_id=SUBJECT, kind="boundary_changed",
                             summary="Family boundaries changed: " + describe(new, {}), payload={"policy": new}, actor_id=by)
    return new


ANYONE = "*"  # no household on file: anyone but the care recipient may approve (as before boundaries existed)


def may_approve(allowed: list[str], person: str, elder_id: str) -> bool:
    return bool(person) and (person in allowed or (ANYONE in allowed and person != elder_id))


def approvers(policy: dict, roster) -> list[str]:
    """Who may approve: the policy's list, else the primary caregivers, else every member who is not the care recipient."""
    if roster is None:
        return [ANYONE]
    elder = (roster.elder or {}).get("id") if roster else None
    members = [m for m in (roster.members if roster else []) if m.get("id") and m.get("id") != elder]
    chosen = [a for a in _clean(policy)["approvers"] if any(m["id"] == a for m in members)]
    if chosen:
        return chosen
    primary = [m["id"] for m in members if "primary" in str(m.get("role", "")).lower()]
    return primary or [m["id"] for m in members]


def can_manage(policy: dict, roster, person: str) -> bool:
    """Only an approver changes the boundaries (never the care recipient)."""
    return roster is not None and person in approvers(policy, roster)


async def month_spent(session: AsyncSession, family_id: str, *, now: datetime | None = None) -> float:
    from app.specialists.guard import rupees
    from app.tasks.models import Task

    local = clock.ist(now)
    start = local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    rows = (await session.execute(select(Task).where(Task.family_id == family_id, Task.status == "done", Task.created_at >= start))).scalars()
    total = 0.0
    for t in rows:
        if (t.result or {}).get("placed") or (t.result or {}).get("booked"):
            total += rupees((t.details or {}).get("confirmed_total") or (t.result or {}).get("total")) or 0.0
    return total


async def reasons(session: AsyncSession, *, family_id: str, service: str, kind: str, amount: float | None,
                  requested_by: str, elder_id: str, policy: dict | None = None) -> list[str]:
    """Why this order or ride needs an approver's yes (empty: it may go ahead)."""
    p = _clean(policy if policy is not None else await get(session, family_id))
    out = []
    member = p["members"].get(requested_by) or {}
    if kind == "ride" and member.get("can_ride") is False:
        out.append("this person is not allowed to book rides on their own")
    if kind != "ride" and member.get("can_order") is False:
        out.append("this person is not allowed to order on their own")
    category = CATEGORY_OF.get(service, "ride" if kind == "ride" else "grocery")
    if category in p["approval_categories"]:
        out.append(f"the family asked to approve every {category} {'booking' if kind == 'ride' else 'order'}")
    if amount:
        limit = p["elder_ride_limit"] if kind == "ride" else p["elder_order_limit"]
        if requested_by == elder_id and amount > limit:
            out.append(f"₹{amount:.0f} is above the ₹{limit} limit for the care recipient's own {'rides' if kind == 'ride' else 'orders'}")
        if p["anyone_over"] and amount > p["anyone_over"]:
            out.append(f"₹{amount:.0f} is above the family's ₹{p['anyone_over']} approval limit")
        if p["monthly_cap"]:
            spent = await month_spent(session, family_id)
            if spent + amount > p["monthly_cap"]:
                out.append(f"this month would reach ₹{spent + amount:.0f}, over the ₹{p['monthly_cap']} monthly limit")
    return out
