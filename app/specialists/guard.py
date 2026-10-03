"""Money and safety rules, enforced in code on every agent report.

Agents are told these rules too, but nothing here trusts them: a cart or a placement that breaks a rule
is stopped or flagged before the family can be harmed or charged.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass, field

from app.brain import policy
from app.specialists.contract import Limits

PRICE_TOLERANCE = 10.0  # ₹ the placed total may differ from the confirmed total (rounding, tiny fees)


def qty(value) -> int:
    """A quantity from an agent report: 2, "2", "2 x 1kg" -> 2. Anything unreadable counts as 1."""
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return int(value)
    m = re.search(r"\d+", str(value or ""))
    return int(m.group(0)) if m else 1


def alternatives(raw) -> list[dict]:
    """Alternatives as [{"name", "price"}], whether the agent sent strings or objects."""
    out = []
    for a in raw or []:
        if isinstance(a, str) and a.strip():
            out.append({"name": a.strip()})
        elif isinstance(a, dict) and str(a.get("name") or "").strip():
            out.append({"name": str(a["name"]).strip(), **({"price": a["price"]} if a.get("price") else {})})
    return out


def rupees(text) -> float | None:
    if isinstance(text, (int, float)):
        return float(text)
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)", str(text or "").replace("₹", ""))
    return float(m.group(1).replace(",", "")) if m else None


@dataclass
class Verdict:
    """block: stop, nothing may be placed. approval: a caregiver must OK it. warn: tell the family."""

    block: list[str] = field(default_factory=list)
    approval: list[str] = field(default_factory=list)
    warn: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.block


# ── before the agent starts ────────────────────────────────────────────────────


def check_request(kind: str, agent: str, items: list[dict], limits: Limits, *, pickup: str | None = None, drop: str | None = None) -> Verdict:
    v = Verdict()
    if kind == "order":
        if not items:
            v.block.append("no items: list each item with brand/size and quantity first")
        text = " ".join(str(i.get("name", "")) for i in items)
        for c in policy.order_conflicts(text, limits.allergies, limits.never_order):
            v.block.append(f"blocked by the care record: {c}")
        for i in items:
            q = qty(i.get("qty"))
            if q < 1 or q > 50:
                v.block.append(f"quantity {q} for {i.get('name')} is not sensible")
    elif kind == "ride":
        if not (pickup and drop):
            v.block.append("a ride needs pickup and drop")
    return v


# ── after the agent built a cart or found fares ────────────────────────────────


def cart_fingerprint(report: dict) -> str:
    items = sorted((str(i.get("name", "")).strip().lower(), qty(i.get("qty"))) for i in report.get("items") or [])
    raw = json.dumps({"items": items, "total": rupees(report.get("total"))}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


UNIT_WORDS = {"pack", "packet", "pouch", "bottle", "strip", "tablet", "tablets", "box", "fresh", "with", "and", "the"}


def _keywords(name: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", (name or "").lower()) if len(w) >= 4 and w not in UNIT_WORDS]


def missing_items(requested: list[dict], items: list[dict]) -> list[str]:
    """Requested items with no line in the cart (by their main words)."""
    names = " ".join(str(i.get("name", "")).lower() for i in items if i.get("available", True) is not False)
    return [str(r.get("name")) for r in requested if _keywords(str(r.get("name"))) and not any(k in names for k in _keywords(str(r.get("name"))))]


def rx_covers(requested: list[dict], rx_on_file: list[str], cart: list[dict] | None = None) -> bool:
    """Every requested medicine has a prescription on file for the same medicine (first word) and strength.
    The strength is taken from the request, else from the matching cart line ('Telma' asked, 'Telma 80mg' in cart)."""
    if not requested:
        return False
    for q in requested:
        name = str(q.get("name", ""))
        first = (re.findall(r"[a-z]+", name.lower()) or [""])[0]
        want = _strengths(name) or {x for i in cart or [] if first and first in str(i.get("name", "")).lower() for x in _strengths(str(i.get("name", "")))}
        ok = False
        for rx in rx_on_file:
            if (re.findall(r"[a-z]+", rx.lower()) or [""])[0] != first:
                continue
            have = _strengths(rx)
            if not want or not have or want & have:  # a prescription saved without a strength covers the medicine
                ok = True
                break
        if not ok:
            return False
    return True


def _strengths(text: str) -> set[str]:
    """Strengths as numbers ('40mg', '40 mg', 'Telma 40' -> {'40'}). Bare numbers count when no unit is written."""
    t = text or ""
    with_unit = re.findall(r"(\d+(?:\.\d+)?)\s*(?:mg|mcg|ml|g|iu)\b", t, re.I)
    if with_unit:
        return {n.rstrip("0").rstrip(".") if "." in n else n for n in with_unit}
    bare = re.findall(r"(?<![\d.])(\d{1,4}(?:\.\d+)?)(?![\d.]|\s*(?:'s|s\b|x\b|tab|strip|pack|kg))", t, re.I)
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in bare}


def check_cart(kind: str, agent: str, requested: list[dict], report: dict, limits: Limits) -> Verdict:
    v = Verdict()
    if kind == "ride":
        if report.get("surge"):
            v.warn.append("surge pricing: say the fare plainly before anyone chooses")
        fares = [rupees(o.get("fare")) for o in report.get("options") or [] if rupees(o.get("fare"))]
        if fares and min(fares) > limits.budget and limits.requester_is_elder:
            v.approval.append(f"cheapest fare ₹{min(fares):.0f} is above the ₹{limits.budget} limit: a caregiver must OK it")
        return v

    items = report.get("items") or []
    if limits.cod_only and report.get("cod_available") is False:
        v.block.append("Cash on delivery is not offered for this cart")
    text = " ".join(str(i.get("name", "")) for i in items)
    for c in policy.order_conflicts(text, limits.allergies, limits.never_order):
        v.block.append(f"the cart has something the care record forbids: {c}")
    asked = {str(r.get("name", "")).lower(): qty(r.get("qty")) for r in requested}
    for i in items:
        q = qty(i.get("qty"))
        want = max(asked.values()) if asked else 1
        if q > max(limits.max_qty, want):
            v.block.append(f"{q} × {i.get('name')} is more than was asked or allowed")
    if agent == "pharmacy":
        for r in requested:
            want = _strengths(str(r.get("name", "")))
            name = str(r.get("name", "")).split()[0].lower() if r.get("name") else ""
            match = [i for i in items if name and name in str(i.get("name", "")).lower()]
            if not match:
                v.block.append(f"{r.get('name')} is not in the cart: medicines are never substituted")
            elif want and not any(want & _strengths(str(i.get("name", ""))) for i in match):
                v.block.append(f"{r.get('name')}: the cart has a different strength; medicines are never substituted")
        if report.get("needs_prescription") and not rx_covers(requested, limits.rx_on_file):
            v.block.append("this medicine needs a prescription upload first (none on file for it)")
    else:
        missing = missing_items(requested, items)
        if missing:
            v.warn.append(f"not in the cart (unavailable): {', '.join(missing)}; say so before they confirm")
    total = rupees(report.get("total"))
    if total and total > limits.budget and limits.requester_is_elder:
        v.approval.append(f"total ₹{total:.0f} is above the ₹{limits.budget} limit: a caregiver must OK it")
    used = str(report.get("address_used") or "")
    pin = str((limits.place or {}).get("pincode") or "").strip()
    if pin:
        if not used:
            v.warn.append(f"the store did not show the delivery address; check it is {(limits.place or {}).get('nickname')} ({pin}) before confirming")
        else:
            pins = re.findall(r"(?<!\d)\d{3}\s?\d{3}(?!\d)", used)
            if pins and pin not in {p.replace(" ", "") for p in pins}:
                v.block.append(f"the delivery address '{used}' is not the saved place ({(limits.place or {}).get('nickname')}, {pin})")
            elif not pins:
                v.warn.append(f"the store shows the address as '{used}' with no pincode; read it out and make sure it is {(limits.place or {}).get('nickname')} ({pin})")
    return v


# ── confirm token: the only key that lets an agent place ───────────────────────


def _secret() -> bytes:
    return (os.getenv("TASK_CONFIRM_SECRET") or os.getenv("KAVACH_API_SECRET") or "kavach-dev-confirm").encode()


def confirm_token(task_id: str, fingerprint: str, by: str) -> str:
    return hmac.new(_secret(), f"{task_id}|{fingerprint}|{by}".encode(), hashlib.sha256).hexdigest()[:32]


def token_valid(task_id: str, fingerprint: str, by: str, token: str | None) -> bool:
    return bool(token) and hmac.compare_digest(confirm_token(task_id, fingerprint, by), token or "")


# ── after placing ──────────────────────────────────────────────────────────────


def check_placed(kind: str, report: dict, confirmed_total: float | None) -> Verdict:
    v = Verdict()
    method = str(report.get("payment_method") or "")
    if kind == "order" and method and not re.search(r"cash|cod|pay on delivery|deliver", method, re.I):
        v.warn.append(f"payment shows '{method}', not cash on delivery: tell the caregiver now")
    total = rupees(report.get("total"))
    if kind == "order" and confirmed_total and total and abs(total - confirmed_total) > PRICE_TOLERANCE:
        v.warn.append(f"placed total ₹{total:.0f} differs from the confirmed ₹{confirmed_total:.0f}: tell the caregiver now")
    return v
