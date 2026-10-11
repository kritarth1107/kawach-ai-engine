"""The specialist agents: shopping, pharmacy, rides.

Each has its own rules, playbooks (per service), model, step budget and report schema. None of them
talks to people; their reports go through the guard and then to the brain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.tasks import practice
from app.tasks.models import Task
from app.tasks.skills import COMMON, SKILLS, schema_for

SHOPPING_RULES = """SHOPPING AGENT RULES
- Build exactly the cart asked for: item, brand, pack size, quantity. Never add extras, memberships, tips or donations.
- If an item is unavailable, add nothing for it and report the closest alternatives with prices; never substitute on your own.
- Report every line with its price, the fees, the total to pay and whether Cash on Delivery is offered."""

PHARMACY_RULES = """PHARMACY AGENT RULES
- Match the medicine exactly: brand, strength (mg/mcg/ml) and form. A different strength or brand is a different medicine: never substitute; report alternatives instead.
- If the site asks for a prescription, stop and report needs_prescription=true. Never upload anything yourself.
- Prefer the pack size asked for (strip of 10/15); never more strips than asked.
- Report every line with its price, the fees, the total, Cash on Delivery availability and the delivery estimate."""

RIDES_AGENT_RULES = """RIDES AGENT RULES
- Pickup and drop exactly as given. Payment must be Cash. Never book without being told this is the booking step.
- Report each vehicle option with fare and pickup time, and say plainly if surge pricing applies.
- Cancelling: confirm a cancellation that costs nothing (₹0 / free). If it shows a real charge, do not confirm unless the goal
  says the family agreed to it: report cancel_fee with the amount."""


def ride_end(d: dict, end: str) -> str:
    """A ride's pickup or drop as the app must be given it: the saved place's full address when the person meant one (an AI
    decision at the start), else their words (a landmark the app finds)."""
    place = d.get(f"{end}_place") or {}
    return str(place.get("full") or d.get(end) or "")


def cancel_terms(d: dict) -> str:
    """How a cancel run treats a cancellation fee (founder 2026-10-12: a free cancel is never asked about again)."""
    if d.get("fee_ok"):
        return (f"The family agreed to the cancellation fee of {d.get('cancel_fee') or 'the amount shown'}: confirm the cancellation "
                "and report cancelled=true and cancel_fee. If the fee is now higher, do NOT confirm: report cancel_fee.")
    return ("If the store asks to confirm a cancellation that costs nothing (no fee, ₹0, free), confirm it and report "
            "cancelled=true. If it shows a charge for cancelling, do NOT confirm: report cancel_fee with the amount in rupees. "
            "If it no longer lets you cancel, report cancelled=false and problem with the store's words.")


@dataclass(frozen=True)
class Specialist:
    name: str
    services: tuple[str, ...]
    rules: str
    model_env: str
    steps: dict

    @property
    def model(self) -> str | None:
        return os.getenv(self.model_env) or os.getenv("BROWSER_AGENT_LLM") or None

    def hints(self, service: str, learned: list[str]) -> str:
        text = f"{COMMON}\n\n{self.rules}\n\nABOUT {SKILLS[service]['label'].upper()}:\n{SKILLS[service]['hints']}"
        if learned:
            text += "\n\nLEARNED FROM EARLIER RUNS:\n" + "\n".join(f"- {n}" for n in learned)
        return text

    def schema(self, task: Task) -> dict:
        s = schema_for("ride" if task.kind == "ride" else "order", task.phase)
        if task.phase == "place":
            s["properties"]["price_changed"] = {"type": "boolean"}
            s["properties"]["new_total"] = {"type": "string"}
            if practice.on(task):
                s["properties"].update(practice.SCHEMA)
        return s

    def goal(self, task: Task) -> str:
        d = task.details or {}
        label = SKILLS[task.service]["label"]
        if task.kind == "ride":
            pick, drop = ride_end(d, "pickup"), ride_end(d, "drop")
            what = f"from '{pick}' to '{drop}'" + (f", vehicle {d['vehicle']}" if d.get("vehicle") else "")
            # Live 2026-10-12: a saved home went to the app as words, the app set the city and the ride was booked there.
            ends = (f" Set the pickup to exactly '{pick}' and the drop to exactly '{drop}': type the full address and choose the "
                    "suggestion that matches it (never a city or area alone). Report pickup_used and drop_used exactly as the app "
                    "shows them.")
            if d.get("ends_fix"):
                ends += f" The last run set {d['ends_fix']}, which is wrong: set it again carefully."
            if task.phase == "browse":
                return (f"On {label}, without logging in and without booking: enter the pickup and drop {what} and report the ride options "
                        f"with fares and pickup times.{ends} If the site needs a login before it shows fares, stop and report login_required=true.")
            if task.phase == "prepare":
                return (f"On {label}, find the ride options {what} with cash payment. Do NOT book.{ends} Report the options with fares, "
                        "surge, and whether you are logged in.")
            if task.phase == "place":
                choice = d.get("choice") or "the option the family confirmed"
                return (f"On {label}, book the ride {what}: {choice}, payment Cash.{ends} Before you book, look at the pickup the app "
                        f"shows: if it is only a city or an area, or another place than '{pick}', do NOT book: report "
                        "pickup_mismatch=true and pickup_used. After booking, wait for the driver to be assigned and report booked, "
                        "ride_id, driver (name), vehicle (model and colour), plate, pickup_eta, ride_pin (the PIN / OTP shown for "
                        "starting the trip, if any), status, pickup_used and drop_used, all exactly as the app shows them.")
            if task.phase == "cancel":
                return f"On {label}, cancel the current ride {(task.result or {}).get('ride_id', '')}. " + cancel_terms(d)
            return task.goal
        items = "; ".join(f"{i.get('qty', 1)} x {i.get('name')}" for i in d.get("items") or []) or task.goal
        place = (d.get("limits") or {}).get("place") or {}
        # By the address line, not the label: the account may have several "Home"s (live run 2026-10-11: the agent took the
        # account's old "Home" in another area instead of the saved place).
        where = ((f" Deliver only to the account's saved address that reads like: '{place.get('full')}' (pincode {place.get('pincode')}). "
                  "Several saved addresses can have similar labels such as 'Home': choose by this address line, never by the label; "
                  "if none matches, stop and report it.") if place.get("full") else
                 f" Deliver only to the saved address '{place.get('nickname')}' ({place.get('pincode')}); if it is not in the account, stop and report it."
                 if place.get("pincode") else (f" Deliver to the saved address near {d['area']}." if d.get("area") else " Deliver to the account's saved home address."))
        if task.phase == "browse":
            spot = (f"{place.get('full')} (pincode {place.get('pincode')})" if place.get("full") else
                    f"pincode {place.get('pincode')}" if place.get("pincode") else d.get("area") or "the family's area")
            # numbered, so for_item can be the number (live 2026-10-11: the agent wrote "4 x <name>" and the picker saw nothing)
            numbered = "; ".join(f"item {n}: {i.get('name')} ({i.get('qty', 1)} wanted)" for n, i in enumerate(d.get("items") or [], 1)) or task.goal
            return (f"On {label}, without logging in: set the delivery location to {spot} with the site's location picker, then "
                    f"search for each of these: {numbered}. For each item report up to 3 matching products, best match first (for example a single "
                    f"can and a pack of 6): the exact name, the pack size exactly as the product card shows it (e.g. '6 x 300 ml'), "
                    f"the price of that pack, whether it is in stock, for_item (the item's number, e.g. '1') and exact_match (false for a "
                    f"different brand or flavour, e.g. Coke Zero or Pepsi when Diet Coke was asked; list those only if the exact product "
                    f"is not there). Report "
                    f"deliverable (does {label} deliver to this location) and the eta if shown. If an item is missing, list up to 3 "
                    f"alternatives with prices. Do not add to the cart, do not open checkout, do not log in.")
        if task.phase == "prepare":
            if place.get("full"):
                # First order on a freshly logged-in account: the family's own address from Kavach may be added once.
                where = (f" Deliver to the account's saved address that reads like: '{place.get('full')}' (pincode {place.get('pincode')}); "
                         "choose it by this address line, never by a label such as 'Home' (the account may have several). If the account has "
                         "no such address, add this one and use it (use the account holder's name and number if asked).")
            return (f"On {label}, first open the cart and remove anything already in it (items left from earlier). Then put exactly "
                    f"these in the cart: {items}.{where} Go to checkout up to the payment step and check that "
                    f"Cash/Pay on Delivery is offered. Do NOT place the order. Report items with prices, total with fees, cod_available, address_used, eta.")
        if task.phase == "place" and practice.on(task):
            return practice.browser_goal(label, items, d.get("confirmed_total") or (task.result or {}).get("total"), where)
        if task.phase == "place":
            total = d.get("confirmed_total") or (task.result or {}).get("total")
            return (f"On {label}, the cart for: {items} is ready and the family confirmed a total of {total}. Open the cart (if it is empty, "
                    f"add exactly these items again first). Place the order now with Cash on "
                    f"Delivery only: choose Cash / Pay on Delivery as the payment method, never a saved card, UPI or wallet; if cash on "
                    f"delivery is not offered, do NOT place and report problem='cod_unavailable'.{where} If the cart items changed or the total is now more than {total} by over ₹10, do NOT place: report "
                    f"price_changed=true and new_total. Otherwise report placed, order_id, payment_method, total and eta, and pressed_place=true "
                    "once you pressed the final place-order / pay button (also when no confirmation page came; then wait for it and "
                    "report what the page shows).")
        if task.phase == "cancel":
            return f"On {label}, open the order {(task.result or {}).get('order_id', 'just placed')} and cancel it. " + cancel_terms(d)
        return task.goal


SHOPPING = Specialist("shopping", ("swiggy", "instamart", "zepto", "blinkit", "zomato"), SHOPPING_RULES, "SHOPPING_AGENT_LLM",
                      {"prepare": 60, "otp": 40, "place": 30, "cancel": 30})
PHARMACY = Specialist("pharmacy", ("apollo", "1mg", "pharmeasy"), PHARMACY_RULES, "PHARMACY_AGENT_LLM",
                      {"prepare": 60, "otp": 40, "place": 30, "cancel": 30})
RIDES = Specialist("rides", ("uber", "ola", "rapido"), RIDES_AGENT_RULES, "RIDES_AGENT_LLM",
                   {"prepare": 45, "otp": 40, "place": 30, "cancel": 30})

SPECIALISTS = {s.name: s for s in (SHOPPING, PHARMACY, RIDES)}


def specialist_for(service: str) -> Specialist:
    for s in SPECIALISTS.values():
        if service in s.services:
            return s
    raise ValueError(f"no specialist handles {service}")
