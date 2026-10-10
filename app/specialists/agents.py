"""The specialist agents: shopping, pharmacy, rides.

Each has its own rules, playbooks (per service), model, step budget and report schema. None of them
talks to people; their reports go through the guard and then to the brain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
- Cancelling: if any fee is shown, do not confirm the cancel; report the fee."""


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
        return s

    def goal(self, task: Task) -> str:
        d = task.details or {}
        label = SKILLS[task.service]["label"]
        if task.kind == "ride":
            what = f"from {d.get('pickup')} to {d.get('drop')}" + (f", vehicle {d['vehicle']}" if d.get("vehicle") else "")
            if task.phase == "browse":
                return (f"On {label}, without logging in and without booking: enter the pickup and drop {what} and report the ride options "
                        f"with fares and pickup times. If the site needs a login before it shows fares, stop and report login_required=true.")
            if task.phase == "prepare":
                return f"On {label}, find the ride options {what} with cash payment. Do NOT book. Report the options with fares, surge, and whether you are logged in."
            if task.phase == "place":
                choice = d.get("choice") or "the option the family confirmed"
                return f"On {label}, book the ride {what}: {choice}, payment Cash. Report booked, ride_id, driver and status."
            if task.phase == "cancel":
                return f"On {label}, cancel the current ride {(task.result or {}).get('ride_id', '')}. If a cancellation fee is shown, do NOT confirm: report cancel_fee. Otherwise cancel and report cancelled."
            return task.goal
        items = "; ".join(f"{i.get('qty', 1)} x {i.get('name')}" for i in d.get("items") or []) or task.goal
        place = (d.get("limits") or {}).get("place") or {}
        where = (f" Deliver only to the saved address '{place.get('nickname')}' ({place.get('pincode')}); if it is not in the account, stop and report it."
                 if place.get("pincode") else (f" Deliver to the saved address near {d['area']}." if d.get("area") else " Deliver to the account's saved home address."))
        if task.phase == "browse":
            spot = (f"{place.get('full')} (pincode {place.get('pincode')})" if place.get("full") else
                    f"pincode {place.get('pincode')}" if place.get("pincode") else d.get("area") or "the family's area")
            return (f"On {label}, without logging in: set the delivery location to {spot} with the site's location picker, then "
                    f"search for: {items}. For each item report up to 3 matching products, best match first (for example a single "
                    f"can and a pack of 6): the exact name, the pack size exactly as the product card shows it (e.g. '6 x 300 ml'), "
                    f"the price of that pack, whether it is in stock, for_item (which item asked for it matches) and exact_match (false for a "
                    f"different brand or flavour, e.g. Coke Zero or Pepsi when Diet Coke was asked; list those only if the exact product "
                    f"is not there). Report "
                    f"deliverable (does {label} deliver to this location) and the eta if shown. If an item is missing, list up to 3 "
                    f"alternatives with prices. Do not add to the cart, do not open checkout, do not log in.")
        if task.phase == "prepare":
            if place.get("full"):
                # First order on a freshly logged-in account: the family's own address from Kavach may be added once.
                where = (f" Deliver to the saved address '{place.get('nickname')}' ({place.get('pincode')}). If the account has no address "
                         f"with that pincode, add this one and use it: {place.get('full')} (use the account holder's name and number if asked).")
            return (f"On {label}, put exactly these in the cart: {items}.{where} Go to checkout up to the payment step and check that "
                    f"Cash/Pay on Delivery is offered. Do NOT place the order. Report items with prices, total with fees, cod_available, address_used, eta.")
        if task.phase == "place":
            total = d.get("confirmed_total") or (task.result or {}).get("total")
            return (f"On {label}, the cart for: {items} is ready and the family confirmed a total of {total}. Open the cart (if it is empty, "
                    f"add exactly these items again first). Place the order now with Cash on "
                    f"Delivery only: choose Cash / Pay on Delivery as the payment method, never a saved card, UPI or wallet; if cash on "
                    f"delivery is not offered, do NOT place and report problem='cod_unavailable'.{where} If the cart items changed or the total is now more than {total} by over ₹10, do NOT place: report "
                    f"price_changed=true and new_total. Otherwise report placed, order_id, payment_method, total and eta.")
        if task.phase == "cancel":
            return (f"On {label}, open the order {(task.result or {}).get('order_id', 'just placed')} and cancel it. If a cancellation "
                    f"fee is shown, do NOT confirm: report cancel_fee. Otherwise cancel and report cancelled.")
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
