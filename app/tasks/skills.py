"""Skill packs: what a browser agent should know about each service.

These are hints, not scripts. The agent reaches the goal however the page looks today; after each
finished run, what it learned is appended (task_skill_notes) and fed to the next run.
"""

from __future__ import annotations

COMMON = """You are acting for an Indian family on their own account, to help an elderly parent. Rules that never bend:
- Payment is Cash on Delivery (COD) / Pay on Delivery only. Never enter card, UPI or wallet details. If COD is not offered, stop and report cod_available=false.
- Never place, confirm or pay for an order, and never book a ride, unless the task explicitly says this step is the placing step.
- Log in only with the mobile number given in the task. Once the site says it sent a code (it now asks for the OTP), stop and report needs_otp=true with where it was sent (masked). Report needs_otp=true only when a code was really sent; a login screen with an empty number field is not that. Do not guess codes.
- If you see a CAPTCHA you cannot pass, a "suspicious activity" wall, or the site will not load, stop and report blocked=true with what you saw.
- Use the account's saved delivery address that matches the expected pincode or area when given; never add a new address unless told.
- Prefer the exact item, brand and pack size asked for. If it is unavailable, pick nothing and report the closest alternatives with prices.
- Read prices, fees and totals from the page; never estimate them.
- If something is unclear, stop and report it rather than guessing."""

SKILLS: dict[str, dict] = {
    "swiggy": {
        "label": "Swiggy",
        "kind": "food",
        "start_url": "https://www.swiggy.com/",
        "hints": "Food delivery. Search the restaurant first, then the dish inside its menu. Customisations (size, add-ons) open a sheet: choose the default unless told. The cart is at the top right. Checkout shows bill details: item total, delivery fee, platform fee, GST, To Pay. 'Pay on Delivery' may sit under 'More payment options'. Orders are under Account > Orders; a placed order can be cancelled from the order page only in the first minutes, sometimes with a fee shown before you confirm.",
    },
    "instamart": {
        "label": "Swiggy Instamart",
        "kind": "grocery",
        "start_url": "https://www.swiggy.com/instamart",
        "hints": "Quick-commerce groceries on swiggy.com/instamart. Set or confirm the delivery location first; availability depends on the store for that pincode. Search each item, check brand and pack size, use '+' to raise quantity. The cart shows item total, handling fee, delivery fee and To Pay. Pay on Delivery is offered for many orders below a value limit.",
    },
    "zepto": {
        "label": "Zepto",
        "kind": "grocery",
        "start_url": "https://www.zepto.com/",
        "hints": "Quick-commerce groceries. Location must be set first (top left). The website blocks some datacenter traffic; if pages stay blank, report blocked. Items show MRP and selling price; use the selling price. COD may be called 'Cash on Delivery' at payment.",
    },
    "blinkit": {
        "label": "Blinkit",
        "kind": "grocery",
        "start_url": "https://blinkit.com/",
        "hints": "Quick-commerce groceries. Detect or set location first. Search, check pack size, 'ADD' then '+'. The cart opens as a side panel; 'Proceed' leads to address and payment. Cash on delivery appears as 'Cash' or 'Pay on delivery' when allowed. Login is phone + OTP.",
    },
    "zomato": {
        "label": "Zomato",
        "kind": "food",
        "start_url": "https://www.zomato.com/",
        "hints": "Food delivery. Choose the city/area, search the restaurant, open 'Order Online'. Customisations open a sheet. Checkout shows the bill and payment options; pick cash/pay on delivery only. Past orders are under the profile menu.",
    },
    "apollo": {
        "label": "Apollo 24|7 Pharmacy",
        "kind": "pharmacy",
        "start_url": "https://www.apollopharmacy.in/",
        "hints": "Pharmacy. Search the medicine by brand and strength; match the strength exactly (e.g. 500 mg, not 1000 mg) and the pack (strip of 10/15). Prescription medicines ask for a prescription upload at checkout: stop and report needs_prescription=true. Delivery address and slot come before payment; choose COD.",
    },
    "1mg": {
        "label": "Tata 1mg",
        "kind": "pharmacy",
        "start_url": "https://www.1mg.com/",
        "hints": "Pharmacy. Search by brand + strength; check the 'strip of N' pack. Rx items require a prescription: stop and report needs_prescription=true. COD is shown as 'Cash on Delivery' when available for the pincode.",
    },
    "pharmeasy": {
        "label": "PharmEasy",
        "kind": "pharmacy",
        "start_url": "https://pharmeasy.in/",
        "hints": "Pharmacy. Search medicine, open the product, verify strength and pack, 'Add to cart'. Rx items need a prescription: stop and report needs_prescription=true. Choose COD at payment.",
    },
    "uber": {
        "label": "Uber",
        "kind": "ride",
        "start_url": "https://m.uber.com/",
        "hints": "Ride booking on the mobile web (m.uber.com). Enter pickup and drop, wait for the vehicle list with fares (UberGo, Auto, Moto, Premier). Pick the type asked for, or the cheapest car if not said. Payment must be Cash. Surge shows as a higher fare or a note: report it. Booking is 'Confirm/Request'. A requested ride can be cancelled from the trip screen; a fee may apply after the driver is assigned: report the fee before confirming a cancel.",
    },
    "ola": {
        "label": "Ola",
        "kind": "ride",
        "start_url": "https://book.olacabs.com/",
        "hints": "Ride booking on book.olacabs.com. Set pickup and drop, choose category (Auto, Mini, Prime), payment Cash. 'Book' requests the ride. Cancel from the ride screen; report any cancellation fee before confirming.",
    },
    "rapido": {
        "label": "Rapido",
        "kind": "ride",
        "start_url": "https://www.rapido.bike/",
        "hints": "Bike taxi, auto and cab. The website may only offer app links; if booking is not possible on the web, stop and report web_booking_unavailable=true.",
    },
}

SERVICES = sorted(SKILLS)


def schema_for(kind: str, phase: str) -> dict:
    """What the agent must report back, per phase."""
    base = {
        "logged_in": {"type": "boolean"},
        "needs_otp": {"type": "boolean"},
        "otp_sent_to": {"type": "string"},
        "blocked": {"type": "boolean"},
        "problem": {"type": "string", "description": "What stopped you, in one sentence; empty if nothing"},
    }
    if kind == "ride":
        extra = {
            "options": {"type": "array", "items": {"type": "object", "properties": {"type": {"type": "string"}, "fare": {"type": "string"}, "eta": {"type": "string"}}}},
            "surge": {"type": "boolean"},
            "booked": {"type": "boolean"},
            "ride_id": {"type": "string"},
            "driver": {"type": "string"},
            "cancelled": {"type": "boolean"},
            "cancel_fee": {"type": "string"},
            "status": {"type": "string"},
        }
    else:
        extra = {
            "items": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "qty": {"type": "integer"}, "price": {"type": "string"}, "available": {"type": "boolean"}}}},
            "alternatives": {"type": "array", "items": {"type": "string"}},
            "total": {"type": "string"},
            "fees": {"type": "string"},
            "cod_available": {"type": "boolean"},
            "needs_prescription": {"type": "boolean"},
            "address_used": {"type": "string"},
            "eta": {"type": "string"},
            "placed": {"type": "boolean"},
            "order_id": {"type": "string"},
            "payment_method": {"type": "string"},
            "cancelled": {"type": "boolean"},
            "cancel_fee": {"type": "string"},
            "status": {"type": "string"},
        }
    return {"type": "object", "properties": {**base, **extra}, "required": ["logged_in", "needs_otp", "blocked", "problem"]}
