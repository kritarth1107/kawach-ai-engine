"""Practice orders (founder 2026-10-11): for testing and training, an order goes through every step a real one does, up to
the store's final place-order button, and stops there: nothing is placed, the cart is emptied, and the person hears it was
a practice order.

Which families: the ops setting PRACTICE_ORDER_FAMILIES (comma-separated family ids). A task created while its family is
listed keeps the mark (details.practice) even if the setting changes before it is placed, and a listed family's task is a
practice even without the mark: either one is enough, so a test order is never placed by a change of setting midway.

What stops where:
- connector: the exact cart is built and checked again on the store (address, cash on delivery, total), then emptied;
  the place call is never made.
- Blinkit saved steps: checkout, cash on delivery chosen in the payment frame, then stop before Pay Now.
- other saved place steps: not run.
- browser agent: checkout, cash on delivery chosen, stop at the final button without pressing it, then empty the cart.
A practice run that still reports a placed order is treated as an accident: the caregiver is told at once to check the app.
"""

from __future__ import annotations

import os


def family_listed(family_id: str) -> bool:
    ids = {x.strip() for x in os.getenv("PRACTICE_ORDER_FAMILIES", "").split(",") if x.strip()}
    return bool(ids) and family_id.removeprefix("shadow:") in ids


def on(task) -> bool:
    return bool((task.details or {}).get("practice")) or family_listed(task.family_id)


def browser_goal(label: str, items: str, total, where: str) -> str:
    """The place step of a practice order for the browser agent."""
    return (f"On {label}, the cart for: {items} is ready and the family confirmed a total of {total}. THIS IS A PRACTICE RUN: do "
            "every step of placing the order EXCEPT the very last click. Open the cart (if it is empty, add exactly these items "
            f"again first) and go to checkout.{where} Open the payment step and choose Cash / Pay on Delivery as the payment "
            "method (never a card, UPI or wallet). When the page shows the final button that would place the order (for example "
            "'Place Order', 'Pay ₹…', 'Confirm Order', 'Pay on Delivery ₹…'), STOP: do NOT click it, do not press Enter on it. "
            "Report reached_final_step=true, final_button (its exact text), total, payment_method, eta and address_used. Then go "
            "back to the cart and remove every item so the cart is empty, and report cart_emptied. If cash on delivery is not "
            f"offered, report problem='cod_unavailable'. If the cart items changed or the total is now more than {total} by over "
            "₹10, report price_changed=true and new_total. Never report placed=true: nothing must be placed.")


SCHEMA = {
    "reached_final_step": {"type": "boolean", "description": "The final place-order button was on screen and NOT pressed"},
    "final_button": {"type": "string", "description": "That button's exact text"},
    "cart_emptied": {"type": "boolean"},
}
