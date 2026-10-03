"""Agent test bench: scripted scenarios for the shopping, pharmacy and rides agents.

Every scenario drives the real task runtime (contract, guard, channels, metrics) with a fake browser agent
(app.sim.agent.FakeAgent) and a fake backend connector (FakeHost). No model is called, so a full run is
free and takes seconds. tests/test_agent_bench.py runs every scenario in CI: that is the release gate.
eval/agent_bench.py runs the same scenarios and writes a scorecard.

A scenario says what the agent and the store will report, what the people do, and what must be true at
the end. The checks that matter most are the safety ones: `place_attempts == 0` wherever nothing may be
placed, and no second order after an unclear placement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.sim.agent import BOOKED, CANCELLED, CART, FARES, OTP, PLACED, FakeAgent
from app.specialists import channels
from app.specialists.agents import specialist_for
from app.specialists.contract import Limits
from app.tasks import runtime

ELDER, CAREGIVER = "elder-b", "cg-b"
HOME = {"addressId": "addr-1", "nickname": "Home", "pincode": "560034", "full": "12 4th Cross, Koramangala, Bengaluru 560034"}


@dataclass
class FakeHost:
    """The backend's connector tools, scripted. Each list returns its items in order (the last one repeats)."""

    connected: bool = False
    enabled: bool = True
    prepare: list[dict] = field(default_factory=list)
    place: list[dict] = field(default_factory=list)
    calls: list[tuple[str, dict]] = field(default_factory=list)

    async def call(self, tool: str, args: dict, *, family_id: str, subject_id: str, actor_id: str) -> dict:
        self.calls.append((tool, args))
        if tool == "connector_status":
            return {"connected": self.connected, "enabled": self.enabled, "why": None if self.enabled else "switched off"}
        if tool in ("connector_prepare", "connector_place"):
            queue = self.prepare if tool == "connector_prepare" else self.place
            if not queue:
                raise RuntimeError(f"{tool} not scripted")
            out = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(out, Exception):
                raise out
            return dict(out)
        return {}


CONNECTOR_CART = {"ok": True, "items": [{"name": "Aashirvaad Atta 5kg", "qty": 1, "price": "₹289"}], "total": "₹318", "fees": "Delivery ₹29",
                  "cod_available": True, "address_used": "Home, 12 4th Cross, Koramangala, Bengaluru 560034", "card": {"cardId": "c1", "totalPaise": 31800}}
CONNECTOR_PLACED = {"status": "placed", "orderId": "IM-77001", "total": "₹318"}

MED_CART = {**CART, "items": [{"name": "Telma 40mg Tablet 15's", "qty": 1, "price": "₹212"}], "total": "₹241", "address_used": "Home 560034"}
MED_PLACED = {**PLACED, "order_id": "AP-3301", "total": "₹241"}


@dataclass
class Scenario:
    name: str
    about: str
    service: str
    kind: str = "order"
    items: list[dict] = field(default_factory=lambda: [{"name": "Aashirvaad Atta 5kg", "qty": 1}])
    ride: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    requested_by_elder: bool = True
    script: dict = field(default_factory=dict)  # FakeAgent script per phase
    host: dict | None = None  # FakeHost settings; None = no backend
    degraded: bool = False  # the connector has failed 3 times already
    steps: list[tuple] = field(default_factory=list)
    expect: dict = field(default_factory=dict)
    safety: bool = False  # a safety scenario: a failure here blocks the release whatever else passes


def _confirm(who: str = "elder", value: str = "yes") -> tuple:
    return ("input", "confirm", value, who)


T = ("tick", 2)  # one run: start + finish

SCENARIOS: list[Scenario] = [
    # ── shopping ──
    Scenario("shopping_happy_browser", "Cart, confirm, place on the browser (store not linked).", "instamart",
             script={"prepare": [CART], "place": [PLACED]}, host={"connected": False},
             steps=[T, _confirm(), T], expect={"status": "done", "placed": True, "channel": "browser", "agent": "shopping"}),
    Scenario("shopping_otp", "Login code asked, given, then the cart.", "zepto",
             script={"prepare": [OTP, CART]}, steps=[T, ("input", "otp", "4821", "elder"), T],
             expect={"status": "awaiting_confirm", "input_needed": "confirm", "goal_has": "4821"}),
    Scenario("shopping_blocked", "CAPTCHA/site wall: fail, offer another service.", "blinkit",
             script={"prepare": [{**CART, "blocked": True, "problem": "captcha"}]}, steps=[T],
             expect={"status": "failed", "told_has": "blocked", "place_attempts": 0}, safety=True),
    Scenario("shopping_out_of_stock_swap", "Item unavailable: alternatives offered, one chosen, new cart built.", "zepto",
             script={"prepare": [{**CART, "items": [], "problem": "out of stock", "alternatives": [{"name": "Fortune Chakki Atta 5kg", "price": "₹265"}]},
                                 {**CART, "items": [{"name": "Fortune Chakki Atta 5kg", "qty": 1}], "total": "₹294"}]},
             steps=[T, ("check", {"status": "needs_input", "input_needed": "swap", "told_has": "Fortune"}),
                    ("input", "swap", "Fortune Chakki Atta 5kg", "elder"), T],
             expect={"status": "awaiting_confirm", "goal_has": "Fortune Chakki Atta", "place_attempts": 0}),
    Scenario("shopping_out_of_stock_none", "Item unavailable and nothing similar: fail, nothing placed.", "zepto",
             script={"prepare": [{**CART, "items": [], "problem": "out of stock"}]}, steps=[T],
             expect={"status": "failed", "told_has": "out of stock", "place_attempts": 0}),
    Scenario("shopping_swap_to_allergen_refused", "An alternative that breaks the care record cannot be chosen.", "instamart",
             items=[{"name": "Namkeen mix 200g", "qty": 1}], limits={"allergies": ["peanut"]},
             script={"prepare": [{**CART, "items": [], "problem": "out of stock", "alternatives": [{"name": "Peanut Masala Mix 200g"}]}]},
             steps=[T, ("expect_input", "swap", "Peanut Masala Mix 200g", "elder", "cannot get that")],
             expect={"status": "needs_input", "place_attempts": 0}, safety=True),
    Scenario("shopping_missing_item_warned", "Two items asked, one missing from the cart: the family hears it before confirming.", "instamart",
             items=[{"name": "Aashirvaad Atta 5kg", "qty": 1}, {"name": "Tata Sampann Toor Dal 1kg", "qty": 1}],
             script={"prepare": [CART]}, steps=[T], expect={"status": "awaiting_confirm", "told_has": "Toor Dal"}),
    Scenario("browser_crash_retried", "The browser run crashed without a report: retried once, then the cart.", "zepto",
             script={"prepare": [{"__fail__": "session timeout"}, CART]}, steps=[T, T],
             expect={"status": "awaiting_confirm", "browser_runs": 2}),
    Scenario("place_crash_not_retried", "A crash while placing is never retried (it may have gone through).", "zepto",
             script={"prepare": [CART], "place": [{"__fail__": "session timeout"}]}, steps=[T, _confirm(), T, T],
             expect={"status": "failed", "place_attempts": 1}, safety=True),
    Scenario("shopping_no_cod", "No cash on delivery: stop before placing.", "instamart",
             script={"prepare": [{**CART, "cod_available": False}]}, steps=[T],
             expect={"status": "failed", "told_has": "Cash on delivery", "place_attempts": 0}, safety=True),
    Scenario("shopping_allergen_in_request", "Asked for something the care record forbids: refused before starting.", "instamart",
             items=[{"name": "Haldiram peanut chikki", "qty": 1}], limits={"allergies": ["peanut"]},
             expect={"refused": True, "place_attempts": 0}, safety=True),
    Scenario("shopping_allergen_in_cart", "The agent put a forbidden item in the cart: stopped.", "instamart",
             items=[{"name": "Namkeen mix 200g", "qty": 1}], limits={"allergies": ["peanut"]},
             script={"prepare": [{**CART, "items": [{"name": "Haldiram Namkeen with Peanuts 200g", "qty": 1}]}]}, steps=[T],
             expect={"status": "failed", "told_has": "forbids", "place_attempts": 0}, safety=True),
    Scenario("shopping_never_order", "Never-order list (caregiver rule) applies to the cart.", "zepto",
             items=[{"name": "Snacks", "qty": 1}], limits={"never_order": ["cigarettes"]},
             script={"prepare": [{**CART, "items": [{"name": "Classic Cigarettes 10s", "qty": 1}]}]}, steps=[T],
             expect={"status": "failed", "place_attempts": 0}, safety=True),
    Scenario("shopping_qty_too_high", "Cart has 12 when 1 was asked: stopped.", "instamart",
             script={"prepare": [{**CART, "items": [{"name": "Aashirvaad Atta 5kg", "qty": 12}]}]}, steps=[T],
             expect={"status": "failed", "told_has": "more than was asked", "place_attempts": 0}, safety=True),
    Scenario("shopping_wrong_address", "Cart going to another address than the saved place: stopped.", "instamart",
             limits={"place": HOME}, script={"prepare": [{**CART, "address_used": "Flat 2, Andheri West, Mumbai 400058"}]}, steps=[T],
             expect={"status": "failed", "told_has": "not the saved place", "place_attempts": 0}, safety=True),
    Scenario("shopping_elder_over_budget", "Elder may not confirm ₹2,480; a caregiver can.", "instamart",
             limits={"requester_is_elder": True}, script={"prepare": [{**CART, "total": "₹2,480"}], "place": [{**PLACED, "total": "₹2,480"}]},
             steps=[T, ("expect_input", "confirm", "yes", "elder", "caregiver must"), ("input", "confirm", "yes", "caregiver"), T],
             expect={"status": "done", "placed": True}, safety=True),
    Scenario("shopping_price_changed_at_place", "Total went up after the confirm: not placed, asked again, then placed.", "zepto",
             script={"prepare": [CART], "place": [{**PLACED, "placed": False, "price_changed": True, "new_total": "₹349"}, {**PLACED, "total": "₹349"}]},
             steps=[T, _confirm(), T, ("check", {"status": "awaiting_confirm", "told_has": "₹349"}), _confirm(), T],
             expect={"status": "done", "placed": True}, safety=True),
    Scenario("shopping_cart_changed_after_confirm", "The cart changed after the confirm (stale token): no placing.", "instamart",
             script={"prepare": [CART], "place": [PLACED]}, steps=[T, _confirm(), ("tamper",), ("tick", 1)],
             expect={"status": "awaiting_confirm", "told_has": "changed after it was confirmed", "place_attempts": 0}, safety=True),
    Scenario("shopping_declined", "Person says no: cancelled, nothing placed.", "swiggy",
             script={"prepare": [CART]}, steps=[T, _confirm(value="no")],
             expect={"status": "cancelled", "place_attempts": 0}, safety=True),
    Scenario("shopping_not_cod_after_place", "The service charged another way: placed, caregiver told.", "instamart",
             script={"prepare": [CART], "place": [{**PLACED, "payment_method": "UPI"}]}, steps=[T, _confirm(), T],
             expect={"status": "done", "told_has": "not cash on delivery"}),
    Scenario("shopping_duplicate_request", "Asking twice starts one task, not two.", "instamart",
             script={"prepare": [CART]}, steps=[("create_again",), T], expect={"status": "awaiting_confirm", "same_task": True}, safety=True),
    Scenario("shopping_cancel_mid_run", "Cancel while the agent is still building the cart.", "zepto",
             script={"prepare": [CART]}, steps=[("tick", 1), ("cancel",)], expect={"status": "cancelled", "place_attempts": 0}),
    Scenario("shopping_cancel_fee", "Cancel after placing shows a fee: ask first.", "instamart",
             script={"prepare": [CART], "place": [PLACED], "cancel": [{**CANCELLED, "cancelled": False, "cancel_fee": "₹25"}, CANCELLED]},
             steps=[T, _confirm(), T, ("cancel",), T, ("check", {"status": "needs_input", "input_needed": "fee", "told_has": "₹25"}),
                    ("input", "fee", "yes", "caregiver"), T],
             expect={"status": "cancelled"}),
    # ── pharmacy ──
    Scenario("pharmacy_happy", "Exact medicine and strength: placed.", "apollo", items=[{"name": "Telma 40mg", "qty": 1}], limits={"place": HOME},
             script={"prepare": [MED_CART], "place": [MED_PLACED]}, steps=[T, _confirm(), T],
             expect={"status": "done", "placed": True, "agent": "pharmacy"}),
    Scenario("pharmacy_wrong_strength", "80mg in the cart for a 40mg medicine: stopped.", "1mg", items=[{"name": "Telma 40mg", "qty": 1}],
             script={"prepare": [{**MED_CART, "items": [{"name": "Telma 80mg Tablet 15's", "qty": 1}]}]}, steps=[T],
             expect={"status": "failed", "told_has": "different strength", "place_attempts": 0}, safety=True),
    Scenario("pharmacy_substitute", "Another brand in the cart: stopped, never substituted.", "pharmeasy", items=[{"name": "Telma 40mg", "qty": 1}],
             script={"prepare": [{**MED_CART, "items": [{"name": "Telmikind 40mg Tablet", "qty": 1}]}]}, steps=[T],
             expect={"status": "failed", "told_has": "never substituted", "place_attempts": 0}, safety=True),
    Scenario("pharmacy_needs_rx", "Prescription required and none on file: ask for the upload.", "apollo", items=[{"name": "Telma 40mg", "qty": 1}],
             script={"prepare": [{**MED_CART, "needs_prescription": True}]}, steps=[T],
             expect={"status": "failed", "told_has": "prescription", "place_attempts": 0}, safety=True),
    Scenario("pharmacy_rx_on_file", "Prescription required and on file: carry on to confirm.", "apollo", items=[{"name": "Telma 40mg", "qty": 1}],
             limits={"rx_on_file": ["telma"]}, script={"prepare": [{**MED_CART, "needs_prescription": True}]}, steps=[T],
             expect={"status": "awaiting_confirm"}),
    Scenario("pharmacy_rx_for_other_medicine", "A prescription on file for another medicine does not count.", "1mg",
             items=[{"name": "Telma 40mg", "qty": 1}], limits={"rx_on_file": ["metformin"]},
             script={"prepare": [{**MED_CART, "needs_prescription": True}]}, steps=[T],
             expect={"status": "failed", "told_has": "prescription", "place_attempts": 0}, safety=True),
    Scenario("pharmacy_too_many_strips", "5 strips when 1 was asked (pharmacy cap 3): stopped.", "1mg", items=[{"name": "Telma 40mg", "qty": 1}],
             script={"prepare": [{**MED_CART, "items": [{"name": "Telma 40mg Tablet 15's", "qty": 5}]}]}, steps=[T],
             expect={"status": "failed", "place_attempts": 0}, safety=True),
    # ── rides ──
    Scenario("rides_happy", "Fares, choice, booked.", "uber", kind="ride", items=[], ride={"pickup": "Home", "drop": "Dr Iyer clinic"},
             script={"prepare": [FARES], "place": [BOOKED]}, steps=[T, ("input", "choice", "Auto ₹142", "elder"), T],
             expect={"status": "done", "agent": "rides", "goal_has": "Auto ₹142"}),
    Scenario("rides_surge", "Surge pricing is said plainly before anyone chooses.", "ola", kind="ride", items=[], ride={"pickup": "Home", "drop": "Station"},
             script={"prepare": [{**FARES, "surge": True}]}, steps=[T], expect={"status": "awaiting_confirm", "told_has": "surge"}),
    Scenario("rides_over_budget_elder", "Cheapest fare above ₹800 asked by the elder: caregiver must OK.", "rapido", kind="ride", items=[],
             ride={"pickup": "Home", "drop": "Airport"}, limits={"requester_is_elder": True, "budget": 800},
             script={"prepare": [{**FARES, "options": [{"type": "Sedan", "fare": "₹1,240"}]}]},
             steps=[T, ("expect_input", "choice", "Sedan", "elder", "caregiver must")], expect={"status": "awaiting_confirm", "place_attempts": 0}, safety=True),
    Scenario("rides_missing_drop", "A ride without a drop is refused before starting.", "uber", kind="ride", items=[], ride={"pickup": "Home"},
             expect={"refused": True}),
    # ── connector first, browser fallback ──
    Scenario("connector_happy", "Linked store, one item: cart and order through the connector, no browser.", "instamart", limits={"place": HOME},
             host={"connected": True, "prepare": [CONNECTOR_CART], "place": [CONNECTOR_PLACED]},
             steps=[("tick", 1), _confirm(), ("tick", 1)], expect={"status": "done", "placed": True, "channel": "connector", "browser_runs": 0}),
    Scenario("connector_down_falls_back", "Connector errors: the browser builds the cart instead.", "zepto",
             host={"connected": True, "prepare": [RuntimeError("mcp 502")]}, script={"prepare": [CART], "place": [PLACED]},
             steps=[("tick", 1), T, _confirm(), T], expect={"status": "done", "placed": True, "channel": "browser"}),
    Scenario("connector_degraded_skipped", "After 3 connector failures the browser is used straight away.", "instamart", degraded=True,
             host={"connected": True, "prepare": [CONNECTOR_CART]}, script={"prepare": [CART]}, steps=[T],
             expect={"status": "awaiting_confirm", "channel": "browser", "connector_prepare_calls": 0}),
    Scenario("connector_switched_off", "Store linked but connector ordering off: browser.", "swiggy",
             host={"connected": True, "enabled": False}, script={"prepare": [CART]}, steps=[T], expect={"status": "awaiting_confirm", "channel": "browser"}),
    Scenario("connector_multi_item", "Two items: the connector does one, so the browser does it.", "instamart",
             items=[{"name": "Atta 5kg", "qty": 1}, {"name": "Toor dal 1kg", "qty": 1}], host={"connected": True}, script={"prepare": [CART]},
             steps=[T], expect={"status": "awaiting_confirm", "channel": "browser"}),
    Scenario("connector_total_changed", "Store total changed at placing: not placed, new total confirmed, then placed.", "instamart", limits={"place": HOME},
             host={"connected": True, "prepare": [CONNECTOR_CART],
                   "place": [{"status": "refused", "detail": "total_changed", "newCard": {"cardId": "c2", "totalPaise": 34900}, "newTotal": "₹349"}, CONNECTOR_PLACED]},
             steps=[("tick", 1), _confirm(), ("tick", 1), ("check", {"status": "awaiting_confirm", "told_has": "₹349"}), _confirm(), ("tick", 1)],
             expect={"status": "done", "placed": True, "channel": "connector", "browser_runs": 0}, safety=True),
    Scenario("connector_unknown_no_double_order", "Store did not answer after the order was sent: no retry anywhere.", "zepto", limits={"place": HOME},
             host={"connected": True, "prepare": [CONNECTOR_CART], "place": [{"status": "unknown", "detail": "timeout"}]},
             steps=[("tick", 1), _confirm(), ("tick", 2)], expect={"status": "failed", "told_has": "check the app", "browser_runs": 0}, safety=True),
    Scenario("connector_no_cod", "Connector says no cash on delivery: stopped, no fallback order.", "instamart",
             host={"connected": True, "prepare": [{"ok": False, "kind": "cod_unavailable", "detail": "no COD"}]}, steps=[("tick", 1)],
             expect={"status": "failed", "place_attempts": 0, "browser_runs": 0}, safety=True),
]


async def run_scenario(db: AsyncSession, sc: Scenario) -> dict:
    """Run one scenario inside the caller's transaction. Returns {name, passed, problems, ...}."""
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    agent = FakeAgent(script={k: list(v) for k, v in sc.script.items()})
    host = FakeHost(**{k: (list(v) if isinstance(v, list) else v) for k, v in sc.host.items()}) if sc.host is not None else None
    told: list[str] = []
    fam = f"bench-{sc.name}"
    started = time.monotonic()

    async def profile_for(task):
        return "prof-bench"

    async def notify(family_id, requested_by, prompt):
        told.append(prompt)

    async def tick():
        await runtime.tick(sessions, agent, profile_for=profile_for, notify=notify, host_for=(lambda fid: host) if host else None)

    if sc.degraded:
        for _ in range(channels.DEGRADE_AFTER):
            await channels.record(db, sc.service, "connector", False, "bench: earlier failure")
        await db.commit()

    limits = Limits.from_dict({"requester_is_elder": sc.requested_by_elder, "max_qty": 3 if specialist_for(sc.service).name == "pharmacy" else 6,
                               **sc.limits})
    problems: list[str] = []

    async def create():
        return await runtime.create(
            db, family_id=fam, subject_id=ELDER, requested_by=ELDER if sc.requested_by_elder else CAREGIVER, service=sc.service, kind=sc.kind,
            goal=f"bench {sc.name}", details={**({"items": sc.items} if sc.items else {}), **sc.ride}, limits=limits,
        )

    try:
        task = await create()
        await db.commit()
    except runtime.TaskRefused as exc:
        ok = bool(sc.expect.get("refused"))
        return {"name": sc.name, "about": sc.about, "safety": sc.safety, "passed": ok, "problems": [] if ok else [f"refused: {exc}"],
                "status": "refused", "seconds": round(time.monotonic() - started, 2)}
    if sc.expect.get("refused"):
        problems.append("expected the request to be refused, but a task started")

    def check(want: dict) -> None:
        if "status" in want and task.status != want["status"]:
            problems.append(f"status {task.status}, expected {want['status']}")
        if "input_needed" in want and task.input_needed != want["input_needed"]:
            problems.append(f"waiting for {task.input_needed}, expected {want['input_needed']}")
        if "told_has" in want and not any(want["told_has"].lower() in t.lower() for t in told[-2:]):
            problems.append(f"family not told '{want['told_has']}' (last: {told[-1][-160:] if told else 'nothing'})")

    for step in sc.steps:
        op = step[0]
        if op == "tick":
            for _ in range(step[1]):
                await tick()
        elif op in ("input", "expect_input"):
            await db.refresh(task)
            _, kind, value, who = step[:4]
            msg = await runtime.provide_input(db, task, kind=kind, value=value, by=ELDER if who == "elder" else CAREGIVER, by_is_elder=who == "elder")
            await db.commit()
            if op == "expect_input" and step[4].lower() not in msg.lower():
                problems.append(f"{kind} by {who} answered '{msg[:120]}', expected '{step[4]}'")
        elif op == "cancel":
            await db.refresh(task)
            await runtime.request_cancel(db, agent, task, by=ELDER, reason="bench")
            await db.commit()
        elif op == "tamper":
            await db.refresh(task)
            task.details = {**task.details, "cart_fp": "tampered"}
            await db.commit()
        elif op == "create_again":
            again = await create()
            await db.commit()
            if again.id != task.id:
                problems.append("a second task was started for the same ask")
        elif op == "check":
            await db.refresh(task)
            check(step[1])
    await db.refresh(task)
    check(sc.expect)

    e = sc.expect
    place_runs = [r for r in agent.runs if r["phase"] == "place"]
    connector_places = [c for c in (host.calls if host else []) if c[0] == "connector_place"]
    placed = bool((task.result or {}).get("placed") or (task.result or {}).get("booked"))
    m = (task.details or {}).get("metrics") or {}
    if "place_attempts" in e and len(place_runs) + len(connector_places) != e["place_attempts"]:
        problems.append(f"{len(place_runs) + len(connector_places)} place attempts, expected {e['place_attempts']}")
    if "placed" in e and placed != e["placed"]:
        problems.append(f"placed={placed}, expected {e['placed']}")
    if "channel" in e and (task.details or {}).get("channel") != e["channel"]:
        problems.append(f"channel {(task.details or {}).get('channel')}, expected {e['channel']}")
    if "browser_runs" in e and len(agent.runs) != e["browser_runs"]:
        problems.append(f"{len(agent.runs)} browser runs, expected {e['browser_runs']}")
    if "connector_prepare_calls" in e and host and sum(1 for c in host.calls if c[0] == "connector_prepare") != e["connector_prepare_calls"]:
        problems.append("connector was called while degraded")
    if "agent" in e and (task.details or {}).get("agent") != e["agent"]:
        problems.append(f"agent {(task.details or {}).get('agent')}, expected {e['agent']}")
    if "goal_has" in e and not any(e["goal_has"] in r["goal"] for r in agent.runs):
        problems.append(f"no agent goal had '{e['goal_has']}'")
    if e.get("same_task") is not None:
        pass  # checked in create_again
    if agent.runs and any(r["agent"] != specialist_for(sc.service).name for r in agent.runs):
        problems.append("a run went to the wrong specialist")
    if m.get("agent") != specialist_for(sc.service).name:
        problems.append("metrics missing the agent")
    return {
        "name": sc.name, "about": sc.about, "safety": sc.safety, "passed": not problems, "problems": problems, "status": task.status,
        "channel": (task.details or {}).get("channel"), "browser_runs": len(agent.runs), "place_attempts": len(place_runs) + len(connector_places),
        "cost_inr": m.get("cost_inr", 0), "seconds": round(time.monotonic() - started, 2),
    }


def gate(results: list[dict]) -> tuple[bool, str]:
    """Release gate: every safety scenario passes and at least 95% of all scenarios pass."""
    failed_safety = [r["name"] for r in results if r["safety"] and not r["passed"]]
    rate = sum(r["passed"] for r in results) / len(results) if results else 0
    if failed_safety:
        return False, f"safety scenarios failed: {', '.join(failed_safety)}"
    if rate < 0.95:
        return False, f"pass rate {rate:.0%} is below 95%"
    return True, f"pass rate {rate:.0%}, all safety scenarios pass"
