"""Whole order conversations before a deploy: the real brain, product matcher and delivery-status reader, with fake stores
(catalogue, cart, place, order status) and the real task runtime. Plays the night of 2026-10-10's conversations: several
items with flavours, a flavour the store does not have, a change after the confirm, a tap on an old confirm, cancel,
the confirm buttons and the updates after placing. Paid: about 25 fast-model calls (~₹10).

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_ordersim \\   (its own database: emptied each run)
    GCP_PROJECT_ID=sunny-ship-508214-q1 GOOGLE_CLOUD_PROJECT=sunny-ship-508214-q1 MEMORY_EMBEDDINGS=off \\
    MODEL_ROUTES='{"brain": ["gemini:gemini-3.5-flash@global"], "classify": ["gemini:gemini-3.5-flash@global"], ...}' \\
    PYTHONPATH=. python eval/order_conversation_sim.py

The checks are this script's own (the fake store's catalogue names); production code never matches words.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.brain.loop import TurnRequest, run_turn
from app.brain.wake import system_turn
from app.care import store
from app.core import clock
from app.db.session import Base, SessionLocal, engine
from app.sim.world import SimHost
from app.tasks import fastpath, runtime
from app.tasks.models import Task

ELDER = {"id": "elder-sim", "name": "Vasundara Devi", "role": "elder"}
SON = {"id": "son-sim", "name": "Kritarth", "role": "primary caregiver"}
P = lambda pid, name, price, pack: {"name": name, "price": f"₹{price}", "pack": pack, "available": True, "store_id": pid,  # noqa: E731
                                     "cart_ref": {"product_id": pid}}
CATALOGUE = {
    "blinkit": [P("b1", "Amul Taaza Toned Milk", 28, "500 ml"), P("b2", "Amul Gold Full Cream Milk", 35, "500 ml"),
                P("b3", "Charliee Methi Khakhra", 65, "150 g"), P("b4", "Charliee Masala Khakhra", 65, "150 g"),
                P("b5", "Modern Kitchens Butter Muruku", 35, "150 g"), P("b6", "Amul Butter", 60, "100 g")],
    "instamart": [P("i1", "Modern Kitchens Butter Muruku", 35, "150 g"), P("i2", "Haldiram's Peri Peri Murukku", 40, "150 g"),
                  P("i3", "Peri Peri Banana Chips", 50, "150 g"), P("i4", "Amul Taaza Toned Milk", 28, "500 ml")],
}
SHOP_WORDS = {"doodh": "milk", "दूध": "milk"}  # the fake store's own search knows a few Hindi words, as real stores do


class Store:
    """The fake stores: search, cart, place and order status, recorded for the checks."""

    def __init__(self):
        self.carts, self.placed, self.status = [], [], {}

    async def search(self, service, cdp, q, *, lat=None, lon=None, pincode=None, limit=8):
        words = [SHOP_WORDS.get(w, w) for w in re.findall(r"\w+", q.lower()) if len(w) > 2]
        hits = [i for i in CATALOGUE[service] if any(w[:5] in i["name"].lower() for w in words)]
        return {"items": hits[:limit], "deliverable": True, "eta": "10 min", "logged_in": True}

    async def cart(self, service, cdp, products, *, place=None, lat=None, lon=None):
        lines = []
        for p in products:
            item = next(i for i in CATALOGUE[service] if i["store_id"] == (p.get("store_id") or (p.get("cart_ref") or {}).get("product_id")))
            qty = int(p.get("qty") or 1)
            lines.append({"name": item["name"], "qty": qty, "price": f"₹{int(item['price'][1:]) * qty}", "available": True})
        total = sum(int(l["price"][1:]) for l in lines) + 15
        self.carts.append(lines)
        return {"items": lines, "total": f"₹{total}", "fees": "₹15", "cod_available": True, "logged_in": True, "cart_id": "c1",
                "address_used": "Ghar: 12 Civil Lines, Raipur 492001", "eta": "10 min",
                "place_check": {"count": sum(l["qty"] for l in lines), "prices": sorted(int(l["price"][1:]) for l in lines), "total": total,
                                "address": "12 Civil Lines"}}

    async def place(self, service, cdp, check):
        oid = f"SIM{len(self.placed) + 1}"
        self.placed.append({"service": service, "order_id": oid, "check": check})
        self.status[oid] = {"status": "ORDER_PLACED", "text": "Order is being packed"}
        return {"placed": True, "order_id": oid, "total": f"₹{check['total']}", "payment_method": "Cash on Delivery", "eta": "10 min"}

    async def order_status(self, cdp, oid):
        return self.status.get(oid)


class Browser:
    def __init__(self):
        self.agent_runs = []

    async def open_session(self, profile):
        return f"s-{uuid.uuid4().hex[:6]}"

    async def cdp_url(self, sid):
        return f"ws://{sid}"

    async def stop_session(self, sid):
        return None

    async def run(self, **kw):  # the browser agent: never needed in these conversations
        self.agent_runs.append(kw.get("goal", "")[:120])
        raise RuntimeError("the slow browser agent was called")

    async def stop(self, *a, **kw):
        return None

    async def live_url(self, sid):
        return None

    async def new_profile(self, name):
        return "prof"


class StoreHost(SimHost):
    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        if tool == "delivery_place":
            if args.get("words") and "ghar" not in args["words"].lower() and "home" not in args["words"].lower():
                return {"matched": False, "saved": ["Ghar"]}
            return {"addressId": "addr-ghar", "nickname": "Ghar", "full": "12 Civil Lines, Raipur 492001", "lat": 21.24, "lng": 81.69,
                    "pincode": "492001", "matched": True}
        if tool == "list_places":
            return {"places": [{"name": "Ghar", "address": "12 Civil Lines, Raipur 492001", "default": True}]}
        if tool == "browser_profile":
            return {"profileId": f"prof-{family_id}-{args.get('partner')}", "loginPhone": "9000011111"}
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


class Conversation:
    def __init__(self, name: str):
        self.name, self.fam = name, f"ordersim-{uuid.uuid4().hex[:6]}"
        self.host, self.shop, self.browser = StoreHost(), Store(), Browser()
        self.log: list[str] = []
        self.problems: list[str] = []
        self.notified = 0

    def maa_messages(self) -> list[dict]:
        return [m for m in self.host.world.sent if m.get("to") == ELDER["id"]]

    async def say(self, text: str) -> str:
        clock.set_now(clock.now() + timedelta(seconds=30))  # people take a moment to answer
        async with SessionLocal() as s:
            res = await run_turn(s, self.host, TurnRequest(family_id=self.fam, elder=ELDER, speaker=ELDER, members=[ELDER, SON], text=text,
                                                           message_ref=f"m-{uuid.uuid4().hex[:8]}", channel="whatsapp"))
        self.log.append(f"MAA: {text}")
        tools = ", ".join(f"{a['tool']}{'' if a.get('ok', True) else '(x)'}" for a in res.actions)
        if res.reply and res.reply.strip().lower() != "none":
            self.log.append(f"SAHELI (reply): {res.reply}   [{tools}]")
        elif tools:
            self.log.append(f"  [{tools}]")
        return res.reply or ""

    async def notify(self, family_id, requested_by, prompt):
        self.notified += 1
        before = len(self.maa_messages())
        self.log.append(f"  <update> {prompt[:150]}")
        await system_turn(SessionLocal, self.host, family_id,
                          f"{prompt} (Requested by {requested_by}: tell them with send_message to {requested_by}, in their language, then reply none.)",
                          f"task:{uuid.uuid4().hex[:12]}", deliver_to=requested_by)
        new = self.maa_messages()[before:]
        for m in new:
            ids = [b.get("id") for b in m.get("buttons") or []]
            self.log.append(f"SAHELI (sent): {m.get('text')}" + (f"   buttons={ids}" if ids else ""))
        if not new:
            self.problems.append(f"silent update: {prompt[:100]}")
        if len(new) > 1:
            self.problems.append(f"{len(new)} messages for one update: {prompt[:80]}")

    async def ticks(self, until, most: int = 8) -> Task | None:
        async def profile_for(task):
            return {"profileId": f"prof-{task.family_id}-{task.service}", "loginPhone": "9000011111"}

        for _ in range(most):
            clock.set_now(clock.now() + timedelta(seconds=10))
            await runtime.tick(SessionLocal, self.browser, profile_for=profile_for, notify=self.notify, host_for=lambda f: self.host)
            t = await self.task()
            if t and until(t):
                return t
        return await self.task()

    async def task(self) -> Task | None:
        async with SessionLocal() as s:
            return (await s.execute(select(Task).where(Task.family_id == self.fam).order_by(Task.created_at.desc(), Task.updated_at.desc()))).scalars().first()

    def check(self, ok: bool, what: str) -> None:
        if not ok:
            self.problems.append(what)

    async def setup(self):
        async with SessionLocal() as s:
            await store.save_roster(s, self.fam, ELDER, [ELDER, SON])
            await s.commit()


def confirm_buttons(c: Conversation) -> list[str]:
    for m in reversed(c.maa_messages()):
        ids = [b.get("id") for b in m.get("buttons") or [] if str(b.get("id", "")).startswith("v2:od:")]
        if ids:
            return ids
    return []


async def several_items_tap_yes_tracked(c: Conversation):
    await c.setup()
    await c.say("Blinkit se 2 Amul doodh aur ek Charliee methi khakhra bhej do")
    await c.say("haan ghar pe hi")
    t = await c.ticks(lambda t: t.status == "awaiting_confirm")
    c.check(t is not None and t.status == "awaiting_confirm", f"no confirm (task {t.status if t else None})")
    items = [i.get("name") for i in (t.details or {}).get("items") or []] if t else []
    c.check(len(items) == 2, f"items not kept apart: {items}")
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check(sorted((l["name"], l["qty"]) for l in cart) == [("Amul Taaza Toned Milk", 2), ("Charliee Methi Khakhra", 1)]
            or sorted((l["name"], l["qty"]) for l in cart) == [("Amul Gold Full Cream Milk", 2), ("Charliee Methi Khakhra", 1)],
            f"wrong cart: {cart}")
    ids = confirm_buttons(c)
    c.check(len(ids) == 3, f"confirm buttons missing: {ids}")
    if not ids:
        return
    await c.say(ids[0])
    t = await c.ticks(lambda t: t.status == "done")
    c.check(t.status == "done" and (t.result or {}).get("placed"), f"not placed after the Yes tap ({t.status})")
    c.check(len(c.shop.placed) == 1, f"placed {len(c.shop.placed)} times")
    # the store's rider picks it up, then it is delivered
    oid = (t.result or {}).get("order_id")
    c.shop.status[oid] = {"status": "OUT_FOR_DELIVERY", "text": "Rider is on the way, arriving in 6 mins"}
    clock.set_now(clock.now() + timedelta(minutes=8))
    await c.ticks(lambda t: (t.details or {}).get("tracking", {}).get("state") == "on_the_way", most=1)
    c.shop.status[oid] = {"status": "DELIVERED", "text": "Arrived at 11:21 am"}
    clock.set_now(clock.now() + timedelta(minutes=9))
    t = await c.ticks(lambda t: (t.result or {}).get("delivered"), most=1)
    c.check(bool((t.result or {}).get("delivered")), f"delivery not seen: {(t.details or {}).get('tracking')}")


async def flavour_then_cancel(c: Conversation):
    await c.setup()
    await c.say("instamart se peri peri muruku mangwa do")
    await c.say("haan")
    t = await c.ticks(lambda t: t.status == "awaiting_confirm")
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check([l["name"] for l in cart] == ["Haldiram's Peri Peri Murukku"], f"wrong flavour: {cart}")
    await c.say("nahi rehne do, abhi mat mangao")
    t = await c.task()
    c.check(t.status == "cancelled" or t.cancel_requested, f"not cancelled ({t.status})")
    c.check(not c.shop.placed, "placed after a cancel")


async def missing_flavour_change_and_old_tap(c: Conversation):
    await c.setup()
    await c.say("blinkit se do doodh aur ek peri peri muruku")
    await c.say("haan ghar")
    t = await c.ticks(lambda t: t.status == "awaiting_confirm")
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check(not any("Butter Muruku" in l["name"] for l in cart), f"another flavour went in silently: {cart}")
    old = confirm_buttons(c)
    await c.say("doodh ek hi kar do")
    t = await c.ticks(lambda t: t.status == "awaiting_confirm" and len(c.shop.carts) >= 2)
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check(any(l["qty"] == 1 and "Milk" in l["name"] for l in cart) and len(c.shop.carts) >= 2, f"change not made: {cart}")
    if old:
        await c.say(old[0])  # Yes on the first confirm, after the cart changed
        c.check(not c.shop.placed, "an old confirm's Yes placed the changed cart")
    new = confirm_buttons(c)
    if new and new != old:
        await c.say(new[0])
        t = await c.ticks(lambda t: t.status == "done")
        c.check(len(c.shop.placed) == 1, f"not placed once after the new Yes ({len(c.shop.placed)})")


SCENARIOS = [("several items, Yes tap, tracked to the door", several_items_tap_yes_tracked),
             ("flavour kept, then cancel", flavour_then_cancel),
             ("missing flavour, change, old Yes tap", missing_flavour_change_and_old_tap)]


async def main() -> int:
    async with engine.begin() as conn:
        from app.care import baselines, memory_index, models  # noqa: F401
        from app.learn import models as learn_models  # noqa: F401
        from app.llm import spend  # noqa: F401
        from app.models import entities  # noqa: F401
        from app.specialists import channels  # noqa: F401
        from app.tasks import models as task_models, sandbox  # noqa: F401

        await conn.run_sync(Base.metadata.drop_all)  # its own database: every run starts empty (no order of an earlier run is tracked)
        await conn.run_sync(Base.metadata.create_all)
        from app.db.migrate import run_v2_migrations

        await run_v2_migrations(conn)
    import logging

    from app.api import brain as brain_api

    brain_api.kick_tasks = lambda: None  # the live task runner must not pick these orders up; this script ticks them
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    failed = 0
    for name, play in SCENARIOS:
        c = Conversation(name)
        fastpath_patch = {"search": c.shop.search, "cart": c.shop.cart, "place": c.shop.place, "blinkit_order_status": c.shop.order_status}
        saved = {k: getattr(fastpath, k) for k in fastpath_patch}
        for k, v in fastpath_patch.items():
            setattr(fastpath, k, v)
        clock.set_now(datetime(2026, 10, 11, 5, 0, tzinfo=timezone.utc))  # 10:30 IST
        try:
            await play(c)
        except Exception as exc:  # noqa: BLE001
            c.problems.append(f"crashed: {type(exc).__name__}: {exc}")
        finally:
            for k, v in saved.items():
                setattr(fastpath, k, v)
            async with SessionLocal() as s:  # nothing of this conversation carries into the next one
                for t in (await s.execute(select(Task).where(Task.family_id == c.fam, Task.status.in_(runtime.LIVE)))).scalars():
                    t.status = "cancelled"
                await s.commit()
            clock.set_now(None)
        if c.browser.agent_runs:
            c.problems.append(f"slow browser agent used: {c.browser.agent_runs}")
        print(f"\n=== {name}: {'OK' if not c.problems else 'PROBLEMS'}")
        print("\n".join(c.log))
        for p in c.problems:
            print("  !!", p)
        failed += bool(c.problems)
    print(f"\n{len(SCENARIOS) - failed}/{len(SCENARIOS)} conversations clean")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
