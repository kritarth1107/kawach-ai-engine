"""Whole order conversations before a deploy: the real brain, product matcher and delivery-status reader, with fake stores
(catalogue, cart, place, order status) and the real task runtime. Plays the night of 2026-10-10's conversations: several
items with flavours, a flavour the store does not have, a change after the confirm, a tap on an old confirm, cancel,
the confirm buttons and the updates after placing; and after an order: did it come, how was it (remembered), and a late
order told to the caregiver. Paid: about 40 fast-model calls (~₹10).

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
from app.brain import wake
from app.brain.wake import system_turn
from app.care import store
from app.care.models import OpenLoop
from app.core import clock
from app.db.session import Base, SessionLocal, engine
from app.sim.world import SimHost
from app.tasks import aftercare, fastpath, runtime
from app.tasks.models import Task

ELDER = {"id": "elder-sim", "name": "Vasundara Devi", "role": "elder"}
SON = {"id": "son-sim", "name": "Kritarth", "role": "primary caregiver"}
P = lambda pid, name, price, pack: {"name": name, "price": f"₹{price}", "pack": pack, "available": True, "store_id": pid,  # noqa: E731
                                     "cart_ref": {"product_id": pid}}
CATALOGUE = {
    "blinkit": [P("b1", "Amul Taaza Toned Milk", 28, "500 ml"), P("b2", "Amul Gold Full Cream Milk", 35, "500 ml"),
                P("b3", "Charliee Methi Khakhra", 65, "150 g"), P("b4", "Charliee Masala Khakhra", 65, "150 g"),
                P("b5", "Modern Kitchens Butter Muruku", 35, "150 g"), P("b6", "Amul Butter", 60, "100 g"),
                # what Blinkit listed for "RiteBite Choco Berry" on 2026-10-11 (a Bangalore family's order)
                P("b7", "RiteBite Max Protein Daily Choco Berry 10 g Protein Bar", 80, "50 g"),
                P("b8", "RiteBite Max Protein Choco Berry 30 g Protein Bar", 177, "100 g"),
                P("b9", "Gokul Full Cream Milk", 37, "500 ml"), P("b10", "Britannia Marie Gold Biscuits", 35, "250 g"),
                P("b11", "Mother Dairy Toned Milk", 29, "500 ml")],
    "instamart": [P("i1", "Modern Kitchens Butter Muruku", 35, "150 g"), P("i2", "Haldiram's Peri Peri Murukku", 40, "150 g"),
                  P("i3", "Peri Peri Banana Chips", 50, "150 g"), P("i4", "Amul Taaza Toned Milk", 28, "500 ml"),
                  # a 4-pack next to the single bar: "4 bars" must be 4 x the single, never 4 x the 4-pack
                  P("i5", "RiteBite Max Protein Daily Choco Berry 10g Protein Bar", 316, "50 g x 4"),
                  P("i6", "RiteBite Max Protein Daily Choco Berry 10g Protein Bar", 79, "50 g"),
                  P("i7", "Aavin Toned Milk", 24, "500 ml"), P("i8", "Britannia Marie Gold Biscuits", 34, "250 g"),
                  P("i9", "Mother Dairy Toned Milk", 28, "500 ml")],
    "zepto": [P("z1", "RiteBite Max Protein 30g Protein Bar Choco Berry", 176, "1 pack (100 g)"),
              P("z2", "Amul Taaza Toned Milk", 29, "500 ml"), P("z3", "Aavin Toned Milk", 25, "500 ml"),
              P("z4", "Britannia Marie Gold Biscuits", 33, "250 g")],
}
SHOP_WORDS = {"doodh": "milk", "दूध": "milk"}  # the fake store's own search knows a few Hindi words, as real stores do


class Store:
    """The fake stores: search, cart, place and order status, recorded for the checks."""

    def __init__(self, address: str = "Ghar: 12 Civil Lines, Raipur 492001"):
        self.carts, self.placed, self.status = [], [], {}
        self.address = address
        self.searched: list[tuple] = []  # (service, lat, lon): a store that needs the point must never get an empty one
        self.broken: set[str] = set()  # stores whose saved look-up fails (the browser agent looks instead)

    async def search(self, service, cdp, q, *, lat=None, lon=None, pincode=None, limit=8):
        self.searched.append((service, lat, lon))
        if fastpath.SITES[service].needs_point and not fastpath.has_point(lat, lon):
            raise fastpath.NoPoint(f"{service}: no point")  # as the real step does
        if service in self.broken:
            raise fastpath.FastPathError(f"{service} search answered 500")
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
                "address_used": self.address, "eta": "10 min",
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
    """The cloud browser: sessions for the saved steps, and a browser agent that reads the fake store. The agent is used only
    when a saved step cannot run (no map point, a broken step, a store without a saved cart or place step); slow[service] =
    how many polls (10 s each) its agent takes."""

    def __init__(self, shop: "Store | None" = None):
        self.shop, self.agent_runs, self.slow = shop, [], {}
        self._runs: dict[str, dict] = {}

    async def open_session(self, profile):
        return f"s-{uuid.uuid4().hex[:6]}"

    async def cdp_url(self, sid):
        return f"ws://{sid}"

    async def stop_session(self, sid):
        return None

    async def run(self, *, goal, hints, schema, session_id, profile_id, start_url, max_steps, metadata, llm=None, flash=False):
        from app.tasks.browser_use import AgentRun

        service, phase = metadata.get("service"), metadata.get("phase")
        self.agent_runs.append(f"{service}:{phase}")
        async with SessionLocal() as s:
            task = await s.get(Task, uuid.UUID(str(metadata["task"])))
            out = await self._report(task, service, phase)
        tid = f"bu-{uuid.uuid4().hex[:6]}"
        self._runs[tid] = {"out": out, "polls": 0, "need": self.slow.get(service, 1)}
        return AgentRun(task_id=tid, session_id=session_id or f"s-{tid}", status="created")

    async def _report(self, task: Task, service: str, phase: str) -> dict:
        base = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": ""}
        d = task.details or {}
        if phase == "browse":  # the agent sets the location by the written address, then searches each item
            items = []
            for n, it in enumerate(d.get("items") or [], 1):
                words = [SHOP_WORDS.get(w, w) for w in re.findall(r"\w+", str(it.get("name")).lower()) if len(w) > 2]
                items += [{k: x[k] for k in ("name", "pack", "price", "available")} | {"for_item": str(n)}
                          for x in CATALOGUE[service] if any(w[:5] in x["name"].lower() for w in words)][:3]
            return {**base, "logged_in": False, "deliverable": True, "eta": "12 min", "items": items, "location_set": self.shop.address}
        if phase == "prepare":
            from app.specialists import channels

            # a browser agent's look-up has no store ids: its products are found again by name, as the agent does on the site
            byname = {x["name"]: x for x in CATALOGUE[service]}
            chosen = [{**byname.get(c.get("name"), {}), **c, "qty": channels._qty_for(d.get("items") or [], c)}
                      for c in d.get("chosen") or []]
            chosen = [{**c, "store_id": c.get("store_id") or byname.get(c.get("name"), {}).get("store_id")} for c in chosen]
            cart = await self.shop.cart(service, None, chosen)
            cart.pop("place_check", None)
            return {**base, **cart}
        if phase == "place":
            total = int(re.sub(r"\D", "", str(d.get("confirmed_total") or "0")) or 0)
            return {**base, **await self.shop.place(service, None, {"total": total})}
        return {**base, "problem": f"unexpected phase {phase}"}

    async def poll(self, tid):
        from app.tasks.browser_use import AgentRun

        r = self._runs[tid]
        r["polls"] += 1
        if r["polls"] < r["need"]:
            return AgentRun(task_id=tid, session_id="", status="started")
        return AgentRun(task_id=tid, session_id="", status="finished", output=dict(r["out"]), steps=8, path=["https://store.test/"])

    async def stop(self, *a, **kw):
        return None

    async def live_url(self, sid):
        return None

    async def new_profile(self, name):
        return "prof"


class Family:
    """A family of the test set: who writes, in which city and language, and its saved place (with or without a map point)."""

    def __init__(self, label: str, *, elder: dict, carer: dict | None, speaker: str, place: dict, heal: dict | None = None,
                 home_words: tuple[str, ...] = ()):
        self.label, self.elder, self.carer, self.place, self.heal = label, elder, carer, place, heal
        self.speaker = carer if speaker == "carer" and carer else elder
        self.members = [elder] + ([carer] if carer else [])
        # the fake backend's own way of knowing the saved place in its words (production asks the brain to name it)
        self.home_words = ("ghar", "home", "घर", place["nickname"].lower()) + home_words


RAIPUR = Family("Raipur, Hinglish/Hindi, elder with a caregiver, place with a map point", elder=ELDER, carer=SON, speaker="elder",
                place={"addressId": "addr-ghar", "nickname": "Ghar", "full": "12 Civil Lines, Raipur 492001", "pincode": "492001",
                       "lat": 21.24, "lng": 81.69, "geoPrecision": "address"}, home_words=("civil",))


class StoreHost(SimHost):
    def __init__(self, family: Family = RAIPUR):
        super().__init__()
        self.family, self.place_calls = family, 0

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        f = self.family
        if tool == "delivery_place":
            words = str(args.get("words") or "").lower()
            if args.get("exact") and words and words != f.place["nickname"].lower():  # the engine's AI sent an exact name
                return {"matched": False, "saved": [f.place["nickname"]]}
            if words and not args.get("exact") and not any(w in words for w in f.home_words):  # the fake's own book
                return {"matched": False, "saved": [f.place["nickname"]]}
            self.place_calls += 1
            place = dict(f.place)
            if f.heal and self.place_calls > 1:  # saved without a point; the backend finds one when asked again
                place.update(f.heal)
            return {**place, "matched": True}
        if tool == "list_places":
            return {"places": [{"name": f.place["nickname"], "address": f.place["full"], "isDefault": True}]}
        if tool == "browser_profile":
            return {"profileId": f"prof-{family_id}-{args.get('partner')}", "loginPhone": "9000011111"}
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


class Conversation:
    def __init__(self, name: str, family: Family = RAIPUR, *, agent_ok: tuple[str, ...] = ()):
        self.name, self.fam = name, f"ordersim-{uuid.uuid4().hex[:6]}"
        self.family = family
        self.shop = Store(address=f"{family.place['nickname']}: {family.place['full']}")
        self.host, self.browser = StoreHost(family), Browser(self.shop)
        self.agent_ok = agent_ok  # "service:phase" runs of the browser agent this conversation expects
        self.log: list[str] = []
        self.problems: list[str] = []
        self.notified = 0

    def maa_messages(self) -> list[dict]:
        return self.messages_to(self.family.speaker["id"])

    def messages_to(self, person: str) -> list[dict]:
        return [m for m in self.host.world.sent if m.get("to") == person]

    async def wake(self) -> int:
        """The scheduler's wake-up run (due follow-ups), as production runs it every few minutes. Returns how many messages
        went out."""
        before = len(self.host.world.sent)
        await wake.wake_due(SessionLocal, lambda fid: self.host)
        for m in self.host.world.sent[before:]:
            self.log.append(f"SAHELI (wake → {'Maa' if m.get('to') == ELDER['id'] else m.get('to')}): {m.get('text')}")
            if str(m.get("text") or "").count("?") > 2:  # "आ गया? सब ठीक है न?" is one ask
                self.problems.append(f"more than one question in one message: {m.get('text')}")
        return len(self.host.world.sent) - before

    async def facts(self, domain: str) -> list:
        async with SessionLocal() as s:
            return await store.facts(s, self.fam, self.family.elder["id"], domains=[domain], statuses=("active",))

    async def say(self, text: str) -> str:
        clock.set_now(clock.now() + timedelta(seconds=30))  # people take a moment to answer
        f = self.family
        async with SessionLocal() as s:
            res = await run_turn(s, self.host, TurnRequest(family_id=self.fam, elder=f.elder, speaker=f.speaker, members=f.members, text=text,
                                                           message_ref=f"m-{uuid.uuid4().hex[:8]}", channel="whatsapp"))
        self.log.append(f"{f.speaker['name'].upper()}: {text}")
        self.said_at = clock.now()
        tools = ", ".join(f"{a['tool']}{'' if a.get('ok', True) else '(x)'}" for a in res.actions)
        if res.reply and res.reply.strip().lower() != "none":
            self.log.append(f"SAHELI (reply): {res.reply}   [{tools}]")
        elif tools:
            self.log.append(f"  [{tools}]")
        return res.reply or ""

    async def notify(self, family_id, requested_by, prompt):
        self.notified += 1
        before = len(self.messages_to(requested_by))
        self.log.append(f"  <update to {'Maa' if requested_by == ELDER['id'] else requested_by}> {prompt[:150]}")
        await system_turn(SessionLocal, self.host, family_id,
                          f"{prompt} (Requested by {requested_by}: tell them with send_message to {requested_by}, in their language, then reply none.)",
                          f"task:{uuid.uuid4().hex[:12]}", deliver_to=requested_by)
        new = self.messages_to(requested_by)[before:]
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
            await store.save_roster(s, self.fam, self.family.elder, self.family.members)
            await s.commit()

    def sent_with_times(self) -> list[dict]:
        return self.messages_to(self.family.speaker["id"])

    def option_lists(self) -> list[str]:
        """Messages that read out options to choose from (two or more lines with a price) before the one confirm. The
        sim's own check; production never reads words."""
        out = []
        for m in self.messages_to(self.family.speaker["id"]):
            text = str(m.get("text") or "")
            if any(str(b.get("id", "")).startswith("v2:od:") for b in m.get("buttons") or []):
                break  # the one confirm: everything after it is about that cart
            priced = [ln for ln in text.splitlines() if "₹" in ln or re.search(r"\brs\.?\s*\d", ln, re.I)]
            if len(priced) >= 2:
                out.append(text)
        return out


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
    # she got it; three hours later Saheli asks how it was, one question at a time, and remembers the answers
    await c.say("haan mil gaya, sab theek hai")
    t = await c.task()
    c.check(((t.result or {}).get("feedback") or {}).get("arrived") == "yes", f"arrival not saved: {(t.result or {}).get('feedback')}")
    clock.set_now(clock.now() + timedelta(hours=3, minutes=5))
    t = await c.task()
    asked = await c.wake()
    already = int((((t.result or {}).get("feedback") or {}).get("answers")) or 0)
    c.check(asked == 1 or already > 0, "no 'how was it?' question (and they had not talked about it)")
    await c.say("khakhra bahut accha laga, pehli baar khaya")
    await c.say("doodh to roz wala hi hai")
    await c.say("haan agli baar bhi yahi khakhra lana")
    t = await c.task()
    fb = (t.result or {}).get("feedback") or {}
    likes = {f.value.get("item"): f.value for f in await c.facts("preference")}
    khakhra = likes.get("Charliee Methi Khakhra") or {}
    c.check(khakhra.get("liked") == "yes" and khakhra.get("usual") == "new", f"khakhra not remembered as a liked first try: {likes}")
    c.check(any(v.get("usual") == "usual" for k, v in likes.items() if "Milk" in str(k)), f"milk not remembered as usual: {likes}")
    c.check(int(fb.get("answers") or 0) <= aftercare.MAX_QUESTIONS + 1, f"too many questions: {fb}")
    c.check(khakhra.get("again") == "yes", f"'get the same khakhra again' not saved: {khakhra}")


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


async def late_order_tells_the_caregiver(c: Conversation):
    await c.setup()
    await c.say("blinkit se ek Amul butter mangwa do")
    await c.say("haan ghar")
    await c.ticks(lambda t: t.status == "awaiting_confirm")
    ids = confirm_buttons(c)
    c.check(len(ids) == 3, f"confirm buttons missing: {ids}")
    if not ids:
        return
    await c.say(ids[0])
    t = await c.ticks(lambda t: t.status == "done")
    c.check(bool((t.result or {}).get("placed")), "not placed")
    # the store keeps saying "being packed"; 30 minutes later Saheli asks whether it came
    clock.set_now(clock.now() + timedelta(minutes=31))
    await c.ticks(lambda t: True, most=1)
    c.check(await c.wake() == 1, "no 'has it come?' question")
    await c.say("nahi aaya abhi tak")
    t = await c.task()
    c.check(not (t.details or {}).get("aftercare", {}).get("caregiver_due"), "caregiver told before it was late")
    # ten minutes later it is long overdue: asked again, still not here → Kritarth is told
    clock.set_now(clock.now() + timedelta(minutes=11))
    c.check(await c.wake() == 1, "not asked again when it became late")
    await c.say("abhi bhi nahi aaya")
    before = len(c.messages_to(SON["id"]))
    await c.ticks(lambda t: True, most=1)
    c.check(len(c.messages_to(SON["id"])) > before, "the caregiver was not told about the late order")


# --- every family, not one (founder 2026-10-11): cities, languages, places with and without a map point, self-care ---------

BANGALORE = Family("Bangalore, English, a son orders for his mother, place saved WITHOUT a map point (found at order time)",
                   elder={"id": "elder-blr", "name": "Sudha", "role": "elder"},
                   carer={"id": "son-blr", "name": "Vish", "role": "primary caregiver"}, speaker="carer",
                   place={"addressId": "addr-blr", "nickname": "Home", "pincode": "560092",
                          "full": "74 4th cross 4c sector amrutnagar byatarayanapura bangalore karnataka india, 560092"},
                   heal={"lat": 13.0617, "lng": 77.5929, "geoPrecision": "pincode"}, home_words=("amrutnagar",))
DELHI = Family("Delhi, Hindi (Devanagari), elder with a caregiver, place with a map point",
               elder={"id": "elder-del", "name": "Kamla Devi", "role": "elder"},
               carer={"id": "son-del", "name": "Rohit", "role": "primary caregiver"}, speaker="elder",
               place={"addressId": "addr-del", "nickname": "Ghar", "pincode": "110024", "full": "B-14, Lajpat Nagar 2, New Delhi, Delhi 110024",
                      "lat": 28.5672, "lng": 77.2365, "geoPrecision": "area"}, home_words=("lajpat",))
MUMBAI = Family("Mumbai, Marathi, self-care (no caregiver), place with a map point",
                elder={"id": "self-bom", "name": "Sunanda Joshi", "role": "elder"}, carer=None, speaker="elder",
                place={"addressId": "addr-bom", "nickname": "Ghar", "pincode": "400050",
                       "full": "Flat 5, Sai Darshan, Bandra West, Mumbai, Maharashtra 400050", "lat": 19.0596, "lng": 72.8295,
                       "geoPrecision": "address"}, home_words=("घरी", "bandra"))
CHENNAI = Family("Chennai, Tamil, elder with a caregiver, place WITHOUT a map point and none found (the browser agent looks)",
                 elder={"id": "elder-maa", "name": "Meenakshi", "role": "elder"},
                 carer={"id": "son-maa", "name": "Karthik", "role": "primary caregiver"}, speaker="elder",
                 place={"addressId": "addr-maa", "nickname": "Veedu", "pincode": "600017", "full": "Flat 9, T. Nagar, Chennai, Tamil Nadu 600017"},
                 home_words=("வீடு", "வீட்டு", "veedu", "nagar"))
KOLKATA = Family("Kolkata, Bengali, elder with a caregiver daughter, one store slow (its saved step broken)",
                 elder={"id": "elder-ccu", "name": "Mitali", "role": "elder"},
                 carer={"id": "dau-ccu", "name": "Ria", "role": "primary caregiver"}, speaker="elder",
                 place={"addressId": "addr-ccu", "nickname": "Bari", "pincode": "700029",
                        "full": "12 Hindustan Park, Gariahat, Kolkata, West Bengal 700029", "lat": 22.5186, "lng": 88.3639, "geoPrecision": "address"},
                 home_words=("বাড়ি", "বাড়িতে", "bari", "gariahat"))


def _script(text: str, lo: int, hi: int) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text)


async def _one_confirm_without_lists(c: Conversation, until_most: int = 14) -> Task | None:
    t = await c.ticks(lambda t: t.status == "awaiting_confirm" and t.input_needed == "confirm", most=until_most)
    c.check(t is not None and t.status == "awaiting_confirm" and t.input_needed == "confirm",
            f"no one confirm (task {t.service if t else None} {t.status if t else None}/{t.input_needed if t else None})")
    for text in c.option_lists():
        c.problems.append(f"options listed to choose from: {text[:160]}")
    ids = confirm_buttons(c)
    c.check(len(ids) == 3, f"confirm buttons missing: {ids}")
    confirm = next((m for m in reversed(c.maa_messages()) if any(str(b.get("id", "")).startswith("v2:od:") for b in m.get("buttons") or [])), None)
    # the amount in any script: "₹331", "৮৩ টাকা", "₹ ८३"
    c.check(confirm is not None and any(ch.isdigit() for ch in str(confirm.get("text"))), f"the confirm has no amount: {confirm}")
    return t


def _points_sent(c: Conversation) -> None:
    empty = [s for s in c.shop.searched if fastpath.SITES[s[0]].needs_point and not fastpath.has_point(s[1], s[2])]
    c.check(not empty, f"a store's own search was sent without the map point: {empty}")


async def bangalore_ritebite_no_store_named(c: Conversation):
    """The order of 2026-10-11: Saheli went quiet for 6 minutes, then listed two Blinkit packs to choose from."""
    await c.setup()
    await c.say("Hey can u order a rite bite choco berry")
    await c.say("Yes please order 4 of it")
    t = await _one_confirm_without_lists(c)
    _points_sent(c)
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check(len(cart) == 1 and cart[0]["qty"] == 4 and "RiteBite" in cart[0]["name"], f"not 4 of one bar: {cart}")
    chosen = ((t.details or {}).get("chosen") or []) if t else []
    c.check(not any("x 4" in str(x.get("pack")) for x in chosen), f"4 x a 4-pack: {chosen}")
    ids = confirm_buttons(c)
    if ids:
        await c.say(ids[0])
        t = await c.ticks(lambda t: t.status == "done", most=8)
        c.check(len(c.shop.placed) == 1 and t.status == "done", f"not placed once after Yes ({len(c.shop.placed)}, {t.status})")


async def delhi_hindi_no_store_then_no(c: Conversation):
    await c.setup()
    await c.say("मुझे दो पैकेट मदर डेयरी दूध चाहिए")
    await c.say("हाँ, घर पर ही")
    await _one_confirm_without_lists(c)
    _points_sent(c)
    confirm = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    c.check(_script(str(confirm), 0x0900, 0x097F), f"the confirm is not in Hindi: {confirm}")
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check([(l["name"], l["qty"]) for l in cart] == [("Mother Dairy Toned Milk", 2)], f"wrong cart: {cart}")
    await c.say("नहीं, रहने दो")
    t = await c.task()
    c.check(t.status == "cancelled" or t.cancel_requested, f"not cancelled ({t.status})")
    c.check(not c.shop.placed, "placed after a no")


async def mumbai_marathi_self_care_named_store_lacks_it(c: Conversation):
    """A store they named does not have it: one question before another app, then the one confirm."""
    await c.setup()
    await c.say("झेप्टो वरून दोन पाकीट गोकुळ दूध पाठवा")
    await c.say("हो, घरीच")
    await c.ticks(lambda t: False, most=8)  # the named store looks, then the other two (80 s of ticks)
    async with SessionLocal() as s:  # the two other stores start together: look at all of them, not the latest one
        asking = (await s.execute(select(Task).where(Task.family_id == c.fam, Task.status == "awaiting_confirm",
                                                     Task.input_needed == "go"))).scalars().first()
    c.check(asking is not None and asking.service == "blinkit", f"no question about Blinkit ({asking.service if asking else None})")
    for text in c.option_lists():
        c.problems.append(f"options listed to choose from: {text[:160]}")
    await c.say("हो, चालेल")
    t = await _one_confirm_without_lists(c)
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check([(l["name"], l["qty"]) for l in cart] == [("Gokul Full Cream Milk", 2)], f"wrong cart: {cart}")
    ids = confirm_buttons(c)
    if ids:
        await c.say(ids[0])
        t = await c.ticks(lambda t: t.status == "done", most=8)
        c.check(len(c.shop.placed) == 1, f"not placed once ({len(c.shop.placed)})")


async def chennai_tamil_no_point_browser_agent_looks(c: Conversation):
    await c.setup()
    await c.say("ஆவின் பால் இரண்டு பாக்கெட் வேண்டும்")
    await c.say("ஆமாம், வீட்டுக்கே அனுப்புங்க")
    await _one_confirm_without_lists(c, until_most=16)
    _points_sent(c)
    c.check(not [s for s in c.shop.searched if fastpath.SITES[s[0]].needs_point], "a saved look-up ran without a point")
    c.check(any(r.endswith(":browse") for r in c.browser.agent_runs), f"the browser agent did not look: {c.browser.agent_runs}")
    confirm = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    c.check(_script(str(confirm), 0x0B80, 0x0BFF), f"the confirm is not in Tamil: {confirm}")
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check([(l["name"], l["qty"]) for l in cart] == [("Aavin Toned Milk", 2)], f"wrong cart: {cart}")
    ids = confirm_buttons(c)
    if ids:
        await c.say(ids[0])
        t = await c.ticks(lambda t: t.status == "done", most=8)
        c.check(len(c.shop.placed) == 1, f"not placed once ({len(c.shop.placed)})")


async def kolkata_bengali_slow_store_not_waited_for(c: Conversation):
    """One store's saved step is broken and its browser agent takes 5 minutes: the others answer in seconds, and the
    confirm comes from them within about 2 minutes; nobody waits in silence."""
    await c.setup()
    c.shop.broken.add("zepto")
    c.browser.slow["zepto"] = 30
    await c.say("দুই প্যাকেট মারি বিস্কুট পাঠাও")
    await c.say("হ্যাঁ, বাড়িতেই পাঠাও")
    asked_at = c.said_at
    t = await _one_confirm_without_lists(c, until_most=20)
    waited = (clock.now() - asked_at).total_seconds()
    c.check(waited <= 180, f"the confirm took {waited:.0f} s of a slow store's time")
    confirm = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    c.check(_script(str(confirm), 0x0980, 0x09FF), f"the confirm is not in Bengali: {confirm}")
    c.check(t is not None and t.service in ("blinkit", "instamart"), f"picked the slow store: {t.service if t else None}")


SCENARIOS = [("several items, Yes tap, tracked to the door, how was it", several_items_tap_yes_tracked, RAIPUR, ()),
             ("flavour kept, then cancel", flavour_then_cancel, RAIPUR, ()),
             ("missing flavour, change, old Yes tap", missing_flavour_change_and_old_tap, RAIPUR, ()),
             ("late order: asked, checked again, caregiver told", late_order_tells_the_caregiver, RAIPUR, ()),
             ("Bangalore: RiteBite x4, no store named, place without a point", bangalore_ritebite_no_store_named, BANGALORE, ("zepto:prepare", "zepto:place")),
             ("Delhi: Hindi, no store named, then no", delhi_hindi_no_store_then_no, DELHI, ("zepto:prepare",)),
             ("Mumbai: Marathi, self-care, the named store lacks it", mumbai_marathi_self_care_named_store_lacks_it, MUMBAI, ()),
             ("Chennai: Tamil, no point anywhere: the agent looks", chennai_tamil_no_point_browser_agent_looks, CHENNAI,
              # no store ids from the agent's look-up: the agent also builds the cart and places, on whichever store is best
              tuple(f"{s}:{p}" for s in ("blinkit", "instamart", "zepto") for p in ("browse", "prepare", "place"))),
             ("Kolkata: Bengali, one store slow", kolkata_bengali_slow_store_not_waited_for, KOLKATA, ("zepto:browse",))]


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
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    for name, play, family, agent_ok in SCENARIOS:
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        c = Conversation(name, family, agent_ok=agent_ok)
        print(f"\n--- {family.label}", flush=True)
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
                for loop in (await s.execute(select(OpenLoop).where(OpenLoop.family_id == c.fam, OpenLoop.status == "open"))).scalars():
                    loop.status = "cancelled"
                await s.commit()
            clock.set_now(None)
        unexpected = [r for r in c.browser.agent_runs if r not in c.agent_ok]
        if unexpected:
            c.problems.append(f"slow browser agent used: {unexpected}")
        print(f"\n=== {name}: {'OK' if not c.problems else 'PROBLEMS'}")
        print("\n".join(c.log))
        for p in c.problems:
            print("  !!", p)
        failed += bool(c.problems)
    ran = sum(1 for n, *_ in SCENARIOS if not only or any(o.lower() in n.lower() for o in only))
    print(f"\n{ran - failed}/{ran} conversations clean")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
