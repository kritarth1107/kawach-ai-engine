"""Whole order conversations before a deploy: the real brain, product matcher and delivery-status reader, with fake stores
(catalogue, cart, place, order status) and the real task runtime. Plays the night of 2026-10-10's conversations: several
items with flavours, a flavour the store does not have, a change after the confirm, a tap on an old confirm, cancel,
the confirm buttons and the updates after placing; and after an order: did it come, how was it (remembered), and a late
order told to the caregiver. Founder 2026-10-11: what was found is always listed, numbered, every store the person named
reported, and the person picks (in their own words) before the one confirm; a placed order is never "cancelled": the
caregiver is told. Paid: about 60 fast-model calls (~₹15).

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_ordersim \\   (its own database: emptied each run)
    GCP_PROJECT_ID=sunny-ship-508214-q1 GOOGLE_CLOUD_PROJECT=sunny-ship-508214-q1 MEMORY_EMBEDDINGS=off \\
    MODEL_ROUTES='{"brain": ["gemini:gemini-3.5-flash@global"], "classify": ["gemini:gemini-3.5-flash@global"], ...}' \\
    PYTHONPATH=. python eval/order_conversation_sim.py

The checks are this script's own (the fake store's catalogue names); production code never matches words.
"""

from __future__ import annotations

import asyncio
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

ELDER = {"id": "elder-sim", "name": "Shanti Devi", "role": "elder"}
SON = {"id": "son-sim", "name": "Mohan", "role": "primary caregiver"}
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
                P("b11", "Mother Dairy Toned Milk", 29, "500 ml"), P("b12", "Karachi Bakery Fruit Biscuits", 200, "400 g")],
    "instamart": [P("i1", "Modern Kitchens Butter Muruku", 35, "150 g"), P("i2", "Haldiram's Peri Peri Murukku", 40, "150 g"),
                  P("i3", "Peri Peri Banana Chips", 50, "150 g"), P("i4", "Amul Taaza Toned Milk", 28, "500 ml"),
                  # a 4-pack next to the single bar: "4 bars" must be 4 x the single, never 4 x the 4-pack
                  P("i5", "RiteBite Max Protein Daily Choco Berry 10g Protein Bar", 316, "50 g x 4"),
                  P("i6", "RiteBite Max Protein Daily Choco Berry 10g Protein Bar", 79, "50 g"),
                  P("i7", "Aavin Toned Milk", 24, "500 ml"), P("i8", "Britannia Marie Gold Biscuits", 34, "250 g"),
                  P("i9", "Mother Dairy Toned Milk", 28, "500 ml"), P("i10", "Karachi Bakery Fruit Biscuits", 158, "400 g")],
    "zepto": [P("z1", "RiteBite Max Protein 30g Protein Bar Choco Berry", 176, "1 pack (100 g)"),
              P("z2", "Amul Taaza Toned Milk", 29, "500 ml"), P("z3", "Aavin Toned Milk", 25, "500 ml"),
              P("z4", "Britannia Marie Gold Biscuits", 33, "250 g")],
    # pharmacies: the strength asked, another strength and another brand (a medicine is offered only exactly as asked)
    "apollo": [P("a1", "Dolo 650 Tablet", 33, "15 tablets"), P("a2", "Dolo 500 Tablet", 30, "15 tablets")],
    # no magnesium citrate here: the closest is another supplement (a live 1mg order, 2026-10-12)
    "1mg": [P("m3", "Calcium and Vitamin D3 Veg Tablets with Zinc, Magnesium & Alfalfa", 369, "60 tablets"),
            P("m1", "Dolo 650 Tablet", 31, "15 tablets"), P("m2", "Calpol 650 Tablet", 28, "15 tablets")],
    "pharmeasy": [P("e1", "Crocin Advance 500 Tablet", 25, "15 tablets")],
}
F = lambda pid, name, price, rest, eta: {**P(pid, name, price, "1 plate"), "restaurant": rest, "restaurantId": rest.lower().replace(" ", "-"),  # noqa: E731
                                         "eta": eta}
CATALOGUE.update({
    "swiggy": [F("s1", "Paneer Butter Masala", 249, "Punjabi Tadka", "35 min"), F("s2", "Paneer Butter Masala", 289, "Delhi Darbar", "40 min"),
               F("s3", "Butter Naan", 45, "Punjabi Tadka", "35 min")],
    "zomato": [F("x1", "Paneer Butter Masala", 239, "Shree Annapoorna", "30 min"), F("x2", "Kadai Paneer", 259, "Shree Annapoorna", "30 min")],
})
# fares each ride app shows (Rapido has no ride near the pickup)
RIDES = {"uber": [{"type": "Uber Go", "fare": "₹236", "eta": "6 min"}, {"type": "Auto", "fare": "₹142", "eta": "4 min"}],
         "ola": [{"type": "Auto", "fare": "₹128", "eta": "5 min"}, {"type": "Mini", "fare": "₹210", "eta": "7 min"}],
         "rapido": []}
SHOP_WORDS = {"doodh": "milk", "दूध": "milk"}  # the fake store's own search knows a few Hindi words, as real stores do


class Store:
    """The fake stores: search, cart, place and order status, recorded for the checks."""

    def __init__(self, address: str = "Ghar: 12 Civil Lines, Raipur 492001"):
        self.carts, self.placed, self.status = [], [], {}
        self.address = address
        self.searched: list[tuple] = []  # (service, lat, lon): a store that needs the point must never get an empty one
        self.broken: set[str] = set()  # stores whose saved look-up fails (the browser agent looks instead)
        self.away: set[str] = set()  # stores that do not deliver to the family's place
        self.needs_code: set[str] = set()  # stores whose account is not logged in: a login code comes first
        self.codes_refused: dict[str, int] = {}  # how many login codes a store's boxes do not take first (PharmEasy, 2026-10-13)
        self.cancel_mode: dict[str, str] = {}  # how a store answers a cancel: ok (default), fee0, fee25, refuse
        self.cancelled: list[str] = []
        self.logged_in: set[str] = set()
        self.rides: list[dict] = []  # rides booked
        self.city_pickup: str | None = None  # the ride app sets the pickup to this city once (live 2026-10-12)
        self.fares_need_login: set[str] = set()  # ride apps that show no fares to a guest (Ola, live 2026-10-12)
        # ride apps whose logged-in fare look-up books anyway (live 2026-10-12 21:52 IST: "Confirm & Book" pressed to make a
        # login appear): the run's own pages show the booking, its report leaves it out
        self.books_on_look: set[str] = set()

    async def search(self, service, cdp, q, *, lat=None, lon=None, pincode=None, limit=8):
        self.searched.append((service, lat, lon))
        if fastpath.SITES[service].needs_point and not fastpath.has_point(lat, lon):
            raise fastpath.NoPoint(f"{service}: no point")  # as the real step does
        if service in self.broken:
            raise fastpath.FastPathError(f"{service} search answered 500")
        if service in self.away:
            return {"items": [], "deliverable": False, "eta": None, "logged_in": True}
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
                "address_used": f"{place['nickname']}: {place['full']}" if (place or {}).get("full") else self.address, "eta": "10 min",
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
        self.shop, self.agent_runs, self.slow, self.goals = shop, [], {}, []
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
        self.goals.append((f"{service}:{phase}", goal))
        async with SessionLocal() as s:
            task = await s.get(Task, uuid.UUID(str(metadata["task"])))
            out = await self._report(task, service, phase, goal)
        tid = f"bu-{uuid.uuid4().hex[:6]}"
        path, self._next_path = getattr(self, "_next_path", None) or ["https://store.test/"], None
        self._runs[tid] = {"out": out, "polls": 0, "need": self.slow.get(f"{service}:{phase}", self.slow.get(service, 1)), "path": path}
        return AgentRun(task_id=tid, session_id=session_id or f"s-{tid}", status="created")

    async def _report(self, task: Task, service: str, phase: str, goal: str = "") -> dict:
        base = {"logged_in": True, "needs_otp": False, "blocked": False, "problem": ""}
        d = task.details or {}
        if phase == "check":  # the account's own list: a ride or order made since the look-up began
            made = [r for r in self.shop.rides if r["service"] == service and r.get("by_mistake")]
            if not made:
                return {**base, "this_task": False}
            gone = service in self.shop.cancelled
            return {**base, "this_task": True, "ride_id": made[-1]["id"], "cancelled": gone,
                    "status": "Cancelled" if gone else "Driver on the way"}
        if task.kind == "ride":
            from app.specialists.agents import ride_end

            fares = [dict(o) for o in RIDES.get(service, [])]
            ends = {"pickup_used": ride_end(d, "pickup"), "drop_used": ride_end(d, "drop")}  # the app shows what it was given
            if phase == "prepare" and self.shop.city_pickup:
                # the app chose the city for the pickup (live 2026-10-12): its fares are for another route
                ends["pickup_used"], self.shop.city_pickup = self.shop.city_pickup, None
                return {**base, **ends, "options": [{**o, "fare": f"₹{int(o['fare'][1:]) + 50}"} for o in fares]}
            if phase == "browse" and service in self.shop.fares_need_login:
                return {**base, **ends, "logged_in": False, "options": [], "login_required": True,
                        "problem": "The site requires login before showing fares."}
            if phase == "browse":  # guest fares
                return {**base, **ends, "logged_in": False, "options": fares, **({} if fares else {"problem": "no rides near the pickup"})}
            if phase == "prepare" and service in self.shop.books_on_look:
                self.shop.books_on_look.discard(service)
                ref = f"CRN{10000000001 + len(self.shop.rides)}"
                self.shop.rides.append({"service": service, "choice": "Auto", "pickup": ends["pickup_used"], "by_mistake": True, "id": ref})
                self._next_path = ["https://book.olacabs.com/", "https://book.olacabs.com/confirm-ride-p2p?x",
                                   f"https://book.olacabs.com/ride-detail/{ref}?x", "https://book.olacabs.com/?x"]
                return {**base, **ends, "options": [{**o, "fare": f"₹{int(o['fare'][1:]) + 3}"} for o in fares], "booked": None, "ride_id": None}
            if phase == "prepare":  # logged in: the fares a few rupees up
                return {**base, **ends, "options": [{**o, "fare": f"₹{int(o['fare'][1:]) + 3}"} for o in fares]}
            if phase == "place":
                self.shop.rides.append({"service": service, "choice": d.get("choice"), "pickup": ends["pickup_used"]})
                return {**base, **ends, "booked": True, "ride_id": f"R{len(self.shop.rides)}", "driver": "Suresh", "vehicle": "White Dzire",
                        "plate": "CG04 AB 1234", "pickup_eta": "4 min", "ride_pin": "4931", "status": "driver assigned"}
        if phase == "cancel":
            mode = self.shop.cancel_mode.get(service, "ok")
            if mode in ("fee0", "fee25") and not d.get("fee_ok"):
                return {**base, "cancelled": False, "cancel_fee": "₹0" if mode == "fee0" else "₹25"}
            if mode == "refuse":
                return {**base, "cancelled": False, "problem": "the order is already packed and cannot be cancelled"}
            self.shop.cancelled.append(service)
            return {**base, "cancelled": True}
        if phase in ("prepare", "otp") and service in self.shop.needs_code and service not in self.shop.logged_in:
            # a code given to this run: typed into the boxes already, or handed to the agent to type
            if "Enter this login code" not in goal and "already typed into the code boxes" not in goal:  # the store sends one
                self.code_sent_at = getattr(self, "polls_total", 0)
                return {**base, "logged_in": False, "needs_otp": True, "otp_sent_to": "phone ending 1111"}
            if self.shop.codes_refused.get(service):  # its code boxes did not take this one
                self.shop.codes_refused[service] -= 1
                return {**base, "logged_in": False, "needs_otp": False, "problem": "code not accepted: the boxes kept 3 of 4 digits"}
            self.shop.logged_in.add(service)
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
            cart = await self.shop.cart(service, None, chosen, place=(d.get("limits") or {}).get("place"))
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
        return AgentRun(task_id=tid, session_id="", status="finished", output=dict(r["out"]), steps=8, path=list(r.get("path") or ["https://store.test/"]))

    async def stop(self, *a, **kw):
        return None

    async def live_url(self, sid):
        return None

    async def new_profile(self, name):
        return "prof"


class Family:
    """A family of the test set: who writes, in which city and language, and its saved place (with or without a map point)."""

    def __init__(self, label: str, *, elder: dict, carer: dict | None, speaker: str, place: dict, heal: dict | None = None,
                 home_words: tuple[str, ...] = (), carer_place: dict | None = None):
        self.label, self.elder, self.carer, self.place, self.heal = label, elder, carer, place, heal
        self.carer_place = carer_place  # the caregiver's own home (their default), when they live elsewhere
        self.speaker = carer if speaker == "carer" and carer else elder
        self.members = [elder] + ([carer] if carer else [])
        # the fake backend's own way of knowing the saved place in its words (production asks the brain to name it)
        self.home_words = ("ghar", "home", "घर", place["nickname"].lower()) + home_words


RAIPUR = Family("Raipur, Hinglish/Hindi, elder with a caregiver, place with a map point", elder=ELDER, carer=SON, speaker="elder",
                place={"addressId": "addr-ghar", "nickname": "Ghar", "full": "12 Civil Lines, Raipur 492001", "pincode": "492001",
                       "lat": 21.25, "lng": 81.63, "geoPrecision": "address"}, home_words=("civil",))


class StoreHost(SimHost):
    def __init__(self, family: Family = RAIPUR):
        super().__init__()
        self.family, self.place_calls, self.saves = family, 0, []

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        f = self.family
        if tool == "delivery_place" and f.carer_place:
            # two homes: the care recipient's and the caregiver's own; each person's default is their own (whose home: defaultFor)
            homes = {f.place["nickname"].lower(): {**f.place, "defaultFor": [f.elder["id"]]},
                     f.carer_place["nickname"].lower(): {**f.carer_place, "defaultFor": [f.carer["id"]]}}
            words = str(args.get("words") or "").lower()
            if not words:
                mine = f.carer_place if subject_id == (f.carer or {}).get("id") else f.place
                return {**homes[mine["nickname"].lower()], "matched": None}
            hit = homes.get(words) if args.get("exact") else None
            return {**hit, "matched": True} if hit else {"matched": False, "saved": [p["nickname"] for p in (f.place, f.carer_place)]}
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
        if tool == "save_place":
            self.saves.append(args)
            return {"ok": True}
        if tool == "list_places":
            if f.carer_place:
                his = subject_id == (f.carer or {}).get("id")
                return {"places": [{"name": f.place["nickname"], "address": f.place["full"], "isDefault": not his},
                                   {"name": f.carer_place["nickname"], "address": f.carer_place["full"], "isDefault": his}]}
            return {"places": [{"name": f.place["nickname"], "address": f.place["full"], "isDefault": True}]}
        if tool == "browser_profile":
            return {"profileId": f"prof-{family_id}-{args.get('partner')}", "loginPhone": "9000011111"}
        if tool == "connector_status":
            return {"connected": False}  # no linked store: the saved steps and the browser agent
        if tool == "ride_place":
            return {}  # no map point for a ride's places here: the browser agent looks the fares up
        return await super().call(tool, args, family_id=family_id, subject_id=subject_id, actor_id=actor_id)


class Conversation:
    def __init__(self, name: str, family: Family = RAIPUR, *, agent_ok: tuple[str, ...] = ()):
        self.name, self.fam = name, f"ordersim-{uuid.uuid4().hex[:6]}"
        self.family = family
        self.shop = Store(address=f"{family.place['nickname']}: {family.place['full']}")
        self.host, self.browser = StoreHost(family), Browser(self.shop)
        self.agent_ok = agent_ok  # "service:phase" runs of the browser agent this conversation expects
        self.log: list[str] = []
        self.location_asked = False  # Saheli's last reply asked for a WhatsApp location
        self.last_buttons: list[dict] = []  # the one-tap answers on Saheli's last reply
        self.problems: list[str] = []
        self.notified = 0
        self.prompts: list[str] = []  # the task updates Saheli was given (to check every store was reported)
        self.tick_no = 0
        self.prompt_ticks: list[tuple[int, str, str]] = []  # (tick, person, update) as each update went out
        self.hold = False  # updates wait (as one does while the person's own message is being answered)
        self.held: list[tuple] = []
        self.dropped: list[str] = []  # updates dropped at delivery because their task had moved on

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

    async def say(self, text: str, who: dict | None = None, location: dict | None = None) -> str:
        clock.set_now(clock.now() + timedelta(seconds=30))  # people take a moment to answer
        f = self.family
        speaker = who or f.speaker
        if location:  # as the backend sends a shared pin: the text tag and the data
            text = text or f"[location lat={location['lat']} lng={location['lng']}]"
            location = {"name": "", "address": "", **location}
        async with SessionLocal() as s:
            res = await run_turn(s, self.host, TurnRequest(family_id=self.fam, elder=f.elder, speaker=speaker, members=f.members, text=text,
                                                           message_ref=f"m-{uuid.uuid4().hex[:8]}", channel="whatsapp", location=location))
        self.location_asked = bool(res.location_request)
        self.last_buttons = list(res.buttons or [])
        self.log.append(f"{speaker['name'].upper()}: {text}" + ("  (shared a location)" if location else ""))
        self.said_at = clock.now()
        tools = ", ".join(f"{a['tool']}{'' if a.get('ok', True) else '(x)'}" for a in res.actions)
        if res.reply and res.reply.strip().lower() != "none":
            self.log.append(f"SAHELI (reply): {res.reply}   [{tools}]" + ("  [Send location button]" if res.location_request else "")
                            + (f"   buttons={[b.get('id') for b in res.buttons]}" if res.buttons else ""))
        elif tools:
            self.log.append(f"  [{tools}]")
        return res.reply or ""

    async def notify(self, family_id, requested_by, prompt):
        self.notified += 1
        self.prompts.append(prompt)
        self.prompt_ticks.append((self.tick_no, requested_by, prompt))
        check = runtime.delivery_check()  # as production's notify: the update is checked again when its turn comes
        if self.hold:
            self.log.append(f"  <update held: the person's own message is being answered> {prompt[:120]}")
            self.held.append((family_id, requested_by, prompt, check))
            return
        await self._deliver(family_id, requested_by, prompt, check)

    async def release(self) -> None:
        """The updates that waited for the person's own turn are delivered now, each checked again first."""
        self.hold, held, self.held = False, self.held, []
        for family_id, person, prompt, check in held:
            await self._deliver(family_id, person, prompt, check)

    async def _deliver(self, family_id, requested_by, prompt, check):
        before = len(self.messages_to(requested_by))
        self.log.append(f"  <update to {'Maa' if requested_by == ELDER['id'] else requested_by}> {prompt[:150]}")
        await system_turn(SessionLocal, self.host, family_id,
                          f"{prompt} (For person {requested_by}, who asked: send it to them now with send_message (to {requested_by}), in their language. After that send, reply none.)",
                          f"task:{uuid.uuid4().hex[:12]}", deliver_to=requested_by, still_true=check)
        new = self.messages_to(requested_by)[before:]
        if not new and check is not None:
            async with SessionLocal() as s:
                if not await check(s):
                    self.dropped.append(prompt)
                    self.log.append("  (dropped at delivery: the task had moved on)")
                    return
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
            self.tick_no += 1
            self.browser.polls_total = self.tick_no
            await runtime.tick(SessionLocal, self.browser, profile_for=profile_for, notify=self.notify, host_for=lambda f: self.host)
            t = await self.task()
            if t and until(t):
                return t
        return await self.task()

    async def task(self) -> Task | None:
        async with SessionLocal() as s:
            return (await s.execute(select(Task).where(Task.family_id == self.fam).order_by(Task.created_at.desc(), Task.updated_at.desc()))).scalars().first()

    async def tasks(self) -> list[Task]:
        async with SessionLocal() as s:
            return list((await s.execute(select(Task).where(Task.family_id == self.fam).order_by(Task.created_at))).scalars())

    async def ticks_any(self, until, most: int = 8) -> list[Task]:
        """Ticks until `until(every task of the family)` holds."""
        async def profile_for(task):
            return {"profileId": f"prof-{task.family_id}-{task.service}", "loginPhone": "9000011111"}

        for _ in range(most):
            clock.set_now(clock.now() + timedelta(seconds=10))
            self.tick_no += 1
            self.browser.polls_total = self.tick_no
            await runtime.tick(SessionLocal, self.browser, profile_for=profile_for, notify=self.notify, host_for=lambda f: self.host)
            ts = await self.tasks()
            if until(ts):
                return ts
        return await self.tasks()

    def check(self, ok: bool, what: str) -> None:
        if not ok:
            self.problems.append(what)

    async def setup(self):
        async with SessionLocal() as s:
            await store.save_roster(s, self.fam, self.family.elder, self.family.members)
            await s.commit()

    def sent_with_times(self) -> list[dict]:
        return self.messages_to(self.family.speaker["id"])

    def options_msgs(self) -> list[str]:
        """Messages to the person that list numbered options (a line starting with a number, in any script) before the one
        confirm. The sim's own check; production never reads words."""
        out = []
        for m in self.messages_to(self.family.speaker["id"]):
            text = str(m.get("text") or "")
            if any(str(b.get("id", "")).startswith("v2:od:") for b in m.get("buttons") or []):
                break  # the one confirm: everything after it is about that cart
            if any(re.match(r"\s*[*_•\-]*\s*[(\[]?\d{1,2}(?!\d)", ln) for ln in text.splitlines()):
                out.append(text)
        return out


def confirm_buttons(c: Conversation) -> list[str]:
    for m in reversed(c.maa_messages()):
        ids = [b.get("id") for b in m.get("buttons") or [] if str(b.get("id", "")).startswith("v2:od:")]
        if ids:
            return ids
    return []


def _offered(ts: list[Task]) -> dict[str, tuple[Task, dict]]:
    """The numbered options waiting for the person's pick, by their number (the tasks' own record)."""
    return {n: (t, o) for t in ts if t.status == "awaiting_confirm" and t.input_needed == "go"
            for n, o in ((t.details or {}).get("offer") or {}).items()}


async def _options_then_confirm(c: Conversation, pick: str, *, want: str = "s", stores: tuple[str, ...] = (), until_most: int = 14,
                                must_confirm: bool = True) -> Task | None:
    """Founder 2026-10-11: what was found goes out numbered, every store of a comparison in ONE update (with why a store has
    nothing); the person picks in their own words; that store builds the cart; then the one confirm with the amount.
    want: the option the pick means ("s" = the suggested one, else its number); stores: labels the update must report."""
    ts = await c.ticks_any(lambda ts: bool(_offered(ts)) and not any(t.status in ("queued", "running") for t in ts), most=until_most)
    offered = _offered(ts)
    c.check(bool(offered), f"no options offered ({[(t.service, t.status, t.input_needed) for t in ts]})")
    c.check(bool(c.options_msgs()), "the person was not shown numbered options")
    update = next((p for p in reversed(c.prompts) if "nothing is ordered" in p and ("Looked on" in p or "Found on" in p)), "")
    for label in stores:
        c.check(f"- {label}" in update, f"{label} not reported in the options update")
    target = (next((x for x in offered.values() if x[1].get("s")), None) if want == "s" else offered.get(want) if want.isdigit()
              else next((x for _, x in sorted(offered.items(), key=lambda kv: int(kv[0])) if x[0].service == want), None))
    c.check(target is not None, f"option {want} was not offered: {sorted(offered)}")
    await c.say(pick)
    t = await c.ticks(lambda t: t.status == "awaiting_confirm" and t.input_needed == "confirm", most=until_most)
    if target is not None:
        picked = next((x for x in await c.tasks() if x.id == target[0].id), None)
        got = {str(p.get("oid")) for p in ((picked.details or {}).get("chosen") or [])} if picked else set()
        c.check(set(target[1]["oids"]) <= got, f"'{pick}' did not pick option {want} ({target[1].get('text')}): got {got}")
    if must_confirm:
        c.check(t is not None and t.status == "awaiting_confirm" and t.input_needed == "confirm",
                f"no one confirm (task {t.service if t else None} {t.status if t else None}/{t.input_needed if t else None})")
        ids = confirm_buttons(c)
        c.check(len(ids) == 3, f"confirm buttons missing: {ids}")
        confirm = next((m for m in reversed(c.maa_messages()) if any(str(b.get("id", "")).startswith("v2:od:") for b in m.get("buttons") or [])), None)
        # the amount in any script: "₹331", "৮৩ টাকা", "₹ ८३"
        c.check(confirm is not None and any(ch.isdigit() for ch in str(confirm.get("text"))), f"the confirm has no amount: {confirm}")
    return t


async def _pick_again_if_asked(c: Conversation, pick: str) -> None:
    """After a change at the confirm, a changed item's options may come back: the person picks again."""
    ts = await c.ticks_any(lambda ts: bool(_offered(ts)) or any(t.status == "awaiting_confirm" and t.input_needed == "confirm" for t in ts), most=8)
    if _offered(ts):
        await c.say(pick)


async def several_items_tap_yes_tracked(c: Conversation):
    await c.setup()
    await c.say("Blinkit se 2 Amul doodh aur ek Charliee methi khakhra bhej do")
    await c.say("haan ghar pe hi")
    t = await _options_then_confirm(c, "jo aap suggest kar rahe ho wahi bhej do")
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
    await _options_then_confirm(c, "haan wahi peri peri wala")
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
    await _options_then_confirm(c, "doodh pehla wala, muruku rehne do")
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check(not any("Butter Muruku" in l["name"] for l in cart), f"another flavour went in silently: {cart}")
    old = confirm_buttons(c)
    await c.say("doodh ek hi kar do")
    await _pick_again_if_asked(c, "pehla wala")
    await c.ticks(lambda t: t.status == "awaiting_confirm" and t.input_needed == "confirm" and len(c.shop.carts) >= 2)
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check(any(l["qty"] == 1 and "Milk" in l["name"] for l in cart) and len(c.shop.carts) >= 2, f"change not made: {cart}")
    if old:
        await c.say(old[0])  # Yes on the first confirm, after the cart changed
        c.check(not c.shop.placed, "an old confirm's Yes placed the changed cart")
    new = confirm_buttons(c)
    if new and new != old:
        await c.say(new[0])
        await c.ticks(lambda t: t.status == "done")
        c.check(len(c.shop.placed) == 1, f"not placed once after the new Yes ({len(c.shop.placed)})")


async def late_order_tells_the_caregiver(c: Conversation):
    await c.setup()
    await c.say("blinkit se ek Amul butter mangwa do")
    await c.say("haan ghar")
    await _options_then_confirm(c, "haan wahi")
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
    # ten minutes later it is long overdue: asked again, still not here → Mohan is told
    clock.set_now(clock.now() + timedelta(minutes=11))
    c.check(await c.wake() == 1, "not asked again when it became late")
    await c.say("abhi bhi nahi aaya")
    before = len(c.messages_to(SON["id"]))
    await c.ticks(lambda t: True, most=1)
    c.check(len(c.messages_to(SON["id"])) > before, "the caregiver was not told about the late order")


async def three_stores_named_one_does_not_deliver_then_cancel_after_placing(c: Conversation):
    """The founder's own chat of 2026-10-11 (21:12-21:17), played for another family: Karachi Bakery biscuits on all three
    stores; one store does not deliver there. Every store is reported, the person picks one that is not the suggestion,
    and after it is placed she asks to cancel: never 'cancelled'; her caregiver is told to cancel it in the app."""
    await c.setup()
    c.shop.away.add("zepto")
    await c.say("Karachi Bakery ke fruit biscuit chahiye, Blinkit, Instamart aur Zepto teeno pe dekh lo")
    await c.say("haan ghar pe")
    await _options_then_confirm(c, "Blinkit wala le lo", want="2", stores=("Blinkit", "Swiggy Instamart", "Zepto"))
    t = await c.task()
    c.check(t is not None and t.service == "blinkit", f"not the store she picked: {t.service if t else None}")
    ids = confirm_buttons(c)
    if not ids:
        return
    await c.say(ids[0])
    t = await c.ticks(lambda t: t.status == "done", most=8)
    c.check(len(c.shop.placed) == 1 and t.status == "done", f"not placed once ({len(c.shop.placed)}, {t.status})")
    before = len(c.messages_to(SON["id"]))
    await c.say("arre cancel kar do, galti se ho gaya")
    t = await c.task()
    c.check(t.status == "done" and ((t.details or {}).get("cancel_ask") or {}).get("elder"), f"not kept as placed with the ask: {t.status}")
    c.check(not any(r.endswith(":cancel") for r in c.browser.agent_runs), "a cancel was run on the store")
    await c.ticks(lambda t: True, most=1)
    told = len(c.messages_to(SON["id"])) - before
    c.check(told == 1, f"the caregiver got {told} messages about cancelling it in the app, not one")


# --- every family, not one (founder 2026-10-11): cities, languages, places with and without a map point, self-care ---------

BANGALORE = Family("Bangalore, English, a son orders for his mother, place saved WITHOUT a map point (found at order time)",
                   elder={"id": "elder-blr", "name": "Lakshmi", "role": "elder"},
                   carer={"id": "son-blr", "name": "Arjun", "role": "primary caregiver"}, speaker="carer",
                   place={"addressId": "addr-blr", "nickname": "Home", "pincode": "560011",
                          "full": "21 5th main 4th block jayanagar south bangalore karnataka india, 560011"},
                   heal={"lat": 12.9299, "lng": 77.5826, "geoPrecision": "pincode"}, home_words=("jayanagar",))
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


PUNE = Family("Pune, English, a son ordering for himself; his mother in Nagpur talks to Saheli too",
              elder={"id": "elder-ngp", "name": "Sudha", "role": "elder"},
              carer={"id": "son-pnq", "name": "Nikhil", "role": "primary caregiver"}, speaker="carer",
              place={"addressId": "addr-ngp", "nickname": "Ghar", "pincode": "440010", "full": "7 Sample Nagar, Nagpur, Maharashtra 440010",
                     "lat": 21.13, "lng": 79.07, "geoPrecision": "address"},
              carer_place={"addressId": "addr-pnq", "nickname": "Pune flat", "pincode": "411038",
                           "full": "B-402 Sample Heights, Kothrud, Pune, Maharashtra 411038", "lat": 18.50, "lng": 73.81,
                           "geoPrecision": "address"})


def _script(text: str, lo: int, hi: int) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text)


async def _pin_if_asked(c: Conversation, pin: dict | None, *, rather: str = "") -> None:
    """Saheli asked for the pickup as a WhatsApp location (founder 2026-10-12): share it, or say to use the saved address."""
    c.check(c.location_asked, "the pickup was not asked as a WhatsApp location")
    if c.location_asked:
        await (c.say("", location=pin) if pin else c.say(rather))


def _points_sent(c: Conversation) -> None:
    empty = [s for s in c.shop.searched if fastpath.SITES[s[0]].needs_point and not fastpath.has_point(s[1], s[2])]
    c.check(not empty, f"a store's own search was sent without the map point: {empty}")


async def bangalore_ritebite_no_store_named(c: Conversation):
    """The order of 2026-10-11: Saheli went quiet for 6 minutes. Every store's options come in one update; he picks. Whom
    it is for is not in his first words: asked with the address in one line (2026-10-12), so the answer says both."""
    await c.setup()
    await c.say("Hey can u order a rite bite choco berry")
    await c.say("Yes please order 4 of it, it's for Amma")
    t = await _options_then_confirm(c, "the first one please", want="1", stores=("Blinkit", "Swiggy Instamart", "Zepto"))
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
    await _options_then_confirm(c, "पहला वाला", want="1", stores=("Blinkit", "Swiggy Instamart", "Zepto"))
    _points_sent(c)
    shown = c.options_msgs()[-1] if c.options_msgs() else ""
    c.check(_script(shown, 0x0900, 0x097F), f"the options are not in Hindi: {shown}")
    confirm = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    c.check(_script(str(confirm), 0x0900, 0x097F), f"the confirm is not in Hindi: {confirm}")
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check([(l["name"], l["qty"]) for l in cart] == [("Mother Dairy Toned Milk", 2)], f"wrong cart: {cart}")
    await c.say("नहीं, रहने दो")
    t = await c.task()
    c.check(t.status == "cancelled" or t.cancel_requested, f"not cancelled ({t.status})")
    c.check(not c.shop.placed, "placed after a no")


async def mumbai_marathi_self_care_named_store_lacks_it(c: Conversation):
    """A store they named does not have it: the other stores look too, and every store's options come in one update (the
    named one first, said plainly); she says yes to the suggested one, then the one confirm."""
    await c.setup()
    await c.say("झेप्टो वरून दोन पाकीट गोकुळ दूध पाठवा")
    await c.say("हो, घरीच")
    await _options_then_confirm(c, "हो, चालेल", want="s", stores=("Blinkit", "Swiggy Instamart", "Zepto"))
    update = next((p for p in reversed(c.prompts) if "Looked on" in p), "")
    c.check("They asked for Zepto" in update, f"the store she named is not said first: {update[:160]}")
    cart = c.shop.carts[-1] if c.shop.carts else []
    c.check([(l["name"], l["qty"]) for l in cart] == [("Gokul Full Cream Milk", 2)], f"wrong cart: {cart}")
    ids = confirm_buttons(c)
    if ids:
        await c.say(ids[0])
        t = await c.ticks(lambda t: t.status == "done", most=8)
        c.check(len(c.shop.placed) == 1, f"not placed once ({len(c.shop.placed)})")
        # self care: asking to cancel a placed order → she cancels it in the store's app (nobody else to tell)
        await c.say("रद्द करा ना, चुकून झाली ऑर्डर")
        t = await c.ticks(lambda t: t.status == "cancelled", most=4)
        c.check(t.status == "cancelled" and c.shop.cancelled == ["blinkit"], f"her own cancel was not done on Blinkit: {t.status} {c.shop.cancelled}")


async def chennai_tamil_no_point_browser_agent_looks(c: Conversation):
    await c.setup()
    await c.say("ஆவின் பால் இரண்டு பாக்கெட் வேண்டும்")
    await c.say("ஆமாம், வீட்டுக்கே அனுப்புங்க")
    await _options_then_confirm(c, "முதல் ஒன்று", want="1", stores=("Blinkit", "Swiggy Instamart", "Zepto"), until_most=16)
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
        await c.ticks(lambda t: t.status == "done", most=8)
        c.check(len(c.shop.placed) == 1, f"not placed once ({len(c.shop.placed)})")


async def kolkata_bengali_slow_store_not_waited_for(c: Conversation):
    """One store's saved step is broken and its browser agent takes 5 minutes: the others answer in seconds, and their
    options go out within about 2 minutes with the slow store reported as too slow; nobody waits in silence."""
    await c.setup()
    c.shop.broken.add("zepto")
    c.browser.slow["zepto"] = 30
    await c.say("দুই প্যাকেট মারি বিস্কুট পাঠাও")
    await c.say("হ্যাঁ, বাড়িতেই পাঠাও")
    asked_at = c.said_at
    t = await _options_then_confirm(c, "প্রথমটা", want="1", stores=("Blinkit", "Swiggy Instamart", "Zepto"), until_most=20)
    waited = (clock.now() - asked_at).total_seconds()
    c.check(waited <= 240, f"the options and the confirm took {waited:.0f} s of a slow store's time")
    update = next((p for p in reversed(c.prompts) if "Looked on" in p), "")
    c.check("- Zepto: its website did not answer in time" in update, f"the slow store is not reported: {update[:200]}")
    confirm = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    c.check(_script(str(confirm), 0x0980, 0x09FF), f"the confirm is not in Bengali: {confirm}")
    c.check(t is not None and t.service in ("blinkit", "instamart"), f"picked the slow store: {t.service if t else None}")


# --- the founder's rules everywhere (2026-10-11): every service, store mix, elder and caregiver; one regression per root cause


def _digits(text: str) -> str:
    """The digits of a message in any script (₹१३१ → 131): the sim's own check."""
    import unicodedata

    return "".join(str(unicodedata.digit(ch)) if unicodedata.digit(ch, None) is not None else ch for ch in str(text or ""))


def _idle(ts: list[Task]) -> bool:
    return not any(t.status in ("queued", "running") for t in ts)


async def delhi_ride_cheapest_then_cancel_after_booking(c: Conversation):
    """Options (rides: the fares), every app reported, one confirm with the fare, cancel after booking: an elder in Hindi
    asks for the cheapest auto; every ride app is looked at; the one with no ride is said with why; she picks Ola's auto;
    the one confirm names the fare now; booked; then she asks to cancel: never "cancelled", her son is told once."""
    await c.setup()
    await c.say("मुझे घर से एम्स अस्पताल जाना है, सबसे सस्ता ऑटो बुला दो")
    await _pin_if_asked(c, {"lat": 28.5676, "lng": 77.2369})
    ts = await c.ticks_any(lambda ts: bool(_offered(ts)) and _idle(ts), most=10)
    c.check(any(((t.details or {}).get("pickup_place") or {}).get("lat") == 28.5676 for t in ts), "her pin is not the pickup")
    c.check([s_.get("place") for s_ in c.host.saves] == ["Ghar"], f"her pin was not saved to Ghar: {c.host.saves}")
    offered = _offered(ts)
    c.check({t.service for t in ts} == {"uber", "ola", "rapido"}, f"not every ride app looked: {sorted(t.service for t in ts)}")
    update = next((p for p in reversed(c.prompts) if "Looked on" in p), "")
    for label in ("Ola", "Rapido", "Uber"):
        c.check(f"- {label}" in update, f"{label} not reported in the fares update")
    c.check(bool(c.options_msgs()), "the fares were not shown numbered")
    c.check(any(t.service == "ola" and "Auto" in o["text"] for t, o in offered.values()), f"Ola's auto not offered: {offered}")
    await c.say("ओला वाला ऑटो बुक कर दो")
    ts = await c.ticks_any(lambda ts: any(t.service == "ola" and t.input_needed == "choice" for t in ts), most=8)
    ola = next((t for t in ts if t.service == "ola"), None)
    c.check(ola is not None and ola.input_needed == "choice", f"no one confirm with the fare: {[(t.service, t.status, t.input_needed) for t in ts]}")
    confirm = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    c.check("131" in _digits(confirm), f"the one confirm does not name the fare now (₹131): {confirm}")
    await c.say("हाँ, बुक कर दो")
    await c.ticks_any(lambda ts: any(t.service == "ola" and t.status == "done" for t in ts), most=8)
    c.check([r["service"] for r in c.shop.rides] == ["ola"], f"not booked once on Ola: {c.shop.rides}")
    carer = c.family.carer["id"]
    before = len(c.messages_to(carer))
    await c.say("अरे रद्द कर दो, मुझे अब नहीं जाना")
    await c.ticks_any(lambda ts: True, most=1)
    ola = next(t for t in await c.tasks() if t.service == "ola")
    c.check(ola.status == "done" and ((ola.details or {}).get("cancel_ask") or {}).get("elder"), f"not kept booked with her ask: {ola.status}")
    told = len(c.messages_to(carer)) - before
    c.check(told == 1, f"her son got {told} messages to cancel the ride in the app, not one")
    c.check(not any(r.endswith(":cancel") for r in c.browser.agent_runs), "a cancel was run on the ride app")


async def chennai_ride_from_home_the_app_set_the_city_first(c: Conversation):
    """Live 2026-10-12 (another family): the pickup went to the ride app as words, the app set the city, the ride was booked
    there, and Saheli said it was their address. A Tamil elder books an auto on Uber from home: the app gets the saved
    address; the first logged-in look sets the city, so it is set again (its fares never reach her); booked from home; she
    asks whether it comes to her house and hears where the app set it."""
    await c.setup()
    c.shop.city_pickup = "Chennai, Tamil Nadu"
    await c.say("வீட்டிலிருந்து அப்பல்லோ ஆஸ்பத்திரிக்கு ஊபர்ல ஆட்டோ புக் பண்ணுங்க")
    await _pin_if_asked(c, None, rather="லொகேஷன் அனுப்ப தெரியாது, வீட்டு முகவரியையே பயன்படுத்துங்க")
    ts = await c.ticks_any(lambda ts: bool(_offered(ts)) and _idle(ts), most=10)
    uber = next((t for t in ts if t.service == "uber"), None)
    full = c.family.place["full"]
    c.check(uber is not None and ((uber.details or {}).get("pickup_place") or {}).get("full") == full,
            f"the pickup is not the saved home: {(uber.details or {}).get('pickup_place') if uber else None}")
    await c.say("ஆட்டோ")
    ts = await c.ticks_any(lambda ts: any(t.service == "uber" and t.input_needed == "choice" for t in ts), most=10)
    prepares = [g for k, g in c.browser.goals if k == "uber:prepare"]
    c.check(len(prepares) == 2 and "The last run set the pickup to 'Chennai, Tamil Nadu'" in (prepares[-1] if prepares else ""),
            f"the pickup was not set again after the app chose the city: {len(prepares)} looks")
    c.check(not any("192" in _digits(m.get("text")) for m in c.maa_messages()), "a fare from the wrong pickup reached her")
    await c.say("சரி, புக் பண்ணுங்க")
    await c.ticks_any(lambda ts: any(t.service == "uber" and t.status == "done" for t in ts), most=8)
    c.check([r["pickup"] for r in c.shop.rides] == [full], f"not booked once from home: {c.shop.rides}")
    booked = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    # founder 2026-10-12: the booking details never reached him; every one goes out (the PIN, the plate, the driver)
    c.check("4931" in _digits(booked) and "1234" in _digits(booked), f"the booked message lacks the ride PIN or the plate: {booked}")
    reply = await c.say("வண்டி என் வீட்டுக்கே வருமா? எந்த அட்ரஸ் போட்டிருக்கீங்க?")
    c.check(any(w in reply for w in ("Nagar", "நகர்", "Flat 9", "600017")), f"she did not hear the address the app set: {reply}")


async def bangalore_ride_he_cancelled_it_himself_in_the_app(c: Conversation):
    """Live 2026-10-12: he cancelled a booked ride in the app himself and said so; the task still read booked (counted in the
    month's spend; "did the cab come?" due next morning). A Bangalore son books an Ola cab for himself, then says he cancelled
    it in the Ola app: it is recorded, nothing is run on the app, and nothing more is asked."""
    await c.setup()
    await c.say("Book me an Ola mini from home to Bangalore airport")
    await _pin_if_asked(c, {"lat": 12.9301, "lng": 77.5828})
    await c.ticks_any(lambda ts: bool(_offered(ts)) and _idle(ts), most=10)
    await c.say("the mini")
    await c.ticks_any(lambda ts: any(t.service == "ola" and t.input_needed == "choice" for t in ts), most=10)
    await c.say("yes book it")
    await c.ticks_any(lambda ts: any(t.service == "ola" and t.status == "done" for t in ts), most=8)
    c.check([r["service"] for r in c.shop.rides] == ["ola"], f"not booked once on Ola: {c.shop.rides}")
    booked = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    c.check("4931" in _digits(booked), f"the booked message lacks the ride PIN: {booked}")
    await c.say("I just cancelled it myself in the Ola app, plans changed")
    ola = next(t for t in await c.tasks() if t.service == "ola")
    c.check(ola.status == "cancelled" and ((ola.details or {}).get("in_app") or {}).get("what") == "cancelled",
            f"his own cancel in the app was not recorded: {ola.status} {(ola.details or {}).get('in_app')}")
    c.check(not any(r.endswith(":cancel") for r in c.browser.agent_runs), "a cancel was run on the app he already cancelled in")
    before = len(c.host.world.sent)
    await c.ticks_any(lambda ts: True, most=2)
    clock.set_now(clock.now() + timedelta(hours=3))  # past the "did the cab come?" time
    await c.wake()
    c.check(len(c.host.world.sent) == before, f"something more was said about it: {c.host.world.sent[before:]}")


async def bangalore_ola_for_tomorrow_the_look_up_booked_it(c: Conversation):
    """Founder 2026-10-12, 21:52 IST (his own chat; played for another family): an Ola for the next afternoon; Ola shows no
    fares without a login; the logged-in fare look-up pressed "Confirm & Book" and an auto came to his door; its report
    left it out, and Saheli said "nothing was booked". Now the booking is found on the run's own pages, owned in the next
    message, cancelled on Ola, never denied, and nothing is booked again."""
    await c.setup()
    c.shop.fares_need_login.add("ola")
    c.shop.books_on_look.add("ola")
    await c.say("Book me an Ola from home to Bangalore airport tomorrow at 3 pm, it's for me")
    await _pin_if_asked(c, {"lat": 12.9301, "lng": 77.5828})
    await c.ticks_any(lambda ts: any(t.service == "ola" and t.input_needed == "go" for t in ts) and _idle(ts), most=10)
    seen = len(c.maa_messages())
    await c.say("Ok go ahead")
    await c.ticks_any(lambda ts: any(t.service == "ola" and t.status in ("cancelled", "failed", "done") for t in ts) and _idle(ts), most=14)
    ola = next(t for t in await c.tasks() if t.service == "ola")
    c.check(bool((ola.result or {}).get("by_mistake") and (ola.result or {}).get("booked")), f"the booking was not recorded: {ola.result}")
    c.check("ola" in c.shop.cancelled and ola.status == "cancelled", f"not cancelled on Ola: {ola.status} {c.shop.cancelled}")
    said = " ".join(m.get("text") or "" for m in c.maa_messages()[seen:]).lower()
    c.check("mistake" in said, f"the mistake was not owned: {said[-400:]}")
    c.check(not any(x in said for x in ("nothing was booked", "nothing has been booked", "nothing is booked")), f"denied: {said[-400:]}")
    c.check(not any(r.endswith(":place") for r in c.browser.agent_runs), "booked again")
    reply = (await c.say("You booked it, I had to cancel it myself from the app")).lower()
    c.check(not any(x in reply for x in ("nothing was booked", "nothing has been booked", "was not booked", "wasn't booked")),
            f"denied again: {reply}")


async def kolkata_cab_for_6_tomorrow(c: Conversation):
    """Founder 2026-10-12: "book a cab at xyz time ... just book it that time ... send the booked details". A Bengali elder
    asks at night for an Uber at 6 tomorrow from home to Howrah station: the pickup as a location, the fares now, her pick,
    ONE question with everything and the most she will pay, then nothing until its time; the fare is looked at 25 minutes
    before; booked without asking; every detail goes to her."""
    from app.tasks import later

    await c.setup()
    clock.set_now(datetime(2026, 10, 11, 20, 0, tzinfo=clock.IST))
    await c.say("কাল সকাল ৬টায় বাড়ি থেকে হাওড়া স্টেশন যাব, উবার বুক করে দাও")
    await _pin_if_asked(c, {"lat": 22.5188, "lng": 88.3641})
    ts = await c.ticks_any(lambda ts: bool(_offered(ts)) and _idle(ts), most=10)
    uber = next((t for t in ts if t.service == "uber"), None)
    c.check(uber is not None and later.of(uber).get("at", "").startswith("2026-10-12T06:00"), f"not set for 6 tomorrow: {later.of(uber) if uber else None}")
    await c.say("অটোটা")
    t = await c.task()
    c.check(t.input_needed == "later", f"no one question for later after her pick: {t.status}/{t.input_needed}")
    tap = next((b["id"] for b in c.last_buttons if b["id"].startswith("v2:lt:yes")), None)
    c.check(tap is not None, f"the question for later has no one-tap answers: {c.last_buttons}")
    await c.say(tap or "হ্যাঁ, ঠিক আছে")
    t = next(x for x in await c.tasks() if x.service == "uber")
    lt = later.of(t)
    c.check(t.status in ("scheduled", "queued", "needs_input") and lt.get("max"), f"not set for later with a limit: {t.status} {lt}")
    c.check(not c.shop.rides, "booked before its time")
    if lt.get("login_at"):
        clock.set_now(later.parse_at(lt["login_at"]) + timedelta(minutes=1))
        await c.ticks_any(lambda ts: all(x.status == "scheduled" for x in ts if x.service == "uber"), most=6)
    clock.set_now(later.parse_at(lt["book_at"]) - later.CHECK_BEFORE + timedelta(minutes=1))
    await c.ticks_any(lambda ts: any(later.of(x).get("state") == "ready" for x in ts), most=8)
    t = next(x for x in await c.tasks() if x.service == "uber")
    clock.set_now(later.parse_at(later.of(t)["book_at"]) + timedelta(seconds=30))
    await c.ticks_any(lambda ts: any(x.service == "uber" and x.status == "done" for x in ts), most=8)
    c.check([r["service"] for r in c.shop.rides] == ["uber"], f"not booked once at its time: {c.shop.rides}")
    booked = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    c.check("4931" in _digits(booked), f"the booked message lacks the ride PIN: {booked}")


async def mumbai_milk_for_7_tomorrow(c: Conversation):
    """"order milk at 7 tomorrow": a Marathi self-care lady orders at night for the morning; the options now, ONE question
    with the most she will pay; at 06:23 the cart is built (within her limit: nothing to say); at 06:50 it is placed and every
    detail goes to her."""
    from app.tasks import later

    await c.setup()
    clock.set_now(datetime(2026, 10, 11, 21, 0, tzinfo=clock.IST))
    await c.say("उद्या सकाळी ७ वाजता दोन पाकीट गोकुळ दूध पाठवा")
    await c.say("हो, घरीच")
    ts = await c.ticks_any(lambda ts: bool(_offered(ts)) and _idle(ts), most=10)
    c.check(any(later.of(t).get("at", "").startswith("2026-10-12T07:00") for t in ts), "not set for 7 tomorrow")
    await c.say("पहिला")
    t = next((x for x in await c.tasks() if x.input_needed == "later"), None) or await c.task()
    c.check(t.input_needed == "later", f"no one question for later after her pick: {t.status}/{t.input_needed}")
    await c.say("हो चालेल, जास्तीत जास्त १०० रुपये")  # her own limit, in words
    t = next((x for x in await c.tasks() if later.of(x).get("by")), None)
    c.check(t is not None and later.of(t).get("max") == 100, f"her own limit was not kept: {later.of(t) if t else None}")
    t = next((x for x in await c.tasks() if later.of(x).get("by")), None)
    c.check(t is not None and t.status in ("scheduled", "queued", "needs_input"), f"not set for later: {t.status if t else None}")
    if t is None:
        return
    lt = later.of(t)
    if lt.get("login_at"):
        clock.set_now(later.parse_at(lt["login_at"]) + timedelta(minutes=1))
        await c.ticks_any(lambda ts: all(x.status == "scheduled" for x in ts if x.id == t.id), most=6)
    clock.set_now(later.parse_at(lt["book_at"]) - later.CHECK_BEFORE + timedelta(minutes=1))
    await c.ticks_any(lambda ts: any(later.of(x).get("state") == "ready" for x in ts), most=8)
    c.check(not c.shop.placed, "placed before its time")
    clock.set_now(later.parse_at(lt["book_at"]) + timedelta(seconds=30))
    await c.ticks_any(lambda ts: any(x.id == t.id and x.status == "done" for x in ts), most=8)
    c.check(len(c.shop.placed) == 1, f"not placed once at its time: {c.shop.placed}")
    placed = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    c.check("SIM1" in placed or "1" in _digits(placed), f"no order details: {placed}")


async def pune_son_orders_for_himself_maa_hears_nothing(c: Conversation):
    """Founder 2026-10-12, from his mother's phone: his own Karachi Bakery biscuits, ordered from his own chat to his own
    home, were recorded as hers and she was asked twice whether they came ("it's confusing"). A Pune son orders biscuits for
    himself; his mother in Nagpur talks to Saheli too. The order is his, to his own flat; the follow-up asks him; she hears
    nothing about it at any point."""
    await c.setup()
    maa = c.family.elder["id"]
    await c.say("Namaste beta, aaj thandi hai", who=c.family.elder)  # she talks to Saheli (the old rule then asked her)
    heard = len(c.messages_to(maa))
    await c.say("Order 2 packs of Karachi Bakery fruit biscuits for me from Instamart")
    await c.say("yes, my Pune flat")
    t = await _options_then_confirm(c, "the first one")
    son = c.family.carer["id"]
    c.check(t is not None and t.subject_id == son and t.requested_by == son, f"not recorded as his own order: {t.subject_id if t else None}")
    place = (((t.details or {}).get("limits") or {}).get("place") or {}) if t else {}
    c.check(place.get("nickname") == c.family.carer_place["nickname"], f"not to his own flat: {place.get('nickname')}")
    ids = confirm_buttons(c)
    if not ids:
        return
    await c.say(ids[0])
    t = await c.ticks(lambda t: t.status == "done", most=8)
    oid = (t.result or {}).get("order_id")
    c.shop.status[oid] = {"status": "DELIVERED", "text": "Delivered at your door"}
    clock.set_now(clock.now() + timedelta(minutes=20))
    await c.ticks(lambda t: (t.result or {}).get("delivered"), most=2)
    clock.set_now(clock.now() + timedelta(hours=4))
    await c.wake()
    clock.set_now(clock.now() + timedelta(hours=4))
    await c.wake()
    c.check(len(c.messages_to(maa)) == heard, f"his mother heard about his own order: {[m.get('text') for m in c.messages_to(maa)[heard:]]}")
    asks = [m.get("text") for m in c.messages_to(son) if "?" in str(m.get("text"))]
    c.check(bool(asks), "nobody was asked whether it came")


async def bangalore_food_two_apps_named_caregiver_cancels_in_app(c: Conversation):
    """Food and a caregiver: the two apps he named are both reported (dishes from restaurants as options); he picks the
    Zomato one; after it is placed he asks to cancel: he is pointed to the Zomato app and nobody else is messaged."""
    await c.setup()
    await c.say("Check Swiggy and Zomato for one paneer butter masala for Amma")
    await c.say("yes, home")
    t = await _options_then_confirm(c, "the Zomato one please", want="zomato", stores=("Swiggy", "Zomato"))
    c.check({x.service for x in await c.tasks()} == {"swiggy", "zomato"}, "a store he did not name was searched")
    c.check(t is not None and t.service == "zomato", f"not the app he picked: {t.service if t else None}")
    ids = confirm_buttons(c)
    if not ids:
        return
    await c.say(ids[0])
    t = await c.ticks(lambda t: t.status == "done", most=8)
    c.check(len(c.shop.placed) == 1 and t.status == "done", f"not placed once ({len(c.shop.placed)}, {t.status})")
    before = len(c.prompts)
    c.shop.cancel_mode["zomato"] = "refuse"
    await c.say("Actually cancel it, she already ate")
    await c.ticks(lambda t: (t.details or {}).get("cancel_ask", {}).get("state") not in ("cancelling", None), most=4)
    t = await c.task()
    c.check(t.status == "done" and (t.details or {}).get("cancel_ask", {}).get("state") == "refused",
            f"his cancel after placing went to Zomato, which refused: {t.status} {(t.details or {}).get('cancel_ask')}")
    c.check(any(r == "zomato:cancel" for r in c.browser.agent_runs), "Zomato was not asked to cancel it")
    c.check(not [p for p in c.prompts[before:] if p.startswith("[Order needs you]")], "someone else was messaged about his own cancel")


async def chennai_tamil_medicine_exact_strength_only(c: Conversation):
    """Medicines: every pharmacy reported; only the exact medicine and strength asked is offered (never Dolo 500 or another
    brand); she picks; one confirm with the amount; placed."""
    await c.setup()
    await c.say("டோலோ 650 மாத்திரை ஒரு அட்டை வேண்டும்")
    await c.say("ஆமாம், வீட்டுக்கே அனுப்புங்க")
    ts = await c.ticks_any(lambda ts: bool(_offered(ts)) and _idle(ts), most=10)
    texts = [o["text"] for _, o in _offered(ts).values()]
    c.check(texts and all("Dolo 650" in x for x in texts), f"a medicine other than the one asked was offered: {texts}")
    await _options_then_confirm(c, "முதல் ஒன்று", want="1", stores=("Apollo 24|7 Pharmacy", "PharmEasy", "Tata 1mg"))
    ids = confirm_buttons(c)
    if ids:
        await c.say(ids[0])
        await c.ticks(lambda t: t.status == "done", most=8)
        c.check(len(c.shop.placed) == 1, f"not placed once ({len(c.shop.placed)})")


async def kolkata_two_of_three_named_login_code_asked_at_once(c: Conversation):
    """Only the two stores she named are searched and both are reported; she picks Zepto, whose account needs a login
    code: the ask goes out in the same step the store sends the code; then the one confirm."""
    await c.setup()
    c.shop.needs_code.add("zepto")
    await c.say("শুধু ব্লিংকিট আর জেপ্টোতে দুই প্যাকেট মারি বিস্কুট দেখো")
    await c.say("হ্যাঁ, বাড়িতেই পাঠাও")
    ts = await c.ticks_any(lambda ts: bool(_offered(ts)) and _idle(ts), most=10)
    c.check({t.service for t in ts} == {"blinkit", "zepto"}, f"stores searched: {sorted(t.service for t in ts)}")
    update = next((p for p in reversed(c.prompts) if "Looked on" in p), "")
    c.check("- Blinkit" in update and "- Zepto" in update and "Instamart" not in update, f"not exactly the stores she named: {update[:200]}")
    await c.say("জেপ্টোরটা নাও")
    ts = await c.ticks_any(lambda ts: any(t.service == "zepto" and t.input_needed == "otp" for t in ts), most=8)
    code_tick = c.tick_no
    asks = [k for k, person, p in c.prompt_ticks if person == c.family.speaker["id"] and "login code" in p]
    c.check(bool(asks) and asks[-1] == code_tick, f"the login-code ask went out in step {asks[-1] if asks else None}; the store sent it in step {code_tick}")
    await c.say("কোড 4821")
    ts = await c.ticks_any(lambda ts: any(t.service == "zepto" and t.input_needed == "confirm" for t in ts), most=8)
    c.check(any(t.service == "zepto" and t.input_needed == "confirm" for t in ts), "no confirm after the login code")
    ids = confirm_buttons(c)
    if ids:
        await c.say(ids[0])
        await c.ticks(lambda t: t.status == "done", most=8)
        c.check(len(c.shop.placed) == 1, f"not placed once ({len(c.shop.placed)})")


async def kolkata_first_code_did_not_go_in(c: Conversation):
    """Live 2026-10-13 (PharmEasy): the store's code boxes did not take the code; the person was asked for a code six times.
    Now the first code that does not go in is asked for once more, plainly, and the second goes through to the confirm."""
    await c.setup()
    c.shop.needs_code.add("zepto")
    c.shop.codes_refused["zepto"] = 1
    await c.say("জেপ্টো থেকে দুই প্যাকেট মারি বিস্কুট পাঠাও")
    await c.say("হ্যাঁ, বাড়িতেই পাঠাও")
    await c.ticks_any(lambda ts: bool(_offered(ts)) and _idle(ts), most=10)
    await c.say("প্রথমটা নাও")
    await c.ticks_any(lambda ts: any(t.service == "zepto" and t.input_needed == "otp" for t in ts), most=8)
    seen = len(c.maa_messages())
    await c.say("কোড 4821")
    ts = await c.ticks_any(lambda ts: any(t.service == "zepto" and t.input_needed == "otp" and int((t.details or {}).get("code_fails") or 0) == 1
                                          for t in ts) and _idle(ts), most=8)
    c.check(any(int((t.details or {}).get("code_fails") or 0) == 1 for t in ts), "the code that did not go in was not counted")
    c.check(len(c.maa_messages()) > seen, "she was not told the code did not go in")
    await c.say("নতুন কোড 7390")
    ts = await c.ticks_any(lambda ts: any(t.service == "zepto" and t.input_needed == "confirm" for t in ts), most=8)
    c.check(any(t.service == "zepto" and t.input_needed == "confirm" for t in ts), "no confirm after the second code")
    asks = [p for k, person, p in c.prompt_ticks if person == c.family.speaker["id"] and "code" in p and "ask" in p.lower()]
    c.check(len(asks) <= 2, f"asked for a code {len(asks)} times")


async def raipur_approval_at_once_then_cancel_while_placing(c: Conversation):
    """Above the family's limit for her: her son's approval is asked in the first step after her yes; he approves; while it
    is being placed she asks to cancel: she hears it cannot be stopped, then ONE message with the outcome; he is told once
    to cancel it in the app; never "cancelled"."""
    from app.care import boundaries

    await c.setup()
    async with SessionLocal() as s:
        await boundaries.save(s, c.fam, {"elder_order_limit": 100}, by=SON["id"])
        await s.commit()
    c.browser.slow["zepto:place"] = 3
    await c.say("Zepto se 4 packet Marie biscuit bhej do")
    await c.say("haan ghar pe")
    await _options_then_confirm(c, "haan wahi bhej do", want="s")
    ids = confirm_buttons(c)
    if not ids:
        return
    await c.say(ids[0])
    t = await c.task()
    c.check(t.input_needed == "approve", f"no approval asked: {t.status}/{t.input_needed}")
    step = c.tick_no + 1
    await c.ticks(lambda t: True, most=1)
    asks = [k for k, person, p in c.prompt_ticks if person == SON["id"] and p.startswith("[Approval needed]")]
    c.check(bool(asks) and asks[0] == step, f"the approval ask went out in step {asks[:1]}, not the first step after her yes ({step})")
    await c.say("haan theek hai, mangwa do", who=SON)
    t = await c.ticks(lambda t: t.phase == "place" and t.status == "running", most=4)
    c.check(t.phase == "place" and t.status == "running", f"not placing: {t.status}/{t.phase}")
    before_her, before_son = len(c.maa_messages()), len(c.messages_to(SON["id"]))
    reply = await c.say("arre ruko, cancel kar do")
    c.check(not any(w in reply.lower() for w in ("cancel kar diya", "cancel ho gaya", "cancelled")), f"said cancelled: {reply}")
    t = await c.ticks(lambda t: t.status == "done", most=6)
    c.check(t.status == "done" and len(c.shop.placed) == 1, f"placing did not finish once ({t.status}, {len(c.shop.placed)})")
    after = c.maa_messages()[before_her:]
    c.check(len(after) == 1, f"{len(after)} updates to her after her cancel, not one: {[m.get('text') for m in after]}")
    son = c.messages_to(SON["id"])[before_son:]
    c.check(len(son) == 1, f"her son got {len(son)} messages about her cancel, not one")
    c.shop.cancel_mode["zepto"] = "fee25"
    await c.say("haan cancel kar do", who=SON)
    t = await c.ticks(lambda t: t.input_needed == "fee", most=4)
    c.check(t.input_needed == "fee", f"the ₹25 fee was not asked: {t.status}/{t.input_needed}")
    fee_asks = [m for m in c.messages_to(SON["id"]) if re.search(r"(?<![\d.])25(?![\d])", _digits(m.get("text")))]
    c.check(len(fee_asks) == 1, f"the fee was asked {len(fee_asks)} times, not once")
    await c.say("haan theek hai, 25 de denge", who=SON)
    t = await c.ticks(lambda t: t.status == "cancelled", most=4)
    c.check(t.status == "cancelled" and c.shop.cancelled == ["zepto"], f"not cancelled on Zepto with the fee: {t.status} {c.shop.cancelled}")


async def bangalore_1mg_closest_named_then_free_cancel(c: Conversation):
    """A live 1mg order (2026-10-12), played for a made-up family: no exact product, so the closest is offered marked as not
    exactly it; the confirm and the "placed" message name what was bought, never his words; then he asks to cancel and 1mg
    shows a ₹0 fee: it goes through without a second question, and the one result follows."""
    await c.setup()
    c.shop.cancel_mode["1mg"] = "fee0"
    await c.say("Order magnesium citrate tablets from Tata 1mg for Amma")
    await c.say("yes, home")
    ts = await c.ticks_any(lambda ts: bool(_offered(ts)) and _idle(ts), most=8)
    c.check(bool(_offered(ts)), "no options offered")
    await c.say("ok the first one")
    await c.ticks(lambda t: t.status == "awaiting_confirm" and t.input_needed == "confirm", most=8)
    ids = confirm_buttons(c)
    c.check(len(ids) == 3, f"confirm buttons missing: {ids}")
    if not ids:
        return
    await c.say(ids[0])
    t = await c.ticks(lambda t: t.status == "done", most=8)
    placed = c.maa_messages()[-1].get("text") if c.maa_messages() else ""
    c.check("calcium" in placed.lower(), f"the placed message does not name what was bought: {placed}")
    before = len(c.maa_messages())
    await c.say("Please cancel that 1mg order")
    t = await c.ticks(lambda t: t.status in ("cancelled", "done") and (t.details or {}).get("cancel_ask", {}).get("state") in ("cancelled", "refused"), most=6)
    c.check(t.status == "cancelled" and c.shop.cancelled == ["1mg"], f"not cancelled on 1mg: {t.status} {c.shop.cancelled}")
    after = c.maa_messages()[before:]
    c.check(len(after) == 1, f"{len(after)} messages after his cancel, not one result: {[m.get('text') for m in after]}")
    c.check(not any("₹0" in str(m.get("text")) or "fee" in str(m.get("text")).lower() for m in after), "a free cancel was asked about")


async def mumbai_marathi_cancel_while_the_cart_is_built(c: Conversation):
    """One true status: the cart becomes ready while she says cancel; that update waits for her turn and is dropped at
    delivery, so she hears only that it was stopped and nothing was ordered."""
    await c.setup()
    await c.say("ब्लिंकिटवरून दोन पाकीट अमूल ताजा दूध पाठवा")
    await c.say("हो, घरीच")
    await c.ticks_any(lambda ts: bool(_offered(ts)) and _idle(ts), most=8)
    await c.say("पहिला")
    c.hold = True
    await c.ticks(lambda t: t.status == "awaiting_confirm" and t.input_needed == "confirm", most=6)
    c.check(bool(c.held), "no update was waiting (the cart was not ready)")
    await c.say("नको, रद्द करा, आत्ता नको")
    await c.release()
    t = await c.task()
    c.check(t.status == "cancelled", f"not stopped: {t.status}")
    c.check(len(c.dropped) == 1, f"{len(c.dropped)} updates dropped at delivery; the stale cart update should be")
    c.check(not confirm_buttons(c), "the stale cart confirm reached her after she cancelled")
    c.check(not c.shop.placed, "placed after a cancel")


SCENARIOS = [("several items, Yes tap, tracked to the door, how was it", several_items_tap_yes_tracked, RAIPUR, ()),
             ("flavour kept, then cancel", flavour_then_cancel, RAIPUR, ()),
             ("missing flavour, change, old Yes tap", missing_flavour_change_and_old_tap, RAIPUR, ()),
             ("late order: asked, checked again, caregiver told", late_order_tells_the_caregiver, RAIPUR, ()),
             ("three stores named, one does not deliver, cancel after placing", three_stores_named_one_does_not_deliver_then_cancel_after_placing,
              RAIPUR, ()),
             ("Bangalore: RiteBite x4, no store named, place without a point", bangalore_ritebite_no_store_named, BANGALORE, ("zepto:prepare", "zepto:place")),
             ("Delhi: Hindi, no store named, then no", delhi_hindi_no_store_then_no, DELHI, ("zepto:prepare",)),
             ("Mumbai: Marathi, self-care, the named store lacks it", mumbai_marathi_self_care_named_store_lacks_it, MUMBAI,
              ("blinkit:cancel",)),
             ("Chennai: Tamil, no point anywhere: the agent looks", chennai_tamil_no_point_browser_agent_looks, CHENNAI,
              # no store ids from the agent's look-up: the agent also builds the cart and places, on whichever store is picked
              tuple(f"{s}:{p}" for s in ("blinkit", "instamart", "zepto") for p in ("browse", "prepare", "place"))),
             ("Kolkata: Bengali, one store slow", kolkata_bengali_slow_store_not_waited_for, KOLKATA, ("zepto:browse",)),
             # the founder's rules everywhere: one regression per root cause, across services, store mixes and people
             ("Delhi: Hindi ride, cheapest app, cancel after booking", delhi_ride_cheapest_then_cancel_after_booking, DELHI,
              ("uber:browse", "ola:browse", "rapido:browse", "ola:prepare", "ola:place", "uber:prepare", "uber:place")),
             ("Pune: a son orders for himself, his mother hears nothing", pune_son_orders_for_himself_maa_hears_nothing, PUNE, ()),
             ("Bangalore: food on the two apps he named, Zomato will not cancel: he is sent to its app",
              bangalore_food_two_apps_named_caregiver_cancels_in_app, BANGALORE,
              ("zomato:prepare", "zomato:place", "zomato:cancel", "swiggy:prepare", "swiggy:place", "swiggy:cancel")),
             ("Bangalore: 1mg, the closest named as bought, a free cancel not asked again", bangalore_1mg_closest_named_then_free_cancel,
              BANGALORE, ("1mg:prepare", "1mg:place", "1mg:cancel")),
             ("Chennai: Tamil ride from home, the app set the city first", chennai_ride_from_home_the_app_set_the_city_first, CHENNAI,
              ("uber:browse", "uber:prepare", "uber:place")),
             ("Bangalore: his Ola ride, then he cancelled it himself in the app", bangalore_ride_he_cancelled_it_himself_in_the_app,
              BANGALORE, ("ola:browse", "ola:prepare", "ola:place")),
             ("Kolkata: Bengali cab for 6 tomorrow, booked then without asking", kolkata_cab_for_6_tomorrow, KOLKATA,
              ("uber:browse", "uber:prepare", "uber:place")),
             ("Mumbai: Marathi milk for 7 tomorrow, placed then", mumbai_milk_for_7_tomorrow, MUMBAI, ()),
             ("Chennai: Tamil medicine, exact strength only", chennai_tamil_medicine_exact_strength_only, CHENNAI,
              tuple(f"{s}:{p}" for s in ("apollo", "1mg", "pharmeasy") for p in ("prepare", "place"))),
             ("Kolkata: Bengali, two of three named, login code at once", kolkata_two_of_three_named_login_code_asked_at_once, KOLKATA,
              ("zepto:prepare", "zepto:place")),
             ("Kolkata: the first login code did not go in: asked once more, then the confirm", kolkata_first_code_did_not_go_in, KOLKATA,
              ("zepto:prepare", "zepto:place")),
             ("Raipur: approval at once, cancel while placing, her son's cancel with a fee", raipur_approval_at_once_then_cancel_while_placing,
              RAIPUR, ("zepto:prepare", "zepto:place", "zepto:cancel")),
             ("Mumbai: Marathi, cancel while the cart is built", mumbai_marathi_cancel_while_the_cart_is_built, MUMBAI, ()),
             ("Bangalore: an Ola for tomorrow, the fare look-up booked it: owned, cancelled, never denied",
              bangalore_ola_for_tomorrow_the_look_up_booked_it, BANGALORE, ("ola:browse", "ola:prepare", "ola:check", "ola:cancel"))]


async def main() -> int:
    async with engine.begin() as conn:
        from app.care import baselines, memory_index, models, why  # noqa: F401  (why: the trace_turns table)
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
        # the read-only store check after a run with the family's login (founder 2026-10-12) follows any expected run there
        unexpected = [r for r in c.browser.agent_runs if r not in c.agent_ok
                      and not (r.endswith(":check") and any(o.startswith(r.split(":")[0] + ":") for o in c.agent_ok))]
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
