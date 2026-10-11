"""Orders work for every family, not one (founder 2026-10-11, after a Bangalore family's order: Blinkit and Zepto look-ups
answered 400 for a place saved with a pincode only, the picker saw no listings because the browser agent tagged them
"4 x …", the person heard nothing for 6 minutes, and then got a list of options to choose from)."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.specialists.contract import Limits
from app.tasks import fastpath, runtime
from tests.test_tasks import CART, ELDER, FAM, FOUND, OTP, FakeAgent, LoginHarness

RITEBITE = "RiteBite Choco Berry protein bar"
BAR_50 = {"name": "RiteBite Max Protein Daily Choco Berry 10 g Protein Bar", "pack": "50 g", "price": "₹80", "available": True}
BAR_100 = {"name": "RiteBite Max Protein Choco Berry 30 g Protein Bar", "pack": "100 g", "price": "₹177", "available": True}
COMBO = {"name": "RiteBite Max Protein Daily Choco Almond Protein Bar + RiteBite Max Protein Daily Choco Berry Protein Bar Combo",
         "pack": "50 g + 50 g", "price": "₹151", "available": True}
# what the Blinkit browser agent reported live on 2026-10-11 (task 51492e5b): the quantity inside for_item
BLINKIT_LIVE = {**FOUND, "eta": "8 mins", "location_set": "560092 (Sahakar Nagar, Byatarayanapura)",
                "items": [{**x, "for_item": f"4 x {RITEBITE}"} for x in (BAR_50, BAR_100, COMBO)]}
BANGALORE = {"addressId": "pl-blr", "nickname": "Home", "pincode": "560092", "full": "74 4th cross 4c sector amrutnagar bangalore karnataka india, 560092"}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


async def _bars(db, service="blinkit", **details):
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service=service, kind="order", goal="RiteBite bars",
                             details={"items": [{"name": RITEBITE, "qty": 4}], **details}, limits=Limits(place=BANGALORE))
    await db.commit()
    return t


async def test_a_quantity_in_the_agents_tag_still_reaches_the_picker_and_saheli_picks_one(db, at, sessions, monkeypatch, fake_matcher):
    """Live 2026-10-11: the picker was given no listings ("No listings available") and the person got two Blinkit packs to
    choose from. The tag is read as the asked item, the AI picks one, and it goes on to the cart without a question."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 14:52")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [BLINKIT_LIVE]}), "9000012345")
    t = await _bars(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert fake_matcher.calls and len(fake_matcher.calls[-1][1][0]) == 3, "the picker saw every listing"
    assert t.phase == "prepare" and t.status == "queued" and h.told == [], "no list: Saheli picked herself"
    assert t.details["chosen"][0]["name"] == BAR_50["name"] and t.details["chosen"][0]["exact_match"] is True
    browse_goal = h.agent.runs[0]["goal"]
    assert f"item 1: {RITEBITE} (4 wanted)" in browse_goal and "for_item (the item's number" in browse_goal


def test_tags_become_the_asked_names_whatever_the_agent_wrote():
    names = ["Diet Coke", "Amul Taaza milk"]
    found = [{"name": "a", "for_item": "2"}, {"name": "b", "for_item": "3 x Diet Coke"}, {"name": "c", "for_item": "Diet Coke"},
             {"name": "d", "for_item": "something else"}, {"name": "e"}]
    runtime._canon_for_item(names, found)
    assert [x.get("for_item") for x in found] == ["Amul Taaza milk", "Diet Coke", "Diet Coke", None, None]
    one = [{"name": "x", "for_item": "4 x RiteBite"}]
    runtime._canon_for_item([RITEBITE], one)
    assert one[0]["for_item"] == RITEBITE, "one item asked: every product is for it"


async def test_the_ai_down_twice_takes_the_stores_top_listing_and_the_confirm_says_closest(db, at, sessions, monkeypatch, fake_matcher):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 15:00")

    async def down(*a, **k):
        fake_matcher.calls.append("down")
        return None

    from app.tasks import matcher
    monkeypatch.setattr(matcher, "choose", down)
    cart = {**CART, "items": [{"name": BAR_50["name"], "qty": 4, "price": "₹320", "available": True}], "total": "₹336"}
    h = LoginHarness(sessions, FakeAgent(script={"browse": [{**FOUND, "items": [BAR_50, BAR_100]}], "prepare": [cart]}), "9000012345")
    t = await _bars(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert fake_matcher.calls.count("down") == 2, "asked twice"
    assert t.phase == "prepare" and h.told == [] and t.details["chosen"][0]["exact_match"] is False
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and "closest match" in h.told[-1] and "ONE short question" in h.told[-1]


def _compare3(db_tasks):
    return {t.service: t for t in db_tasks}


async def _compare(db, item=RITEBITE, qty=4):
    out = []
    for s in runtime.COMPARE_STORES["grocery"]:
        out.append(await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service=s, kind="order", goal="bars",
                                        details={"items": [{"name": item, "qty": qty}], "compare": "g-blr"}, limits=Limits(place=BANGALORE)))
    await db.commit()
    return _compare3(out)


async def test_no_store_has_it_exactly_the_closest_one_is_picked_and_the_confirm_says_so(db, at, sessions, fake_matcher):
    """Never a numbered list: with only other variants anywhere, the cheapest closest one goes to the one confirm."""
    at("2026-10-11 15:00")
    fake_matcher.script = {"Amul Taaza milk 1 litre": {"exact": [], "closest": "Amul Gold Milk 1 L"}}
    gold = {"name": "Amul Gold Milk 1 L", "price": "₹68", "available": True}
    agent = FakeAgent(script={"browse:blinkit": [{**FOUND, "items": [{**gold, "price": "₹70"}]}],
                              "browse:instamart": [{**FOUND, "items": [gold]}],
                              "browse:zepto": [{**FOUND, "items": [{"name": "Mother Dairy Milk 1 L", "price": "₹66", "available": True}]}],
                              "prepare": [{**CART, "items": [{"name": "Amul Gold Milk 1 L", "qty": 1, "price": "₹68", "available": True}], "total": "₹77"}]})
    h = LoginHarness(sessions, agent, "9000012345")
    g = await _compare(db, item="Amul Taaza milk 1 litre", qty=1)
    await h.tick(); await h.tick()
    for t in g.values():
        await db.refresh(t)
    assert g["instamart"].phase == "prepare" and g["blinkit"].status == g["zepto"].status == "cancelled"
    assert h.told == [], "no options list"
    await h.tick(); await h.tick()
    await db.refresh(g["instamart"])
    assert g["instamart"].status == "awaiting_confirm" and "closest match" in h.told[-1] and "Amul Gold Milk 1 L" in h.told[-1]


async def test_a_slow_comparison_says_once_that_it_is_still_looking(db, at, sessions):
    """Live 2026-10-11: 'checking Blinkit, Instamart and Zepto' at 2:52, then nothing until he wrote 'Okay' at 2:58."""
    at("2026-10-11 14:52")
    agent = FakeAgent(script={"browse": [FOUND]}, finish_after_polls=99)  # every store's browser is slow
    h = LoginHarness(sessions, agent, "9000012345")
    await _compare(db)
    await h.tick()
    at("2026-10-11 14:53")
    await h.tick()
    assert h.told == []
    at("2026-10-11 14:54:30")
    await h.tick()
    assert len(h.told) == 1 and "Still looking on Blinkit, Swiggy Instamart, Zepto" in h.told[0] and "Do not ask anything" in h.told[0]
    at("2026-10-11 14:56")
    await h.tick()
    assert len(h.told) == 1, "one line for the whole comparison, not one per store or per minute"


async def test_a_slow_store_does_not_hold_up_the_others_for_long(db, at, sessions):
    at("2026-10-11 15:00")
    agent = FakeAgent(script={"browse:blinkit": [{**FOUND, "items": [BAR_50]}], "browse": [FOUND]})
    h = LoginHarness(sessions, agent, "9000012345")
    g = await _compare(db)
    await h.tick()
    agent.finish_after_polls = 99  # Instamart and Zepto keep loading
    agent._polls[[r["task"] for r in agent.runs if r["service"] == "blinkit"][0]] = 98
    await h.tick()
    at("2026-10-11 15:01:20")
    await h.tick()
    await db.refresh(g["blinkit"])
    assert g["blinkit"].status == "awaiting_confirm", "waits a little for the others"
    at("2026-10-11 15:02:00")
    await h.tick()
    for t in g.values():
        await db.refresh(t)
    assert g["blinkit"].phase == "prepare" and g["instamart"].status == g["zepto"].status == "cancelled"


async def test_the_picked_store_cannot_build_the_cart_so_the_next_best_takes_over_without_asking(db, at, sessions):
    """Live 2026-10-11: the picked store failed after the others were dropped, and the brain read the dropped store's old
    options out as a list. The next best store of the comparison takes over with its own pick."""
    at("2026-10-11 15:00")
    nothing = {**CART, "items": [], "problem": "the item could not be added", "alternatives": []}
    agent = FakeAgent(script={"browse:blinkit": [{**FOUND, "items": [BAR_50]}],
                              "browse:instamart": [{**FOUND, "items": [{**BAR_50, "price": "₹79"}]}],
                              "browse:zepto": [{**FOUND, "items": [BAR_100]}],
                              "prepare:instamart": [nothing],
                              "prepare:blinkit": [{**CART, "items": [{"name": BAR_50["name"], "qty": 4, "price": "₹320", "available": True}], "total": "₹336"}]})
    h = LoginHarness(sessions, agent, "9000012345")
    g = await _compare(db)
    await h.tick(); await h.tick()
    for t in g.values():
        await db.refresh(t)
    assert g["instamart"].phase == "prepare", "₹79 x 4 is the best"
    await h.tick(); await h.tick()
    for t in g.values():
        await db.refresh(t)
    assert g["instamart"].status == "failed" and g["blinkit"].status == "queued" and g["blinkit"].phase == "prepare"
    assert h.told == [], "nobody is asked; the next store builds the cart"
    await h.tick(); await h.tick()
    await db.refresh(g["blinkit"])
    assert g["blinkit"].status == "awaiting_confirm" and "₹336" in h.told[-1] and "Blinkit" in h.told[-1]
    assert g["zepto"].status == "cancelled"


async def test_no_number_to_log_in_with_is_said_once_and_no_other_store_is_tried(db, at, sessions):
    at("2026-10-11 15:00")
    agent = FakeAgent(script={"browse:blinkit": [{**FOUND, "items": [BAR_50]}], "browse:instamart": [{**FOUND, "items": [], "deliverable": False}],
                              "browse:zepto": [{**FOUND, "items": [BAR_100]}],
                              "prepare": [{**CART, "items": [], "problem": "login needed", "logged_in": False}]})
    h = LoginHarness(sessions, agent, None)
    g = await _compare(db)
    for _ in range(4):
        await h.tick()
    for t in g.values():
        await db.refresh(t)
    assert g["blinkit"].status == "failed" and g["zepto"].status == "cancelled"
    assert len(h.told) == 1 and "no phone number to log in with" in h.told[0]


class FastAgent(FakeAgent):
    async def open_session(self, profile_id):
        return "fs-1"

    async def cdp_url(self, session_id):
        return "https://fs-1.cdp.test"


class PlaceHost:
    def __init__(self, point):
        self.point, self.calls = point, []

    async def call(self, tool, args, **kw):
        self.calls.append(tool)
        if tool == "delivery_place":
            return {**BANGALORE, **self.point}
        if tool == "connector_status":
            return {"connected": False}
        return {}


async def _fast_tick(sessions, agent, host):
    told = []

    async def notify(f, p, text):
        told.append(text)

    async def profile_for(task):
        return {"profileId": f"prof-{task.service}", "loginPhone": "9000012345"}

    await runtime.tick(sessions, agent, profile_for=profile_for, notify=notify, host_for=lambda fid: host)
    return told


async def test_a_place_without_a_map_point_never_sends_an_empty_one(db, at, sessions, monkeypatch):
    """Live 2026-10-11 tasks 51492e5b and d4961ba6: lat/lng were empty and Blinkit and Zepto answered 400 (an ops alert each)."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 14:52")
    searched = []

    async def search(service, cdp, q, lat=None, lon=None, pincode=None, **k):
        searched.append((service, lat, lon))
        return {"deliverable": True, "eta": "8 mins", "items": [BAR_50]}

    monkeypatch.setattr(fastpath, "search", search)
    host = PlaceHost({"lat": None, "lng": None})
    agent = FastAgent(script={"browse": [FOUND]})
    t = await _bars(db)
    await _fast_tick(sessions, agent, host)
    await db.refresh(t)
    assert searched == [] and "ops_alert" not in host.calls, "no saved look-up without the point, and no alarm"
    assert any("no map point" in x["note"] for x in t.history) and [r["phase"] for r in agent.runs] == ["browse"]

    # the backend found the point now (every place gets one): the saved look-up runs with it
    host2 = PlaceHost({"lat": 13.0617, "lng": 77.5929, "geoPrecision": "pincode"})
    t2 = await _bars(db, service="zepto")
    await _fast_tick(sessions, FastAgent(script={}), host2)
    await db.refresh(t2)
    assert searched == [("zepto", 13.0617, 77.5929)] and t2.details["limits"]["place"]["lat"] == 13.0617


async def test_does_not_deliver_at_the_pincodes_centre_is_checked_by_the_agent_with_the_address(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 15:00")

    async def search(service, cdp, q, **k):
        return {"deliverable": False, "eta": None, "items": []}

    monkeypatch.setattr(fastpath, "search", search)
    agent = FastAgent(script={"browse": [{**FOUND, "items": [BAR_50]}]})
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="bars",
                             details={"items": [{"name": RITEBITE, "qty": 4}]},
                             limits=Limits(place={**BANGALORE, "lat": 13.0617, "lng": 77.5929, "geoPrecision": "pincode"}))
    await db.commit()
    told = await _fast_tick(sessions, agent, PlaceHost({}))
    await db.refresh(t)
    assert told == [] and [r["phase"] for r in agent.runs] == ["browse"], "the agent checks; nobody hears 'not delivered'"


async def test_one_familys_expired_link_does_not_switch_the_connector_off_for_everyone(db, at, sessions, monkeypatch):
    """Live 2026-10-11: a family's Zepto link failed with InvalidGrantError 5 times and Zepto's connector was degraded
    for every family for 30 minutes."""
    from app.specialists import channels

    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 15:00")

    class ExpiredHost:
        async def call(self, tool, args, **kw):
            if tool == "connector_status":
                return {"connected": True, "enabled": True}
            if tool == "connector_search":
                return {"ok": False, "kind": "auth_expired", "detail": "InvalidGrantError"}
            return {}

    for n in range(4):
        t = await runtime.create(db, family_id=f"fam-x{n}", subject_id=ELDER, requested_by=ELDER, service="zepto", kind="order",
                                 goal="bars", details={"items": [{"name": RITEBITE, "qty": 1}]})
        await db.commit()
        await runtime._connector_lookup(db, ExpiredHost(), t)
    assert await channels.healthy(db, "zepto", "connector"), "still on for the other families"


async def test_a_store_error_inside_a_200_is_a_failed_look_up_not_nothing_found(monkeypatch):
    """Live 2026-10-11: Instamart answered 'not serviceable' at 0,0 and the order said 'Could not find it on Swiggy
    Instamart'; the daily check said 'no products' for milk."""
    answers = []

    async def evaluate(cdp, url, js, **k):
        return answers.pop(0)

    monkeypatch.setattr(fastpath, "cdp_evaluate", evaluate)
    real_sleep = fastpath.asyncio.sleep

    async def no_wait(s):
        await real_sleep(0)

    monkeypatch.setattr(fastpath.asyncio, "sleep", no_wait)
    answers[:] = [{"status": 200, "error": "search statusCode 7", "products": []}] * 2
    with pytest.raises(fastpath.FastPathError, match="statusCode 7"):
        await fastpath.search("instamart", "ws://x", "milk", lat=12.97, lon=77.59)
    answers[:] = [{"status": 200, "error": "search statusCode 7", "products": []},
                  {"status": 200, "serviceable": True, "products": [{"name": "Nandini Milk", "price": 27}]}]
    got = await fastpath.search("instamart", "ws://x", "milk", lat=12.97, lon=77.59)
    assert got["items"][0]["name"] == "Nandini Milk", "a passing hiccup: once more"
    answers[:] = [{"status": 200, "serviceable": True, "note": "no dishes found for this search", "products": []}]
    assert (await fastpath.search("swiggy", "ws://x", "unicorn curry", lat=12.97, lon=77.59))["items"] == [], "a real empty result"
    for lat, lon in ((None, None), ("", ""), (0, 0)):
        with pytest.raises(fastpath.NoPoint):
            await fastpath.search("blinkit", "ws://x", "milk", lat=lat, lon=lon, pincode="560092")
    answers[:] = [{"status": 200, "serviceable": True, "products": [{"name": "Dolo 650", "price": 30}]}]
    assert (await fastpath.search("apollo", "ws://x", "dolo", pincode="560092"))["items"], "pharmacies work by pincode"


def test_the_confirm_buttons_and_the_tap_reply_speak_the_persons_language():
    """Sim 2026-10-11: a Tamil and a Bengali confirm came with English buttons and an English 'placing the order now'."""
    from app.care import outcomes

    tamil = outcomes.buttons("order", "t1:abc", None, "சுவிக்கி இன்ஸ்டாமார்ட்டில் ஆவின் பால் 2 பாக்கெட் தயார். ஆர்டர் செய்யட்டுமா?")
    assert [b["title"] for b in tamil] == ["ஆம், ஆர்டர் செய்", "மாற்று", "ரத்து செய்"] and tamil[0]["id"] == "v2:od:yes:t1:abc"
    bengali = outcomes.buttons("order", "t2", None, "মোট ৮৩ টাকা। অর্ডার করে দেব কি?")
    assert bengali[0]["title"] == "হ্যাঁ, অর্ডার করো"
    assert outcomes.buttons("order", "t3", None, "Shall I place the order?")[0]["title"] == "Yes, order it"
    hindi = outcomes.buttons("order", "t4", None, "क्या मैं ऑर्डर कर दूँ?")
    assert hindi[0]["title"] == "हाँ, मंगाओ"
    from app.brain import guards
    for script in ("tamil", "bengali", "telugu", "kannada", "gujarati", "gurmukhi", "malayalam", "odia"):
        tap = guards.canned({"script": script}, outcomes.ORDER_TAP)
        assert guards.script_of(tap["yes"]) == script and guards.script_of(tap["cancel"]) == script, script
        assert all(len(t) <= 20 for t in outcomes.ORDER_BUTTONS[script]), script
