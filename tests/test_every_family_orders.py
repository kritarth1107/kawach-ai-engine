"""Orders work for every family, not one (founder 2026-10-11, after a Bangalore family's order: Blinkit and Zepto look-ups
answered 400 for a place saved with a pincode only, the picker saw no listings because the browser agent tagged them
"4 x …", the person heard nothing for 6 minutes, and then got a list of options to choose from)."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.specialists.contract import Limits
from app.tasks import fastpath, runtime
from tests.test_tasks import CART, ELDER, FAM, FOUND, OTP, FakeAgent, LoginHarness, _pick

RITEBITE = "RiteBite Choco Berry protein bar"
BAR_50 = {"name": "RiteBite Max Protein Daily Choco Berry 10 g Protein Bar", "pack": "50 g", "price": "₹80", "available": True}
BAR_100 = {"name": "RiteBite Max Protein Choco Berry 30 g Protein Bar", "pack": "100 g", "price": "₹177", "available": True}
COMBO = {"name": "RiteBite Max Protein Daily Choco Almond Protein Bar + RiteBite Max Protein Daily Choco Berry Protein Bar Combo",
         "pack": "50 g + 50 g", "price": "₹151", "available": True}
# what the Blinkit browser agent reported live on 2026-10-11: the quantity inside for_item
BLINKIT_LIVE = {**FOUND, "eta": "8 mins", "location_set": "560011 (Jayanagar 4th Block)",
                "items": [{**x, "for_item": f"4 x {RITEBITE}"} for x in (BAR_50, BAR_100, COMBO)]}
BANGALORE = {"addressId": "pl-blr", "nickname": "Home", "pincode": "560011", "full": "21 5th main 4th block jayanagar bangalore karnataka india, 560011"}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


async def _bars(db, service="blinkit", **details):
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service=service, kind="order", goal="RiteBite bars",
                             details={"items": [{"name": RITEBITE, "qty": 4}], **details}, limits=Limits(place=BANGALORE))
    await db.commit()
    return t


async def test_a_quantity_in_the_agents_tag_still_reaches_the_picker_and_saheli_suggests_one(db, at, sessions, monkeypatch, fake_matcher):
    """Live 2026-10-11: the picker was given no listings ("No listings available"). The tag is read as the asked item, the AI
    ranks every listing, and the person gets them numbered with Saheli's suggestion marked (founder 2026-10-11: options at
    all times)."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 14:52")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [BLINKIT_LIVE]}), "9000012345")
    t = await _bars(db)
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert fake_matcher.calls and len(fake_matcher.calls[-1][1][0]) == 3, "the picker saw every listing"
    assert t.status == "awaiting_confirm" and t.input_needed == "go" and len(h.told) == 1
    assert "1. RiteBite Max Protein Daily Choco Berry 10 g Protein Bar (50 g) ₹80 each; 4 = ₹320 ← suggested" in h.told[0] and "3. " in h.told[0]
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
    assert t.input_needed == "go" and t.details["chosen"][0]["exact_match"] is False
    assert "1. RiteBite Max Protein Daily Choco Berry 10 g Protein Bar (50 g) ₹80 each; 4 = ₹320 ← suggested" in h.told[0], "the store's own order"
    await _pick(db, t)
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


async def test_no_store_has_it_exactly_the_closest_ones_are_offered_marked_similar(db, at, sessions, fake_matcher):
    """With only other variants anywhere, each store's closest one is offered marked similar (the cheapest suggested), and
    the one confirm says it is the closest match."""
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
    assert len(h.told) == 1 and "1. Amul Gold Milk 1 L ₹68 [similar: not exactly Amul Taaza milk 1 litre] ← suggested" in h.told[0]
    assert "2. Amul Gold Milk 1 L ₹70 [similar: not exactly Amul Taaza milk 1 litre]" in h.told[0] and "- Zepto: nothing exactly this" in h.told[0]
    await _pick(db, g["instamart"], "1")
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
    assert g["blinkit"].status == "awaiting_confirm" and h.told == [], "waits a little for the others"
    at("2026-10-11 15:03:05")
    await h.tick()
    for t in g.values():
        await db.refresh(t)
    assert g["blinkit"].input_needed == "go" and g["instamart"].status == g["zepto"].status == "cancelled"
    assert len(h.told) == 1 and "1. RiteBite Max Protein Daily Choco Berry 10 g Protein Bar (50 g) ₹80 each; 4 = ₹320 ← suggested" in h.told[0]
    assert "- Swiggy Instamart: its website did not answer in time" in h.told[0], "every store is reported, the slow ones too"


async def test_the_picked_store_cannot_build_the_cart_so_the_other_stores_options_go_back_to_the_person(db, at, sessions):
    """Live 2026-10-11: the picked store failed after the others were dropped. The other stores' options go back to the
    person in one update (nothing is picked for them), and the store they pick then builds the cart."""
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
    assert "1. RiteBite Max Protein Daily Choco Berry 10 g Protein Bar (50 g) ₹79 each; 4 = ₹316 ← suggested" in h.told[0], "₹79 x 4 is the best"
    await _pick(db, g["instamart"], "1")
    await h.tick(); await h.tick()
    for t in g.values():
        await db.refresh(t)
    assert g["instamart"].status == "failed" and (g["blinkit"].status, g["blinkit"].input_needed) == ("awaiting_confirm", "go")
    assert (g["zepto"].status, g["zepto"].input_needed) == ("awaiting_confirm", "go")
    again = h.told[-1]
    assert len(h.told) == 2 and "Swiggy Instamart, which they picked, could not build the cart" in again
    assert "1. RiteBite Max Protein Daily Choco Berry 10 g Protein Bar (50 g) ₹80 each; 4 = ₹320 ← suggested" in again and "Swiggy Instamart [task" not in again
    await _pick(db, g["blinkit"], "1")
    await h.tick(); await h.tick()
    await db.refresh(g["blinkit"])
    await db.refresh(g["zepto"])
    assert g["blinkit"].status == "awaiting_confirm" and "₹336" in h.told[-1] and "Blinkit" in h.told[-1]
    assert g["zepto"].status == "cancelled"


async def test_no_number_to_log_in_with_is_said_once_and_no_other_store_is_tried(db, at, sessions):
    at("2026-10-11 15:00")
    agent = FakeAgent(script={"browse:blinkit": [{**FOUND, "items": [BAR_50]}], "browse:instamart": [{**FOUND, "items": [], "deliverable": False}],
                              "browse:zepto": [{**FOUND, "items": [BAR_100]}],
                              "prepare": [{**CART, "items": [], "problem": "login needed", "logged_in": False}]})
    h = LoginHarness(sessions, agent, None)
    g = await _compare(db)
    await h.tick(); await h.tick()
    await _pick(db, g["blinkit"], "1")
    await h.tick(); await h.tick()
    for t in g.values():
        await db.refresh(t)
    assert g["blinkit"].status == "failed" and g["zepto"].status == "cancelled"
    assert len(h.told) == 2 and "no phone number to log in with" in h.told[1], "said once; no other store offered"


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
    """Live 2026-10-11: lat/lng were empty and Blinkit and Zepto answered 400 (an ops alert each)."""
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
    host2 = PlaceHost({"lat": 12.9299, "lng": 77.5826, "geoPrecision": "pincode"})
    t2 = await _bars(db, service="zepto")
    await _fast_tick(sessions, FastAgent(script={}), host2)
    await db.refresh(t2)
    assert searched == [("zepto", 12.9299, 77.5826)] and t2.details["limits"]["place"]["lat"] == 12.9299


async def test_does_not_deliver_at_the_pincodes_centre_is_checked_by_the_agent_with_the_address(db, at, sessions, monkeypatch):
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 15:00")

    async def search(service, cdp, q, **k):
        return {"deliverable": False, "eta": None, "items": []}

    monkeypatch.setattr(fastpath, "search", search)
    agent = FastAgent(script={"browse": [{**FOUND, "items": [BAR_50]}]})
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="bars",
                             details={"items": [{"name": RITEBITE, "qty": 4}]},
                             limits=Limits(place={**BANGALORE, "lat": 12.9299, "lng": 77.5826, "geoPrecision": "pincode"}))
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
            await fastpath.search("blinkit", "ws://x", "milk", lat=lat, lon=lon, pincode="560011")
    answers[:] = [{"status": 200, "serviceable": True, "products": [{"name": "Dolo 650", "price": 30}]}]
    assert (await fastpath.search("apollo", "ws://x", "dolo", pincode="560011"))["items"], "pharmacies work by pincode"


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


async def test_a_login_code_that_never_came_is_asked_for_again(db, at, sessions):
    """Live 2026-10-11: 'Haven't got it' — Saheli promised a new code, nothing could send one, and the order sat until the
    ladder cancelled it. task_input resend runs a fresh login (at most twice)."""
    at("2026-10-11 16:30")
    agent = FakeAgent(script={"browse": [{**FOUND, "items": [BAR_100]}], "prepare": [OTP]})
    h = LoginHarness(sessions, agent, "9000012345")
    t = await _bars(db, service="zepto")
    for _ in range(4):
        await h.tick()
    await db.refresh(t)
    assert t.input_needed == "otp" and "task_input kind resend" in h.told[-1]
    out = await runtime.provide_input(db, t, kind="resend", by=ELDER, by_is_elder=False)
    await db.commit()
    assert out.startswith("asking the store for a new login code") and t.phase == "prepare" and t.status == "queued"
    await h.tick()
    goal = agent.runs[-1]["goal"]
    assert "only log in" in goal and "never arrived" in goal and "enter the mobile number 9000012345" in goal
    await h.tick()
    await db.refresh(t)
    assert t.input_needed == "otp" and "NEW login code" in h.told[-1]
    await runtime.provide_input(db, t, kind="resend", by=ELDER, by_is_elder=False); await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    third = await runtime.provide_input(db, t, kind="resend", by=ELDER, by_is_elder=False)
    assert "already asked for 2 times" in third and t.input_needed == "otp", "at most twice; then ask whether the number is right"
    assert "not waiting" in await runtime.provide_input(db, await _bars(db, service="instamart"), kind="resend", by=ELDER, by_is_elder=False)


async def test_the_ai_decides_which_saved_place_the_words_mean(monkeypatch):
    """No word list on people's words: the AI picks the saved place, and its exact name goes to the backend."""
    from app.brain import tools
    from app.tasks import matcher

    class Host:
        def __init__(self):
            self.calls = []

        async def call(self, tool, args, **kw):
            self.calls.append((tool, args))
            return {"places": [{"name": "Home", "address": "21 5th main, Bangalore 560011", "isDefault": True},
                               {"name": "Beta's flat", "address": "B-7, Vijay Nagar, Indore 452010"}]}

    async def ai(words, places):
        return {"ghar": "Home", "beta ke ghar": "Beta's flat"}.get(words)

    monkeypatch.setattr(matcher, "which_place", ai)
    kw = {"family_id": "f", "subject_id": "e", "actor_id": "e"}
    assert await tools.saved_place_args(Host(), "ghar", **kw) == {"words": "Home", "exact": True}
    assert await tools.saved_place_args(Host(), "beta ke ghar", **kw) == {"words": "Beta's flat", "exact": True}
    assert await tools.saved_place_args(Host(), "the station", **kw) == {"words": "the station", "exact": True}, "not saved: asked"
    assert await tools.saved_place_args(Host(), "", **kw) == {"words": ""}

    async def down(words, places):
        raise RuntimeError("model down")

    monkeypatch.setattr(matcher, "which_place", down)
    assert await tools.saved_place_args(Host(), "ghar", **kw) == {"words": "ghar"}, "model down: the backend's own lookup"


async def test_naming_one_store_during_a_comparison_drops_the_others(db, at, sessions):
    """Live 2026-10-11: he wrote 'Zepto' while three stores looked; Saheli said 'checking only on Zepto' and all three went
    on. task_input only: that store goes on alone (its own options for them to pick, or a fresh look-up), the others are
    dropped."""
    at("2026-10-11 16:49")
    agent = FakeAgent(script={"browse": [FOUND]}, finish_after_polls=99)
    h = LoginHarness(sessions, agent, "9000012345")
    g = await _compare(db)
    await h.tick()
    out = await runtime.provide_input(db, g["zepto"], kind="only", by=ELDER, by_is_elder=False)
    await db.commit()
    for t in g.values():
        await db.refresh(t)
    assert out.startswith("only Zepto now") and runtime.compare_group(g["zepto"]) is None
    assert g["blinkit"].status == g["instamart"].status == "cancelled" and g["zepto"].status in ("queued", "running")
    # already looked and picked: straight on to the cart
    at("2026-10-11 17:00")
    agent2 = FakeAgent(script={"browse:zepto": [{**FOUND, "items": [BAR_100]}], "browse": [FOUND]}, finish_after_polls=99)
    h2 = LoginHarness(sessions, agent2, "9000012345")
    g2 = {t.service: t for t in [await runtime.create(db, family_id="fam-only", subject_id=ELDER, requested_by=ELDER, service=s, kind="order",
                                                        goal="bars", details={"items": [{"name": RITEBITE, "qty": 4}], "compare": "g-only"},
                                                        limits=Limits(place=BANGALORE)) for s in runtime.COMPARE_STORES["grocery"]]}
    await db.commit()
    await h2.tick()  # every store starts looking
    agent2._polls[[r["task"] for r in agent2.runs if r["service"] == "zepto"][0]] = 98  # Zepto answers, the others are slow
    await h2.tick()
    z = g2["zepto"]
    await db.refresh(z)
    assert z.status == "awaiting_confirm" and z.input_needed == "go", "Zepto looked; the comparison waits for the others"
    said = await runtime.provide_input(db, z, kind="only", by=ELDER, by_is_elder=False); await db.commit()
    await db.refresh(z)
    assert (z.status, z.input_needed) == ("awaiting_confirm", "go") and runtime.compare_group(z) is None and z.details["only_store"]
    assert "1. RiteBite Max Protein Choco Berry 30 g Protein Bar (100 g) ₹177 each; 4 = ₹708 ← suggested" in said, "its own options to pick from"
    assert "not a store of a running comparison" in await runtime.provide_input(db, await _bars(db, service="blinkit"), kind="only", by=ELDER, by_is_elder=False)


async def test_the_brain_sees_only_sahelis_pick_while_the_cart_is_built(db, at, sessions, monkeypatch):
    """Live 2026-10-11 16:54: he wrote 'Ok' while the Blinkit cart was built; the task line listed every product found, and
    the brain read them out as options again. Once they picked, it sees only the pick."""
    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 16:54")
    h = LoginHarness(sessions, FakeAgent(script={"browse": [BLINKIT_LIVE]}, finish_after_polls=99), "9000012345")
    t = await _bars(db)
    await h.tick(); h.agent.finish_after_polls = 1; await h.tick()
    await db.refresh(t)
    assert "options sent, waiting for their pick" in runtime.describe(t) and "₹177" in runtime.describe(t)
    await _pick(db, t, "1")
    line = runtime.describe(t)
    assert "picked (not in a cart yet" in line and "4 x RiteBite Max Protein Daily Choco Berry 10 g Protein Bar (50 g) ₹80" in line
    assert "₹177" not in line and "found (not in a cart yet)" not in line, line


def test_a_total_is_an_amount_read_off_the_checkout():
    """Live 2026-10-11: a Blinkit agent reported '₹320 (estimated based on ₹80 per item; full bill summary hidden)' and
    that went into the one confirm as the total."""
    assert runtime._plain_amount("₹320") and runtime._plain_amount("₹ 1,234.50") and runtime._plain_amount("Rs. 99") and runtime._plain_amount(318)
    assert not runtime._plain_amount("₹320 (estimated based on ₹80 per item; full bill summary hidden)")
    assert not runtime._plain_amount("shown at checkout") and not runtime._plain_amount("") and not runtime._plain_amount(None)


async def test_an_estimated_total_gets_one_look_at_the_checkout(db, at, sessions):
    at("2026-10-11 16:59")
    estimated = {**CART, "items": [{"name": "Aashirvaad Atta 5kg", "qty": 1, "price": "₹289", "available": True}],
                 "total": "₹289 (estimated; bill summary hidden)"}
    h = LoginHarness(sessions, FakeAgent(script={"prepare": [estimated, CART]}), "9000012345")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order", goal="Atta",
                             details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]})
    await db.commit()
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "queued" and t.details["total_only"] and h.told == [], "no confirm with an estimate"
    await h.tick(); await h.tick()
    await db.refresh(t)
    assert t.status == "awaiting_confirm" and "₹318" in h.told[-1] and "estimated" not in h.told[-1]


async def test_after_pay_now_an_unclear_place_reads_the_order_list_and_never_says_it_failed(db, at, sessions, monkeypatch):
    """Live 2026-10-11 17:05: the agent pressed Pay Now (cash), ran out of steps on the confirmation page and reported not
    placed; Saheli said 'could not be placed, no amount was charged' while Blinkit was packing order 3800000001."""
    from app.specialists.contract import Limits as L

    class PlaceAgent(FakeAgent):
        async def cdp_url(self, sid):
            return "ws://placed"

    looked = []

    async def order_list(cdp, total, within_min=15, names=None):
        looked.append((total, names))
        return listed

    monkeypatch.setattr(fastpath, "blinkit_recent_order", order_list)
    ran_out = {"logged_in": True, "needs_otp": False, "blocked": False, "placed": False,
               "problem": "Reached maximum steps while waiting for order confirmation after clicking Pay Now."}
    cart = {**CART, "items": [{"name": BAR_50["name"], "qty": 4, "price": "₹320", "available": True}], "total": "₹334"}
    for listed, want in (("3800000001", "done"), (None, "failed")):
        at("2026-10-11 17:03")
        h = LoginHarness(sessions, PlaceAgent(script={"prepare": [cart], "place": [ran_out]}), "9000012345")
        t = await runtime.create(db, family_id=f"fam-place-{want}", subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order",
                                 goal="bars", details={"items": [{"name": RITEBITE, "qty": 4}], "chosen": [{**BAR_50, "store_id": "b7"}]},
                                 limits=L(place=BANGALORE))
        await db.commit()
        await h.tick(); await h.tick()
        await db.refresh(t)
        await runtime.provide_input(db, t, kind="confirm", value="yes", by=ELDER, by_is_elder=True); await db.commit()
        await h.tick(); await h.tick()
        await db.refresh(t)
        assert t.status == want, (want, t.status, t.history[-2:])
        if want == "done":
            assert t.result["placed"] and t.result["order_id"] == "3800000001" and "Placed" in h.told[-1]
            assert looked[-1][1] == [BAR_50["name"]], "matched by the picked product when the amount differs"
        else:
            assert "not clear" in h.told[-1] and "Nothing was charged" not in h.told[-1] and "do not say it failed" in h.told[-1]


def test_a_place_report_that_never_pressed_the_button_is_not_placed():
    t = runtime.Task(service="blinkit", kind="order", phase="place", details={"limits": {}}, result={}, family_id="f")
    status, msg = runtime._outcome(t, {"placed": False, "pressed_place": False, "problem": "cash on delivery was not offered"})
    assert status == "failed" and "never pressed" in msg and "Nothing was charged" in msg


async def test_the_saved_place_is_asked_twice_and_a_disagreement_means_asking_the_person(monkeypatch):
    from app.tasks import matcher

    places = [{"name": "Home", "isDefault": True}, {"name": "Beta's flat"}]
    answers = []

    class Reply:
        def __init__(self, text):
            self.text = text

    async def complete(role, **kw):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return Reply(a)

    monkeypatch.setattr(matcher.router, "complete", complete)
    answers[:] = ['{"place": 1}', '{"place": 1}']
    assert await matcher.which_place("ghar", places) == "Home"
    answers[:] = ['{"place": 0}', '{"place": 2}']
    assert await matcher.which_place("beta ke ghar", places) == "Beta's flat", "one answer found it, the other did not: used"
    answers[:] = ['{"place": "Beta\'s flat"}', '{"place": 0}']
    assert await matcher.which_place("beta ke ghar", places) == "Beta's flat", "a name instead of a number"
    answers[:] = ['{"place": 1}', '{"place": 2}']
    assert await matcher.which_place("flat", places) is None, "two different places: ask"
    answers[:] = ['{"place": 0}', '{"place": 0}']
    assert await matcher.which_place("station", places) is None
    answers[:] = [RuntimeError("down"), RuntimeError("down")]
    with pytest.raises(RuntimeError):
        await matcher.which_place("ghar", places)


async def test_a_task_another_runner_is_in_the_middle_of_is_left_alone(db, at, sessions, monkeypatch):
    """Live 2026-10-11 13:08: a fast step committed (its new browser recorded), which let the scheduler's runner take the
    same still-queued task; both started a browser run on one browser and the second got 400 "Agent session already has a
    running task". The first runner now claims the task for that time."""
    from datetime import timedelta

    from sqlalchemy import select

    from app.core import clock
    from app.tasks.models import Task

    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-11 18:40")
    seen = []

    async def search(service, cdp, q, **k):
        row = (await db.execute(select(Task.details).where(Task.id == t.id).execution_options(populate_existing=True))).scalar_one()
        seen.append((row or {}).get("claim"))
        return {"deliverable": True, "eta": "10 min", "items": [BAR_50]}

    monkeypatch.setattr(fastpath, "search", search)
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="bars",
                             details={"items": [{"name": RITEBITE, "qty": 4}]},
                             limits=Limits(place={**BANGALORE, "lat": 12.9299, "lng": 77.5826, "geoPrecision": "address"}))
    await db.commit()
    await _fast_tick(sessions, FastAgent(script={}), PlaceHost({}))
    await db.refresh(t)
    assert seen and seen[0], "claimed while the fast step works (after its commit)"
    assert t.details.get("claim") is None and t.input_needed == "go", "and released when the runner is done"
    await _pick(db, t)
    assert t.status == "queued" and t.phase == "prepare"

    # another runner's fresh claim: left alone; a claim older than CLAIM_FOR (a crashed runner): taken
    t.details = {**t.details, "claim": clock.now().isoformat()}
    await db.commit()
    agent = FastAgent(script={"prepare": [OTP]})
    await _fast_tick(sessions, agent, PlaceHost({}))
    await db.refresh(t)
    assert agent.runs == [] and t.status == "queued", "left alone"
    at("2026-10-11 18:44")
    await _fast_tick(sessions, agent, PlaceHost({}))
    await db.refresh(t)
    assert len(agent.runs) == 1 and t.status == "running" and t.details.get("claim") is None
    assert runtime.CLAIM_FOR == timedelta(minutes=3)
