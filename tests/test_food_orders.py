"""Food orders (real Swiggy runs from Maa's WhatsApp, 2026-10-11): a food cart holds one restaurant, so every dish of an
order comes from one place, and the browser agent is told which restaurant and how many of each."""

from app.tasks import runtime

D = lambda pid, name, rest, price: {"name": name, "store_id": pid, "restaurant": rest, "restaurantId": rest.lower(), "price": price}  # noqa: E731


def test_the_restaurant_that_has_everything_is_chosen():
    groups = [("paneer butter masala", [D("p1", "Paneer Butter Masala", "Indian Bawarchi", "₹149"), D("p2", "Paneer Butter Masala", "Patiala Plates", "₹189")]),
              ("roti", [D("r1", "Tawa Roti", "Jassie's Kitchen", "₹20"), D("r2", "Tandoori Roti", "Patiala Plates", "₹25")])]
    matched = {"paneer butter masala": {"exact": ["p1", "p2"]}, "roti": {"exact": ["r1", "r2"]}}
    picks, find = runtime._one_restaurant(groups, matched)
    assert {a: p["restaurant"] for a, p in picks.items()} == {"paneer butter masala": "Patiala Plates", "roti": "Patiala Plates"}
    assert find == []


def test_the_main_dish_decides_when_no_place_has_everything():
    groups = [("paneer butter masala", [D("p1", "Paneer Butter Masala", "Indian Bawarchi", "₹149")]),
              ("roti", [D("r1", "Tawa Roti", "Jassie's Kitchen", "₹20")])]
    matched = {"paneer butter masala": {"exact": ["p1"]}, "roti": {"exact": ["r1"]}}
    picks, find = runtime._one_restaurant(groups, matched)
    assert list(picks) == ["paneer butter masala"] and picks["paneer butter masala"]["restaurant"] == "Indian Bawarchi"
    assert find == ["roti"], "the agent looks for roti in Indian Bawarchi's own menu"


async def test_a_two_dish_food_order_goes_to_one_restaurant_and_the_agent_is_told(db, at):
    at("2026-10-11 01:00")
    t = await runtime.create(db, family_id="fam-food", subject_id="e", requested_by="e", service="swiggy", kind="order", goal="Paneer and roti",
                             details={"items": [{"name": "paneer butter masala", "qty": 1}, {"name": "roti", "qty": 2}]})
    t.phase = "browse"
    found = [{**D("p1", "Paneer Butter Masala", "Indian Bawarchi", "₹149"), "for_item": "paneer butter masala"},
             {**D("r1", "Tawa Roti", "Jassie's Kitchen", "₹20"), "for_item": "roti"}]
    t.details = {**t.details, "matched": {"paneer butter masala": {"exact": ["p1"]}, "roti": {"exact": ["r1"]}}}
    status, _ = runtime._browse_outcome(t, {"items": found, "deliverable": True})
    assert status == "queued" and [c["restaurant"] for c in t.details["chosen"]] == ["Indian Bawarchi"]
    assert t.details["menu_find"] == ["roti"] and not t.details.get("missing_items")

    class Agent:
        def __init__(self):
            self.goal = None

        async def run(self, **kw):
            self.goal = kw["goal"]
            from app.tasks.browser_use import AgentRun
            return AgentRun(task_id="a1", session_id="s1", status="running")

    a = Agent()

    async def profile_for(task):
        return {"profileId": "p", "loginPhone": "9000012345"}

    await runtime._start_run(db, a, t, profile_for)
    assert "ONE restaurant: Indian Bawarchi" in a.goal and "1 x Paneer Butter Masala ₹149" in a.goal
    assert "find in the same restaurant's menu and add: 2 x roti" in a.goal and "never items from another restaurant" in a.goal


async def test_a_cart_without_a_total_gets_one_look_at_the_checkout_before_the_confirm(db, at):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.sim.agent import CART, FakeAgent

    at("2026-10-11 01:30")
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    no_total = {**CART, "total": None, "fees": None}
    agent = FakeAgent(script={"prepare": [no_total, CART]})
    told = []

    async def notify(f, r, p):
        told.append(p)

    t = await runtime.create(db, family_id="fam-food", subject_id="e", requested_by="e", service="instamart", kind="order", goal="atta",
                             details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]})
    await db.commit()
    for _ in range(4):
        await runtime.tick(sessions, agent, profile_for=lambda task: _p(), notify=notify)
    await db.refresh(t)
    assert "open the cart and go to checkout up to the payment step. Change nothing" in agent.runs[1]["goal"]
    assert t.status == "awaiting_confirm" and t.result["total"] == "₹318" and "total ₹318" in told[-1]


async def _p():
    return {"profileId": "p", "loginPhone": "9000012345"}


async def test_the_swiggy_look_up_keeps_restaurant_and_closed_state(monkeypatch):
    from app.tasks import fastpath

    async def fake_eval(cdp, url, js, **kw):
        return {"status": 200, "products": [{"name": "Masala Dosa", "price": 120, "available": False, "id": "d1", "restaurant": "Sagar Ratna",
                                             "restaurantId": "555", "closed": True, "opens": "Opens at 7 am"},
                                            {"name": "Masala Dosa", "price": 90, "available": True, "id": "d2", "restaurant": "Anand", "restaurantId": "777",
                                             "closed": False, "opens": ""}]}

    monkeypatch.setattr(fastpath, "cdp_evaluate", fake_eval)
    out = await fastpath.search("swiggy", "ws://x", "masala dosa", lat=21.2, lon=81.6)
    a, b = out["items"]
    assert a["restaurantId"] == "555" and a["closed"] is True and a["opens"] == "Opens at 7 am"
    assert b["restaurantId"] == "777" and "closed" not in b and "opens" not in b


async def test_connector_food_look_up_searches_side_dishes_in_the_main_dishes_restaurant(db, at):
    at("2026-10-11 02:00")
    calls = []

    class Host:
        async def call(self, tool, args, *, family_id, subject_id, actor_id):
            calls.append((tool, args))
            if tool == "connector_search" and args["item"] == "paneer butter masala":
                return {"ok": True, "items": [{"name": "Paneer Butter Masala", "price": "₹149",
                                               "ref": {"menuItemId": "m1", "restaurantId": 11, "restaurantName": "Indian Bawarchi"}}]}
            if tool == "connector_search":
                return {"ok": True, "items": [{"name": "Tawa Roti", "price": "₹12",
                                               "ref": {"menuItemId": "m2", "restaurantId": 11, "restaurantName": args.get("restaurant") or "?"}}]}
            return {}

    t = await runtime.create(db, family_id="fam-food", subject_id="e", requested_by="e", service="swiggy", kind="order", goal="paneer, roti",
                             details={"items": [{"name": "paneer butter masala", "qty": 1}, {"name": "roti", "qty": 2}]})
    out = await runtime._connector_lookup(db, Host(), t)
    roti_search = next(a for tool, a in calls if tool == "connector_search" and a["item"] == "roti")
    assert roti_search["restaurant"] == "Indian Bawarchi", "the side dish is looked for in the main dish's restaurant"
    assert {i["restaurantId"] for i in out["items"]} == {"11"} and out["items"][0]["restaurant"] == "Indian Bawarchi"


async def test_a_browser_only_family_skips_the_connector(db, at, monkeypatch):
    from app.specialists import channels

    monkeypatch.setenv("BROWSER_ONLY_FAMILIES", "fam-food")
    t = await runtime.create(db, family_id="fam-food", subject_id="e", requested_by="e", service="swiggy", kind="order", goal="dosa",
                             details={"items": [{"name": "masala dosa"}]})

    class Host:
        async def call(self, *a, **kw):
            raise AssertionError("the connector is not asked")

    assert (await channels.pick_channel(db, Host(), t))[0] == "browser"


async def test_a_named_restaurant_without_the_dish_offers_other_places_on_the_same_app(db, at):
    at("2026-10-11 02:00")
    t = await runtime.create(db, family_id="fam-food", subject_id="e", requested_by="e", service="swiggy", kind="order", goal="paneer",
                             details={"items": [{"name": "paneer butter masala", "qty": 1, "restaurant": "Haldiram"}]})
    t.phase = "browse"
    t.details = {**t.details, "matched": {"paneer butter masala": {"exact": [], "closest": None}}}
    found = [D("p1", "Paneer Butter Masala", "Indian Bawarchi", "₹149"), D("p2", "Paneer Butter Masala", "Patiala Plates", "₹318")]
    status, message = runtime._browse_outcome(t, {"items": found, "deliverable": True})
    assert status == "awaiting_confirm" and t.input_needed == "go"
    assert message.startswith("Haldiram on Swiggy does not have it now. Other places on Swiggy have it: For paneer butter masala: [p1]")
    assert "Indian Bawarchi" in message and not t.details.get("fell_back")


async def test_a_cart_on_another_address_is_switched_once_then_stopped(db, at, monkeypatch):
    from app.tasks import matcher

    at("2026-10-11 02:30")
    t = await runtime.create(db, family_id="fam-food", subject_id="e", requested_by="e", service="swiggy", kind="order", goal="paneer",
                             details={"items": [{"name": "paneer butter masala"}]})
    t.phase = "prepare"
    t.details = {**t.details, "limits": {**t.details["limits"], "place": {"nickname": "Home", "full": "C504, Sunita Park, Labhandih, Raipur 492001", "pincode": "492001"}}}
    seen = []

    async def different(saved, pin, shown):
        seen.append(shown)
        return False

    monkeypatch.setattr(matcher, "same_place", different)
    out = {"items": [{"name": "Paneer Butter Masala", "qty": 1}], "total": "₹200", "address_used": "Home: 504, Block C (lilly), Purena"}
    status, message = await runtime._check_address(t, out, "awaiting_confirm", "The cart is ready")
    assert status == "queued" and message is None and t.details["address_fix_pending"] is True and t.input_needed is None
    goal = runtime._goal(t)
    assert "The delivery address is wrong" in goal and "C504, Sunita Park" in goal and "never by a label" in goal
    status, message = await runtime._check_address(t, out, "awaiting_confirm", "The cart is ready")
    assert status == "failed" and "keeps using another delivery address" in message and seen == [out["address_used"]] * 2

    async def same(saved, pin, shown):
        return True

    monkeypatch.setattr(matcher, "same_place", same)
    t.details = {**t.details, "address_fix": None}
    assert (await runtime._check_address(t, out, "awaiting_confirm", "ok")) == ("awaiting_confirm", "ok")


def test_the_agent_is_given_the_address_line_not_the_label():
    from app.specialists.agents import specialist_for

    class T:
        service, kind, phase, family_id = "swiggy", "order", "browse", "fam-x"
        details = {"items": [{"name": "dosa", "qty": 1}], "limits": {"place": {"nickname": "Home", "full": "C504, Sunita Park, Raipur 492001", "pincode": "492001"}}}
        result = {}
        goal = "dosa"

    T.phase = "place"
    T.details = {**T.details, "confirmed_total": "₹200"}
    goal = specialist_for("swiggy").goal(T())
    assert "reads like: 'C504, Sunita Park, Raipur 492001'" in goal and "never by the label" in goal
