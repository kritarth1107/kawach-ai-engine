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
