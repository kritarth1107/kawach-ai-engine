"""Never ordered: tobacco, gutka, vapes, alcohol, illegal drugs (founder 2026-10-13: "get me classic ice brust" was being
looked up on three stores; "it should not even try to order; say directly it can't and suggest an alternative"). Two AI
judgements, each enough: the brain's own on each item, and a separate check of the names. Code acts on their answers;
no list is matched against anyone's words. No task, no search, no address question; a store listing that is restricted
is never shown or carted."""

import pytest

from app.brain import tools
from app.care import restricted, store
from app.sim.world import SimHost
from app.tasks import matcher, runtime
from app.tasks.models import Task

FAM, ME = "fam-rs", "self-rs"
REAL_CHECK = matcher.restricted_kinds
M = {"id": ME, "name": "Vikram", "role": "elder"}
# what the separate AI check answers for these names, faked (the real model is exercised in eval/order_lang_eval.py)
CHECK = {"classic ice brust": "tobacco", "kingfisher ultra 2 bottles": "alcohol", "rajnigandha pan masala": "gutka",
         "vape pod mint": "vape", "ganja": "illegal", "cadbury flake": None, "kingfisher packaged drinking water": None,
         "apple cider vinegar": None, "nicotex nicotine gum": None, "sugar-free mints": None}


@pytest.fixture(autouse=True)
def ai_check(monkeypatch):
    seen = []

    async def kinds(names):
        seen.append(list(names))
        return [CHECK.get(n.lower()) for n in names]

    monkeypatch.setattr(matcher, "restricted_kinds", kinds)
    return seen


def _ctx(db, ref="m1"):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=M, speaker=M, members=[M], message_ref=ref)


async def _start(db, name, restricted_said=None, **extra):
    await store.save_roster(db, FAM, M, [M])
    item = {"name": name, **({"restricted": restricted_said} if restricted_said else {})}
    return await tools.start_task(_ctx(db), {"kind": "order", "goal": name, "items": [item], "for_whom": ME, **extra})


@pytest.mark.parametrize("name,kind", [("classic ice brust", "tobacco"), ("Kingfisher Ultra 2 bottles", "alcohol"),
                                       ("Rajnigandha pan masala", "gutka"), ("vape pod mint", "vape")])
async def test_the_ai_check_alone_refuses_before_anything_starts(db, at, name, kind):
    at("2026-10-13 14:59")
    out = await _start(db, name, category="grocery")
    assert out["status"] == "not started" and out["refused"] == kind and out["order"] == "none"
    assert out["alternatives"] == list(restricted.substitutes(kind)) and "confirm_address" not in out
    assert not await runtime.live_tasks(db, FAM)
    events = await store.events(db, FAM, ME, kinds=["order_refused"])
    assert events and events[-1].payload["kind"] == kind and events[-1].payload["by_check"]


async def test_the_brains_own_judgement_alone_refuses_too(db, at):
    at("2026-10-13 15:00")
    out = await _start(db, "Gold Flake Kings", restricted_said="tobacco", service="blinkit")  # the check knows nothing of it
    assert out["refused"] == "tobacco" and not await runtime.live_tasks(db, FAM)


async def test_an_illegal_drug_is_refused_and_never_talked_about(db, at):
    at("2026-10-13 15:01")
    out = await _start(db, "ganja", category="grocery")
    assert out["refused"] == "illegal" and "alternatives" not in out and "say nothing about the substance" in out["say"]


@pytest.mark.parametrize("name", ["Cadbury Flake", "Kingfisher packaged drinking water", "apple cider vinegar", "Nicotex nicotine gum",
                                  "sugar-free mints"])
async def test_harmless_things_that_share_a_name_are_ordered(db, at, name):
    at("2026-10-13 15:02")
    out = await _start(db, name, service="blinkit")
    assert "refused" not in out and out["status"] in ("not started yet", "started")


async def test_a_restricted_item_added_to_a_running_order_is_refused(db, at):
    at("2026-10-13 15:03")
    await store.save_roster(db, FAM, M, [M])
    t = await runtime.create(db, family_id=FAM, subject_id=ME, requested_by=ME, service="blinkit", kind="order", goal="mints",
                             details={"items": [{"name": "sugar-free mints"}], "for_set": True})
    t.status, t.phase, t.input_needed = "awaiting_confirm", "browse", "go"
    await db.commit()
    out = await tools.task_input(_ctx(db, "m2"), {"task_id": str(t.id), "kind": "add", "items": [{"name": "classic ice brust"}]})
    assert out["refused"] == "tobacco" and "nothing was added" in out["next"]
    assert [i["name"] for i in t.details["items"]] == ["sugar-free mints"]


async def test_a_restricted_store_listing_is_never_shown_or_picked(db, at, monkeypatch):
    """A "mint" search that returns a cigarette brand: the matcher marks it; it is dropped from what is shown."""
    at("2026-10-13 15:04")

    async def choose(items, listings, **kw):
        return [{"exact": [1], "closest": None, "restricted": [0], "why": "mints; listing 1 is cigarettes"}]

    monkeypatch.setattr(matcher, "choose", choose)
    t = Task(service="blinkit", kind="order", status="queued", phase="browse", goal="mints", details={"items": [{"name": "mints"}]},
             result={})
    report = {"items": [{"name": "Classic Ice Burst Cigarettes (10)", "price": "₹190", "available": True},
                        {"name": "Polo Mint Roll", "price": "₹10", "available": True}]}
    t.result = dict(report)
    await runtime._ai_pick(t, report, None)
    names = [i["name"] for i in t.result["items"]]
    assert names == ["Polo Mint Roll"], "the saved result, a new dict (never changed in place)"
    assert any("restricted listing" in h["note"] for h in t.history)


def test_the_check_is_given_the_lists_as_reference_never_matches_them_in_code():
    assert "cigarette" in matcher.RESTRICTED_PROMPT.format(reference="cigarette").lower()
    import inspect

    src = inspect.getsource(tools._restricted)
    assert "category(" not in src, "the guard acts on the AI's answers, not on a word list"


async def test_when_everything_the_store_has_for_it_is_restricted_the_task_ends_with_the_refusal(db, at, monkeypatch):
    """The third check: a brand name both earlier checks missed; the store's listings for it are all cigarettes."""
    at("2026-10-13 15:05")

    async def choose(items, listings, **kw):
        return [{"exact": [], "closest": None, "restricted": [0, 1], "why": "both are cigarettes"}]

    monkeypatch.setattr(matcher, "choose", choose)
    t = Task(service="blinkit", kind="order", status="running", phase="browse", goal="Ice Burst", details={"items": [{"name": "Ice Burst"}]},
             result={})
    report = {"items": [{"name": "Classic Ice Burst Cigarettes (10)", "price": "₹190", "available": True},
                        {"name": "Classic Ice Burst Cigarettes (20)", "price": "₹380", "available": True}]}
    t.result = dict(report)
    await runtime._ai_pick(t, report, None)
    status, said = runtime._outcome(t, t.result)
    assert status == "failed" and "NOT ordered, and never will be" in said and "mints" in said
    assert t.result["items"] == []


async def test_the_separate_check_answers_once_per_order_not_on_every_call(monkeypatch):
    from app.llm import router

    calls = []

    class Reply:
        text = '{"kinds": ["none"]}'

    async def complete(*a, **k):
        calls.append(k.get("timeout_s"))
        return Reply()

    monkeypatch.setattr(matcher, "_KIND_CACHE", {})
    monkeypatch.setattr(router, "complete", complete)
    assert await REAL_CHECK(["Amul butter"]) == [None]
    assert await REAL_CHECK(["amul butter"]) == [None]
    assert len(calls) == 1 and calls[0] <= 6, "asked once, and never waited on for long"


async def test_through_a_real_look_up_the_restricted_listing_is_gone_from_what_is_saved(db, at, monkeypatch):
    """Sim 2026-10-14: the listing was left out of what was offered, but the saved result still held it (an in-place change
    to an already-written result was never written). The whole tick, a fast look-up on Blinkit, the saved task checked."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.sim.agent import FakeAgent
    from app.tasks import fastpath

    class Agent(FakeAgent):
        async def open_session(self, p):
            return "s1"

        async def cdp_url(self, s):
            return "ws://x"

    monkeypatch.setenv("TASK_BROWSE_FIRST", "on")
    at("2026-10-14 01:00")
    listings = [{"name": "Classic Ice Burst Menthol Mint Cigarettes", "price": "₹190", "available": True, "store_id": "b13"},
                {"name": "Polo Mint Roll", "price": "₹10", "available": True, "store_id": "b14"}]

    async def search(service, cdp, q, **kw):
        return {"items": [dict(x) for x in listings], "deliverable": True, "eta": "10 min", "logged_in": True}

    async def choose(items, ls, **kw):
        return [{"exact": [1], "closest": None, "restricted": [0], "why": "listing 1 is cigarettes"}]

    monkeypatch.setattr(fastpath, "search", search)
    monkeypatch.setattr(matcher, "choose", choose)
    t = await runtime.create(db, family_id=FAM, subject_id=ME, requested_by=ME, service="blinkit", kind="order", goal="mints",
                             details={"items": [{"name": "mints"}], "for_set": True})
    t.details = {**t.details, "limits": {**(t.details.get("limits") or {}), "place": {
        "addressId": "a1", "nickname": "Ghar", "full": "12 Sample Lines", "lat": 26.85, "lng": 80.95, "pincode": "226001"}}}
    await db.commit()
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")

    async def notify(*a):
        return None

    async def prof(task):
        return "p"

    for _ in range(3):
        await runtime.tick(sessions, Agent(), profile_for=prof, notify=notify)
    await db.refresh(t)
    assert t.status == "awaiting_confirm"
    assert [i["name"] for i in t.result["items"]] == ["Polo Mint Roll"]
    assert [i["name"] for i in t.details["found"]] == ["Polo Mint Roll"]
