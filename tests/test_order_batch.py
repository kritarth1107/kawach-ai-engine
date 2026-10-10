"""Order improvements of 2026-10-10 night: confirm buttons, family tastes for the matcher, delivery tracking after placing,
the daily store health check and the product-page hint for the browser agent."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.care import outcomes, store
from app.care.models import OpenLoop
from app.tasks import fastpath, runtime, store_health, tracking

FAM, ELDER, SON = "fam-ob", "elder-ob", "son-ob"


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


async def _placed(db, service="instamart", eta="12 min", order_id="OID1"):
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service=service, kind="order",
                             goal="Amul milk 1L", details={"items": [{"name": "Amul milk 1L", "qty": 1}]})
    t.status, t.phase = "done", "place"
    t.result = {"placed": True, "order_id": order_id, "total": "₹68", "eta": eta}
    await runtime.delivery_followup(db, t)
    tracking.start(t)
    await db.commit()
    return t


class StatusHost:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    async def call(self, tool, args, *, family_id, subject_id, actor_id):
        self.calls.append((tool, args))
        return self.answers.pop(0) if self.answers else {"found": False}


def _judge(states):
    states = list(states)

    async def fake(status, now):
        return states.pop(0) if states else {"state": "unknown", "minutes_left": None}
    return fake


async def test_connector_order_is_told_on_the_way_once_and_delivered_moves_to_how_was_it(db, at, sessions, monkeypatch):
    at("2026-10-10 20:00")
    t = await _placed(db)
    told = []

    async def notify(fam, person, text):
        told.append((person, text))

    host = StatusHost([{"found": True, "status": "PICKED_UP", "currentStatus": "Out for delivery"},
                       {"found": True, "status": "PICKED_UP", "currentStatus": "Out for delivery"},
                       {"found": True, "status": "DELIVERED", "currentStatus": "Order delivered by Suyash"}])
    monkeypatch.setattr(tracking, "judge", _judge([{"state": "on_the_way", "minutes_left": 6}, {"state": "on_the_way", "minutes_left": 3},
                                                   {"state": "delivered", "minutes_left": None}]))
    assert await tracking.sweep(sessions, None, notify, host_for=lambda f: host) == 0, "not due yet"
    at("2026-10-10 20:05")
    assert await tracking.sweep(sessions, None, notify, host_for=lambda f: host) == 1
    assert host.calls[0] == ("connector_order_status", {"store": "instamart", "order_id": "OID1"})
    assert len(told) == 1 and told[0][0] == ELDER and "on the way, arriving in about 6 min" in told[0][1] and "ask nothing" in told[0][1]
    at("2026-10-10 20:10")
    await tracking.sweep(sessions, None, notify, host_for=lambda f: host)
    assert len(told) == 1, "on the way is told once"
    at("2026-10-10 20:15")
    await tracking.sweep(sessions, None, notify, host_for=lambda f: host)
    assert len(told) == 2 and "was delivered" in told[1][1]
    assert "ask whether they got it and everything is all right" in told[1][1], "she ordered it for herself: asked at once"
    await db.refresh(t)
    assert t.result["delivered"] is True and t.details["tracking"]["done"]
    loop = (await db.execute(select(OpenLoop).where(OpenLoop.dedupe_key == f"delivery:{t.id}"))).scalar_one()
    assert loop.status == "open" and loop.detail["stage"] == "feedback", "next: how was it"
    assert loop.wake_at.isoformat().startswith("2026-10-10T17:45"), "groceries: asked 3 hours after it came (IST 23:15 → deferred by quiet hours)"
    at("2026-10-10 20:30")
    assert await tracking.sweep(sessions, None, notify, host_for=lambda f: host) == 0, "finished: no more looks"


async def test_a_store_cancellation_is_told_plainly_and_the_task_is_cancelled(db, at, sessions, monkeypatch):
    at("2026-10-10 20:00")
    t = await _placed(db)
    told = []

    async def notify(fam, person, text):
        told.append(text)

    host = StatusHost([{"found": True, "status": "CANCELLED", "currentStatus": "Cancelled by store: items out of stock"}])
    monkeypatch.setattr(tracking, "judge", _judge([{"state": "cancelled", "minutes_left": None}]))
    at("2026-10-10 20:05")
    await tracking.sweep(sessions, None, notify, host_for=lambda f: host)
    await db.refresh(t)
    assert t.status == "cancelled" and t.result["store_cancelled"] is True
    assert "store cancelled it (Cancelled by store: items out of stock)" in told[0] and "offer to order it again" in told[0]


async def test_an_order_not_in_the_list_stops_being_looked_for(db, at, sessions, monkeypatch):
    at("2026-10-10 20:00")
    t = await _placed(db)
    judged = []

    async def judge(status, now):
        judged.append(status)
        return {"state": "unknown", "minutes_left": None}

    monkeypatch.setattr(tracking, "judge", judge)
    host = StatusHost([])

    async def notify(*a):
        raise AssertionError("nothing to tell")

    for minute in (5, 10, 15, 20, 25):
        at(f"2026-10-10 20:{minute:02d}")
        await tracking.sweep(sessions, None, notify, host_for=lambda f: host)
    await db.refresh(t)
    assert t.details["tracking"]["done"] and len(host.calls) == tracking.NOT_FOUND_LIMIT and not judged


class LookAgent:
    def __init__(self):
        self.opened, self.stopped = [], []

    async def open_session(self, profile):
        self.opened.append(profile)
        return f"s{len(self.opened)}"

    async def cdp_url(self, sid):
        return f"ws://{sid}"

    async def stop_session(self, sid):
        self.stopped.append(sid)


async def test_blinkit_is_looked_at_a_few_times_in_the_familys_browser(db, at, sessions, monkeypatch):
    at("2026-10-10 20:00")
    t = await _placed(db, service="blinkit", eta="10 min", order_id="BK9")
    assert t.details["tracking"]["next_at"].startswith("2026-10-10T14:37"), "first look a little before the store's time (IST 20:07)"
    agent, looked = LookAgent(), []

    async def status(cdp, oid):
        looked.append((cdp, oid))
        return {"status": "ORDER_PLACED", "text": "Order is being packed"}

    async def profile_for(task):
        return {"profileId": "prof-maa"}

    monkeypatch.setattr(fastpath, "blinkit_order_status", status)
    monkeypatch.setattr(tracking, "judge", _judge([{"state": "preparing", "minutes_left": None}] * 5))

    async def notify(*a):
        raise AssertionError("preparing is not told")

    for when in ("20:08", "20:20", "20:40", "21:00"):
        at(f"2026-10-10 {when}")
        await tracking.sweep(sessions, agent, notify, profile_for=profile_for)
    await db.refresh(t)
    assert len(looked) == tracking.BROWSER_LOOKS and looked[0] == ("ws://s1", "BK9") and agent.opened == ["prof-maa"] * 3
    assert agent.stopped == ["s1", "s2", "s3"], "every browser look is closed"
    assert t.details["tracking"]["done"]
    loop = (await db.execute(select(OpenLoop).where(OpenLoop.dedupe_key == f"delivery:{t.id}"))).scalar_one()
    assert loop.status == "open", "not seen delivered: the follow-up still asks after the delivery time"


async def test_shadow_and_untracked_orders_are_not_tracked(db, at):
    at("2026-10-10 20:00")
    t = await _placed(db, service="zepto")
    assert "tracking" not in t.details
    s = await runtime.create(db, family_id="shadow:x", subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order",
                             goal="milk", details={"items": [{"name": "milk"}]})
    s.result = {"placed": True, "order_id": "1"}
    tracking.start(s)
    assert "tracking" not in s.details


def test_the_eta_given_at_placing_sets_the_first_look():
    assert tracking._eta_minutes("10-15 min") == 15 and tracking._eta_minutes("8 mins") == 8
    assert tracking._eta_minutes("Tomorrow 9 AM") is None and tracking._eta_minutes(None) is None


# --- order confirm buttons ---------------------------------------------------------------------------------------------


async def _awaiting(db):
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="milk",
                             details={"items": [{"name": "milk"}]})
    t.status, t.input_needed, t.phase = "awaiting_confirm", "confirm", "prepare"
    t.result = {"items": [{"name": "Amul Taaza 1L", "qty": 1}], "total": "₹68"}
    from app.specialists import guard

    t.details = {**t.details, "cart_fp": guard.cart_fingerprint(t.result)}
    await db.commit()
    return t


def test_order_buttons_carry_the_task_and_cart():
    b = outcomes.buttons("order", "abc:1234abcd", {"language": "hi"})
    assert [x["id"] for x in b] == ["v2:od:yes:abc:1234abcd", "v2:od:change:abc:1234abcd", "v2:od:cancel:abc:1234abcd"]
    assert outcomes.parse_button("v2:od:yes:abc:1234abcd") == ("od", "yes", "abc:1234abcd")


async def test_tapping_yes_confirms_the_cart_it_was_shown_with(db, at, monkeypatch):
    at("2026-10-10 20:00")
    t = await _awaiting(db)
    key = f"{t.id}:{t.details['cart_fp'][:8]}"
    reply, note = await outcomes.handle_order_button(db, family_id=FAM, speaker_id=ELDER, elder_id=ELDER, code="yes", task_id=key, profile=None)
    await db.commit()
    await db.refresh(t)
    assert reply and note is None and t.phase == "place" and t.details["confirmed_by"] == ELDER


async def test_a_yes_on_an_older_confirm_places_nothing(db, at):
    at("2026-10-10 20:00")
    t = await _awaiting(db)
    reply, note = await outcomes.handle_order_button(db, family_id=FAM, speaker_id=ELDER, elder_id=ELDER, code="yes",
                                                     task_id=f"{t.id}:00000000", profile=None)
    await db.commit()
    await db.refresh(t)
    assert reply is None and "changed since" in note and t.status == "awaiting_confirm" and not t.details.get("confirmed_by")


async def test_change_goes_to_the_brain_and_cancel_drops_it(db, at, monkeypatch):
    at("2026-10-10 20:00")
    t = await _awaiting(db)
    reply, note = await outcomes.handle_order_button(db, family_id=FAM, speaker_id=ELDER, elder_id=ELDER, code="change", task_id=str(t.id), profile=None)
    assert reply is None and "what they want to change" in note
    cancelled = []

    async def request_cancel(session, agent, task, *, by, reason):
        cancelled.append((task.id, by, reason))

    monkeypatch.setattr(runtime, "request_cancel", request_cancel)
    reply, note = await outcomes.handle_order_button(db, family_id=FAM, speaker_id=ELDER, elder_id=ELDER, code="cancel", task_id=str(t.id), profile=None)
    assert reply and cancelled == [(t.id, ELDER, "tapped cancel")]
    other, _ = await outcomes.handle_order_button(db, family_id="fam-other", speaker_id=ELDER, elder_id=ELDER, code="yes", task_id=str(t.id), profile=None)
    assert other == outcomes.ORDER_TAP["english"]["done"], "another family's order is never touched"


def test_the_confirm_question_asks_for_the_buttons_with_the_cart_check():
    class T:
        id = "tid"
        service = "blinkit"
        details = {"cart_fp": "abcdef1234567890", "chosen": [], "limits": {}}

    assert "key: tid:abcdef12" in runtime._confirm_ask(T(), {"total": "₹68"})


# --- family tastes, product pages --------------------------------------------------------------------------------------


async def test_family_tastes_list_what_they_bought_and_like(db, at):
    at("2026-10-10 20:00")
    old = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order", goal="khakhra",
                               details={"items": [{"name": "methi khakhra"}], "chosen": [{"name": "Charliee Methi Khakhra", "pack": "200 g"}]})
    old.status, old.result = "done", {"placed": True, "order_id": "1"}
    await db.commit()
    now = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order", goal="khakhra",
                               details={"items": [{"name": "khakhra"}]})
    tastes = await runtime.family_tastes(db, now)
    assert any("bought before on Swiggy Instamart: Charliee Methi Khakhra (200 g)" in x for x in tastes)


def test_blinkit_product_page_for_the_browser_agent():
    assert runtime.product_page("blinkit", {"store_id": "12345"}) == "https://blinkit.com/prn/x/prid/12345"
    assert runtime.product_page("blinkit", {"cart_ref": {"product_id": 9}}) == "https://blinkit.com/prn/x/prid/9"
    assert runtime.product_page("instamart", {"store_id": "1"}) is None


# --- daily store health check ------------------------------------------------------------------------------------------


class GuestAgent(LookAgent):
    pass


async def test_store_health_retries_a_cold_page_before_alerting(monkeypatch):
    tries: dict[str, int] = {}

    async def search(service, cdp, q, **kw):
        tries[service] = tries.get(service, 0) + 1
        if service == "instamart" and tries[service] == 1:
            raise fastpath.FastPathError("instamart search answered 202")
        if service == "1mg":
            raise fastpath.FastPathError("Failed to parse URL")
        return {"items": [{"name": q}]}

    async def fares(service, cdp, **kw):
        return {"options": [{"type": "auto", "fare": "₹90"}]}

    monkeypatch.setattr(fastpath, "search", search)
    monkeypatch.setattr(fastpath, "fares", fares)
    alerts = []

    async def alert(subject, text):
        alerts.append((subject, text))

    agent = GuestAgent()
    out = await store_health.check(agent, alert, retry_wait_s=0)
    assert out["instamart"] == "ok", "a cold first page is tried once more"
    assert out["1mg"].startswith("failed") and tries["1mg"] == 2
    assert len(alerts) == 1 and alerts[0][0] == "Store check: saved look-up broke on 1mg"
    assert agent.opened == [None] and agent.stopped == ["s1"], "one guest browser, closed after"


async def test_store_health_is_quiet_when_all_stores_work(monkeypatch):
    async def search(service, cdp, q, **kw):
        return {"items": [{"name": q}]}

    async def fares(service, cdp, **kw):
        return {"login_required": True}

    monkeypatch.setattr(fastpath, "search", search)
    monkeypatch.setattr(fastpath, "fares", fares)

    async def alert(*a):
        raise AssertionError("no alert")

    out = await store_health.check(GuestAgent(), alert, retry_wait_s=0)
    assert set(out.values()) == {"ok"}
    assert await store_health.check(object()) == {"skipped": "no cloud browser"}


# --- Instamart: a sent order with no clear answer is looked up in the linked account -------------------------------------


async def test_an_unclear_instamart_place_is_found_in_the_order_list(db, at, monkeypatch):
    at("2026-10-10 20:00")
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order", goal="milk",
                             details={"items": [{"name": "milk"}]})
    t.phase, t.agent_session = "place", "s1"
    t.details = {**t.details, "fast_place_check": {"total": 68, "count": 1}, "place_started": "2026-10-10T14:29:00+00:00"}
    await db.commit()

    async def place(service, cdp, check):
        return {"placed": False, "unclear": True}

    monkeypatch.setattr(fastpath, "place", place)
    host = StatusHost([{"orderId": "250600000000001", "eta": "18 mins"}])
    out = await runtime._fast_place(db, LookAgent(), t, None, host)
    assert out["placed"] and out["order_id"] == "250600000000001" and out["eta"] == "18 mins"
    tool, args = host.calls[0]
    assert tool == "connector_recent_order" and args["store"] == "instamart" and args["total"] == 68
    from datetime import datetime
    assert args["since_ms"] == int(datetime.fromisoformat("2026-10-10T14:29:00+00:00").timestamp() * 1000), "since the place was sent"
    host = StatusHost([{"orderId": None}])
    out = await runtime._fast_place(db, LookAgent(), t, None, host)
    assert out.get("unclear") and not out.get("placed"), "not found: stays unclear, never placed again"


# --- a tap reaches the order without a model call -----------------------------------------------------------------------


async def test_a_yes_tap_in_chat_places_without_the_model(db, at, monkeypatch):
    from app.brain.loop import TurnRequest, run_turn
    from app.sim.world import SimHost

    at("2026-10-10 20:00")
    t = await _awaiting(db)
    E = {"id": ELDER, "name": "Vasundara", "role": "elder"}
    await store.save_roster(db, FAM, E, [E])

    class Shadow(SimHost):
        shadow = True

    res = await run_turn(db, Shadow(), TurnRequest(family_id=FAM, elder=E, speaker=E, members=[E],
                                                  text=f"v2:od:yes:{t.id}:{t.details['cart_fp'][:8]}", message_ref="b1"))
    await db.refresh(t)
    assert res.model == "none" and res.actions[0]["tool"] == "order_button" and t.phase == "place"


# --- an order's later update is not taken for a repeat of the earlier one --------------------------------------------


async def test_a_delivered_update_is_not_blocked_as_a_repeat_of_placed(db, at):
    from app.brain import tools

    at("2026-10-10 20:00")
    E = {"id": ELDER, "name": "Vasundara", "role": "elder"}
    placed = "आपका ब्लिंकिट ऑर्डर हो गया है (आईडी: SIM1)। सामान 10 मिनट में घर पहुँच जाएगा।"
    await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="assistant", text=placed, meta={"proactive": True})
    await db.commit()
    at("2026-10-10 20:15")
    update = tools.TurnCtx(session=db, host=None, family_id=FAM, elder=E, speaker={"id": "saheli-scheduler", "role": "system"}, members=[E],
                           message_ref="task:abc")
    assert not [p for p in await tools.send_problems(update, ELDER, "आपका ब्लिंकिट सामान घर पहुँच गया है।") if "already sent" in p]
    assert [p for p in await tools.send_problems(update, ELDER, placed) if "already sent" in p], "the same message twice is still stopped"
    nudge = tools.TurnCtx(session=db, host=None, family_id=FAM, elder=E, speaker={"id": "saheli-scheduler", "role": "system"}, members=[E],
                          message_ref="loop:x")
    assert [p for p in await tools.send_problems(nudge, ELDER, "आपका ब्लिंकिट सामान घर पहुँच गया है।") if "already sent" in p]


async def test_a_tap_is_answered_in_the_script_the_confirm_was_written_in(db, at):
    from app.brain.loop import TurnRequest, run_turn
    from app.sim.world import SimHost

    at("2026-10-11 10:30")
    t = await _awaiting(db)
    E = {"id": ELDER, "name": "Vasundara", "role": "elder"}
    await store.save_roster(db, FAM, E, [E])
    await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="blinkit se ek Amul butter mangwa do", meta={})
    await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="assistant", text="ब्लिंकिट पर अमूल बटर तैयार है, ₹75। ऑर्डर कर दूँ?", meta={})
    await db.commit()

    class Shadow(SimHost):
        shadow = True

    res = await run_turn(db, Shadow(), TurnRequest(family_id=FAM, elder=E, speaker=E, members=[E],
                                                  text=f"v2:od:yes:{t.id}:{t.details['cart_fp'][:8]}", message_ref="b2"))
    assert res.reply == outcomes.ORDER_TAP["devanagari"]["yes"]
