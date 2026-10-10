"""After an order (founder 2026-10-11): Saheli asks whether it came; a long-overdue order the person says has not come is
told to the caregiver; once it came she asks how it was, one question at a time, and remembers what they liked, whether it
was a first try, and what never to order again."""

import json
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import tools, wake
from app.care import store
from app.care.models import OpenLoop
from app.core import clock
from app.sim.world import SimHost
from app.tasks import aftercare, runtime

FAM, ELDER, SON = "fam-ac", "elder-ac", "son-ac"
E = {"id": ELDER, "name": "Vasundara Devi", "role": "elder"}
S = {"id": SON, "name": "Kritarth", "role": "primary caregiver"}
CART = [{"name": "Amul Taaza Toned Milk", "qty": 2, "price": "₹56"}, {"name": "Charliee Methi Khakhra", "qty": 1, "price": "₹65"}]


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


async def _placed(db, *, service="blinkit", eta="10 min", items=None, requested_by=ELDER, members=(E, S), talks=True):
    await store.save_roster(db, FAM, E, list(members))
    if talks:
        await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="doodh mangwa do", meta={})
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=requested_by, service=service, kind="order",
                             goal="Milk and khakhra", details={"items": [{"name": "milk", "qty": 2}, {"name": "methi khakhra"}]})
    t.status, t.phase = "done", "place"
    t.result = {"placed": True, "order_id": "OD1", "total": "₹136", "eta": eta, "items": items or CART}
    await runtime.delivery_followup(db, t)
    await db.commit()
    return t


async def _loop(db, t) -> OpenLoop:
    return (await db.execute(select(OpenLoop).where(OpenLoop.dedupe_key == f"delivery:{t.id}").order_by(OpenLoop.created_at.desc()))).scalars().first()


def _ctx(db, speaker=E):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=speaker, members=[E, S], message_ref="m1")


async def _say(db, args, speaker=E) -> dict:
    out, err = await tools.run(_ctx(db, speaker), "order_feedback", args)
    assert not err, out
    await db.commit()
    return json.loads(out)


async def test_every_order_gets_an_arrival_question_after_the_stores_time(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db)
    loop = await _loop(db, t)
    assert loop.detail["stage"] == "arrival" and loop.owner_id == ELDER
    assert loop.wake_at == clock.now() + timedelta(minutes=30), "10 min store time + 20 min"
    assert "1. Amul Taaza Toned Milk x2; 2. Charliee Methi Khakhra" in loop.title and f"[task {t.id}]" in loop.title
    at("2026-10-11 10:30")
    prompt = await wake.wake_prompt(db, loop, ELDER)
    assert prompt.startswith("[Order follow-up] The Blinkit order for person elder-ac")
    assert "placed at 10:00; the store said it would come by about 10:10" in prompt and "has it come all right?" in prompt
    assert f"order_feedback with task_id {t.id}" in prompt and "This store gives no live status." in prompt


async def test_the_person_it_is_for_is_asked_if_they_talk_to_saheli(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db, requested_by=SON)
    assert (await _loop(db, t)).owner_id == ELDER, "Kritarth ordered it; Maa receives it and talks to Saheli"
    loop = await _loop(db, t)
    assert "(Kritarth ordered it for them)" in await wake.wake_prompt(db, loop, ELDER)


async def test_someone_who_never_writes_is_not_asked(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db, requested_by=SON, talks=False)
    assert (await _loop(db, t)).owner_id == SON


async def test_it_came_then_how_was_it_one_question_at_a_time_into_memory(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db)
    at("2026-10-11 10:30")
    loop = await _loop(db, t)
    loop.detail = {**loop.detail, "wakes": 1}  # the arrival question went out
    await db.commit()
    r = await _say(db, {"task_id": str(t.id), "arrived": "yes"})
    assert "Do not ask how it was now" in r["next"], "just arrived: not tried yet"
    loop = await _loop(db, t)
    assert loop.detail["stage"] == "feedback" and loop.wake_at == clock.now() + timedelta(hours=3)
    assert loop.detail["next_action"] == "say how they liked the order"
    # three hours later: the first "how was it?"
    at("2026-10-11 13:30")
    prompt = await wake.wake_prompt(db, loop, ELDER)
    assert prompt.startswith("[Order feedback]") and "how Amul Taaza Toned Milk, Charliee Methi Khakhra were and whether they liked it" in prompt
    loop.detail = {**loop.detail, "wakes": 1}
    await db.commit()
    r = await _say(db, {"task_id": str(t.id), "items": [{"item": 2, "liked": "yes", "note": "crispy"}, {"item": 1, "liked": "yes"}]})
    assert "Likes Charliee Methi Khakhra (crispy)" in r["saved"]
    assert "whether Amul Taaza Toned Milk is something they usually have" in r["next"], "first time through Saheli: usual or new?"
    r = await _say(db, {"task_id": str(t.id), "items": [{"item": 1, "usual": "usual"}]})
    assert "whether Charliee Methi Khakhra is something they usually have" in r["next"]
    r = await _say(db, {"task_id": str(t.id), "items": [{"item": 2, "usual": "new"}]})
    assert r["next"].startswith("That is all"), "three answers: enough"
    assert await _loop(db, t) is None or (await _loop(db, t)).status == "done"
    facts = {f.key: f for f in await store.facts(db, FAM, ELDER, domains=["preference"], statuses=("active",))}
    k = facts["preference:charliee_methi_khakhra"]
    assert k.text.startswith("Likes Charliee Methi Khakhra; first tried it in Oct 2026 (Blinkit)") and k.value["liked"] == "yes"
    assert facts["preference:amul_taaza_toned_milk"].text == "Likes Amul Taaza Toned Milk; usually has it"
    await db.refresh(t)
    assert t.result["feedback"]["answers"] == 3 and t.result["feedback"]["items"]["2"]["usual"] == "new"
    ev = await store.events(db, FAM, ELDER, kinds=["order_feedback"])
    assert any("Likes Charliee Methi Khakhra" in e.summary for e in ev) and any(e.summary == "Blinkit: it came" for e in ev)


async def test_what_they_did_not_like_is_never_ordered_again_and_the_picker_knows(db, at):
    from app.brain import policy
    from app.specialists.contract import build_limits

    at("2026-10-11 10:00")
    t = await _placed(db, items=[{"name": "Modern Kitchens Butter Muruku", "qty": 1, "price": "₹35"}])
    r = await _say(db, {"task_id": str(t.id), "arrived": "yes", "items": [{"item": 1, "liked": "no", "note": "too oily"}]})
    assert "what was not right about Modern Kitchens Butter Muruku" in r["next"]
    r = await _say(db, {"task_id": str(t.id), "items": [{"item": 1, "again": "no"}]})
    never = (await store.facts(db, FAM, ELDER, domains=["no_order"], statuses=("active",)))[0]
    assert never.value["item"] == "Modern Kitchens Butter Muruku" and "too oily" in never.text
    limits = await build_limits(db, family_id=FAM, subject_id=ELDER, kind="order", agent="shopping", requester_is_elder=True, place={})
    assert policy.order_conflicts("Modern Kitchens Butter Muruku", limits.allergies, limits.never_order), "the cart check stops it"
    assert not policy.order_conflicts("Haldiram Peri Peri Murukku", limits.allergies, limits.never_order), "other muruku is fine"
    nxt = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="blinkit", kind="order", goal="muruku",
                               details={"items": [{"name": "muruku"}]})
    tastes = await runtime.family_tastes(db, nxt)
    assert any("Did not like Modern Kitchens Butter Muruku; do not order it again (too oily)" in x for x in tastes)


async def test_not_come_yet_is_answered_from_the_store_and_checked_again_when_late(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db)
    at("2026-10-11 10:30")
    r = await _say(db, {"task_id": str(t.id), "arrived": "no"})
    assert r["tell"].startswith("It is not late yet.") and "should come by about 10:10" in r["tell"] and "again at 10:40" in r["tell"]
    loop = await _loop(db, t)
    assert loop.detail["stage"] == "arrival" and loop.wake_at.isoformat().startswith("2026-10-11T05:10"), "ask again at 10:40 IST"
    await db.refresh(t)
    assert "caregiver_due" not in (t.details.get("aftercare") or {})


async def test_long_overdue_tells_the_caregiver_once(db, at, sessions):
    at("2026-10-11 10:00")
    t = await _placed(db)
    at("2026-10-11 10:45")
    r = await _say(db, {"task_id": str(t.id), "arrived": "no"})
    assert r["tell"].startswith("Kritarth will be told now") and "do not message them yourself" in r["tell"]
    told = []

    async def notify(fam, person, text):
        told.append((person, text))

    assert await aftercare.sweep(sessions, notify) == 1
    person, text = told[0]
    assert person == SON and text.startswith("[Order needs you] The Blinkit order for person elder-ac")
    assert "it has not come; it was due by 10:10, 35 minutes ago" in text and "order id OD1" in text and "check the Blinkit app" in text
    assert await aftercare.sweep(sessions, notify) == 0, "sent once"
    at("2026-10-11 11:00")
    r = await _say(db, {"task_id": str(t.id), "arrived": "no"})
    assert "already knows" in r["tell"]
    assert await aftercare.sweep(sessions, notify) == 0
    ev = await store.events(db, FAM, ELDER, kinds=["order_caregiver_told"])
    assert len(ev) == 1


async def test_store_says_delivered_but_it_did_not_come_tells_the_caregiver_at_once(db, at, sessions):
    at("2026-10-11 10:00")
    t = await _placed(db)
    t.result = {**t.result, "delivered": True, "tracked_at": clock.now().isoformat()}
    await db.commit()
    at("2026-10-11 10:12")
    r = await _say(db, {"task_id": str(t.id), "arrived": "no"})
    assert r["tell"].startswith("Kritarth will be told now")
    await db.refresh(t)
    assert "the store says it was delivered, but they say it has not come" in t.details["aftercare"]["caregiver_due"]["why"]


async def test_something_wrong_or_missing_goes_to_the_caregiver(db, at, sessions):
    at("2026-10-11 10:00")
    t = await _placed(db)
    at("2026-10-11 10:20")
    r = await _say(db, {"task_id": str(t.id), "arrived": "partly", "problem": "khakhra missing"})
    assert r["tell"].startswith("Kritarth will be told now") and "next" not in r
    told = []

    async def notify(fam, person, text):
        told.append(text)

    await aftercare.sweep(sessions, notify)
    assert "it came with a problem: khakhra missing" in told[0]
    assert (await _loop(db, t)).detail["stage"] == "feedback", "what came is still asked about later"


async def test_self_care_has_no_one_else_to_tell(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db, members=(E,))
    at("2026-10-11 10:50")
    r = await _say(db, {"task_id": str(t.id), "arrived": "no"})
    assert r["tell"].startswith("There is no one else in the family to tell") and "check the Blinkit app" in r["tell"]
    await db.refresh(t)
    assert "caregiver_due" not in (t.details.get("aftercare") or {})


async def test_saying_how_it_was_on_their_own_starts_the_chat_now(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db)
    at("2026-10-11 10:20")
    r = await _say(db, {"items": [{"item": 2, "liked": "yes"}]})  # "khakhra bahut accha hai" (no task id: their latest order)
    assert "how Amul Taaza Toned Milk was" in r["next"], "they are talking about it: the next question now"
    loop = await _loop(db, t)
    assert loop.detail["stage"] == "feedback" and loop.wake_at is None
    await db.refresh(t)
    assert t.result["feedback"]["arrived"] == "yes"


async def test_they_can_stop_the_chat(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db)
    at("2026-10-11 10:20")
    r = await _say(db, {"items": [{"item": 2, "liked": "yes"}], "done": True})
    assert r["next"].startswith("That is all")
    assert (await _loop(db, t)) is None or (await _loop(db, t)).status == "done"


async def test_a_medicine_order_stops_after_it_came(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db, service="1mg", eta="Tomorrow 9 PM", items=[{"name": "Telma 40 Tablet", "qty": 1}])
    loop = await _loop(db, t)
    assert loop.detail["max_wakes"] == 1
    prompt = await wake.wake_prompt(db, loop, ELDER)
    assert "is it the right medicine and strength?" in prompt
    r = await _say(db, {"task_id": str(t.id), "arrived": "yes"})
    assert r["next"] == "Say you are glad in a few words; ask nothing more."
    assert (await _loop(db, t)) is None or (await _loop(db, t)).status == "done"


async def test_feedback_needs_a_placed_order(db, at):
    at("2026-10-11 10:00")
    await store.save_roster(db, FAM, E, [E, S])
    out, err = await tools.run(_ctx(db), "order_feedback", {"arrived": "yes"})
    assert err and "remember (domain preference)" in out


async def test_the_store_saying_delivered_moves_on_and_asks_the_right_person(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db)
    assert await aftercare.store_delivered(db, t, clock.now()) is True, "she ordered it for herself: asked in the update"
    loop = await _loop(db, t)
    assert loop.detail["stage"] == "feedback" and "store_delivered_at" in loop.detail
    t2 = await _placed(db, requested_by=SON, service="instamart")
    assert await aftercare.store_delivered(db, t2, clock.now()) is False, "Kritarth asked; Maa is asked later with how was it"
    loop2 = await _loop(db, t2)
    at("2026-10-11 13:10")
    prompt = await wake.wake_prompt(db, loop2, ELDER)
    assert "The store said it was delivered, but they have not said they got it" in prompt


async def test_an_unanswered_follow_up_asks_once_more_with_how_was_it_then_closes_itself(db, at, sessions):
    at("2026-10-11 10:00")
    t = await _placed(db)
    loop = await _loop(db, t)
    loop.detail = {**loop.detail, "wakes": 1}
    await db.commit()
    prompt = await wake.wake_prompt(db, loop, ELDER)
    assert "they did not answer whether it came" in prompt and "whether it came all right and how" in prompt
    loop.detail = {**loop.detail, "wakes": 2}
    loop.wake_at = clock.now() - timedelta(minutes=1)
    await db.commit()
    at("2026-10-13 10:30")  # past expire_at (2 days)
    stats = await wake.wake_due(sessions, lambda fid: SimHost())
    assert stats["expired"] == 1 and stats["ran"] == 0
    await db.refresh(loop)
    assert loop.status == "expired" and loop.closed_note == "no answer; closed by itself"


async def test_a_follow_up_with_nothing_left_to_ask_is_closed_without_a_message(db, at, sessions):
    at("2026-10-11 10:00")
    t = await _placed(db)
    await aftercare.store_delivered(db, t, clock.now())
    t.result = {**t.result, "feedback": {"arrived": "yes", "answers": 1, "items": {"2": {"liked": "yes"}}}}
    loop = await _loop(db, t)
    loop.wake_at = clock.now()
    await db.commit()
    stats = await wake.wake_due(sessions, lambda fid: SimHost())
    assert stats["ran"] == 0
    await db.refresh(loop)
    assert loop.status == "done" and loop.closed_note == "nothing left to ask"


async def test_usual_buys_are_not_asked_about_again(db, at):
    at("2026-10-09 10:00")
    for _ in range(2):
        old = await _placed(db)
        old.created_at = clock.now() - timedelta(days=1)
    await db.commit()
    at("2026-10-11 10:00")
    t = await _placed(db)
    bought = await aftercare.bought_before(db, t)
    assert bought["amul_taaza_toned_milk"] == 2
    fb = {"answers": 1, "items": {"1": {"liked": "yes"}, "2": {"liked": "yes"}}}
    assert aftercare.next_question(t, {"answers": 1, "items": {"1": {"liked": "yes"}, "2": {"liked": "yes", "usual": "new"}}}, bought) == \
        "whether to get Charliee Methi Khakhra again next time.", "a liked first try: get it again?"
    assert "Amul Taaza Toned Milk is something they usually have" not in (aftercare.next_question(t, fb, bought) or "")
    loop = await _loop(db, t)
    await aftercare.store_delivered(db, t, clock.now())
    prompt = await wake.wake_prompt(db, loop, ELDER)
    assert "Known already, do not ask: Amul Taaza Toned Milk x2 was ordered 2 times before" in prompt or \
           "Amul Taaza Toned Milk was ordered 2 times before (a usual buy)" in prompt


async def test_food_asks_sooner_and_about_the_restaurant(db, at):
    at("2026-10-11 13:00")
    t = await _placed(db, service="swiggy", eta="35 mins", items=[{"name": "Paneer Butter Masala", "qty": 1, "restaurant": "Haldiram's"}])
    loop = await _loop(db, t)
    await aftercare.store_delivered(db, t, clock.now())
    assert loop.wake_at == clock.now() + timedelta(minutes=40)
    r = await _say(db, {"task_id": str(t.id), "items": [{"item": 1, "liked": "yes", "usual": "usual"}]})
    assert "whether they would like food from Haldiram's again" in r["next"]
    r = await _say(db, {"task_id": str(t.id), "order_again": "yes"})
    assert "Likes food from Haldiram's (Swiggy)" in r["saved"]


class AskingBrain:
    """On an order follow-up it asks Maa; otherwise it says nothing."""

    def __init__(self):
        self.seen = []

    async def complete(self, route, *, messages, **kw):
        from app.llm.router import LLMReply, ToolCall

        last = messages[-1]
        if last["role"] == "user":
            text = last["content"][0]["text"]
            self.seen.append(text)
            if "[Order follow-up]" in text:
                return LLMReply(text="", tool_calls=[ToolCall("c1", "send_message", {"to": ELDER, "text": "ब्लिंकिट का सामान आ गया?"})], model="fake")
        return LLMReply(text="none", tool_calls=[], model="fake")


async def test_has_it_come_is_not_held_back_by_an_unanswered_placed_message(db, at, sessions, monkeypatch):
    from app.llm import router

    brain = AskingBrain()
    router.register_provider("fake", brain)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    try:
        at("2026-10-11 10:00")
        t = await _placed(db)
        # the "placed" update went out and she did not answer it (people rarely do)
        await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="assistant", text="आपका ऑर्डर हो गया है, 10 मिनट में आएगा।",
                             meta={"proactive": True, "ref": "task:abc"})
        await db.commit()
        at("2026-10-11 10:31")
        host = SimHost()
        stats = await wake.wake_due(sessions, lambda fid: host)
        assert stats["ran"] == 1 and host.world.sent and host.world.sent[-1]["text"] == "ब्लिंकिट का सामान आ गया?"
        turns = await store.recent_turns(db, FAM, ELDER, limit=3)
        assert str(turns[-1].meta.get("ref", "")).startswith("task:wake:"), "an order follow-up is not a nudge"
    finally:
        router._providers.pop("fake", None)


async def test_the_brain_cannot_close_an_order_follow_up(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db)
    loop = await _loop(db, t)
    out, err = await tools.run(_ctx(db), "close_loop", {"loop_id": str(loop.id), "outcome": "it came"})
    assert not err and json.loads(out)["closed"] is False and "order_feedback" in json.loads(out)["why"]
    assert (await _loop(db, t)).status == "open"


async def test_a_second_ask_after_not_yet_is_its_own_turn(db, at, sessions, monkeypatch):
    from app.llm import router

    brain = AskingBrain()
    router.register_provider("fake", brain)
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "extract": ["fake:m"]}')
    router.reset_breakers()
    try:
        at("2026-10-11 10:00")
        t = await _placed(db)
        at("2026-10-11 10:31")
        host = SimHost()
        await wake.wake_due(sessions, lambda fid: host)
        await store.add_turn(db, family_id=FAM, thread_id=ELDER, role="user", text="nahi aaya", meta={})
        await db.commit()
        await _say(db, {"task_id": str(t.id), "arrived": "no"})  # not late yet: asked again at 10:41
        at("2026-10-11 10:42")
        await wake.wake_due(sessions, lambda fid: host)
        asks = [m for m in host.world.sent if m["to"] == ELDER]
        assert len(asks) == 2, "the second 'has it come?' went out (not swallowed as a repeat of the first wake)"
    finally:
        router._providers.pop("fake", None)


async def test_a_scheduled_turn_cannot_answer_for_them(db, at):
    at("2026-10-11 10:00")
    t = await _placed(db)
    system = {"id": "saheli-scheduler", "name": "Scheduler", "role": "system"}
    ctx = tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=system, members=[E, S], message_ref="task:wake:x:2")
    out, err = await tools.run(ctx, "order_feedback", {"task_id": str(t.id), "arrived": "no"})
    assert err and "Only their own answer" in out
    await db.refresh(t)
    assert "aftercare" not in t.details
