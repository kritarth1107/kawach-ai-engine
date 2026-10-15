"""D-13: links between care-record items (what a medicine is for, who prescribed it, which record it came from) and the
why-trail (one trace per message or task, carried to events, versions, loops, tasks, backend calls and model spend; a
scrubbed record of every turn; the admin "Why?" chain)."""

import json
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import loop, tools
from app.brain.loop import TurnRequest, run_turn
from app.care import digest, links, memory_upkeep, store, versions, why
from app.care.models import CareEvent, CareLink, OpenLoop
from app.core import trace
from app.learn.models import ReplyLog
from app.llm import router, spend
from app.llm.router import LLMReply, ToolCall
from app.sim.agent import CART, PLACED, FakeAgent
from app.tasks import runtime
from app.tasks.models import Task

FAM, ELDER, CG = "fam-d13", "elder-d13", "cg-d13"
E = {"id": ELDER, "name": "Kamla", "role": "elder"}
C = {"id": CG, "name": "Asha", "role": "primary caregiver"}
MED = {"name": "Metformin", "dose": "500 mg", "times": ["08:00"]}


class Host:
    """Answers the backend calls these flows make, and notes which trace each call carried."""

    shadow = False

    def __init__(self):
        self.calls = []

    async def call(self, tool, args, **kw):
        self.calls.append((tool, args, trace.current()))
        if tool == "delivery_place":
            return {"addressId": "a1", "nickname": "Home", "pincode": "492001", "full": "12, Sample Park, Raipur 492001", "matched": None,
                    "saved": ["Home (Raipur 492001)"]}
        if tool == "browser_profile":
            return {"profile_id": "prof-1", "phone": "9000000001"}
        return {"ok": True, "tool": tool}


def ctx(db, speaker=C, host=None, *, channel="whatsapp", ref="m1"):
    return tools.TurnCtx(session=db, host=host or Host(), family_id=FAM, elder=E, speaker=speaker, members=[E, C], channel=channel,
                         message_ref=ref)


async def remember(db, speaker=C, *, details=None, sentence="Metformin 500 mg at 08:00", links_=None, name="Metformin", domain="medicine",
                   channel="whatsapp", ref="m1"):
    args = {"about": ELDER, "domain": domain, "name": name, "details": details if details is not None else dict(MED), "sentence": sentence}
    if links_ is not None:
        args["links"] = links_
    out, err = await tools.run(ctx(db, speaker, channel=channel, ref=ref), "remember", args)
    assert not err, out
    return json.loads(out)


async def record_basics(db):
    await remember(db, domain="condition", name="diabetes", details={"name": "diabetes"}, sentence="Type 2 diabetes")
    await remember(db, domain="doctor", name="Dr Sharma", details={"name": "Dr Sharma", "phone": "9000011111"}, sentence="Dr Sharma, family doctor")


async def active_links(db, subject=ELDER):
    return await links.links_for(db, FAM, subject, statuses=("active",))


# ── links ─────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def no_backend(monkeypatch):
    """The dashboard routes talk to the backend through make_host(): a recording host here."""
    from app.api import dash

    monkeypatch.setattr(dash, "make_host", lambda: Host())


async def test_links_survive_a_dose_change(db, at):
    at("2026-10-11 10:00")
    await record_basics(db)
    out = await remember(db, links_=[{"relation": "for", "to_domain": "condition", "to_name": "diabetes"},
                                     {"relation": "prescribed_by", "to_domain": "doctor", "to_name": "Dr Sharma"}])
    assert {x["result"] for x in out["links"]} == {"created"}
    first = {ln.id for ln in await active_links(db)}
    at("2026-10-12 10:00")
    out = await remember(db, details={**MED, "dose": "1000 mg"}, sentence="Metformin 1000 mg at 08:00")
    assert out["result"] == "superseded"
    now = await active_links(db)
    assert {ln.id for ln in now} == first  # the same links, untouched by the new row for the same key
    rows = await store.facts(db, FAM, ELDER)
    text = digest.care_record("Kamla", rows, links=await links.links_for(db, FAM, ELDER))
    line = next(ln for ln in text.splitlines() if "[medicine:metformin]" in ln)
    assert "1000 mg" in line and "for diabetes" in line and "prescribed by Dr Sharma" in line
    # the dashboard medicine card reads the same, read-only
    from app.api import dash

    view = await dash.overview(FAM, ELDER, db)
    card = next(f for f in view["facts"] if f["key"] == "medicine:metformin" and f["status"] == "active")
    assert [x["text"] for x in card["links"]] == ["for diabetes", "Dr Sharma"]


async def test_a_weaker_source_health_link_waits_as_pending(db, at):
    at("2026-10-11 10:00")
    await record_basics(db)
    await remember(db)  # the caregiver set the medicine
    out = await remember(db, E, links_=[{"relation": "for", "to_domain": "condition", "to_name": "blood pressure"}])
    assert out["links"] == [{"link": "for blood pressure", "result": "pending"}] and "links_note" in out
    pending = await links.links_for(db, FAM, ELDER, statuses=("pending",))
    assert [ln.to_ref for ln in pending] == ["condition:blood_pressure"]
    loop = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "confirm_link"))).scalar_one()
    assert loop.detail["link_id"] == str(pending[0].id)
    text = digest.care_record("Kamla", await store.facts(db, FAM, ELDER), links=await links.links_for(db, FAM, ELDER))
    assert "PENDING link, not in effect until a caregiver confirms: for blood pressure" in text
    # the elder cannot confirm it; the caregiver can
    _, err = await tools.run(ctx(db, E), "confirm_change", {"key": "medicine:metformin", "approve": True, "link_to": "condition:blood_pressure"})
    assert err
    out, err = await tools.run(ctx(db), "confirm_change", {"key": "medicine:metformin", "approve": True, "link_to": "condition:blood_pressure"})
    assert not err and json.loads(out)["result"] == "active"
    assert "condition:blood_pressure" in {ln.to_ref for ln in await active_links(db)}
    assert (await db.get(OpenLoop, loop.id)).status == "done"
    # a re-read never puts a health link into effect on its own; an appointment's doctor is not a health link
    w = await links.write_link(db, family_id=FAM, subject_id=ELDER, from_ref="medicine:metformin", relation="prescribed_by",
                               to_ref="doctor:dr_rao", label="Dr Rao", source_kind="inferred")
    assert w.result == "pending"
    await remember(db, E, domain="appointment", name="eye check", details={"when": "2026-10-20T10:00"}, sentence="Eye check on 20 Oct",
                   links_=[{"relation": "with_doctor", "to_domain": "doctor", "to_name": "Dr Rao"}])
    assert any(ln.from_ref == "appointment:eye_check" and ln.status == "active" for ln in await active_links(db))


async def test_links_given_with_a_pending_fact_are_decided_with_it(db, at):
    at("2026-10-11 10:00")
    await remember(db)
    out = await remember(db, E, details={**MED, "dose": "250 mg"}, sentence="Metformin 250 mg",
                         links_=[{"relation": "for", "to_domain": "condition", "to_name": "diabetes"}])
    assert out["result"] == "pending" and out["links"][0]["result"] == "pending"
    out, err = await tools.run(ctx(db), "confirm_change", {"key": "medicine:metformin", "approve": False})
    assert not err and json.loads(out)["result"] == "retracted"
    assert not await active_links(db)
    assert [ln.status for ln in await links.links_for(db, FAM, ELDER, statuses=("rejected",))] == ["rejected"]


async def test_undo_and_restore_cover_links(db, at):
    at("2026-10-11 10:00")
    await record_basics(db)
    await remember(db, links_=[{"relation": "prescribed_by", "to_domain": "doctor", "to_name": "Dr Sharma"}])
    out, err = await tools.run(ctx(db), "memory_changes", {"what": "Sharma"})
    change = next(c for c in json.loads(out)["changes"] if c["what"].startswith("Link:"))
    assert change["canUndo"] and "prescribed by Dr Sharma" in change["what"]
    out, err = await tools.run(ctx(db), "undo_change", {"id": change["id"], "mode": "undo", "reason": "wrong doctor"})
    assert not err and json.loads(out)["result"] == "ended", out
    assert not await active_links(db)
    last = (await versions.changes(db, FAM, [ELDER], kinds=("link",)))[0]
    assert last.op == "end" and last.undoes == change["id"]
    out, err = await tools.run(ctx(db), "undo_change", {"id": last.id, "mode": "undo"})  # undoing the undo puts it back
    assert not err and json.loads(out)["result"] == "created"
    assert [ln.to_ref for ln in await active_links(db)] == ["doctor:dr_sharma"]
    # restore puts a link back as it was at that version (and it shows in the preview first)
    first = (await versions.changes(db, FAM, [ELDER], kinds=("link",)))[-1]
    cur = (await active_links(db))[0]
    await links.end_link(db, cur, reason="test", source_kind="caregiver_said", stated_by=CG)
    assert (await versions.preview(db, first, "restore"))["effect"] == "change"
    out, err = await tools.run(ctx(db), "undo_change", {"id": first.id, "mode": "restore"})
    assert not err and [ln.to_ref for ln in await active_links(db)] == ["doctor:dr_sharma"]
    # the elder's undo of a health link waits for a caregiver instead of acting
    newest = (await versions.changes(db, FAM, [ELDER], kinds=("link",)))[0]
    out, err = await tools.run(ctx(db, E), "undo_change", {"id": newest.id, "mode": "undo"})
    assert not err and json.loads(out)["result"] == "pending"
    assert [ln.to_ref for ln in await active_links(db)] == ["doctor:dr_sharma"]


async def test_the_same_link_proposal_is_asked_once(db, at):
    """Review: a repeated or changed proposal added a second pending row, the question moved to it, and the first waited
    for ever (a later yes could still apply a stale value)."""
    at("2026-10-11 10:00")
    await record_basics(db)
    await remember(db)  # the caregiver set the medicine
    bp = [{"relation": "for", "to_domain": "condition", "to_name": "blood pressure"}]
    await remember(db, E, links_=bp)
    await remember(db, E, links_=bp)

    async def open_questions():
        rows = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "confirm_link",
                                                        OpenLoop.status == "open"))).scalars().all()
        return {lp.detail["link_id"] for lp in rows}

    first = await links.links_for(db, FAM, ELDER, statuses=("pending",))
    assert len(first) == 1 and await open_questions() == {str(first[0].id)}
    # a one-value relation: a newer proposal settles the older one, so the caregiver is asked once
    w1 = await links.write_link(db, family_id=FAM, subject_id=ELDER, from_ref="medicine:metformin", relation="prescribed_by",
                                to_ref="doctor:dr_rao", label="Dr Rao", source_kind="inferred")
    w2 = await links.write_link(db, family_id=FAM, subject_id=ELDER, from_ref="medicine:metformin", relation="prescribed_by",
                                to_ref="doctor:dr_gupta", label="Dr Gupta", source_kind="inferred")
    assert w1.result == w2.result == "pending"
    assert (await db.get(CareLink, w1.link.id)).status == "ended"
    assert await open_questions() == {str(first[0].id), str(w2.link.id)}
    # the caregiver saying the link straight away settles the waiting proposal and its question
    await remember(db, links_=bp)
    assert "condition:blood_pressure" in {ln.to_ref for ln in await active_links(db)}
    assert [ln.to_ref for ln in await links.links_for(db, FAM, ELDER, statuses=("pending",))] == ["doctor:dr_gupta"]
    assert await open_questions() == {str(w2.link.id)}
    # approving one value of a one-value relation settles any other proposal for it
    w3 = await links.write_link(db, family_id=FAM, subject_id=ELDER, from_ref="medicine:metformin", relation="prescribed_by",
                                to_ref="doctor:dr_sen", label="Dr Sen", source_kind="inferred")
    await links.resolve_pending_link(db, link_id=w3.link.id, approve=True, by=CG)
    assert not await links.links_for(db, FAM, ELDER, statuses=("pending",)) and not await open_questions()


async def test_undoing_a_replacing_link_change_puts_the_earlier_link_back(db, at):
    """Review: undoing 'prescribed by Dr Rao' (which had replaced Dr Sharma) left the medicine with no doctor at all."""
    at("2026-10-11 10:00")
    await remember(db, links_=[{"relation": "prescribed_by", "to_domain": "doctor", "to_name": "Dr Sharma"}])
    await remember(db, links_=[{"relation": "prescribed_by", "to_domain": "doctor", "to_name": "Dr Rao"}])
    v = (await versions.changes(db, FAM, [ELDER], kinds=("link",)))[0]
    assert v.op == "write" and v.value["replaced"]
    pv = await versions.preview(db, v, "undo")
    assert pv["effect"] == "change" and "Dr Sharma" in pv["text"] and "instead of" in pv["text"], pv
    out, err = await tools.run(ctx(db), "undo_change", {"id": v.id, "mode": "undo", "reason": "Dr Rao was a mistake"})
    assert not err and json.loads(out)["result"] == "replaced", out
    assert [ln.to_ref for ln in await active_links(db)] == ["doctor:dr_sharma"]
    # the elder's undo of such a change only proposes putting the earlier link back
    await remember(db, links_=[{"relation": "prescribed_by", "to_domain": "doctor", "to_name": "Dr Rao"}])
    v = (await versions.changes(db, FAM, [ELDER], kinds=("link",)))[0]
    out, err = await tools.run(ctx(db, E), "undo_change", {"id": v.id, "mode": "undo"})
    assert not err and json.loads(out)["result"] == "pending", out
    assert [ln.to_ref for ln in await active_links(db)] == ["doctor:dr_rao"]
    assert [ln.to_ref for ln in await links.links_for(db, FAM, ELDER, statuses=("pending",))] == ["doctor:dr_sharma"]


async def test_a_decision_about_a_fact_never_decides_a_link(db, at):
    """Review: the dashboard's approve of a fact (or a repeated one) fell through to the only waiting link and decided it."""
    at("2026-10-11 10:00")
    await record_basics(db)
    await remember(db)
    await remember(db, E, links_=[{"relation": "for", "to_domain": "condition", "to_name": "blood pressure"}])
    out, err = await tools.run(ctx(db), "confirm_change", {"key": "medicine:metformin", "approve": True})
    got = json.loads(out)
    assert not err and got["result"] == "nothing_pending", got
    assert got["pending_links"] == [{"link_to": "condition:blood_pressure", "link": "for blood pressure"}]
    assert [ln.to_ref for ln in await links.links_for(db, FAM, ELDER, statuses=("pending",))] == ["condition:blood_pressure"]


async def test_a_caregivers_own_record_reaches_their_turn(db, at):
    """Review: a caregiver with their own care record could not be answered at all (the turn read links without
    importing them)."""
    at("2026-10-11 10:00")
    out, err = await tools.run(ctx(db), "remember", {
        "domain": "medicine", "name": "Vitamin D", "about": CG, "sentence": "My vitamin D at 9",
        "details": {"name": "Vitamin D", "dose": "1 tab", "times": ["09:00"]},
        "links": [{"relation": "prescribed_by", "to_domain": "doctor", "to_name": "Dr Rao"}]})
    assert not err, out
    dynamic, _ = await loop.turn_context(db, turn("kal ka plan?", "wamid.OWN1", speaker=C))
    own = dynamic.split("YOUR OWN CARE", 1)[1]
    assert "[medicine:vitamin_d]" in own and "prescribed by Dr Rao" in own


async def test_one_value_relations_replace_the_old_link(db, at):
    at("2026-10-11 10:00")
    await remember(db, links_=[{"relation": "prescribed_by", "to_domain": "doctor", "to_name": "Dr Sharma"}])
    await remember(db, links_=[{"relation": "prescribed_by", "to_domain": "doctor", "to_name": "Dr Rao"}])
    assert [ln.to_ref for ln in await active_links(db)] == ["doctor:dr_rao"]
    ended = (await db.execute(select(CareLink).where(CareLink.family_id == FAM, CareLink.to_ref == "doctor:dr_sharma"))).scalar_one()
    assert ended.status == "ended" and "Dr Rao" in ended.note
    # a medicine can be for more than one condition
    await remember(db, links_=[{"relation": "for", "to_domain": "condition", "to_name": "diabetes"}])
    await remember(db, links_=[{"relation": "for", "to_domain": "condition", "to_name": "PCOS"}])
    assert {ln.to_ref for ln in await active_links(db) if ln.relation == "for"} == {"condition:diabetes", "condition:pcos"}
    # a link that cannot join is reported, never guessed
    out = await remember(db, links_=[{"relation": "with_doctor", "to_domain": "doctor", "to_name": "Dr Rao"}])
    assert out["links"] == [] and out["links_not_saved"]


async def test_forget_and_restore_cover_links(db, at):
    at("2026-10-11 10:00")
    await record_basics(db)
    await remember(db, links_=[{"relation": "prescribed_by", "to_domain": "doctor", "to_name": "Dr Sharma"}])
    # the elder asking to forget a health link only proposes it
    out, err = await tools.run(ctx(db, E), "forget", {"what": "Sharma prescribed"})
    assert not err and json.loads(out).get("links_waiting") == 1
    assert [ln.to_ref for ln in await active_links(db)] == ["doctor:dr_sharma"]
    out, err = await tools.run(ctx(db), "forget", {"what": "Sharma prescribed"})
    assert not err and json.loads(out)["forgotten"] == 1
    assert not await active_links(db)
    ev = (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "memory_forgotten")
                           .order_by(CareEvent.id.desc()))).scalars().first()
    got = await memory_upkeep.restore(db, FAM, ev.id, by=CG, subjects=[ELDER, "family"])
    assert got["restored"] == 1 and [ln.to_ref for ln in await active_links(db)] == ["doctor:dr_sharma"]


async def test_snapshot_and_restore_family_cover_links(db, at):
    import importlib.util
    import sys
    from pathlib import Path

    from sqlalchemy import text

    from app.jobs import restore_family

    spec = importlib.util.spec_from_file_location("backup_mod", Path(__file__).parent.parent / "backup" / "backup.py")
    backup = importlib.util.module_from_spec(spec)
    sys.modules["backup_mod"] = backup
    spec.loader.exec_module(backup)

    at("2026-10-11 10:00")
    await record_basics(db)
    await remember(db, links_=[{"relation": "for", "to_domain": "condition", "to_name": "diabetes"}])
    await remember(db, domain="contact", name="Ankit", details={"name": "Ankit"}, sentence="Ankit, her son",
                   links_=[{"relation": "related_to", "to_domain": "person", "to_name": ELDER, "as": "son"}])
    rows = [json.loads(r[0]) for r in await db.execute(text(backup.family_sql(set(backup.FAMILY_TABLES))))]
    snap = json.loads(json.dumps(next(r for r in rows if r["family_id"] == FAM), default=str))
    assert {(x["from_ref"], x["relation"]) for x in snap["links"]} == {("medicine:metformin", "for"), ("contact:ankit", "related_to")}
    at("2026-10-12 10:00")
    for ln in await active_links(db):
        await links.end_link(db, ln, reason="lost", source_kind="caregiver_said", stated_by=CG)
    p = await restore_family.plan(db, snap)
    assert {(x["from"], x["health"]) for x in p["links"]} == {("medicine:metformin", True), ("contact:ankit", False)}
    out = await restore_family.apply(db, snap, by=CG)
    assert out["done"]["links"] == 1 and out["done"]["links_pending"] == 1  # the health link waits for a caregiver
    assert [ln.from_ref for ln in await active_links(db)] == ["contact:ankit"]


async def test_stopping_a_condition_asks_about_its_medicines(db, at):
    at("2026-10-11 10:00")
    await record_basics(db)
    await remember(db, links_=[{"relation": "for", "to_domain": "condition", "to_name": "diabetes"}])
    # a caregiver in the chat is asked right away
    out, err = await tools.run(ctx(db), "stop", {"about": ELDER, "domain": "condition", "name": "diabetes", "reason": "doctor says it is reversed"})
    r = json.loads(out)
    assert not err and r["linked_medicines"] == ["Metformin"] and "ask" in r
    assert not (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "memory_check"))).scalars().all()
    assert (await store.active_fact(db, FAM, ELDER, "medicine:metformin")) is not None  # nothing stopped on its own
    # from the dashboard, one question goes to the caregiver
    await remember(db, domain="condition", name="diabetes", details={"name": "diabetes"}, sentence="Type 2 diabetes")
    out, err = await tools.run(ctx(db, channel="dashboard"), "stop", {"about": ELDER, "domain": "condition", "name": "diabetes", "reason": "reversed"})
    loop = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "memory_check"))).scalar_one()
    assert "Metformin" in loop.title and loop.detail["kind"] == "condition_stopped"


async def test_the_record_check_finds_links_to_nothing(db, at):
    at("2026-10-11 10:00")
    await remember(db, links_=[{"relation": "for", "to_domain": "condition", "to_name": "thyroid"},
                               {"relation": "prescribed_by", "to_domain": "doctor", "to_name": "Dr Rao"}])
    kinds = {i["kind"]: i for i in await memory_upkeep.health(db, FAM, ELDER)}
    assert "thyroid" in kinds["link_condition_missing"]["ask"] and "Dr Rao" in kinds["link_doctor_missing"]["ask"]
    await remember(db, domain="doctor", name="Rao", details={"name": "Dr. Rao"}, sentence="Dr Rao, endocrinologist")
    kinds = {i["kind"] for i in await memory_upkeep.health(db, FAM, ELDER)}
    assert "link_doctor_missing" not in kinds and "link_condition_missing" in kinds


async def test_the_doctor_brief_and_the_review_carry_links(db, at):
    from app.api import dash, doctor_brief

    at("2026-10-11 10:00")
    await record_basics(db)
    body = dash.FactIn(actor=dash.Actor(id=CG, name="Asha"), domain="medicine", name="Metformin", details=dict(MED),
                       sentence="Metformin 500 mg at 08:00 (from Dr Sharma)", source="prescription",
                       links=[dash.LinkIn(relation="from_record", to_domain="record", to_name="doc-123", label="3 Sep prescription"),
                              dash.LinkIn(relation="prescribed_by", to_domain="doctor", to_name="Dr Sharma")])
    out = await dash.save_fact(FAM, ELDER, body, db)
    assert {x["result"] for x in out["links"]} == {"created"}
    lines = await doctor_brief.care_record_lines(db, FAM, ELDER)
    assert "Metformin 500 mg at 08:00 (from Dr Sharma) — prescribed by Dr Sharma; from the 3 Sep prescription" in lines
    # a medicine already on the schedule gets linked without changing it
    later = await dash.save_links(FAM, ELDER, dash.LinksIn(actor=dash.Actor(id=CG), domain="medicine", name="Metformin 500",
                                                           source="prescription",
                                                           links=[dash.LinkIn(relation="from_record", to_domain="record", to_name="doc-456",
                                                                              label="9 Oct prescription")]), db)
    assert later["key"] == "medicine:metformin" and later["links"][0]["result"] == "replaced"


# ── the why-trail ─────────────────────────────────────────────────────────────


class Script:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    async def complete(self, route, **kw):
        self.calls += 1
        return self.replies.pop(0) if self.replies else LLMReply(text="Theek hai ji.", tool_calls=[], model="fake", usage=USAGE)


USAGE = {"in": 1200, "out": 150}


def say(text_="Ji.", *calls):
    return LLMReply(text=text_ if not calls else "", tool_calls=[ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls)], model="fake", usage=USAGE)


@pytest.fixture
def model(monkeypatch, db):
    holder = {}

    def use(replies=()):
        holder["m"] = Script(replies)
        router.register_provider("fake", holder["m"])
        return holder["m"]

    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["fake:m"], "brain_hard": ["fake:m"], "extract": ["fake:m"], "vision": ["fake:m"]}')
    from app.api import brain as brain_api

    monkeypatch.setattr(brain_api, "kick_tasks", lambda: None)  # the test moves the order itself
    router.reset_breakers()
    spend.reset()
    spend.configure(async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint"))
    yield use
    spend.reset()
    router._providers.pop("fake", None)


class CostlyAgent(FakeAgent):
    """The browser agent's own model use, so the task's runs show up in its spend."""

    async def run(self, **kw):
        await spend.record("worker", "gemini-2.5-flash", {"in": 6000, "out": 400})
        return await super().run(**kw)


def turn(text_, ref, speaker=E):
    return TurnRequest(family_id=FAM, elder=E, speaker=speaker, members=[E, C], text=text_, message_ref=ref)


async def test_one_whatsapp_order_is_one_trace(db, at, model):
    from app.brain.wake import system_turn

    at("2026-10-11 10:00")
    host = Host()
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    order = {"kind": "order", "service": "instamart", "goal": "Atta for Kamla", "items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}],
             "for_whom": ELDER}
    model([say("", ("start_task", order)), say("Ghar ke address pe bhejun?"),
           say("", ("start_task", order)), say("Ji, dekh rahi hoon.")])
    await run_turn(db, host, turn("atta mangwa do", "wamid.A1"))
    at("2026-10-11 10:01")
    r = await run_turn(db, host, turn("haan ghar", "wamid.B2"))
    assert r.trace_id == "wamid.B2"
    task = (await db.execute(select(Task).where(Task.family_id == FAM))).scalar_one()
    assert task.trace_id == "wamid.B2"

    class Updates:  # the task's updates are passed on by Saheli in a system turn, as live
        def __init__(self):
            self.told = []

        async def notify(self, family_id, requested_by, prompt):
            self.told.append(prompt)
            await system_turn(sessions, host, family_id, f"{prompt} (Requested by {requested_by}.)", f"task:{uuid.uuid4().hex[:12]}",
                              deliver_to=requested_by)

    u = Updates()
    agent = CostlyAgent(script={"prepare": [CART], "place": [PLACED]})

    async def profile_for(t):
        return "prof-1"

    async def tick():
        return await runtime.tick(sessions, agent, profile_for=profile_for, notify=u.notify)

    await tick()
    await tick()
    await db.refresh(task)
    assert task.status == "awaiting_confirm", task.history
    at("2026-10-11 10:05")
    model([say("", ("task_input", {"task_id": str(task.id), "kind": "confirm", "value": "yes"})), say("Order kar rahi hoon.")])
    r = await run_turn(db, host, turn("haan order kar do", "wamid.C3"))
    assert r.trace_id == "wamid.C3"
    await tick()
    await tick()
    await db.refresh(task)
    assert task.status == "done" and task.result.get("placed"), task.history
    await spend.flush()

    # one trace reaches the task, the confirm, the spend and the outcome
    confirm = (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "task_confirmed"))).scalar_one()
    assert confirm.trace_id == "wamid.B2" and confirm.payload["via"] == "wamid.C3" and confirm.payload["by"] == ELDER
    assert confirm.payload["total"] == "₹318"
    done = (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "task_done"))).scalars().all()
    assert done and all(e.trace_id == "wamid.B2" for e in done)
    rows = await spend.for_traces(db, ["wamid.B2"])
    assert {r_["role"] for r_ in rows} >= {"brain", "worker"}  # the turn that started it and the task's own runs
    assert any(r_["role"] == "worker" and r_["taskId"] == str(task.id) for r_ in rows)

    out = await why.chain(db, family_id=FAM, task_id=str(task.id))
    assert out["traces"] == ["wamid.B2", "wamid.C3"]
    assert out["start"]["text"] == "haan ghar" and out["start"]["kind"] == "message"
    assert [c["kind"] for c in out["confirms"]][:1] == ["confirmed"] and out["confirms"][0]["total"] == "₹318"
    assert out["outcome"]["placed"] and out["outcome"]["orderId"] == "IM-55821"
    assert out["spend"]["totalInr"] > 0 and out["spend"]["byTask"][str(task.id)] > 0
    tools_used = [t["name"] for x in out["turns"] for t in x["tools"]]
    assert "start_task" in tools_used and "task_input" in tools_used
    assert any(x["kind"] == "system" and x["taskId"] == str(task.id) for x in out["turns"])  # the updates Saheli passed on
    assert all(x["models"] for x in out["turns"] if x["kind"] == "message")
    # the first message (the address question) has its own trace; it is not part of this order's trail
    assert "wamid.A1" not in out["traces"]


async def test_no_login_code_or_secret_is_stored_in_a_trace(db, at, model):
    at("2026-10-11 11:00")
    host = Host()
    t = await runtime.create(db, family_id=FAM, subject_id=ELDER, requested_by=ELDER, service="instamart", kind="order", goal="Atta",
                             details={"items": [{"name": "Aashirvaad Atta 5kg", "qty": 1}]})
    t.status, t.input_needed, t.phase = "needs_input", "otp", "otp"
    await db.commit()
    model([say("", ("task_input", {"task_id": str(t.id), "kind": "otp", "value": "482913"})), say("Code mil gaya, aage badh rahi hoon.")])
    r = await run_turn(db, host, turn("482913", "wamid.OTP1"))
    model([say("", ("note", {"about": ELDER, "topic": "misc", "text": "my otp is 553311"})), say("Theek hai.")])
    await run_turn(db, host, turn("my otp is 553311", "wamid.OTP2"))
    stored = (await db.execute(select(why.TraceTurn).where(why.TraceTurn.family_id == FAM))).scalars().all()
    blob = json.dumps([{"text": s.text, "reply": s.reply, "tools": s.tools} for s in stored], ensure_ascii=False)
    assert "482913" not in blob and "553311" not in blob and "••••" in blob
    logs = (await db.execute(select(ReplyLog).where(ReplyLog.family_id == FAM))).scalars().all()
    assert logs and "482913" not in json.dumps([x.trace for x in logs]) and "553311" not in json.dumps([x.trace for x in logs])
    assert r.trace_id == "wamid.OTP1"


async def test_a_dashboard_change_has_a_trail(db, at):
    from app.api import dash

    at("2026-10-11 12:00")
    body = dash.FactIn(actor=dash.Actor(id=CG, name="Asha"), domain="medicine", name="Metformin", details=dict(MED), sentence="Metformin 500 mg")
    await dash.save_fact(FAM, ELDER, body, db)
    v = (await versions.changes(db, FAM, [ELDER], kinds=("fact",)))[0]
    assert v.trace_id and v.trace_id.startswith("dash:")
    out = await why.chain(db, family_id=FAM, version_id=v.id)
    assert out["focus"]["type"] == "change" and out["turns"][0]["kind"] == "dashboard"
    assert out["turns"][0]["tools"][0]["name"] == "remember" and "Metformin" in out["turns"][0]["tools"][0]["args"]
    # the reminder sync to the backend carried the same trace
    ev = (await db.execute(select(CareEvent).where(CareEvent.family_id == FAM, CareEvent.kind == "dashboard_edit"))).scalar_one()
    assert ev.trace_id == v.trace_id


async def test_backend_calls_and_versions_carry_the_trace(db, at, monkeypatch):
    at("2026-10-11 12:00")
    host = Host()
    with trace.use("wamid.SYNC1", family_id=FAM):
        out, err = await tools.run(ctx(db, host=host), "remember", {"about": ELDER, "domain": "medicine", "name": "Metformin", "details": dict(MED),
                                                                    "sentence": "Metformin 500 mg at 08:00"})
    assert not err
    sync = next(c for c in host.calls if c[0] == "sync_medicine_schedule")
    assert sync[2] == "wamid.SYNC1"
    v = (await versions.changes(db, FAM, [ELDER], kinds=("fact",)))[0]
    assert v.trace_id == "wamid.SYNC1"
    # the real client sends it to the backend with the call
    from app.agents import tool_client

    sent = []

    class Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"ok": True}}

    class Client:
        async def post(self, url, json=None, headers=None):
            sent.append(json)
            return Resp()

    monkeypatch.setattr(tool_client, "_client", lambda: Client())
    with trace.use("wamid.SYNC2"):
        await tool_client.execute_backend_tool(tool="list_places", args={}, family_id=FAM, elder_id=ELDER, actor_user_id=CG)
    await tool_client.execute_backend_tool(tool="list_places", args={}, family_id=FAM, elder_id=ELDER, actor_user_id=CG)
    assert sent[0]["trace_id"] == "wamid.SYNC2" and "trace_id" not in sent[1]


async def test_the_why_view_stays_inside_one_family(db, at):
    at("2026-10-11 12:00")
    with trace.use("wamid.OTHER", family_id="fam-other"):
        await store.record_event(db, family_id="fam-other", subject_id="x", kind="note", summary="other family")
    assert (await why.chain(db, family_id=FAM, trace_id="wamid.OTHER"))["turns"] == []
    assert await why.chain(db, family_id=FAM, event_id=1_000_000_000) is None


def test_background_task_runs_are_not_filed_under_the_turn():
    import contextvars

    with trace.use("wamid.TURN"), trace.collect_spend() as calls:
        c = contextvars.copy_context()
        c.run(trace.detach)
        assert c.run(trace.current) is None and trace.current() == "wamid.TURN"
        c.run(trace.note_spend, "worker", "m", 1.0, 1, 1)
        assert calls == []


def test_scrubbing_keeps_amounts_and_doses():
    assert trace.scrub("482913") == "••••"
    assert trace.scrub("the code from the store is 482913") == "the code from the store is ••••"
    assert trace.scrub("482913 go ahead") == "•••• go ahead", "'g' of 'go' is not a unit"
    assert trace.scrub("482913 ghar pe") == "•••• ghar pe"
    assert trace.scrub("Swiggy ka message aaya hai 482913 wala dekh lo") == "Swiggy ka message aaya hai •••• wala dekh lo"
    assert trace.scrub("Total ₹1250, Metformin 1000 mg, 2500 rupees") == "Total ₹1250, Metformin 1000 mg, 2500 rupees"
    assert trace.scrub('{"date": "2026-10-15", "until": "15/10/2026"}') == '{"date": "2026-10-15", "until": "15/10/2026"}'
    assert trace.scrub("code 4829 and 2026-10-15") == "code •••• and 2026-10-15"
    assert trace.summarize({"task_id": "t1", "kind": "otp", "value": "4829"}) == "task_id=t1, kind=otp, value=••••"
    assert trace.mask({"password": "x", "items": [{"name": "milk"}]}) == {"password": "••••", "items": [{"name": "milk"}]}


async def test_a_follow_up_keeps_the_trail_it_was_opened_in(db, at, model):
    from datetime import timedelta

    from app.brain import wake
    from app.core import clock

    at("2026-10-11 10:00")
    await store.save_roster(db, FAM, E, [E, C])
    with trace.use("wamid.ASK1", family_id=FAM):
        loop = await store.open_loop(db, family_id=FAM, subject_id=ELDER, kind="question", title="Ask how the knee is",
                                     wake_at=clock.now() + timedelta(minutes=5), alert_rule="ask_again")
    await db.commit()
    assert loop.trace_id == "wamid.ASK1"
    at("2026-10-11 10:10")
    model([say("none")])
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    stats = await wake.wake_due(sessions, lambda fid: Host())
    assert stats["ran"] == 1
    woke = (await db.execute(select(why.TraceTurn).where(why.TraceTurn.family_id == FAM))).scalars().all()
    assert [(w.trace_id, w.kind) for w in woke] == [("wamid.ASK1", "system")]
    out = await why.chain(db, family_id=FAM, loop_id=str(loop.id))
    assert out["focus"]["type"] == "loop" and out["turns"][0]["kind"] == "system"


async def test_nightly_work_is_filed_under_the_night(db, at):
    from app.care import dream

    at("2026-10-11 23:30")
    await store.save_roster(db, FAM, E, [E, C])
    await remember(db, links_=[{"relation": "for", "to_domain": "condition", "to_name": "thyroid"}])
    await db.commit()
    sessions = async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")
    roster = await store.roster(db, FAM)
    await dream.dream_family(sessions, roster, "2026-10-11", with_models=False)
    checks = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "memory_check"))).scalars().all()
    assert checks and all(lp.trace_id == f"job:night:2026-10-11:{FAM}" for lp in checks)
    assert any(lp.detail["kind"] == "link_condition_missing" for lp in checks)


async def test_the_admin_endpoints_answer_within_a_family(db, at):
    from fastapi import HTTPException

    from app.api import admin

    at("2026-10-11 12:00")
    with trace.use("wamid.ADM1", family_id=FAM):
        await remember(db, links_=[{"relation": "for", "to_domain": "condition", "to_name": "diabetes"}])
    v = (await versions.changes(db, FAM, [ELDER], kinds=("fact",)))[0]
    out = await admin.admin_why(db, family=FAM, version=v.id)
    assert out["traces"] == ["wamid.ADM1"] and any(c["kind"] == "link" for c in out["changes"])
    with pytest.raises(HTTPException):
        await admin.admin_why(db, family="fam-other", version=v.id)
    with pytest.raises(HTTPException):
        await admin.admin_why(db, family=FAM)
    recent = await admin.admin_why_recent(FAM, db)
    assert recent["changes"] and recent["changes"][0]["traced"]
    rec = await admin.admin_care_record(FAM, ELDER, db)
    med = next(f for f in rec["facts"] if f["key"] == "medicine:metformin")
    assert med["versionId"] and med["links"] == [{"text": "for diabetes", "status": "active"}]
