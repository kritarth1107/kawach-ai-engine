"""Learning A2: turn traces and fine-tuning data, hard-moment routing, grader calibration, drift alarms, rule
proposals, situation-scoped trials, gap specs, failure export, the tuned-vs-base comparison. Scripted models only."""

import json
from datetime import timedelta

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.brain import loop
from app.brain.loop import TurnRequest, run_turn
from app.care import outcomes, store
from app.core import clock
from app.db.migrate import run_v2_migrations
from app.learn import jobs, lessons, playbook, review, timing, tuning
from app.learn.models import LearnReport, PlaybookVersion, ReplyLog, RuleProposal
from app.llm import router, spend
from app.llm.router import LLMReply, ToolCall
from app.sim.world import SimHost
from eval import compare_models, export_failures

ELDER = {"id": "e-a2", "name": "Kamla Sharma", "role": "elder"}
CG = {"id": "c-a2", "name": "Ankit Sharma", "role": "primary caregiver"}


@pytest.fixture
def sessions(db):
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, join_transaction_mode="create_savepoint")


@pytest.fixture(autouse=True)
def clean():
    playbook.reset_cache()
    timing.reset_cache()
    spend.reset()
    yield
    playbook.reset_cache()
    timing.reset_cache()
    spend.reset()


class Model:
    def __init__(self, replies=(), default="Theek hai ji."):
        self.replies, self.default, self.routes = list(replies), default, []

    async def complete(self, route, *, messages, system_dynamic="", **kw):
        self.routes.append(route.model)
        return self.replies.pop(0) if self.replies else LLMReply(text=self.default, tool_calls=[], model="fake")


@pytest.fixture
def model(monkeypatch):
    def use(replies=(), default="Theek hai ji.", routes=None):
        m = Model(replies, default)
        router.register_provider("fake", m)
        monkeypatch.setenv("MODEL_ROUTES", json.dumps(routes or {r: ["fake:m"] for r in ("brain", "extract", "judge", "judge_fast", "learn", "worker")}))
        router.reset_breakers()
        return m

    yield use
    router._providers.pop("fake", None)


def req(text, ref, fam="fam-a2"):
    return TurnRequest(family_id=fam, elder=ELDER, speaker=ELDER, members=[ELDER, CG], text=text, message_ref=ref)


async def _log(db, fam, n, *, situation="chit_chat", score=0.5, days_ago=1, arm="live", version=0, judge=None, tone=None,
               user="Aaj mandir gayi thi", text="Bahut accha ji 🙏", note=None):
    for i in range(n):
        db.add(ReplyLog(family_id=fam, thread_id=ELDER["id"], at=clock.now() - timedelta(days=days_ago, minutes=i), kind="reply",
                        situation=situation, speaker_role="elder", lang="latin-indic", user_text=user, text=text, text_len=len(text),
                        tools=[], playbook_version=version, arm=arm, sent_hour=10, scored_at=clock.now(), replied=True,
                        score=score + (i % 3) * 0.05, tone=tone, judge_pass=judge, judge_note=note))
    await db.commit()


# ── traces and fine-tuning data ──


async def test_turn_trace_logged_and_becomes_an_anonymised_tuning_example(db, at, model):
    at("2026-10-02 10:00")
    await store.save_roster(db, "fam-a2", ELDER, [ELDER, CG])
    await outcomes.set_consent(db, "fam-a2", ELDER["id"], granted=True, by=CG["id"])
    await db.commit()
    model([LLMReply(text="", tool_calls=[ToolCall(id="t1", name="remember", args={"text": "Kamla ji ko mandir jaana pasand hai"})], model="fake"),
           LLMReply(text="Kamla ji, yaad rakh liya 🙏", tool_calls=[], model="fake")])
    await run_turn(db, SimHost(), req("Main Kamla, aaj mandir gayi thi", "t-1"))
    row = (await db.execute(select(ReplyLog).where(ReplyLog.family_id == "fam-a2"))).scalars().one()
    roles = [s["role"] for s in row.trace]
    assert roles[0] == "user" and "tool" in roles and row.trace[-1]["text"].startswith("Kamla ji")
    ex = tuning.to_example(row.trace, situation=row.situation, names=["Kamla Sharma", "Ankit Sharma"], meds=[])
    flat = json.dumps(ex, ensure_ascii=False)
    assert "Kamla" not in flat
    parts = [p for c in ex["contents"] for p in c["parts"]]
    assert any("functionCall" in p for p in parts) and any("functionResponse" in p for p in parts)
    assert ex["contents"][-1]["role"] == "model" and "text" in ex["contents"][-1]["parts"][0]


async def test_tuning_build_consented_only_and_holdout_by_family(db, at):
    at("2026-10-02 10:00")
    fams = [f"fam-t{i}" for i in range(6)]
    for f in fams + ["fam-nope"]:
        await store.save_roster(db, f, ELDER, [ELDER, CG])
    for f in fams:
        await outcomes.set_consent(db, f, ELDER["id"], granted=True, by=CG["id"])
    await db.commit()
    trace = [{"role": "user", "text": "Aaj mandir gayi thi"}, {"role": "model", "text": "Bahut accha ji 🙏"}]
    for f in fams + ["fam-nope"]:
        for i in range(5):
            db.add(ReplyLog(family_id=f, thread_id=ELDER["id"], at=clock.now(), kind="reply", situation="chit_chat", text="x", score=0.8, trace=trace))
    await db.commit()
    split, stats = await tuning.build(db, with_tools=False)
    assert stats["no_consent"] == 5 and stats["examples"] == 30 and stats["holdout_families"] == 1
    assert len(split.holdout) == 5 and len(split.train) + len(split.validation) == 25


def test_tuning_example_with_a_leak_is_dropped():
    trace = [{"role": "user", "text": "call me on 9876543210"}, {"role": "model", "text": "Ok"}]
    assert tuning.to_example(trace, situation="chit_chat", names=[], meds=[]) is not None  # phone numbers are scrubbed
    bad = [{"role": "user", "text": "hi"}, {"role": "tool", "results": []}]
    assert tuning.to_example(bad, situation="chit_chat", names=[], meds=[]) is None  # must end on Saheli's text


# ── hard moments go to the stronger brain (only when configured) ──


def test_role_for(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTES", json.dumps({"brain": ["fake:a"]}))
    assert loop.role_for("emergency") == "brain"
    monkeypatch.setenv("MODEL_ROUTES", json.dumps({"brain": ["fake:a"], "brain_hard": ["fake:b"]}))
    assert loop.role_for("emergency") == "brain_hard" and loop.role_for("chit_chat") == "brain"
    monkeypatch.setenv("BRAIN_ROLE", "brain_tuned")
    assert loop.role_for("emergency") == "brain_tuned"


async def test_hard_moment_uses_brain_hard_route(db, at, model):
    at("2026-10-02 10:00")
    await store.save_roster(db, "fam-a2", ELDER, [ELDER, CG])
    await db.commit()
    m = model(routes={"brain": ["fake:normal"], "brain_hard": ["fake:strong"], "extract": ["fake:normal"], "judge_fast": ["fake:normal"]})
    await run_turn(db, SimHost(), req("Unki bahut yaad aa rahi hai", "h-1"))
    assert m.routes[0] == "strong"
    await run_turn(db, SimHost(), req("Aaj mausam accha hai", "h-2"))
    assert "normal" in m.routes[1:]


async def test_brain_hard_counts_as_essential_spend():
    assert spend.ESSENTIAL_ROLES  # brain* roles stay essential under the soft cap
    assert await spend.gate("brain_hard") == await spend.gate("brain")


# ── grader calibration ──


async def test_calibration_low_agreement_blocks_promotion(db, at, sessions, model, monkeypatch):
    at("2026-10-12 03:00")
    await _log(db, "fam-cal", 20, judge=True)
    verdicts = [{"n": i + 1, "pass": i % 2 == 0} for i in range(20)]
    model([LLMReply(text=json.dumps({"verdicts": verdicts}), tool_calls=[], model="fake")])
    cal = await review.calibrate(sessions)
    assert cal["n"] == 20 and cal["agreement"] == 0.5 and cal["trusted"] is False
    # a canary that scores better is not promoted on untrusted grades
    await db.execute(delete(PlaybookVersion))
    db.add(PlaybookVersion(version=1, status="live", lessons={}, examples={}, created_at=clock.now(), live_since=clock.now() - timedelta(days=30)))
    db.add(PlaybookVersion(version=2, status="canary", lessons={"chit_chat": ["Be brief."]}, examples={}, created_at=clock.now(),
                           canary_since=clock.now() - timedelta(days=8)))
    await db.commit()
    monkeypatch.setenv("LEARN_CANARY_MIN_N", "50")
    await _log(db, "fam-c", 60, arm="canary", version=2, score=0.7)
    await _log(db, "fam-d", 60, arm="live", version=1, score=0.2)
    assert (await lessons.evaluate(sessions))["decision"] == "better_but_grader_untrusted"


async def test_calibration_skips_without_data(db, at, sessions, model):
    at("2026-10-12 03:00")
    model()
    assert "skipped" in await review.calibrate(sessions)


# ── drift ──


async def test_drift_alarm_when_scores_fall(db, at, sessions):
    at("2026-10-12 03:00")
    await _log(db, "fam-dr", 50, days_ago=14, score=0.6)
    await _log(db, "fam-dr", 50, days_ago=2, score=0.1, tone="annoyed")
    out = await review.drift(sessions)
    assert any("average score fell" in a for a in out["alarms"]) and any("annoyed" in a for a in out["alarms"])
    assert (await review.latest(sessions, "drift"))["alarms"] == out["alarms"]


async def test_no_drift_alarm_on_small_or_steady_data(db, at, sessions):
    at("2026-10-12 03:00")
    await _log(db, "fam-st", 50, days_ago=14, score=0.5)
    await _log(db, "fam-st", 50, days_ago=2, score=0.5)
    await _log(db, "fam-st", 3, days_ago=14, situation="grief", score=0.9)
    await _log(db, "fam-st", 3, days_ago=2, situation="grief", score=0.1)  # too few to judge on its own
    out = await review.drift(sessions)
    assert out["alarms"] == [] and "grief" not in out["by"] and "chit_chat" in out["by"]


# ── rule proposals → founder → next playbook ──


async def test_rule_proposal_needs_approval_then_enters_next_playbook(db, at, sessions, model, monkeypatch):
    at("2026-10-11 23:00")
    await db.execute(delete(PlaybookVersion))
    await db.execute(delete(RuleProposal))
    await _log(db, "fam-rp", 6, situation="med_question", judge=False, note="lectured the elder", days_ago=1)
    rule = "When an elder asks about a medicine, say the doctor decides and offer to alert the family if she feels unwell."
    model([LLMReply(text=json.dumps({"proposals": [{"situation": "med_question", "rule": rule, "why": "lecturing", "examples": [1, 2]}]}),
                    tool_calls=[], model="fake")])
    got = await review.propose_rules(sessions)
    assert len(got) == 1
    p = (await db.execute(select(RuleProposal))).scalars().one()
    assert p.status == "proposed" and len(p.evidence) == 2
    # not approved: the weekly draft has nothing to learn from it
    monkeypatch.setenv("LEARN_MIN_EXAMPLES", "1000")
    assert (await lessons.weekly(sessions))["draft"] is None
    assert (await review.decide_rule(sessions, p.id, approve=True, by="founder"))["ok"]
    assert not (await review.decide_rule(sessions, p.id, approve=False, by="founder"))["ok"]  # decided once
    out = await lessons.weekly(sessions)
    pb = await db.get(PlaybookVersion, out["draft"])
    assert rule in pb.lessons["med_question"] and pb.scope == ["med_question"]
    assert pb.gate["passed"]  # the founder's rule may mention alerts; it still goes through the trial
    await db.refresh(p)
    assert p.status == "in_playbook"


async def test_rejected_rule_never_used(db, at, sessions, monkeypatch):
    at("2026-10-11 23:00")
    await db.execute(delete(PlaybookVersion))
    await db.execute(delete(RuleProposal))
    db.add(RuleProposal(created_at=clock.now(), situation="chit_chat", rule="Always reply in English.", status="proposed"))
    await db.commit()
    rid = (await db.execute(select(RuleProposal.id))).scalar_one()
    await review.decide_rule(sessions, rid, approve=False, by="founder")
    monkeypatch.setenv("LEARN_MIN_EXAMPLES", "1000")
    assert (await lessons.weekly(sessions))["draft"] is None


async def test_general_rule_is_trialled_on_every_situation(db, at, sessions, monkeypatch):
    at("2026-10-11 23:00")
    await db.execute(delete(PlaybookVersion))
    await db.execute(delete(RuleProposal))
    db.add(RuleProposal(created_at=clock.now(), situation="general", rule="Never use more than one emoji.", status="approved"))
    await db.commit()
    monkeypatch.setenv("LEARN_MIN_EXAMPLES", "1000")
    pb = await db.get(PlaybookVersion, (await lessons.weekly(sessions))["draft"])
    assert pb.scope == [] and pb.lessons["general"] == ["Never use more than one emoji."]


# ── situation-scoped trials ──


async def test_trial_compares_only_the_situations_it_changed(db, at, sessions, monkeypatch):
    at("2026-10-12 03:00")
    await db.execute(delete(PlaybookVersion))
    db.add(PlaybookVersion(version=1, status="live", lessons={}, examples={}, created_at=clock.now(), live_since=clock.now() - timedelta(days=30)))
    db.add(PlaybookVersion(version=2, status="canary", lessons={"med_question": ["Short."]}, examples={}, scope=["med_question"],
                           created_at=clock.now(), canary_since=clock.now() - timedelta(days=8)))
    await db.commit()
    monkeypatch.setenv("LEARN_CANARY_MIN_N", "50")
    await _log(db, "fam-c", 60, situation="med_question", arm="canary", version=2, score=0.7)
    await _log(db, "fam-d", 60, situation="med_question", arm="live", version=1, score=0.2)
    await _log(db, "fam-c", 200, situation="chit_chat", arm="canary", version=2, score=-0.8)  # unrelated, not its doing
    res = await lessons.evaluate(sessions)
    assert res["decision"] == "better" and res["scope"] == ["med_question"] and res["n_canary"] == 60


# ── gap specs and the founder's view ──


async def test_gap_specs_and_overview(db, at, sessions, model):
    at("2026-10-12 03:00")
    from app.learn import gaps

    await gaps.record(db, family_id="fam-g", user_text="bijli ka bill bhar do", how="said_cannot", names=[])
    await gaps.record(db, family_id="fam-g", user_text="pay my electricity bill", how="said_cannot", names=[])
    await db.commit()
    model([LLMReply(text=json.dumps({"specs": [{"category": "bills_payments", "what": "Pay utility bills", "why": "asked twice", "tool": "BBPS",
                                                 "risks": "money", "size": "M"}]}), tool_calls=[], model="fake")])
    out = await review.gap_specs(sessions)
    assert out["specs"][0]["tool"] == "BBPS"
    view = await jobs.overview(sessions)
    assert view["gapSpecs"]["specs"][0]["category"] == "bills_payments" and "ruleProposals" in view and "drift" in view


# ── failures → replay cases ──


async def test_export_failures_consented_anonymised_deduped(db, at):
    at("2026-10-12 03:00")
    for f in ("fam-fy", "fam-fn"):
        await store.save_roster(db, f, ELDER, [ELDER, CG])
    await outcomes.set_consent(db, "fam-fy", ELDER["id"], granted=True, by=CG["id"])
    await db.commit()
    await _log(db, "fam-fy", 3, judge=False, user="Ankit kab aayega?", text="Kamla ji, pata nahi.", note="cold")
    await _log(db, "fam-fn", 3, judge=False, note="cold")
    cases = await export_failures.collect(db)
    assert len(cases) == 1  # same message once; the other family did not agree to share
    c = cases[0]
    assert "Ankit" not in c["said"] and "Kamla" not in c["bad_reply"] and c["note"] == "cold"


# ── tuned vs base comparison (pure parts) ──


def test_compare_decision_points_and_summary():
    ex = {"systemInstruction": {"parts": [{"text": "S"}]}, "contents": [
        {"role": "user", "parts": [{"text": "goli le li"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "log_dose", "args": {}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "log_dose", "response": {}}}]},
        {"role": "model", "parts": [{"text": "Shabash ji"}]},
    ]}
    pts = compare_models.decision_points(ex)
    assert len(pts) == 1 and pts[0]["calls"] == ["log_dose"] and len(pts[0]["history"]) == 1
    assert compare_models.agrees(["log_dose"], ["log_dose", "log_dose"]) and not compare_models.agrees([], ["remember"])
    rows = [{"base_agrees": True, "tuned_agrees": True, "prefer": "tuned"}, {"base_agrees": True, "tuned_agrees": True, "prefer": "tie"}]
    assert compare_models.summarise(rows)["pass"]
    rows = [{"base_agrees": True, "tuned_agrees": False}] * 10
    assert not compare_models.summarise(rows)["pass"]
    assert not compare_models.summarise([])["pass"]


# ── migrations ──


async def test_v2_migrations_are_idempotent(db):
    conn = await db.connection()
    await run_v2_migrations(conn)
    await run_v2_migrations(conn)
    await db.execute(select(LearnReport).limit(1))
    cols = {r[0] for r in (await db.execute(__import__("sqlalchemy").text(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'reply_log'"))).all()}
    assert "trace" in cols
