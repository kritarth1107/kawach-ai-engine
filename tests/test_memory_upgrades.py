"""Memory upgrades: hybrid recall, importance and fading, rollups, record health, profile card, forget/restore,
style memory, and a small recall benchmark (keyword + a fake meaning model)."""

import hashlib
import re
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from app.brain import tools
from app.care import baselines, digest, memory_index, memory_upkeep, store
from app.care.models import CareEvent, MemoryNote, OpenLoop
from app.core import clock
from app.llm import router, spend
from app.llm.router import LLMReply
from app.sim.world import SimHost

FAM, ELDER, CG = "fam-m", "e-m", "c-m"
E = {"id": ELDER, "name": "Kamla", "role": "elder"}
C = {"id": CG, "name": "Asha", "role": "primary caregiver"}
SYN = {"dard": "pain", "ghutna": "knee", "ghutne": "knee", "ghutno": "knee", "chakkar": "dizzy", "neend": "sleep", "sugar": "glucose",
       "bukhar": "fever", "pet": "stomach", "khansi": "cough", "dawai": "medicine", "goli": "medicine"}


async def fake_embed(texts):
    """A tiny 'meaning' model: words mapped to concepts, hashed into 64 dims (enough to test hybrid ranking)."""
    out = []
    for t in texts:
        v = [0.0] * memory_index.DIM
        for w in re.findall(r"\w+", t.lower()):
            w = SYN.get(w, w)
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        out.append(v)
    return out


@pytest.fixture
def meaning():
    memory_index.set_embedder(fake_embed)
    yield
    memory_index.set_embedder(None)


def ctx(db, speaker=C):
    return tools.TurnCtx(session=db, host=SimHost(), family_id=FAM, elder=E, speaker=speaker, members=[E, C])


async def ev(db, kind, at, summary, **payload):
    clock.set_now(at)
    await store.record_event(db, family_id=FAM, subject_id=ELDER, kind=kind, summary=summary, payload=payload)


def ist(d, hhmm="10:00", m=10):
    h, mi = map(int, hhmm.split(":"))
    return datetime(2026, m, d, h, mi, tzinfo=clock.IST)


async def test_hybrid_recall_finds_meaning_and_weighs_importance(db, at, meaning):
    await ev(db, "symptom", ist(1), "left knee pain while climbing stairs", severity="watch")
    for d in range(2, 20):
        await ev(db, "meal", ist(d), f"had poha and tea on day {d}")
    await ev(db, "mood", ist(20), "happy, grandson visited")
    clock.set_now(ist(21))
    assert await memory_index.index_subject(db, FAM, ELDER) > 0
    hits = await memory_index.search(db, FAM, [ELDER], "ghutne mein wahi dard phir se", limit=3)
    assert hits and "knee pain" in hits[0].text  # found by meaning, despite older and no shared words


def test_fading_spares_important_memories():
    old = clock.now() - timedelta(days=90)
    assert memory_index.decay(old, 0.3) < 0.3 and memory_index.decay(old, 1.0) == 1.0


async def test_forget_and_restore(db, at):
    at("2026-10-02 10:00")
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="family", title="Family", body_md="- fought with Rahul about money\n- loves bhajans")
    await ev(db, "mood", ist(2), "upset after the fight with Rahul about money")
    out, err = await tools.run(ctx(db, E), "forget", {"what": "Rahul money"})
    assert not err and '"forgotten": 2' in out
    db.expire_all()
    note = (await store.notes(db, FAM, [ELDER]))[0]
    assert "Rahul" not in note.body_md and "bhajans" in note.body_md
    assert not [h for h in await store.recall(db, FAM, [ELDER], "Rahul money") if "Rahul" in h.text]
    fe = (await db.execute(select(CareEvent).where(CareEvent.kind == "memory_forgotten"))).scalars().one()
    assert (await memory_upkeep.restore(db, FAM, fe.id, by=CG, subjects=[ELDER, "family"]))["restored"] == 2
    db.expire_all()
    body = (await store.notes(db, FAM, [ELDER]))[0].body_md
    assert body.count("Rahul") == 1 and "bhajans" in body


async def test_record_health_finds_problems_and_asks_once(db, at):
    at("2026-10-02 10:00")
    c = ctx(db)
    await tools.run(c, "remember", {"domain": "medicine", "name": "Metformin", "details": {"dose": "500 mg", "times": ["08:00"]}, "sentence": "Metformin 500 at 8"})
    await tools.run(c, "remember", {"domain": "medicine", "name": "Metformin SR", "details": {"dose": "1000 mg", "times": ["20:00"]}, "sentence": "Metformin SR 1000 at night"})
    await tools.run(c, "remember", {"domain": "medicine", "name": "Ecosprin", "details": {"dose": "75 mg"}, "sentence": "Ecosprin 75"})
    await tools.run(c, "remember", {"domain": "doctor", "name": "Dr Mehra", "details": {"name": "Dr Mehra"}, "sentence": "Dr Mehra is her doctor"})
    await tools.run(c, "remember", {"domain": "contact", "name": "Mehra clinic", "details": {"name": "Mehra"}, "sentence": "Mehra clinic number"})
    kinds = {i["kind"] for i in await memory_upkeep.health(db, FAM, ELDER)}
    assert {"duplicate_medicine", "no_time", "duplicate_person"} <= kinds
    assert await memory_upkeep.ask_about_health(db, FAM, ELDER, CG) >= 3
    await memory_upkeep.ask_about_health(db, FAM, ELDER, CG)
    loops = (await db.execute(select(OpenLoop).where(OpenLoop.family_id == FAM, OpenLoop.kind == "memory_check"))).scalars().all()
    assert len(loops) == len({l.dedupe_key for l in loops})  # asked once each


async def test_profile_card_and_notes_order(db, at):
    at("2026-10-02 10:00")
    c = ctx(db)
    await tools.run(c, "remember", {"domain": "naming", "name": "address_as", "details": {"name": "Kamla ji", "avoid": ["Amma"]}, "sentence": "Call her Kamla ji"})
    await tools.run(c, "remember", {"domain": "allergy", "name": "sulfa", "details": {"allergen": "sulfa"}, "sentence": "Allergic to sulfa drugs"})
    await tools.run(c, "remember", {"domain": "preference", "name": "bhajans", "details": {}, "sentence": "Loves morning bhajans"})
    card = await memory_upkeep.save_profile_card(db, FAM, E)
    assert "Kamla ji" in card and "sulfa" in card and "bhajans" in card
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="diary", title="Diary", body_md="- 2026-10-01: long day")
    block = digest.notes_block(await store.notes(db, FAM, [ELDER]))
    assert block.index("Profile card") < block.index("Preference") if "Preference" in block else "Profile card" in block
    assert "long day" not in block  # the diary stays out of the per-turn context


async def test_weekly_and_monthly_rollups(db, at, monkeypatch):
    class M:
        async def complete(self, route, **kw):
            return LLMReply(text="A steady week: tablets on time, knee pain twice, happy about the grandson.", tool_calls=[], model="fake")

    router.register_provider("fake", M())
    monkeypatch.setenv("MODEL_ROUTES", '{"extract": ["fake:m"]}')
    router.reset_breakers()
    spend.reset()
    diary = "\n".join(f"- 2026-10-{d:02d}: day {d} fine" for d in range(1, 32) if d <= 31)
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="diary", title="Diary", body_md=diary)
    clock.set_now(ist(4, "02:00"))  # a Sunday
    assert (await memory_upkeep.rollups(db, FAM, ELDER))["weekly"]
    clock.set_now(datetime(2026, 11, 1, 2, 0, tzinfo=clock.IST))
    out = await memory_upkeep.rollups(db, FAM, ELDER)
    assert out["monthly"] and out["life"]
    slugs = {n.slug for n in await store.notes(db, FAM, [ELDER])}
    assert {"weekly", "monthly", "life-so-far"} <= slugs
    router._providers.pop("fake", None)


def test_style_from_their_own_messages_and_reactions():
    own = ["haan", "le li goli", "theek hai beta 🙏", "nahi", "aaj chakkar hai"]
    scored = [(120, 0.6, "")] * 6 + [(600, -0.5, "")] * 6
    st = baselines.style_baseline(own, scored)
    assert st["their_words"] <= 3 and st["good_reply_chars"] == 120 and "about 20 words" in st["summary"]


# ── a small recall benchmark: the right memory must be in the top 3 ──

BENCH = [
    ("symptom", "left knee pain on the stairs", "ghutne mein dard"),
    ("vital", "fasting sugar 182 in the morning", "sugar kitni thi"),
    ("symptom", "dizzy after getting up from bed", "chakkar aaya tha kab"),
    ("sleep", "could not sleep, awake till 3 am", "neend ki problem"),
    ("symptom", "dry cough at night for three days", "khansi"),
    ("mood", "missing her late husband on their anniversary", "anniversary pe udaas"),
    ("outcome", "Fall: slipped in the bathroom, bruised elbow", "bathroom mein gire the"),
    ("symptom", "stomach upset after the wedding food", "pet kharab"),
]


async def test_recall_benchmark(db, at, meaning):
    for i, (kind, text, _) in enumerate(BENCH):
        await ev(db, kind, ist(1 + i), text)
    for d in range(10, 28):
        await ev(db, "meal", ist(d), "dal chawal for lunch")
        await ev(db, "dose_taken", ist(d, "08:05"), "Metformin: taken")
    clock.set_now(ist(28))
    await memory_index.index_subject(db, FAM, ELDER)
    found = 0
    for _, text, query in BENCH:
        hits = await memory_index.search(db, FAM, [ELDER], query, limit=3)
        found += any(text.split(":")[-1].strip()[:12] in h.text for h in hits)
    # With the toy meaning model above, 6 of 8 Hinglish paraphrases are found (keywords alone find fewer).
    # eval/memory_bench.py runs the same questions with real Vertex embeddings, where the bar is 8/8.
    assert found >= 6, f"recall found {found}/{len(BENCH)}"


async def test_two_notes_in_one_turn_both_kept(db, at):
    at("2026-10-02 10:00")
    c = ctx(db, E)
    await tools.run(c, "note", {"topic": "food", "text": "loves besan chilla"})
    await tools.run(c, "note", {"topic": "food", "text": "no karela please"})
    body = [n for n in await store.notes(db, FAM, [ELDER]) if n.slug == "food"][0].body_md
    assert "besan chilla" in body and "karela" in body


async def test_memory_health_api(db, at):
    import httpx

    from app.core.security import verify_api_secret
    from app.db.session import get_db
    from app.main import app

    at("2026-10-02 10:00")
    await tools.run(ctx(db), "remember", {"domain": "medicine", "name": "Ecosprin", "details": {"dose": "75 mg"}, "sentence": "Ecosprin 75"})
    await store.upsert_note(db, family_id=FAM, subject_id=ELDER, slug="family", title="Family", body_md="- argued with Rahul")
    await db.commit()

    async def _db():
        yield db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[verify_api_secret] = lambda: None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            h = (await c.get(f"/v2/dash/{FAM}/{ELDER}/memory-health")).json()
            assert any(i["kind"] == "no_time" for i in h["issues"])
            r = (await c.post(f"/v2/dash/{FAM}/{ELDER}/forget", json={"actor": {"id": CG, "name": "Asha"}, "what": "argued Rahul"})).json()
            assert r["forgotten"] == 1
            h = (await c.get(f"/v2/dash/{FAM}/{ELDER}/memory-health")).json()
            fid = h["forgotten"][0]["id"]
            r = (await c.post(f"/v2/dash/{FAM}/{ELDER}/forgotten/{fid}/restore", json={"actor": {"id": CG, "name": "Asha"}})).json()
            assert r["restored"] == 1
    finally:
        app.dependency_overrides.clear()
