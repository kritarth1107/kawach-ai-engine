"""The 200-fact bench stays 200, and a replaced dose is not what search returns."""

import hashlib

import pytest

from app.care import memory_index, store
from eval.recall_200 import bench_cases


def test_vowel_signs_stay_on_the_letter():
    assert "बीपी" in store.query_words("बीपी की गोली की खुराक क्या है")
    assert "বড়ি" in store.query_words("সুগারের বড়ি এর ডোজ কত")
    assert "மாத்திரை" in store.query_words("சர்க்கரை மாத்திரை எவ்வளவு")


def test_the_bench_is_200_facts_across_six_languages():
    cases = bench_cases()
    assert len(cases) == 200
    assert {c["lang"] for c in cases} == {"en", "hi", "hinglish", "bn", "ta", "mixed"}
    assert {c["domain"] for c in cases} >= {"medicine", "allergy", "naming", "preference", "event"}
    assert sum(1 for c in cases if c["old"]) >= 12
    for c in cases:
        assert c["needle"] not in c["query"]
        assert c["needle"] in c["text"]
        if c["old"]:
            assert c["old"] not in c["text"]


async def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * memory_index.DIM
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        out.append(v)
    return out


@pytest.fixture
def meaning():
    memory_index.set_embedder(fake_embed)
    yield
    memory_index.set_embedder(None)


async def test_a_replaced_dose_is_not_returned(db, at, meaning):
    at("2026-10-01 09:00")
    case = next(c for c in bench_cases() if c["old"] and c["lang"] == "en" and c["key"] == "medicine:metformin")
    fam = "bench-unit"
    await store.write_fact(db, family_id=fam, subject_id=case["subject"], domain="medicine", key=case["key"],
                           value={"text": case["old"]}, text=case["old"], source_kind="caregiver_said")
    at("2026-10-10 09:00")
    await store.write_fact(db, family_id=fam, subject_id=case["subject"], domain="medicine", key=case["key"],
                           value={"text": case["text"]}, text=case["text"], source_kind="caregiver_said", replace=True)
    other = "bench-other"
    await store.write_fact(db, family_id=fam, subject_id=other, domain="medicine", key=case["key"],
                           value={"text": "Metformin 20 mg at 08:00. sugar tablet"}, text="Metformin 20 mg at 08:00. sugar tablet",
                           source_kind="caregiver_said")
    await memory_index.index_subject(db, fam, case["subject"])
    await memory_index.index_subject(db, fam, other)
    hits = await memory_index.search(db, fam, [case["subject"]], case["query"], limit=3)
    texts = [h.text for h in hits]
    assert texts and case["needle"] in texts[0]
    assert all(case["old"] not in t for t in texts)
    other_hits = await memory_index.search(db, fam, [other], case["query"], limit=3)
    assert all(case["needle"] not in t for t in [h.text for h in other_hits])


async def test_a_rare_word_outranks_a_newer_common_word(db, at, meaning):
    fam, who = "bench-unit-ki", "bench-hi"
    at("2026-10-01 09:00")
    await store.write_fact(db, family_id=fam, subject_id=who, domain="medicine", key="medicine:vitamin_d",
                           value={"text": "Vitamin D 1000 mg at 08:00. vitamin D"},
                           text="Vitamin D 1000 mg at 08:00. vitamin D", source_kind="caregiver_said")
    for i in range(12):
        at(f"2026-10-1{i % 9 + 1} 10:00")
        await store.write_fact(db, family_id=fam, subject_id=who, domain="occasion", key=f"event:ki-{i}",
                               value={"text": f"note {i} ki yaad"}, text=f"note {i} ki yaad", source_kind="caregiver_said")
    await memory_index.index_subject(db, fam, who)
    hits = await memory_index.search(db, fam, [who], "vitamin D ki dose kya hai", limit=3)
    assert hits and "Vitamin D 1000 mg" in hits[0].text


def test_the_phrase_rank_is_linear_and_unchanged():
    """A scheduler turn's whole prompt is the search query. The old every-start-every-end phrase loop was cubic in its
    length and held the engine's event loop for seconds (15 Oct: the nightly's message week ran past 4 hours)."""
    import random
    import time

    from app.care.memory_index import _phrase_len

    def slow(q, h):
        hay, best = " ".join(h), 0
        for i in range(len(q)):
            for j in range(i + 1, len(q) + 1):
                if j - i > best and " ".join(q[i:j]) in hay:
                    best = j - i
        return best

    rnd = random.Random(11)
    words = ["bp", "ki", "goli", "ol", "oli", "maa", "raat", "गोली", "बीपी", "की", "sugar", "walk"]
    for _ in range(3000):
        q = [rnd.choice(words) for _ in range(rnd.randint(0, 10))]
        h = [rnd.choice(words) for _ in range(rnd.randint(0, 12))]
        assert _phrase_len(q, h) == slow(q, h), (q, h)
    assert _phrase_len(["बीपी", "की", "गोली"], ["सुबह", "बीपी", "की", "गोली", "8", "बजे"]) == 3
    long_q = [rnd.choice(words) for _ in range(2000)]
    t = time.perf_counter()
    for _ in range(50):
        _phrase_len(long_q, [rnd.choice(words) for _ in range(80)])
    assert time.perf_counter() - t < 1.0
