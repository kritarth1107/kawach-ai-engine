"""The synthetic test world (eval/world): it builds the same bytes every time, meets every quota, and seeds cleanly."""

import json

import pytest

from eval.world import make, validate
from eval.world.slice import slice_ids


@pytest.fixture(scope="module")
def world():
    return make.build()


def test_world_is_byte_identical_to_the_lock(world):
    lock = json.loads((make.DATA / "world.lock").read_text())
    assert make.lock(world)["sha256"] == lock["sha256"], "the world changed: run python -m eval.world.make and commit world.lock"


def test_world_meets_every_quota_and_safety_rule(world):
    errs, _ = validate.check(world)
    assert errs == []


def test_slices_cover_every_language_household_and_shape(world):
    ci = set(slice_ids("ci", world))
    assert len(ci) == 120
    chosen = [f for f in world if f["id"] in ci]
    from eval.world import grid as G

    assert {f["language"] for f in chosen} == set(G.LANGS)
    assert {f["household"] for f in chosen} == set(G.HOUSEHOLDS)
    assert {s for f in chosen for s in f["shapes"]} >= set(G.SHAPES)
    assert len(slice_ids("ui", world)) == 40 and len(slice_ids("nudge", world)) == 200


def test_seeder_refuses_a_remote_database():
    from eval.world.seed_engine import assert_local

    with pytest.raises(SystemExit):
        assert_local("postgresql+asyncpg://u:p@10.1.2.3:5432/kawach_ai")
    assert_local("postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_test")


async def test_seeding_a_family_goes_through_the_store(db, world):
    from eval.world.seed_engine import Seeded, seed_family
    from app.care import store

    fam = next(f for f in world if f["shapes"] == [] and f["truth"] and any(t["medicines"] for t in f["truth"].values()))
    out = Seeded()
    try:
        await seed_family(db, fam, out)
    finally:
        from app.core import clock

        clock.set_now(None)
    sid = next(s for s, t in fam["truth"].items() if t["medicines"])
    facts = await store.facts(db, fam["familyId"], sid)
    keys = {f.key for f in facts if f.status == "active"}
    assert {k for k in keys if k.startswith("medicine:")}, keys
    assert out.events > 0 and (await store.roster(db, fam["familyId"])) is not None


def test_the_ai_written_texts_still_match_the_world(world):
    """A change to make.py that shifts names, briefs or story arcs would silently drop the paid texts (cached by hash)."""
    from eval.world.stories import attach_stories
    from eval.world.write import attach

    fams = [dict(f) for f in world]
    probes = sum(len(f["probes"]) for f in fams)
    assert attach(fams) >= 0.98 * probes
    assert attach_stories([f for f in fams if f["tier"] == "story"]) >= 95


async def test_a_paused_family_gets_nothing_saheli_starts():
    """The backend holds every message Saheli sends while the admin console pauses the family (purpose "proactive");
    the simulated backend does the same, so world checks see what a family would really get."""
    from app.sim.world import SimHost, World

    host = SimHost(World(paused=True))
    sent = await host.call("send_whatsapp", {"to": "p1", "text": "How was lunch?"}, family_id="f", subject_id="p1", actor_id="p1")
    assert sent == {"delivered": False, "reason": "paused"} and host.world.sent == []
    tpl = await host.call("send_template", {"to": "p1", "template": {"purpose": "reminder"}}, family_id="f", subject_id="p1", actor_id="p1")
    assert tpl["sent"] is False and host.world.templates == []
    open_ = SimHost(World())
    assert (await open_.call("send_whatsapp", {"to": "p1", "text": "hi"}, family_id="f", subject_id="p1", actor_id="p1"))["delivered"] is True
