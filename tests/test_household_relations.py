"""Saheli is told who each member is, and to whom (world probe judge, 2026-10-13).

The backend sent each member's name and role but not the relationship the family gave, so Saheli guessed: a patient's
husband was called her son and a wife was called mother. The relation is to the member who added them, and where none
is known Saheli is told not to guess."""

from app.api.brain import Person
from app.care import digest
from eval.world.make import load
from eval.world.seed_engine import relations, roster_of

PREM = {"id": "c1", "name": "Prem", "role": "primary caregiver"}
HEMA = {"id": "e1", "name": "Hema", "role": "elder", "relation": "spouse", "relation_to": "c1"}
ASHA = {"id": "c2", "name": "Asha", "role": "caregiver", "relation": "Sister", "relation_to": "c1"}


def test_each_member_is_shown_with_who_they_are_to_whom():
    block = digest.household(HEMA, [HEMA, PREM, ASHA], "c1")
    assert "Care recipient: Hema (id e1), spouse of Prem" in block
    assert "caregiver: Asha (id c2), sister of Prem" in block
    assert "primary caregiver: Prem (id c1)  ← speaking now" in block  # the one who made the family has no relation
    assert "Never guess one" in block


def test_a_relation_whose_member_is_unknown_still_says_so_and_none_is_invented():
    lone = {"id": "e1", "name": "Kamla", "role": "elder", "relation": "mother", "relation_to": "gone"}
    block = digest.household(lone, [lone, PREM], "c1")
    assert "Kamla (id e1), relation given: mother (to the family member who added them)" in block
    plain = digest.household({"id": "e1", "name": "Kamla", "role": "elder"}, [{"id": "e1", "name": "Kamla", "role": "elder"}, PREM], "c1")
    assert "of Prem" not in plain and "relation given" not in plain
    other = {"id": "e1", "name": "Kamla", "role": "elder", "relation": "Other", "relation_to": "c1"}
    assert "other of" not in digest.household(other, [other, PREM], "c1")


def test_self_care_alone_gets_no_relation_lines():
    me = {"id": "s1", "name": "Rhea", "role": "primary caregiver"}
    block = digest.household(me, [me], "s1")
    assert "Never guess" not in block and "of " not in block.split("\n")[1]


def test_the_relation_comes_through_the_api_model():
    p = Person(id="e1", name="Hema", role="elder", relation="spouse", relation_to="c1")
    assert p.model_dump() == {"id": "e1", "name": "Hema", "role": "elder", "relation": "spouse", "relation_to": "c1"}
    assert Person(id="c1").model_dump()["relation"] == ""


def test_the_world_seeds_relations_the_way_the_backend_stores_them():
    by_house: dict[str, dict] = {}
    for f in load():
        by_house.setdefault(f["household"], f)
    surgery = by_house["post_surgery"]
    patient = next(p for p in surgery["people"] if p["careRecipient"])
    spouse = next(p for p in surgery["people"] if p["role"] == "primary caregiver")
    assert relations(surgery)[patient["id"]] == ("spouse", spouse["id"])
    siblings = by_house["siblings_share"]
    second = next(p for p in siblings["people"] if p["role"] == "caregiver")
    assert relations(siblings)[second["id"]][0] in ("brother", "sister")
    for f in by_house.values():
        elder, members = roster_of(f)
        primary = next((m for m in members if m["role"] == "primary caregiver"), None)
        assert all("relation" not in m for m in members if m is primary)
        if primary and primary["id"] != elder["id"]:
            assert elder.get("relation_to") == primary["id"], f["household"]
