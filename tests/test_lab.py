"""Saheli Lab: bot-designed families and the queue the family-player bots answer."""

import asyncio
import copy
import json
from pathlib import Path

import pytest

from eval.lab import families as lab
from eval.lab import queue

EXAMPLE = json.loads((Path(__file__).resolve().parents[1] / "eval/lab/examples/example-verma.json").read_text())


def test_example_family_is_valid_and_converts():
    assert lab.validate(EXAMPLE) == []
    spec = lab.to_spec(EXAMPLE)
    assert spec["events"][0][:3] == (2, "11:00", "ve-usha") and spec["meta"]["group"] == "north"
    assert "Sample messages" in spec["people"][0]["persona"]


@pytest.mark.parametrize("break_it,needle", [
    (lambda f: f["people"][0].update(phone="+91 98765 43210"), "fake"),
    (lambda f: f["meta"].update(group="west"), "group"),
    (lambda f: f["events"].append({"day": 40, "time": "10:00", "who": "ve-usha", "what": "x"}), "bad event"),
    (lambda f: f["events"].append({"day": 3, "time": "10:00", "who": "ve-usha", "what": "x", "expect": {"tools": ["teleport"]}}), "unknown tool"),
    (lambda f: [p.update(recipient=False) for p in f["people"]], "care recipient"),
])
def test_validator_catches_mistakes(break_it, needle):
    f = copy.deepcopy(EXAMPLE)
    break_it(f)
    assert any(needle in e for e in lab.validate(f))


async def test_queue_round_trip(db, monkeypatch):
    """The simulator and the bot run concurrently, so this test uses its own pool (cleaned up after)."""
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from tests.conftest import TEST_DB

    eng = create_async_engine(TEST_DB)
    sessions = async_sessionmaker(eng, expire_on_commit=False)
    monkeypatch.setattr(queue, "SessionLocal", sessions)
    monkeypatch.setenv("SIM_RUN_ID", "test-lab-queue")

    async def bot():
        for _ in range(100):
            got = await queue.claim("north", "Family Player North", limit=5)
            if got:
                return await queue.answer([{"id": r.id, "text": f"reply to {r.person}"} for r in got], "Family Player North")
            await asyncio.sleep(0.05)

    try:
        reply, done = await asyncio.gather(queue.ask("sim", "sys", "prompt", group="north", family="f", person="ve-usha", poll_s=0.05, timeout_s=5), bot())
        assert reply == "reply to ve-usha" and done["answered"] == 1
        assert await queue.ask("sim", "s", "p", group="south_east", poll_s=0.05, timeout_s=0.2) is None  # nobody answered: times out
    finally:
        async with sessions() as s:
            await s.execute(delete(queue.LabRequest).where(queue.LabRequest.run_id == "test-lab-queue"))
            await s.commit()
        await eng.dispose()
