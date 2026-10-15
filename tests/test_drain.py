"""A deploy must finish the turn it already started, and must not start another."""

import asyncio
import signal
from pathlib import Path

import pytest

from app.core.drain import Draining, TurnDrain, chain


class Session:
    def __init__(self) -> None:
        self.rolled_back = False

    async def rollback(self) -> None:
        self.rolled_back = True


@pytest.mark.asyncio
async def test_a_new_turn_is_refused_once_draining():
    gate = TurnDrain()
    async with gate.turn(Session()):
        gate.start()
        with pytest.raises(Draining):
            async with gate.turn(Session()):
                pass


@pytest.mark.asyncio
async def test_shutdown_waits_for_the_open_turn():
    gate = TurnDrain()

    async def work() -> None:
        async with gate.turn(Session()):
            await asyncio.sleep(0.05)

    task = asyncio.create_task(work())
    await asyncio.sleep(0)
    assert await gate.finish(1) is True
    await task


@pytest.mark.asyncio
async def test_a_turn_still_open_at_the_deadline_is_rolled_back():
    gate = TurnDrain()
    session = Session()

    async def work() -> None:
        async with gate.turn(session):
            await asyncio.sleep(5)

    task = asyncio.create_task(work())
    await asyncio.sleep(0)
    assert await gate.finish(0.05) is False
    assert session.rolled_back
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_sigterm_marks_draining_and_keeps_the_previous_handler():
    called: list[int] = []
    gate = TurnDrain()

    def previous(signum: int, _frame: object) -> None:
        called.append(signum)

    chain(previous, gate)(signal.SIGTERM, None)
    assert gate.draining
    assert called == [signal.SIGTERM]


def test_the_process_is_given_two_minutes_to_finish():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text()
    assert '--timeout-graceful-shutdown", "110"' in dockerfile
    workflow = (root / ".github/workflows/deploy-aws.yml").read_text()
    assert ".stopTimeout = 120" in workflow
