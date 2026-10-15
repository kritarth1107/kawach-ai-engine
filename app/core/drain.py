"""On SIGTERM the engine stops starting turns and lets the ones already running finish.

ECS sends SIGKILL when stopTimeout ends (120s on Fargate). A turn still open then has its
session rolled back, so a deploy does not leave a committed tool and no reply.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

log = logging.getLogger(__name__)

# A few seconds inside the 120s stop timeout, so rollback runs before SIGKILL.
WAIT_S = 110


class Draining(Exception):
    """A new turn arrived after SIGTERM. The caller answers 503 and does not write."""


class TurnDrain:
    def __init__(self) -> None:
        self.draining = False
        self._open = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._sessions: set[Any] = set()
        self._lock = asyncio.Lock()

    def start(self) -> None:
        self.draining = True
        log.info("draining: no new turns, %s still running", self._open)

    @asynccontextmanager
    async def turn(self, session: Any) -> AsyncIterator[None]:
        async with self._lock:
            if self.draining:
                raise Draining()
            self._open += 1
            self._idle.clear()
            self._sessions.add(session)
        try:
            yield
        finally:
            async with self._lock:
                self._sessions.discard(session)
                self._open -= 1
                if self._open == 0:
                    self._idle.set()

    async def finish(self, timeout: float = WAIT_S) -> bool:
        """Wait for open turns. Roll back any still open when the wait ends."""
        self.start()
        if self._open == 0:
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
        except TimeoutError:
            await self._rollback()
            return False
        return self._open == 0

    async def _rollback(self) -> None:
        async with self._lock:
            sessions = list(self._sessions)
        for session in sessions:
            rollback = getattr(session, "rollback", None)
            if rollback is None:
                continue
            try:
                await rollback()
                log.warning("rolled back a turn that did not finish before shutdown")
            except Exception:
                log.exception("rollback on shutdown failed")


drain = TurnDrain()


def chain(previous: Callable[..., Any] | int | None, gate: TurnDrain = drain) -> Callable[..., None]:
    """SIGTERM marks the drain, then the handler that was already installed still runs."""

    def handler(signum: int, frame: Any) -> None:
        gate.start()
        if callable(previous):
            previous(signum, frame)

    return handler


def install_sigterm() -> None:
    """Wrap the process SIGTERM handler. Uvicorn installs its own before the app starts."""
    try:
        signal.signal(signal.SIGTERM, chain(signal.getsignal(signal.SIGTERM)))
    except (ValueError, OSError):
        return
