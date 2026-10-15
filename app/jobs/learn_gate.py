"""The draft's replay gate, inside AWS (03:30 IST, after the world and the dream).

GitHub cannot reach the engine, so it no longer runs this. The cron starts the kavach-learn-gate task. That task uses
Gemini in kavach-eval-1015, never kavach-care, on a throwaway database in the same task, capped at ₹30. No regression
starts the canary. A regression rejects the draft. An error leaves it waiting. A canary already running is judged with
the same rules as the night (app.learn.lessons.evaluate).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)


async def _waiting(sessions) -> int | None:
    from sqlalchemy import select

    from app.learn.models import PlaybookVersion

    async with sessions() as session:
        row = (await session.execute(
            select(PlaybookVersion.version).where(PlaybookVersion.status == "draft").order_by(PlaybookVersion.version.desc()).limit(1)
        )).first()
    return int(row[0]) if row else None


async def wait_replay_db() -> None:
    """The throwaway database in this task. Never the production database."""
    import asyncpg

    raw = os.getenv("REPLAY_DATABASE_URL", "").replace("+asyncpg", "").strip()
    if not raw:
        raise RuntimeError("REPLAY_DATABASE_URL is not set")
    last = ""
    for _ in range(max(1, int(os.getenv("LEARN_GATE_DB_TRIES", "40")))):
        try:
            conn = await asyncpg.connect(raw)
            await conn.close()
            return
        except Exception as exc:  # noqa: BLE001 — it is still starting
            last = str(exc)[:160]
            await asyncio.sleep(1)
    raise RuntimeError(f"replay database did not start: {last}")


async def run(sessions=None) -> dict:
    from app.db.session import SessionLocal
    from app.learn import lessons

    sessions = sessions or SessionLocal
    version = await _waiting(sessions)
    gated: dict | None = None
    if version is not None:
        await wait_replay_db()
        gated = await lessons.gate_and_canary(sessions, version)
    canary = await lessons.evaluate(sessions)
    if version is None:
        return {"draft": None, "why": "no draft is waiting", "canary": canary.get("decision") or canary.get("canary")}
    return {"draft": version, "status": (gated or {}).get("status"), "gate": (gated or {}).get("gate"),
            "canary": canary.get("decision") or canary.get("canary")}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        print(json.dumps(asyncio.run(run()), indent=2, default=str))
    except Exception as exc:  # noqa: BLE001 — a failed run must show in the task exit, and the draft stays a draft
        logger.exception("learn gate failed")
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:400]}))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
