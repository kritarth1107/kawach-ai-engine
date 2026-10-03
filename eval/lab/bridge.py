"""Local bridge between the simulator and the Saheli Lab bots (family players, judge).

    LAB_TOKEN=… DATABASE_URL=<training db> .venv/bin/python -m uvicorn eval.lab.bridge:app --host 127.0.0.1 --port 8787

GET  /lab/next?group=north&player=Family%20Player%20North&limit=10   claim up to N requests (role sim or judge)
POST /lab/answer {"player": "...", "answers": [{"id": 12, "text": "..."}]}
GET  /lab/status                                                     counts per run, group and status
All calls need the header X-Lab-Token. Bound to localhost only.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from app.db.session import Base, SessionLocal, engine
from eval.lab import queue

TOKEN = os.getenv("LAB_TOKEN", "")
app = FastAPI(title="Saheli Lab bridge")


def auth(x_lab_token: str = Header(default="")) -> None:
    if not TOKEN or not secrets.compare_digest(x_lab_token, TOKEN):
        raise HTTPException(status_code=401, detail="bad lab token")


@app.on_event("startup")
async def _schema() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/lab/next", dependencies=[Depends(auth)])
async def next_batch(group: str, player: str, limit: int = 10, role: str | None = None) -> dict:
    rows = await queue.claim(group, player, limit=max(1, min(limit, 30)), role=role)
    return {"requests": [{"id": r.id, "role": r.role, "family": r.family, "person": r.person, "system": r.system, "prompt": r.prompt} for r in rows]}


class Answer(BaseModel):
    id: int
    text: str


class Answers(BaseModel):
    player: str
    answers: list[Answer]


@app.post("/lab/answer", dependencies=[Depends(auth)])
async def post_answers(body: Answers) -> dict:
    return await queue.answer([a.model_dump() for a in body.answers], body.player)


@app.get("/lab/status", dependencies=[Depends(auth)])
async def status() -> dict:
    async with SessionLocal() as s:
        rows = (await s.execute(
            select(queue.LabRequest.run_id, queue.LabRequest.group, queue.LabRequest.status, func.count()).group_by(
                queue.LabRequest.run_id, queue.LabRequest.group, queue.LabRequest.status)
        )).all()
    out: dict = {}
    for run, group, st, n in rows:
        out.setdefault(run, {}).setdefault(group, {})[st] = n
    return {"runs": out}
