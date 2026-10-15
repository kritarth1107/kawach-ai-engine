"""One live score every day: the three checks of 3 Oct (trust plan C1.1, C1.2), per family and for everyone.

- Check 1, "the reminder was actually sent" (code): every due dose and reminder slot sent within its window and
  delivered by WhatsApp's own receipt. Score = delivered on time / due; a slot held by rule (already marked taken,
  family paused) is left out of the denominator. The slots come from the backend: the delivery ledger
  (message_deliveries, with wamid and receipts) once it has rows for the day, else the reminder ledger (SaheliNudgeLog,
  where "delivered" means Meta accepted the send; the score says which source it used).
- Check 2, "nothing she was told is missing from memory" (the full-memory judge): per conversation, what the person
  said that should be remembered, checked against memory after it. Score = remembered / should be remembered.
- Check 3, "a correction held on the next reply" (code finds corrections, the judge reads her messages): Saheli's next
  reply after a correction and every message of hers that day, for each correction of the last 7 days. Score = held /
  corrections with messages to check that day.
The overall number is the minimum of the three that have data (one failing area is never hidden by the others); a check
with nothing to measure is "no data", never 100. The judge's pass rate on the day's sample of replies ("replies") is stored
beside them, with how many were graded and how many were sent.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.care import naming as care_naming
from app.care import rejected as care_rejected
from app.care.models import CareFact, FamilyRoster, Turn
from app.learn.models import ReplyLog
from app.core import clock
from app.quality import corrections, judge
from app.quality.models import DailyScore, ReplyVerdict

logger = logging.getLogger(__name__)

CHECKS = ("reminders", "remembered", "corrections")
ALL = "all"
ON_TIME = {"dose_reminder": timedelta(minutes=2), "reminder": timedelta(minutes=15)}
CORRECTION_WINDOW = 7  # days a correction is checked for
MAX_MESSAGES = 30  # of hers per correction per day


def day_bounds(day: str) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(day).replace(tzinfo=clock.IST)
    return start, start + timedelta(days=1)


def pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 1) if den else None


async def put(session: AsyncSession, day: str, family_id: str, check: str, value: float | None, n: int, failures: list | None = None,
              detail: dict | None = None) -> None:
    values = dict(day=date.fromisoformat(day), family_id=family_id, check=check, value=value, n=n, failures=(failures or [])[:200],
                  detail=detail or {}, computed_at=clock.now())
    await session.execute(insert(DailyScore).values(**values).on_conflict_do_update(
        index_elements=[DailyScore.day, DailyScore.family_id, DailyScore.check],
        set_={k: v for k, v in values.items() if k not in ("day", "family_id", "check")}))


# ── check 1 ────────────────────────────────────────────────────────────────────


def _t(v) -> datetime | None:
    if not v:
        return None
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=clock.IST)


def reminders(slots: list[dict], *, receipts: bool = True) -> dict[str, dict]:
    """{family: {"num", "den", "failures", "held"}} from the day's due slots."""
    out: dict[str, dict] = {}
    for s in slots:
        fam = str(s.get("familyId") or "")
        if not fam:
            continue
        row = out.setdefault(fam, {"num": 0, "den": 0, "failures": [], "held": {}})
        if s.get("held"):
            why = str(s.get("heldReason") or "held")
            row["held"][why] = row["held"].get(why, 0) + 1
            continue
        kind = "dose_reminder" if s.get("kind") == "dose_reminder" else "reminder"
        due, sent, delivered, failed = _t(s.get("dueAt")), _t(s.get("sentAt")), _t(s.get("deliveredAt")), _t(s.get("failedAt"))
        row["den"] += 1
        late = sent is not None and due is not None and sent > due + ON_TIME[kind]
        got = delivered is not None if receipts else sent is not None
        if sent is not None and got and not late and failed is None:
            row["num"] += 1
            continue
        if failed is not None or s.get("error"):
            why = f"failed: {s.get('error') or 'send failed'}"
        elif sent is None:
            why = "not sent"
        elif late:
            why = f"sent {int((sent - due).total_seconds() // 60)} min late" if due else "sent late"
        else:
            why = "sent, no delivered receipt from WhatsApp"
        row["failures"].append({"family": fam[:8], "person": s.get("person") or "", "item": str(s.get("item") or "")[:80], "kind": kind,
                                "due": clock.ist(due).strftime("%H:%M") if due else "", "what": why, "ref": s.get("refId")})
    return out


async def check_reminders(host, day: str) -> tuple[dict[str, dict], dict]:
    """Ask the backend for the day's due slots (OPS call: not about one family)."""
    from app.tasks.credits import OPS

    res = await host.call("delivery_day", {"day": day}, family_id=OPS, subject_id=OPS, actor_id=OPS)
    res = res.get("result", res) if isinstance(res, dict) else {}
    if not isinstance(res, dict) or res.get("error"):
        raise RuntimeError(f"delivery_day: {str((res or {}).get('error') if isinstance(res, dict) else res)[:160]}")
    receipts = bool(res.get("receipts", res.get("source") == "message_deliveries"))
    return reminders(list(res.get("slots") or []), receipts=receipts), {"source": res.get("source") or "unknown", "receipts": receipts,
                                                                      "slots": len(res.get("slots") or [])}


# ── check 3 ────────────────────────────────────────────────────────────────────


async def corrections_held(sessions: async_sessionmaker, family_id: str, day: str) -> dict:
    """{"num", "den", "failures", "errors", "checked": [...]} for this family's corrections of the last 7 days, on this day."""
    start, end = day_bounds(day)
    async with sessions() as s:
        found = await corrections.find(s, family_id, start - timedelta(days=CORRECTION_WINDOW - 1), end)
        if not found:
            return {"num": 0, "den": 0, "failures": [], "errors": 0, "checked": []}
        roster = await s.get(FamilyRoster, family_id)
        elder, members = judge.people_of(roster)
        names = judge.names_of(elder, members)
        ids = set(names)
        jobs = []
        for i, c in enumerate(found):
            since = max(c.at, start)
            # a later correction of the same thing replaces this one: her messages after it are checked against that one
            # (13 Oct: a language switched twice in a day failed the first switch for every message after the second)
            later = next((x.at for x in found[i + 1:] if x.subject_id == c.subject_id and x.key == c.key and x.at > c.at), None)
            until = min(end, later) if later else end
            if since >= until:
                continue
            day_msgs = list((await s.execute(
                select(Turn).where(Turn.family_id == family_id, Turn.role == "assistant", Turn.at > since, Turn.at < until).order_by(Turn.at).limit(200)
            )).scalars())
            next_thread = c.by if c.by in ids else c.subject_id
            nxt = (await s.execute(select(Turn).where(Turn.family_id == family_id, Turn.thread_id == next_thread, Turn.role == "assistant",
                                                     Turn.at > c.at).order_by(Turn.at).limit(1))).scalar_one_or_none()
            msgs = [t for t in day_msgs if (t.text or "").strip().lower() not in ("", "none")]
            msgs.sort(key=lambda t: (t.thread_id != c.subject_id, t.at))
            msgs = msgs[:MAX_MESSAGES]
            next_today = nxt is not None and start <= nxt.at < until
            if next_today and nxt is not None and all(t.id != nxt.id for t in msgs):
                msgs.insert(0, nxt)
            if not msgs:
                continue
            terms: list[str] = []
            if c.kind in ("reject", "stop"):
                row = await s.get(CareFact, c.fact_id) if c.fact_id else None
                terms = care_rejected.terms([row]) if row is not None and c.kind == "reject" else []
            jobs.append((c, msgs, nxt.id if next_today and nxt is not None else None, terms))
    results: list[dict] = []
    errors = 0
    sem = asyncio.Semaphore(judge.parallel())

    async def one(c: corrections.Correction, msgs: list[Turn], next_id: int | None, terms: list[str]) -> None:
        nonlocal errors
        code_bad: list[str] = []
        for t in msgs:
            if c.kind == "naming" and t.thread_id == c.subject_id:
                code_bad += [f"used '{w}'" for w in care_naming.uses_avoided(t.text, c.avoid_added)]
            code_bad += [f"used '{w}'" for w in terms if care_naming.uses_avoided(t.text, [w])]
        numbered = [{"n": i + 1, "to": names.get(t.thread_id, t.thread_id[:6]), "at": clock.ist(t.at).strftime("%d %b %H:%M"),
                     "next_reply": t.id == next_id, "text": (t.text or "")[:800]} for i, t in enumerate(msgs)]
        async with sem:
            try:
                judged = await judge.correction_held({**c.as_dict(), "about": names.get(c.subject_id, "the person")}, numbered)
            except Exception as exc:  # noqa: BLE001 — not checked today (counted as an error, never as held)
                logger.warning("correction check failed family=%s key=%s: %s", family_id, c.key, str(exc)[:160])
                errors += 1
                return
        broke = [j for j in judged if j["relevant"] and not j["held"]]
        held = not code_bad and not broke
        results.append({"key": c.key, "kind": c.kind, "person": names.get(c.subject_id, "")[:40].split(" ")[0], "held": held, "messages": len(msgs),
                        "at": c.at.isoformat(), "old": c.old[:160], "new": c.new[:160],
                        "words": (code_bad + [j["words"] for j in broke if j["words"]])[:3]})

    await asyncio.gather(*(one(*j) for j in jobs))
    fails = [{"family": family_id[:8], "person": r["person"], "what": f"correction of {r['key']} not held", "old": r["old"], "words": r["words"]}
             for r in results if not r["held"]]
    return {"num": sum(1 for r in results if r["held"]), "den": len(results), "failures": fails, "errors": errors, "checked": results}


# ── per family, then everyone ──────────────────────────────────────────────────


async def family_checks(sessions: async_sessionmaker, family_id: str, day: str) -> dict:
    """Checks 2 and 3 and the reply pass rate for one family (the judge has graded the day's replies first)."""
    out: dict = {}
    conv = await judge.remembered(sessions, family_id, day)
    ok = [c for c in conv if "error" not in c]
    should = sum(c["should"] for c in ok)
    kept = sum(c["remembered"] for c in ok)
    start, _ = day_bounds(day)
    async with sessions() as s:
        roster = await s.get(FamilyRoster, family_id)
        elder, members = judge.people_of(roster)
        names = judge.names_of(elder, members)
        miss = [{"family": family_id[:8], "person": names.get(c["thread"], "")[:40].split(" ")[0], "what": f"not remembered: {m['what']}",
                 "kind": m["kind"], "said": m["said"]} for c in ok for m in c["missing"]]
        await put(s, day, family_id, "remembered", pct(kept, should), should, miss,
                  {"num": kept, "conversations": len(conv), "errors": len(conv) - len(ok),
                   "invented": [{"person": names.get(c["thread"], "")[:40].split(" ")[0], **x} for c in ok for x in c["invented"]][:30]})
        await s.commit()
    out["remembered"] = {"num": kept, "den": should, "errors": len(conv) - len(ok)}
    held = await corrections_held(sessions, family_id, day)
    async with sessions() as s:
        await put(s, day, family_id, "corrections", pct(held["num"], held["den"]), held["den"], held["failures"],
                  {"num": held["num"], "errors": held["errors"], "checked": held["checked"][:30]})
        vs = list((await s.execute(select(ReplyVerdict).where(ReplyVerdict.family_id == family_id, ReplyVerdict.day == day))).scalars())
        passed = sum(1 for v in vs if v.passed)
        rfails = [{"family": family_id[:8], "person": names.get(v.thread_id, "")[:40].split(" ")[0], "what": ", ".join(v.issues or [])[:120],
                   "reply": v.reply_id, "words": "; ".join(w for i in (v.issues or []) for w in ((v.verdict or {}).get("issues") or {}).get(i, {}).get("words") or [])[:200]}
                  for v in vs if not v.passed]
        sent = int((await s.execute(
            select(func.count()).select_from(ReplyLog).where(
                ReplyLog.family_id == family_id, ReplyLog.at >= start, ReplyLog.at < start + timedelta(days=1))
        )).scalar_one() or 0)
        await put(s, day, family_id, "replies", pct(passed, len(vs)), len(vs), rfails,
                  {"num": passed, "sampled": len(vs), "sent": sent})
        await s.commit()
    out["corrections"] = {"num": held["num"], "den": held["den"], "errors": held["errors"]}
    out["replies"] = {"num": passed, "den": len(vs)}
    del start
    return out


async def finish(sessions: async_sessionmaker, host, day: str) -> dict:
    """After every family's night: check 1 (one call for all), then each family's overall and everyone's ("all")."""
    try:
        by_family, meta = await check_reminders(host, day)
        error = None
    except Exception as exc:  # noqa: BLE001 — no ledger means no check 1 today (shown as an error, not as 100)
        logger.exception("check 1 (reminders) failed")
        by_family, meta, error = {}, {}, str(exc)[:200]
    async with sessions() as s:
        d = date.fromisoformat(day)
        rows = list((await s.execute(select(DailyScore).where(DailyScore.day == d, DailyScore.family_id != ALL))).scalars())
        families = sorted({r.family_id for r in rows} | set(by_family))
        for fam, r1 in by_family.items():
            await put(s, day, fam, "reminders", pct(r1["num"], r1["den"]), r1["den"], r1["failures"], {"num": r1["num"], "held": r1["held"], **meta})
        await s.flush()
        rows = list((await s.execute(select(DailyScore).where(DailyScore.day == d, DailyScore.family_id != ALL))).scalars())
        by = {(r.family_id, r.check): r for r in rows}
        for fam in families:
            vals = [by[(fam, c)].value for c in CHECKS if (fam, c) in by and by[(fam, c)].value is not None]
            await put(s, day, fam, "overall", min(vals) if vals else None, len(vals), [], {c: by[(fam, c)].value for c in CHECKS if (fam, c) in by})
        total: dict[str, float | None] = {}
        for check in (*CHECKS, "replies"):
            rs = [r for r in rows if r.check == check]
            num = sum(int((r.detail or {}).get("num") or 0) for r in rs)
            den = sum(int(r.n or 0) for r in rs)
            fails = [f for r in rs for f in (r.failures or [])]
            detail: dict = {"num": num, "families": len(rs), "errors": sum(int((r.detail or {}).get("errors") or 0) for r in rs)}
            if check == "replies":
                detail["sampled"] = den
                detail["sent"] = sum(int((r.detail or {}).get("sent") or 0) for r in rs)
            if check == "reminders":
                detail.update(meta)
                detail["held"] = {k: sum(int(((r.detail or {}).get("held") or {}).get(k) or 0) for r in rs)
                                  for k in {k for r in rs for k in ((r.detail or {}).get("held") or {})}}
                if error:
                    detail["error"] = error
            await put(s, day, ALL, check, pct(num, den), den, fails, detail)
            total[check] = pct(num, den)
        vals = [v for c, v in total.items() if c in CHECKS and v is not None]
        await put(s, day, ALL, "overall", min(vals) if vals else None, len(vals), [], {c: total.get(c) for c in CHECKS})
        await s.commit()
    return {"day": day, "families": len(families), **{k: v for k, v in total.items()}, "overall": min(vals) if vals else None,
            "reminders_source": meta.get("source"), "reminders_error": error}
