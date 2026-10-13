"""What the admin "Daily score" page shows and what the team's 08:00 IST email says (trust plan C1.2, A2.3, C2.2).

- the live score: the overall number (the minimum of the three checks), each check, and the judge's pass rate on every
  reply, for the day and as a 30-day trend
- every failure, one line each (short family ids and first names only)
- the memory score per family (the minimum across its people), with the unknown truths
- the nightly world score: the 1,000 test families, per language for the code checks and the judge, per household
  type, per topic, recall and language-correct, with a 30-day trend per language
- the judge's calibration (agreement with people)
"""

from __future__ import annotations

import html
import os
from collections import defaultdict
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.quality.daily import ALL, CHECKS
from app.quality.models import DailyScore, EvalScore, MemoryScore

LABELS = {"reminders": "Reminders sent and delivered on time", "remembered": "Nothing she was told is missing from memory",
          "corrections": "Corrections held", "replies": "Replies the full-memory judge passed", "overall": "Overall (lowest of the three)"}
ALERT_PROBLEMS = ("never-use name", "family said is wrong", "wrong_name", "wrong_medicine", "correction_ignored", "medicine", "allergies")


def admin_url() -> str:
    return os.getenv("ADMIN_CONSOLE_URL", "https://kavach-admin-303943038694.asia-south1.run.app").rstrip("/")


def _num(v) -> str:
    return "no data" if v is None else f"{v:.0f}%" if float(v).is_integer() else f"{v:.1f}%"


async def latest_day(session: AsyncSession) -> date | None:
    return (await session.execute(select(DailyScore.day).where(DailyScore.family_id == ALL).order_by(DailyScore.day.desc()).limit(1))).scalar_one_or_none()


async def world(session: AsyncSession, days: int = 30) -> dict:
    rows = list((await session.execute(
        select(EvalScore).where(EvalScore.kind == "nightly", EvalScore.day >= clock.ist().date() - timedelta(days=days)).order_by(EvalScore.day, EvalScore.id)
    )).scalars())
    if not rows:
        return {"latest": None, "trend": []}
    last = rows[-1]
    trend = []
    for r in rows:
        sc = r.scores or {}
        trend.append({"day": r.day.isoformat(), "passed": r.passed, "judge": ((sc.get("judge") or {}).get("overall") or {}).get("pct"),
                      "code": ((sc.get("code") or {}).get("overall") or {}).get("pct"), "recall": (sc.get("recall") or {}).get("pct"),
                      "languages": {k: v.get("pct") for k, v in ((sc.get("judge") or {}).get("languages") or {}).items()}})
    return {"latest": {"day": last.day.isoformat(), "runId": last.run_id, "passed": last.passed, "commit": last.commit[:12], "failures": last.failures,
                       "scores": last.scores}, "trend": trend}


def _nums(d: dict | None) -> str:
    """The few numbers a playbook's gate or canary result carries, as 'key value' pairs."""
    if not isinstance(d, dict):
        return ""
    keep = []
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            keep.append(f"{k} {round(v, 3) if isinstance(v, float) else v}")
        elif isinstance(v, (str, bool)) and k in ("decision", "verdict", "passed", "reason", "why"):
            keep.append(f"{k} {str(v)[:80]}")
    return ", ".join(keep[:6])


async def lessons(session: AsyncSession, days: int = 14) -> dict:
    """The self-learning loop (plan section 10.2): lessons live, in testing, rolled back, and rule proposals the safety gate
    dropped; one line each, for the 08:00 email and the page."""
    from app.learn.models import PlaybookVersion, RuleProposal

    since = clock.now() - timedelta(days=days)
    pbs = list((await session.execute(select(PlaybookVersion).order_by(PlaybookVersion.version.desc()).limit(30))).scalars())
    props = list((await session.execute(select(RuleProposal).where(RuleProposal.created_at >= since).order_by(RuleProposal.created_at.desc()).limit(50))).scalars())

    def line(p) -> str:
        what = "; ".join(x for xs in (p.lessons or {}).values() for x in (xs or []))[:160] or (p.note or "")[:160]
        result = getattr(p, "canary_result", None) or p.canary or p.gate
        return f"v{p.version}: {what}" + (f" ({_nums(result)})" if _nums(result) else "")

    out = {"live": [line(p) for p in pbs if p.status == "live"][:3],
           "testing": [f"{p.status}: " + line(p) for p in pbs if p.status in ("draft", "canary") and p.created_at >= since][:10],
           "rolled_back": [line(p) for p in pbs if p.status in ("rolled_back", "blocked", "rejected") and p.created_at >= since][:10],
           "dropped": [f"{r.rule[:140]} (why: {(r.why or '')[:140]})" for r in props if r.status == "dropped" and (r.decided_by or "") == "auto"][:10]}
    return out


async def page(session: AsyncSession, *, days: int = 30, day: str | None = None) -> dict:
    d = date.fromisoformat(day) if day else await latest_day(session)
    since = (d or clock.ist().date()) - timedelta(days=days - 1)
    rows = list((await session.execute(select(DailyScore).where(DailyScore.day >= since, DailyScore.day <= (d or clock.ist().date())))).scalars())
    trend: dict[str, dict] = defaultdict(dict)
    for r in rows:
        if r.family_id == ALL:
            trend[r.day.isoformat()][r.check] = r.value
    today = {r.check: {"value": r.value, "n": r.n, "detail": r.detail} for r in rows if r.family_id == ALL and r.day == d}
    failures = [{"check": r.check, **f} for r in rows if r.family_id == ALL and r.day == d and r.check != "overall" for f in (r.failures or [])]
    fams: dict[str, dict] = defaultdict(dict)
    for r in rows:
        if r.family_id != ALL and r.day == d:
            fams[r.family_id][r.check] = r.value
    mem = list((await session.execute(select(MemoryScore).where(MemoryScore.day == d, MemoryScore.subject_id == "family")))) if d else []
    memory = []
    for (m,) in mem:
        memory.append({"family": m.family_id[:8], "score": m.score, "stored": m.stored, "recalled": m.recalled, "used": m.used, "unknowns": m.unknowns,
                       "failures": m.failures[:20], "people": (m.detail or {}).get("people")})
        for f in m.failures[:20]:
            failures.append({"check": "memory", "family": m.family_id[:8], "person": f.get("person"), "what": memory_line(f)})
    from app.db.session import SessionLocal
    from app.quality import calibration

    try:
        cal = await calibration.agreement(SessionLocal)
    except Exception:  # noqa: BLE001 — the page must not fail on the calibration lookup
        cal = None
    from app.quality import cases

    replay_counts = await cases.counts(session)
    try:
        learned = await lessons(session)
    except Exception:  # noqa: BLE001 — the page must not fail on the learning lookup
        learned = {}
    return {"day": d.isoformat() if d else None, "today": today, "checks": list(CHECKS), "replay": replay_counts, "lessons": learned,
            "trend": [{"day": k, **v} for k, v in sorted(trend.items())],
            "families": [{"family": f[:8], **v} for f, v in sorted(fams.items(), key=lambda x: (x[1].get("overall") is None, x[1].get("overall") or 0))],
            "memory": sorted(memory, key=lambda m: (m["score"] is None, m["score"] or 0)), "failures": failures[:400],
            "world": await world(session, days), "calibration": cal}


def memory_line(f: dict) -> str:
    if f.get("part") == "used":
        return "used: " + "; ".join(f.get("problems") or [])[:200]
    if f.get("part") == "recalled":
        return f"recalled: {f.get('field')} wrong (record: {str(f.get('expected'))[:80]})"
    return f"stored: {f.get('why') or f.get('check')}"


def failure_line(f: dict) -> str:
    who = " ".join(x for x in (str(f.get("family") or ""), str(f.get("person") or "")) if x)
    what = f.get("what") or ""
    extra = f.get("item") or ""
    words = f.get("words")
    if isinstance(words, list):
        words = "; ".join(str(w) for w in words if w)
    tail = f' "{str(words)[:100]}"' if words else ""
    return f"[{f.get('check')}] {who}: {what}{(' ' + extra) if extra else ''}{tail}".strip()


def is_alert(f: dict) -> bool:
    text = f"{f.get('what') or ''} {f.get('check') or ''}"
    return f.get("check") in ("memory", "replies", "corrections") and any(p in text for p in ALERT_PROBLEMS)


def world_lines(w: dict) -> list[str]:
    last = w.get("latest")
    if not last:
        return ["Nightly world score: no run stored yet."]
    sc = last.get("scores") or {}
    judge = sc.get("judge") or {}
    code = sc.get("code") or {}
    probes = sc.get("probes") or {}
    lines = [f"Nightly world score ({last['day']}, run {last['runId']}): {'PASSED' if last['passed'] else 'FAILED'}",
             f"  judge {_num((judge.get('overall') or {}).get('pct'))} of {(judge.get('overall') or {}).get('n', 0)} judged; "
             f"code checks {_num((code.get('overall') or {}).get('pct'))}; language correct {_num((sc.get('language_correct') or {}).get('pct'))}; "
             f"recall {_num((sc.get('recall') or {}).get('pct'))}; probes run {probes.get('executed', 0)} of {probes.get('selected', 0)}"]
    langs = judge.get("languages") or {}
    if langs:
        lines.append("  per language (judge %, n): " + ", ".join(f"{k} {_num(v.get('pct'))} ({v.get('n')})" for k, v in sorted(langs.items())))
    for f in (last.get("failures") or [])[:10]:
        lines.append(f"  - {f}")
    return lines


async def email(session: AsyncSession, day: str | None = None) -> tuple[str, str, str] | None:
    data = await page(session, days=30, day=day)
    if not data["day"]:
        return None
    t = data["today"]
    over = (t.get("overall") or {}).get("value")
    alerts = [f for f in data["failures"] if is_alert(f)]
    subject = (f"{'ALERT: ' if alerts else ''}Saheli daily score {data['day']}: {_num(over)} "
               f"(reminders {_num((t.get('reminders') or {}).get('value'))}, remembered {_num((t.get('remembered') or {}).get('value'))}, "
               f"corrections {_num((t.get('corrections') or {}).get('value'))})")
    link = f"{admin_url()}/daily-score?day={data['day']}"
    lines = [f"Saheli daily score for {data['day']} (IST)", "", f"Overall: {_num(over)} (the lowest of the three checks)"]
    for i, c in enumerate(CHECKS, 1):
        row = t.get(c) or {}
        det = row.get("detail") or {}
        extra = f"{det.get('num', 0)} of {row.get('n', 0)}"
        if c == "reminders":
            held = sum((det.get("held") or {}).values())
            extra += f"; {held} held by rule; source {det.get('source') or 'unknown'}" + ("" if det.get("receipts", True) else " (Meta accepted, no WhatsApp receipts yet)")
            if det.get("error"):
                extra += f"; ERROR {det['error']}"
        if det.get("errors"):
            extra += f"; {det['errors']} not checked (judge error)"
        lines.append(f"{i}. {LABELS[c]}: {_num(row.get('value'))} ({extra})")
    rep = t.get("replies") or {}
    lines.append(f"Every reply graded by the full-memory judge: {_num(rep.get('value'))} passed ({(rep.get('detail') or {}).get('num', 0)} of {rep.get('n', 0)})")
    lines.append("")
    if data["memory"]:
        lines.append("Memory score per family (lowest across its people; stored / recalled / used):")
        for m in data["memory"]:
            lines.append(f"  {m['family']}: {_num(m['score'])} ({_num(m['stored'])} / {_num(m['recalled'])} / {_num(m['used'])})"
                         + (f"; unknown: {', '.join(m['unknowns'])[:160]}" if m["unknowns"] else ""))
        lines.append("")
    lines += world_lines(data["world"])
    learned = data.get("lessons") or {}
    if any(learned.values()):
        lines.append("")
        lines.append("Lessons (Saheli's self-learning):")
        for label, key in (("live", "live"), ("in testing", "testing"), ("rolled back", "rolled_back"), ("dropped by the safety gate", "dropped")):
            for item in learned.get(key) or []:
                lines.append(f"  - {label}: {item}")
    rc = data.get("replay") or {}
    if rc:
        lines.append(f"Replay cases: {rc.get('open', 0)} open, {rc.get('fixed', 0)} fixed (fixed ones guard every engine deploy)")
    cal = data.get("calibration") or {}
    if cal.get("batch"):
        lines.append(f"Judge calibration {cal['batch']}: {cal.get('labelled', 0)} of {cal.get('size', 0)} labelled"
                     + (f", agreement {cal['agreement'] * 100:.0f}% (bar 85%)" if cal.get("agreement") is not None else ""))
    lines.append("")
    if alerts:
        lines.append(f"NEEDS ATTENTION TODAY ({len(alerts)}): a rejected fact used, or a wrong name, medicine or allergy")
        lines += [f"  - {failure_line(f)}" for f in alerts[:40]]
        lines.append("")
    rest = [f for f in data["failures"] if f not in alerts]
    lines.append(f"Every failure ({len(data['failures'])}):" if data["failures"] else "No failures.")
    lines += [f"  - {failure_line(f)}" for f in rest[:150]]
    if len(rest) > 150:
        lines.append(f"  … and {len(rest) - 150} more on the page")
    lines += ["", f"Open: {link}"]
    text = "\n".join(lines)
    body = "".join(f"<div>{html.escape(ln) or '&nbsp;'}</div>" for ln in lines[:-1])
    page_html = (f"<div style=\"font-family:system-ui,sans-serif;font-size:14px;line-height:1.45\">{body}"
                 f"<p><a href=\"{html.escape(link)}\">Open the Daily score page</a></p></div>")
    return subject[:160], text, page_html


async def send(sessions, host, day: str | None = None, *, force: bool = False) -> dict:
    """The 08:00 email, once a day (a scheduler retry sends nothing twice)."""
    from app.learn.models import LearnReport
    from app.tasks.credits import OPS

    async with sessions() as s:
        built = await email(s, day)
        if built is None:
            return {"sent": False, "why": "no score stored yet"}
        subject, text, page_html = built
        sent_day = subject.split("score ")[1].split(":")[0] if "score " in subject else (day or "")
        if not force:
            done = list((await s.execute(select(LearnReport).where(LearnReport.kind == "morning_report").order_by(LearnReport.at.desc()).limit(5))).scalars())
            if any((r.data or {}).get("day") == sent_day for r in done):
                return {"sent": False, "why": f"already sent for {sent_day}"}
    res = await host.call("team_report", {"subject": subject, "text": text, "html": page_html}, family_id=OPS, subject_id=OPS, actor_id=OPS)
    ok = isinstance(res, dict) and not res.get("error") and (res.get("result", res) or {}).get("sent", True) is not False
    if ok:
        async with sessions() as s:
            s.add(LearnReport(kind="morning_report", at=clock.now(), data={"day": sent_day, "subject": subject}))
            await s.commit()
    return {"sent": ok, "day": sent_day, "subject": subject, "result": res if not ok else None}


async def nightly_failed(host, row: EvalScore) -> dict:
    """The nightly world score broke a bar: email the team now (the deploy gate holds engine deploys until it is fixed)."""
    from app.tasks.credits import OPS

    sc = row.scores or {}
    lines = [f"The nightly world score FAILED on {row.day.isoformat()} (run {row.run_id}, commit {row.commit[:12]}).", "",
             "Broken bars:"] + [f"  - {f}" for f in row.failures[:40]] + ["", *world_lines({"latest": {"day": row.day.isoformat(), "runId": row.run_id,
                                                                                                    "passed": row.passed, "commit": row.commit, "failures": [],
                                                                                                    "scores": sc}}),
                                                          "", "Engine deploys are held until a commit with [fixes-nightly] or the founder's override.",
                                                          f"Open: {admin_url()}/daily-score",
                                                          f"Run: https://github.com/kritarth1107/kawach-ai-engine/actions/runs/{row.run_id}" if row.run_id.isdigit() else ""]
    text = "\n".join(ln for ln in lines if ln is not None)
    return await host.call("team_report", {"subject": f"Nightly world score FAILED {row.day.isoformat()}: {len(row.failures)} bars broken"[:160],
                                           "text": text, "html": "".join(f"<div>{html.escape(ln) or '&nbsp;'}</div>" for ln in text.splitlines())},
                           family_id=OPS, subject_id=OPS, actor_id=OPS)
