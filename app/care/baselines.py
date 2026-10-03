"""Each person's own normal, learned from their last four weeks, and whether their reminders work.

- readings: their usual BP, sugar, pulse, weight (median and spread), so "high" means high *for them*
- rhythm: how many messages a day they usually send, and the hours they are usually awake and writing
- doses: when each dose is really taken (vs its time), and whether the reminder leads to a dose
- suggestions: a reminder time that would fit how they actually live ("night Metformin is taken at ~21:45")

Plain statistics, recomputed every night (the dream job) and stored per person. The brain sees a short
"USUAL FOR …" line; patterns compare against it; reminder-time suggestions go to a caregiver, who decides.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import median

from sqlalchemy import DateTime, String, select
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.care import store
from app.care.domains import slug
from app.care.models import Turn
from app.core import clock
from app.db.session import Base

WINDOW_DAYS = 28
MIN_POINTS = 5


class Baseline(Base):
    __tablename__ = "person_baselines"

    family_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    data: Mapped[dict] = mapped_column(JSONB, default=dict)


def _spread(xs: list[float]) -> dict:
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, max(0, round(p * (len(s) - 1))))]  # noqa: E731
    return {"median": round(median(s), 1), "low": round(q(0.1), 1), "high": round(q(0.9), 1), "n": len(s)}


def _hhmm(m: float) -> str:
    m = int(round(m)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def _minutes(t: str) -> int | None:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(t).strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m and int(m.group(1)) < 24 else None


def vitals_baseline(vitals: list) -> dict:
    by: dict[str, list[list[float]]] = defaultdict(list)
    for e in vitals:
        p = e.payload or {}
        kind = str(p.get("kind") or (e.summary or "").split(" ")[0]).lower()
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(p.get("value") or e.summary))]
        if nums:
            by[kind].append(nums)
    out = {}
    for kind, rows in by.items():
        if len(rows) < MIN_POINTS:
            continue
        if kind == "bp":
            rows = [r for r in rows if len(r) >= 2]
            if len(rows) >= MIN_POINTS:
                out["bp"] = {"systolic": _spread([r[0] for r in rows]), "diastolic": _spread([r[1] for r in rows])}
        else:
            out[kind] = _spread([r[0] for r in rows])
    return out


def rhythm_baseline(turn_times: list[datetime], *, now: datetime) -> dict:
    local = [clock.ist(t) for t in turn_times]
    if len(local) < 10:
        return {}
    days = max(1, min(WINDOW_DAYS, (clock.ist(now).date() - min(local).date()).days))
    per_day: dict = defaultdict(int)
    for t in local:
        per_day[t.date()] += 1
    counts = [per_day.get(clock.ist(now).date() - timedelta(days=b), 0) for b in range(1, days + 1)]
    hours = sorted(t.hour + t.minute / 60 for t in local)
    q = lambda p: hours[min(len(hours) - 1, round(p * (len(hours) - 1)))]  # noqa: E731
    return {"messages_per_day": round(median(counts), 1), "active_from": _hhmm(q(0.05) * 60), "active_to": _hhmm(q(0.95) * 60)}


def dose_baseline(meds: list[dict], doses: list, reminders: list) -> dict:
    """{"<med>@HH:MM": {"taken_at": median HH:MM, "delay_min", "spread_min", "n", "after_reminder": share taken within 60 min}}"""
    out = {}
    for med in meds:
        first = (slug(med["name"]).split("_") or [""])[0]
        times = [t for t in (_minutes(x) for x in med.get("times") or []) if t is not None]
        if not first or not times:
            continue
        taken = [e for e in doses if e.kind == "dose_taken" and first in slug((e.payload or {}).get("medicine") or e.summary)]
        sent = [e for e in reminders if first in slug(e.summary or "")]
        for slot in times:
            delays = []
            for e in taken:
                t = clock.ist(e.at)
                m = t.hour * 60 + t.minute
                if min(times, key=lambda x: abs(x - m)) == slot and abs(m - slot) <= 240:
                    delays.append(m - slot)
            if len(delays) < MIN_POINTS:
                continue
            row = {"taken_at": _hhmm(slot + median(delays)), "delay_min": round(median(delays)), "n": len(delays)}
            s = sorted(delays)
            row["spread_min"] = round(s[int(0.75 * (len(s) - 1))] - s[int(0.25 * (len(s) - 1))])
            slot_sent = [r for r in sent if abs((clock.ist(r.at).hour * 60 + clock.ist(r.at).minute) - slot) <= 30]
            if slot_sent:
                hit = 0
                for r in slot_sent:
                    if any(0 <= (e.at - r.at).total_seconds() <= 3600 for e in taken):
                        hit += 1
                row["after_reminder"] = round(hit / len(slot_sent), 2)
                row["reminders"] = len(slot_sent)
            out[f"{med['name']}@{_hhmm(slot)}"] = row
    return out


EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF🙏]")


def style_baseline(own_texts: list[str], scored: list) -> dict:
    """How this person writes, and which reply lengths worked with them (from their reactions)."""
    if len(own_texts) < 5:
        return {}
    words = [len(t.split()) for t in own_texts if t.strip()]
    emoji = sum(1 for t in own_texts if EMOJI.search(t)) / len(own_texts)
    out = {"their_words": round(median(words)), "emoji_rate": round(emoji, 2)}
    good = [n for n, sc, _ in scored if sc is not None and sc >= 0.3]
    bad = [n for n, sc, _ in scored if sc is not None and sc <= -0.2]
    if len(good) >= 5:
        out["good_reply_chars"] = round(median(good))
    if len(bad) >= 5:
        out["bad_reply_chars"] = round(median(bad))
    target = out.get("good_reply_chars")
    length = ("short" if out["their_words"] <= 8 else "medium" if out["their_words"] <= 25 else "longer")
    if target:
        length = f"about {max(8, round(target / 6))} words (what they responded well to)"
    out["summary"] = (f"replies {length}; they write ~{out['their_words']} words; "
                      + ("they use emoji, a little is fine" if emoji >= 0.3 else "they rarely use emoji, keep it plain"))
    return out


def reminder_suggestions(dose: dict) -> list[dict]:
    """A reminder time that fits how they live: taken consistently ≥45 min late, or the reminder rarely works."""
    out = []
    for key, r in dose.items():
        name, _, slot = key.rpartition("@")
        if r["n"] >= 6 and abs(r["delay_min"]) >= 45 and r.get("spread_min", 999) <= 40:
            new = _hhmm(_minutes(slot) + r["delay_min"] - 5)
            out.append({"key": key, "medicine": name, "now": slot, "suggest": new,
                        "why": f"{name} at {slot} is usually taken around {r['taken_at']} ({r['n']} times)"})
        elif r.get("reminders", 0) >= 6 and r.get("after_reminder", 1) < 0.4:
            out.append({"key": key, "medicine": name, "now": slot, "suggest": None,
                        "why": f"after the {slot} reminder for {name}, the dose follows within an hour only {int(r['after_reminder'] * 100)}% of the time"})
    return out


async def compute(session: AsyncSession, family_id: str, subject_id: str) -> dict:
    now = clock.now()
    since = now - timedelta(days=WINDOW_DAYS)
    evs = await store.events(session, family_id, subject_id, since=since, limit=20000)
    meds = [{"name": str((f.value or {}).get("name") or f.key.split(":", 1)[1]), "times": (f.value or {}).get("times") or []}
            for f in await store.facts(session, family_id, subject_id, domains=["medicine"], statuses=("active",))]
    turn_times = list((await session.execute(
        select(Turn.at).where(Turn.family_id == family_id, Turn.thread_id == subject_id, Turn.role == "user", Turn.at >= since)
    )).scalars())
    dose = dose_baseline(meds, [e for e in evs if e.kind.startswith("dose_")], [e for e in evs if e.kind == "reminder_sent"])
    texts = list((await session.execute(
        select(Turn.text).where(Turn.family_id == family_id, Turn.thread_id == subject_id, Turn.role == "user", Turn.at >= since)
    )).scalars())
    from app.learn.models import ReplyLog

    scored = list((await session.execute(
        select(ReplyLog.text_len, ReplyLog.score, ReplyLog.text).where(ReplyLog.family_id == family_id, ReplyLog.thread_id == subject_id,
                                                                       ReplyLog.score.is_not(None), ReplyLog.at >= since)
    )).all())
    return {
        "style": style_baseline(texts, scored),
        "vitals": vitals_baseline([e for e in evs if e.kind == "vital"]),
        "rhythm": rhythm_baseline(turn_times, now=now),
        "doses": dose,
        "suggestions": reminder_suggestions(dose),
        "window_days": WINDOW_DAYS,
    }


async def save(session: AsyncSession, family_id: str, subject_id: str) -> dict:
    data = await compute(session, family_id, subject_id)
    stmt = insert(Baseline).values(family_id=family_id, subject_id=subject_id, computed_at=clock.now(), data=data)
    await session.execute(stmt.on_conflict_do_update(index_elements=[Baseline.family_id, Baseline.subject_id],
                                                     set_={"computed_at": stmt.excluded.computed_at, "data": stmt.excluded.data}))
    return data


async def get(session: AsyncSession, family_id: str, subject_id: str) -> dict:
    row = await session.get(Baseline, (family_id, subject_id))
    return dict(row.data) if row else {}


def usual_line(name: str, b: dict) -> str:
    """One line for the brain: what is normal for this person."""
    if not b:
        return ""
    bits = []
    bp = (b.get("vitals") or {}).get("bp")
    if bp:
        bits.append(f"BP usually ~{bp['systolic']['median']:.0f}/{bp['diastolic']['median']:.0f}")
    for k, label in (("sugar", "sugar"), ("pulse", "pulse"), ("weight", "weight")):
        v = (b.get("vitals") or {}).get(k)
        if v:
            bits.append(f"{label} usually ~{v['median']:.0f}")
    st = b.get("style") or {}
    if st.get("summary"):
        bits.append(f"how they like replies: {st['summary']}")
    r = b.get("rhythm") or {}
    if r:
        bits.append(f"writes ~{r['messages_per_day']:.0f} messages a day, mostly {r['active_from']}–{r['active_to']}")
    for key, d in list((b.get("doses") or {}).items())[:4]:
        if abs(d["delay_min"]) >= 20:
            bits.append(f"takes {key.replace('@', ' (')}) around {d['taken_at']}")
    return f"USUAL FOR {name.upper()} (last 4 weeks): " + "; ".join(bits) + "." if bits else ""
