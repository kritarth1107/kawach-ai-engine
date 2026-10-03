"""Lab families as JSON files (written by the Family Designer bot), validated and turned into simulator specs.

One file per family in SAHELI_LAB_DIR/families/<key>.json:

{
  "key": "kumar-01", "city": "Lucknow", "setup": "detailed" | "sparse",
  "meta": {"group": "north" | "south_east" | "english", "region": "...", "languages": ["hinglish"],
           "conditions": ["diabetes"], "household": "alone", "personality": ["proud"], "tags": ["dementia"]},
  "people": [{"id": "ku-amma", "name": "...", "role": "elder" | "primary caregiver" | "co-caregiver" | "helper" | "self",
              "relation": "mother", "persona": "...", "phone": "+91 90000 xxxxx", "recipient": true,
              "adherence": 0.8, "reports": 0.6, "answers": 0.7, "chat": [1, 3],
              "style": {"script": "latin" | "devanagari" | …, "typos": 0.2, "emoji": 0.3, "reply_delay_min": [2, 45],
                        "patience": 0.5, "sample_messages": ["…", "…"]}}],
  "truth": "...", "meds": {"ku-amma": [["Metformin", "500 mg", ["08:00", "20:00"]]]},
  "setup_text": "...",
  "events": [{"day": 3, "time": "21:00", "who": "ku-amma", "what": "...", "expect": {"tools": ["log_dose"], "alert": "none"}}]
}
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

LAB_DIR = Path(os.getenv("SAHELI_LAB_DIR", "/home/m4dm4x/OpenBot/Shared/saheli-lab"))
GROUPS = {"north", "south_east", "english"}
ROLES = {"elder", "primary caregiver", "co-caregiver", "helper", "self", "family"}
TIME = re.compile(r"^\d{2}:\d{2}$")
ALERTS = {"none", "red_flag", "no_answer", "safety", "approval"}
KNOWN_TOOLS = None  # filled lazily from app.brain.tools


def _tools() -> set[str]:
    global KNOWN_TOOLS
    if KNOWN_TOOLS is None:
        from app.brain import tools

        KNOWN_TOOLS = {s.name for s in tools.specs()}
    return KNOWN_TOOLS


def validate(f: dict) -> list[str]:
    errs: list[str] = []
    for k in ("key", "city", "setup", "meta", "people", "truth", "meds", "setup_text", "events"):
        if k not in f:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if not re.fullmatch(r"[a-z0-9-]{3,40}", f["key"]):
        errs.append("key must be lowercase letters, digits, dashes")
    if f["setup"] not in ("detailed", "sparse"):
        errs.append("setup must be detailed or sparse")
    if f["meta"].get("group") not in GROUPS:
        errs.append(f"meta.group must be one of {sorted(GROUPS)}")
    ids = set()
    recipients = 0
    for p in f["people"]:
        for k in ("id", "name", "role", "relation", "persona", "phone"):
            if not p.get(k):
                errs.append(f"person missing {k}: {p.get('id')}")
        if p.get("id") in ids:
            errs.append(f"duplicate person id {p.get('id')}")
        ids.add(p.get("id"))
        if p.get("role") not in ROLES:
            errs.append(f"{p.get('id')}: role {p.get('role')} not in {sorted(ROLES)}")
        if not re.fullmatch(r"\+91 90000 \d{5}", str(p.get("phone", ""))):
            errs.append(f"{p.get('id')}: phone must be a fake +91 90000 xxxxx number")
        for k in ("adherence", "reports", "answers"):
            v = p.get(k, 0)
            if not (isinstance(v, (int, float)) and 0 <= v <= 1):
                errs.append(f"{p.get('id')}: {k} must be 0..1")
        c = p.get("chat", [1, 3])
        if not (isinstance(c, list) and len(c) == 2 and 0 <= c[0] <= c[1] <= 12):
            errs.append(f"{p.get('id')}: chat must be [min, max] within 0..12")
        recipients += 1 if p.get("recipient") or p.get("role") == "self" else 0
    if not recipients:
        errs.append("needs at least one care recipient (recipient: true) or a self-care person")
    for who, meds in f["meds"].items():
        if who not in ids:
            errs.append(f"meds for unknown person {who}")
        for m in meds:
            if not (isinstance(m, list) and len(m) == 3 and all(TIME.match(t) for t in m[2])):
                errs.append(f"bad medicine {m} (want [name, dose, ['HH:MM']])")
    tools_ok = _tools()
    for e in f["events"]:
        if not (isinstance(e.get("day"), int) and 1 <= e["day"] <= 30 and TIME.match(str(e.get("time", ""))) and e.get("who") in ids and e.get("what")):
            errs.append(f"bad event {str(e)[:80]}")
            continue
        ex = e.get("expect") or {}
        for t in (ex.get("tools") or []) + (ex.get("tools_none") or []):
            if t not in tools_ok:
                errs.append(f"event day {e['day']}: unknown tool {t}")
        if ex.get("alert") and ex["alert"] not in ALERTS:
            errs.append(f"event day {e['day']}: alert must be one of {sorted(ALERTS)}")
    return errs


def to_spec(f: dict) -> dict:
    """The simulator's family shape (eval/month_families.py)."""
    people = []
    for p in f["people"]:
        style = p.get("style") or {}
        persona = p["persona"]
        if style:
            persona += (f" Writes in {style.get('script', 'their usual script')}; typos {style.get('typos', 0)}; emoji {style.get('emoji', 0)}; "
                        f"patience {style.get('patience', 0.5)}.")
            if style.get("sample_messages"):
                persona += " Sample messages: " + " | ".join(style["sample_messages"][:4])
        people.append({
            "id": p["id"], "name": p["name"], "role": p["role"], "relation": p["relation"], "persona": persona, "phone": p["phone"],
            "recipient": bool(p.get("recipient")), "adherence": p.get("adherence", 0.0), "reports": p.get("reports", 0.0),
            "answers": p.get("answers", 0.7), "chat": tuple(p.get("chat", [1, 3])), "style": style,
        })
    return {
        "key": f["key"], "city": f["city"], "setup": f["setup"], "meta": f["meta"], "people": people, "truth": f["truth"],
        "meds": f["meds"], "setup_text": f["setup_text"],
        "events": [(e["day"], e["time"], e["who"], e["what"], e.get("expect") or {}) for e in f["events"]],
    }


def load(keys: list[str] | None = None, *, directory: Path | None = None) -> tuple[list[dict], dict[str, list[str]]]:
    """(valid specs, {file: errors}) from the families folder."""
    d = directory or (LAB_DIR / "families")
    specs, bad = [], {}
    for path in sorted(d.glob("*.json")):
        if not keys and (path.stem.startswith("example-") or path.stem.startswith("_")):
            continue  # examples and drafts only when named
        try:
            f = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            bad[path.name] = [f"not JSON: {exc}"]
            continue
        if keys and f.get("key") not in keys:
            continue
        errs = validate(f)
        if not errs and f.get("key") != path.stem:
            errs = [f"file name {path.stem!r} differs from key {f.get('key')!r}"]
        if errs:
            bad[path.name] = errs
        elif not keys or f["key"] in keys:
            specs.append(to_spec(f))
    return specs, bad


def coverage(specs: list[dict]) -> dict:
    c: dict[str, Counter] = {k: Counter() for k in ("group", "region", "languages", "conditions", "household", "personality", "tags", "setup")}
    for s in specs:
        m = s["meta"]
        c["group"][m.get("group")] += 1
        c["region"][m.get("region")] += 1
        c["household"][m.get("household")] += 1
        c["setup"][s["setup"]] += 1
        for k in ("languages", "conditions", "personality", "tags"):
            for v in m.get(k) or []:
                c[k][v] += 1
    return {"families": len(specs), **{k: dict(v.most_common()) for k, v in c.items()}}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    specs, bad = load()
    print(json.dumps({"valid": len(specs), "invalid": bad, "coverage": coverage(specs)}, indent=2, ensure_ascii=False))
