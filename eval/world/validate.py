"""World checks: schema, quotas, fake phones only, unique ids, scripts, and no copy of the founder's family.

    PYTHONPATH=. python -m eval.world.validate      # exit 1 on any problem; prints a coverage report
"""

from __future__ import annotations

import re
import sys
from collections import Counter

from app.care.domains import DOMAINS
from eval.world import grid as G
from eval.world.make import load

PHONE = re.compile(r"^(\+9190000\d{5}|\+99\d{10})$")
HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def script_share(text: str, script: str) -> float:
    """Share of letters in the text that belong to the script (digits, punctuation, spaces and emoji are ignored)."""
    ranges = G.SCRIPT_RANGES[script]
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 1.0
    ok = sum(1 for ch in letters if any(a <= ord(ch) <= b for a, b in ranges))
    return ok / len(letters)


def brain_tools() -> set[str]:
    from app.brain import tools

    return set(tools._TOOLS)


def check(families: list[dict]) -> tuple[list[str], list[str]]:
    errs: list[str] = []
    report: list[str] = []
    tools = brain_tools()
    if len(families) != G.PROFILE_FAMILIES + G.STORY_FAMILIES:
        errs.append(f"{len(families)} families, want {G.PROFILE_FAMILIES + G.STORY_FAMILIES}")
    tiers = Counter(f["tier"] for f in families)
    if tiers["profile"] != G.PROFILE_FAMILIES or tiers["story"] != G.STORY_FAMILIES:
        errs.append(f"tiers {dict(tiers)}")
    # ids and phones
    seen: Counter = Counter()
    phones: Counter = Counter()
    for f in families:
        seen[f["id"]] += 1
        seen[f["familyId"]] += 1
        for p in f["people"]:
            seen[p["id"]] += 1
            if p["phone"]:
                phones[p["phone"]] += 1
                if not PHONE.match(p["phone"]):
                    errs.append(f"{f['id']}: phone {p['phone']} is not a fake +91 90000 / +99 number")
            words = {w.lower() for w in re.findall(r"[A-Za-z]+", p["name"])}
            if words & G.BLOCKED_NAMES:
                errs.append(f"{f['id']}: a blocked (real) name in {p['name']!r}")
            if not p["usesWhatsapp"] and p["phone"]:
                errs.append(f"{f['id']}: {p['key']} has no WhatsApp but a phone")
    errs += [f"duplicate id {k}" for k, n in seen.items() if n > 1]
    errs += [f"duplicate phone {k}" for k, n in phones.items() if n > 1]
    # languages and households
    langs = Counter(f["language"] for f in families)
    for lang in G.LANGS:
        if langs[lang] < G.MIN_PER_LANGUAGE:
            errs.append(f"language {lang}: {langs[lang]} families (< {G.MIN_PER_LANGUAGE})")
    story_langs = Counter(f["language"] for f in families if f["tier"] == "story")
    errs += [f"story families in {lang}: {story_langs[lang]} (< 2)" for lang in G.LANGS if story_langs[lang] < 2]
    homes = Counter(f["household"] for f in families)
    for h, share in G.HOUSEHOLDS.items():
        want = share * len(families)
        if abs(homes[h] - want) > 2:
            errs.append(f"household {h}: {homes[h]} (want ~{want:.0f})")
    story_homes = {f["household"] for f in families if f["tier"] == "story"}
    errs += [f"no story family of type {h}" for h in G.HOUSEHOLDS if h not in story_homes]
    mixed = sum(1 for f in families if f["mixed"]) / len(families)
    if abs(mixed - G.MIXED_SHARE) > 0.03:
        errs.append(f"mixed-language families {mixed:.0%} (want {G.MIXED_SHARE:.0%})")
    # health
    subjects = [(f, sid, t) for f in families for sid, t in f["truth"].items()]
    bands = Counter()
    for f, sid, t in subjects:
        n = len(t["medicines"])
        bands["0" if n == 0 else "1-3" if n <= 3 else "4-8" if n <= 8 else "9+"] += 1
        conds = {c["key"] for c in t["conditions"]}
        for m in t["medicines"]:
            if "insulin" in m["name"].lower() and not any(c.startswith("t2_diabetes") for c in conds):
                errs.append(f"{f['id']}: insulin without diabetes")
            if m["prn"] and m["times"]:
                errs.append(f"{f['id']}: an as-needed medicine with times")
            for tm in m["times"]:
                if not HHMM.match(tm):
                    errs.append(f"{f['id']}: bad time {tm}")
        if t["allergies"] != ["none"] and "none" in t["allergies"]:
            errs.append(f"{f['id']}: 'none' next to real allergies")
    # engine and backend references
    for f in families:
        ids = {p["id"] for p in f["people"]}
        talkers = {p["id"] for p in f["people"] if p["usesWhatsapp"]}
        for fact in f["engine"]["facts"]:
            if fact["subject"] not in ids:
                errs.append(f"{f['id']}: fact about someone not in the family")
            if fact["domain"] not in DOMAINS:
                errs.append(f"{f['id']}: unknown domain {fact['domain']}")
        for s in f["backend"]["schedules"]:
            if s["recipientUserId"] not in ids or not HHMM.match(s["time"]):
                errs.append(f"{f['id']}: bad schedule {s['title']} {s['time']}")
        for r in f["backend"]["reminders"]:
            if r["recipientUserId"] not in ids:
                errs.append(f"{f['id']}: reminder for someone not in the family")
        if not any(p["role"] == "primary caregiver" for p in f["people"]):
            errs.append(f"{f['id']}: no primary caregiver")
        if not f["truth"] and "empty_record" not in f["shapes"]:
            errs.append(f"{f['id']}: nobody is cared for")
        # probes
        if f["safety"] and not any(q["topic"] == "crisis" for q in f["probes"]):
            errs.append(f"{f['id']}: safety family without a crisis probe")
        if not f["safety"] and any(q["topic"] == "crisis" for q in f["probes"]):
            errs.append(f"{f['id']}: crisis probe outside a safety family")
        lo, hi = G.PROBES_PER_FAMILY
        if not lo <= len(f["probes"]) - (1 if f["safety"] else 0) <= hi:
            errs.append(f"{f['id']}: {len(f['probes'])} probes")
        for q in f["probes"]:
            if q["who"] not in talkers:
                errs.append(f"{q['id']}: sent by someone without WhatsApp")
            for t in (*q["expect"]["tools_any"], *q["expect"]["tools_none"]):
                if t not in tools:
                    errs.append(f"{q['id']}: expects unknown tool {t}")
            if q.get("text"):
                lang = G.LANGS[q["language"]]
                want = "latin" if lang["roman"] else lang["script"]
                if script_share(q["text"], want) < 0.6:
                    errs.append(f"{q['id']}: text not in {want} script")
                if re.search(r"\+?\d[\d ]{9,}", q["text"]) and not re.search(r"\+9190000|\+99", q["text"]):
                    errs.append(f"{q['id']}: a phone-like number that is not fake")
    # probe topic × language coverage
    cover = Counter((f["language"], q["topic"]) for f in families for q in f["probes"])
    for lang in G.LANGS:
        for topic in G.PROBE_TOPICS:
            if cover[(lang, topic)] == 0:
                errs.append(f"probe topic {topic} never appears in {lang}")
    shapes = Counter(s for f in families for s in f["shapes"])
    errs += [f"shape {s} never appears" for s in G.SHAPES if shapes[s] == 0]
    sleep = Counter(p["sleep"]["kind"] for f in families for p in f["people"])
    uses = Counter("none" if not f["orders"]["uses"] else "some" for f in families)
    rides = [f for f in families if "rides" in f["orders"]["uses"]]
    coords = sum(1 for f in rides if f["places"][0]["lat"] is not None) / max(1, len(rides))
    if coords < 0.6:
        errs.append(f"ride families with coordinates: {coords:.0%} (< 60 %)")
    total_p = sum(sleep.values())
    report += [
        f"families {len(families)} (profile {tiers['profile']}, story {tiers['story']}); people {sum(len(f['people']) for f in families)}; "
        f"probes {sum(len(f['probes']) for f in families)}",
        "languages: " + ", ".join(f"{k} {langs[k]}" for k in G.LANGS),
        "households: " + ", ".join(f"{k} {homes[k]}" for k in G.HOUSEHOLDS),
        f"mixed-language families {mixed:.0%}; voice people {sum(p['voice'] for f in families for p in f['people']) / total_p:.0%}",
        "medicine bands: " + ", ".join(f"{k} {v / max(1, len(subjects)):.0%}" for k, v in sorted(bands.items())),
        "sleep: " + ", ".join(f"{k} {v / total_p:.0%}" for k, v in sleep.items()),
        f"order use none {uses['none'] / len(families):.0%}; ride families {len(rides)} ({coords:.0%} with coordinates)",
        f"safety families {sum(f['safety'] for f in families)}",
        "shapes: " + ", ".join(f"{k} {shapes[k]}" for k in G.SHAPES),
        f"probe texts written: {sum(1 for f in families for q in f['probes'] if q.get('text'))}",
    ]
    return errs, report


def main() -> int:
    errs, report = check(load())
    print("\n".join(report))
    for e in errs[:200]:
        print("ERROR", e)
    print(f"{len(errs)} problems" if errs else "world OK")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
