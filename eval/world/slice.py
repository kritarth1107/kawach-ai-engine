"""Stratified slices of the world: ci (120), ui (40), nudge (200), full (1,000).

ci covers every language, household type, data shape, order use and the safety families' crisis probes at least once;
ui takes the 40 families whose pages look most different; nudge takes 200 families with the most proactive traffic.

    PYTHONPATH=. python -m eval.world.slice ci        # prints the family ids
"""

from __future__ import annotations

import sys

from eval.world import grid as G
from eval.world.make import load

SIZES = {"ci": 120, "ui": 40, "nudge": 200}


def features(f: dict) -> set[str]:
    out = {f"lang:{f['language']}", f"home:{f['household']}", f"tier:{f['tier']}", f"city_tier:{f['city']['tier']}"}
    out |= {f"shape:{s}" for s in f["shapes"]}
    out |= {f"use:{u}" for u in f["orders"]["uses"] or ["none"]}
    out |= {f"sit:{s}" for s in f["orders"]["situations"]}
    out |= {f"style:{p['style']}" for p in f["people"]}
    out |= {f"sleep:{p['sleep']['kind']}" for p in f["people"]}
    if f["safety"]:
        out.add(f"safety:{f['language']}")
    if f["mixed"]:
        out.add("mixed")
    if f.get("nri"):
        out.add(f"nri:{f['nri']}")
    for t in f["truth"].values():
        n = len(t["medicines"])
        out.add("meds:" + ("0" if n == 0 else "1-3" if n <= 3 else "4-8" if n <= 8 else "9+"))
        out |= {f"cond:{c['key']}" for c in t["conditions"]}
        out |= {f"reading:{r}" for r in t["readings"]}
    return out


def cover(families: list[dict], n: int, want: set[str] | None = None) -> list[str]:
    """Greedy set cover over the features (deterministic: ties go to the lower id), then filled round-robin by language."""
    feats = {f["id"]: features(f) for f in families}
    todo = set().union(*feats.values()) if want is None else set(want)
    picked: list[str] = []
    while todo and len(picked) < n:
        best = max(sorted(feats), key=lambda i: (len(feats[i] & todo) if i not in picked else -1))
        if best in picked or not feats[best] & todo:
            break
        picked.append(best)
        todo -= feats[best]
    by_lang: dict[str, list[str]] = {}
    for f in families:
        if f["id"] not in picked:
            by_lang.setdefault(f["language"], []).append(f["id"])
    langs = sorted(by_lang)
    i = 0
    while len(picked) < n and any(by_lang.values()):
        lst = by_lang[langs[i % len(langs)]]
        if lst:
            picked.append(lst.pop(0))
        i += 1
    return sorted(picked)


def slice_ids(name: str, families: list[dict] | None = None) -> list[str]:
    families = families if families is not None else load()
    if name == "full":
        return sorted(f["id"] for f in families)
    if name == "ui":
        want = {x for f in families for x in features(f) if x.split(":")[0] in ("home", "shape", "use", "meds", "tier")}
        return cover(families, SIZES["ui"], want)
    if name == "nudge":
        busy = sorted(families, key=lambda f: (-(len(f["engine"]["loops"]) + sum(len(t["activities"]) + len(t["medicines"]) for t in f["truth"].values())), f["id"]))
        keep = {f["id"] for f in busy[: SIZES["nudge"] // 2]}
        rest = [f for f in families if f["id"] not in keep]
        return sorted(keep | set(cover(rest, SIZES["nudge"] - len(keep))))
    if name == "ci":
        ids = cover(families, SIZES["ci"])
        missing = {f"lang:{lang}" for lang in G.LANGS} - set().union(*(features(f) for f in families if f["id"] in ids))
        assert not missing, missing
        return ids
    raise SystemExit(f"unknown slice {name}: ci | ui | nudge | full")


if __name__ == "__main__":
    print(" ".join(slice_ids(sys.argv[1] if len(sys.argv) > 1 else "ci")))
