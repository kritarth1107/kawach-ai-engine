"""AI writer for the world: persona lines and probe messages in each person's own language, script and style.

One model call per family (paid runs only in the sunny-ship test project, never kavach-care). Every text is checked
(script, length, fake phone numbers only, no blocked real names) and retried once with the problems listed. Results
are cached in eval/world/data/text/<language>.jsonl by a hash of what the writer saw, so a rebuild needs no AI and a
changed family is rewritten.

    DEBUG=false GCP_PROJECT_ID=sunny-ship-508214-q1 GOOGLE_CLOUD_PROJECT=sunny-ship-508214-q1 MEMORY_EMBEDDINGS=off \
    MODEL_ROUTES='{"sim": ["gemini:gemini-3.8-flash"]}' WORLD_MAX_INR=3000 \
    PYTHONPATH=. python -m eval.world.write [ci | ui | nudge | full | w0001 …] [--parallel 16]
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time

from eval.world import grid as G
from eval.world.make import DATA, load
from eval.world.slice import slice_ids
from eval.world.validate import probe_script_ok

TEXT_DIR = DATA / "text"
PROMPT_VERSION = 1
SCRIPT_NAMES = {"devanagari": "Devanagari", "bengali": "Bengali-Assamese", "gurmukhi": "Gurmukhi", "gujarati": "Gujarati", "odia": "Odia",
                "tamil": "Tamil", "telugu": "Telugu", "kannada": "Kannada", "malayalam": "Malayalam", "arabic": "Urdu (Nastaliq/Arabic)",
                "latin": "Latin (Roman letters)"}
STYLE = {
    "chatty": "warm and talkative, two or three sentences", "terse": "very short, a few words, no greeting",
    "never_replies": "short and reluctant", "replies_late": "short, sometimes apologises for the late reply",
    "emoji_heavy": "uses several emoji", "typos": "makes typing mistakes and skips punctuation", "all_caps": "WRITES IN CAPITAL LETTERS where the script has them",
    "voice_only": "speaks (the text is a voice-note transcript: spoken words, fillers, no emoji)", "sends_photos": "short; mentions the photo they send",
}

SYSTEM = """You write test messages for Kavach, a WhatsApp care companion (Saheli) for Indian families. The families are
invented. Write exactly how each real person would type (or speak, for voice notes) on WhatsApp.

Rules:
- Each probe message is written by the person named, in THEIR language and dialect, in the script given. A dialect
  (Marwari, Bhojpuri, Maithili, Chhattisgarhi, Haryanvi, Awadhi, Garhwali, Kumaoni, Dogri, Tulu, Sylheti, Bundeli,
  Kathiawadi) must read like that dialect, not like standard Hindi/Kannada/Bengali/Gujarati.
- Roman-script languages (Hinglish, Tanglish) use Roman letters only. English (Indian) is natural Indian English.
- Follow the person's chat style. Keep medicine names, doses, numbers and links exactly as in the brief (a medicine
  name may be written in the person's script the way they would spell it).
- A voice note is a transcript of speech: no emoji, spoken words, maybe a filler.
- Do not add facts that are not in the brief. Never write a phone number unless the brief gives one. Never use the
  names of real public people.
- A crisis probe is sincere and quiet, not dramatic; a joke idiom is clearly a joke to a native speaker.
- Persona: one line in English describing how the person writes and behaves (age, temperament, habits on chat).

Return JSON only: {"personas": {"<person key>": "<one line>"}, "probes": {"<probe id>": "<message text>"}}"""


def writer_input(fam: dict) -> dict:
    people = {p["id"]: p for p in fam["people"]}
    pp = []
    for p in fam["people"]:
        lang = G.LANGS[p["language"]]
        pp.append({"key": p["key"], "first_name": p["first"], "age": p["age"], "gender": p["gender"], "relation": p["relation"], "role": p["role"],
                   "language": lang["name"], "script": SCRIPT_NAMES["latin" if lang["roman"] else lang["script"]], "style": STYLE[p["style"]],
                   "voice_notes": p["voice"], "lives_in": p["city"], "on_whatsapp": p["usesWhatsapp"]})
    probes = [{"id": q["id"], "by": people[q["who"]]["key"], "topic": q["topic"], "brief": q["brief"], "voice_note": q["voice"]} for q in fam["probes"]]
    # sorted by person id: the same order whether the family was just built or read back from its JSON file
    cared = [f"{people[s]['first']}: " + ", ".join(c["label"] for c in t["conditions"]) for s, t in sorted(fam["truth"].items())]
    return {"household": fam["household"].replace("_", " "), "city": fam["city"]["name"], "cared_for": cared, "people": pp, "probes": probes}


def input_hash(inp: dict) -> str:
    return hashlib.sha256(json.dumps({"v": PROMPT_VERSION, "in": inp}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def problems(fam: dict, out: dict) -> list[str]:
    errs = []
    people = {p["id"]: p for p in fam["people"]}
    keys = {p["key"] for p in fam["people"]}
    if set((out.get("personas") or {}).keys()) != keys:
        errs.append(f"personas must have exactly these keys: {sorted(keys)}")
    texts = out.get("probes") or {}
    for q in fam["probes"]:
        t = texts.get(q["id"])
        if not isinstance(t, str) or not t.strip():
            errs.append(f"{q['id']}: missing")
            continue
        lang = G.LANGS[people[q["who"]]["language"]]
        want = "latin" if lang["roman"] or lang["lang"] == "en" else lang["script"]
        if not probe_script_ok(t, want, q["brief"]):
            errs.append(f"{q['id']}: must be written in {SCRIPT_NAMES[want]} script")
        if len(t) > 700:
            errs.append(f"{q['id']}: too long")
        nums = re.findall(r"\+?\d[\d -]{8,}\d", t)
        if any(not re.sub(r"\D", "", n).startswith(("9190000", "99")) and not re.search(r"\d{1,3}\.\d{3,}", n) for n in nums):
            errs.append(f"{q['id']}: contains a phone-like number")
        if {w.lower() for w in re.findall(r"[A-Za-z]+", t)} & G.BLOCKED_NAMES:
            errs.append(f"{q['id']}: uses a blocked name")
    return errs


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


async def ask(system: str, user: str, tokens: int = 6000) -> str:
    from app.llm import router

    for attempt in range(3):
        try:
            r = await router.complete("sim", system_stable=system, messages=[{"role": "user", "content": [{"type": "text", "text": user}]}],
                                      max_tokens=tokens, effort="low", timeout_s=120)
            return (r.text or "").strip()
        except router.AllModelsFailed as exc:
            if "cap" in str(exc):
                raise
            await asyncio.sleep(4 * (attempt + 1))
        except Exception:  # noqa: BLE001
            await asyncio.sleep(4 * (attempt + 1))
    return ""


def load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for p in sorted(TEXT_DIR.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["family"]] = row
    return cache


def save_cache(cache: dict[str, dict], families: dict[str, dict]) -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    by_lang: dict[str, list[dict]] = {}
    for fid, row in sorted(cache.items()):
        lang = families[fid]["language"] if fid in families else row.get("language", "other")
        by_lang.setdefault(lang, []).append(row)
    for lang, rows in by_lang.items():
        (TEXT_DIR / f"{lang}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def attach(families: list[dict], cache: dict[str, dict] | None = None) -> int:
    """Put cached persona lines and probe texts into the families (only where the cache matches the family)."""
    cache = cache if cache is not None else load_cache()
    n = 0
    for fam in families:
        row = cache.get(fam["id"])
        if not row or row["hash"] != input_hash(writer_input(fam)):
            continue
        for p in fam["people"]:
            p["persona"] = row["personas"].get(p["key"])
        for q in fam["probes"]:
            q["text"] = row["probes"].get(q["id"])
            n += 1
    return n


async def write_family(fam: dict) -> dict | None:
    inp = writer_input(fam)
    user = json.dumps(inp, ensure_ascii=False, indent=1)
    out = parse(await ask(SYSTEM, user))
    errs = problems(fam, out)
    if errs:
        fix = f"{user}\n\nYour previous answer had these problems; fix them and return the whole JSON again:\n- " + "\n- ".join(errs[:20])
        out = parse(await ask(SYSTEM, fix))
        errs = problems(fam, out)
    if errs:
        print(f"  {fam['id']} ({fam['language']}): still {len(errs)} problems: {errs[:3]}")
        return None
    return {"family": fam["id"], "language": fam["language"], "hash": input_hash(inp), "model": os.getenv("MODEL_ROUTES", ""),
            "personas": out["personas"], "probes": {q["id"]: out["probes"][q["id"]] for q in fam["probes"]}}


async def run(ids: list[str], parallel: int) -> int:
    from app.llm import spend

    cap = float(os.getenv("WORLD_MAX_INR", "0") or 0)
    if cap <= 0:
        print("Refusing to start: set WORLD_MAX_INR (₹ this run may spend).")
        return 2
    families = {f["id"]: f for f in load()}
    cache = load_cache()
    todo = [families[i] for i in ids if not (i in cache and cache[i]["hash"] == input_hash(writer_input(families[i])))]
    print(f"{len(ids)} families asked, {len(ids) - len(todo)} cached, {len(todo)} to write (cap ₹{cap:.0f})")
    sem = asyncio.Semaphore(parallel)
    done = failed = 0
    t0 = time.monotonic()

    async def one(fam: dict) -> None:
        nonlocal done, failed
        async with sem:
            if spend.process_spent() >= cap:
                return
            row = await write_family(fam)
            if row:
                cache[fam["id"]] = row
                done += 1
            else:
                failed += 1
            if (done + failed) % 25 == 0:
                save_cache(cache, families)
                print(f"  {done + failed}/{len(todo)} written ({failed} failed), ₹{spend.process_spent():.0f}, {time.monotonic() - t0:.0f}s")

    await asyncio.gather(*(one(f) for f in todo))
    save_cache(cache, families)
    print(f"done: {done} written, {failed} failed, {len(todo) - done - failed} skipped by the cap; spent ₹{spend.process_spent():.0f} in {time.monotonic() - t0:.0f}s")
    return 0 if not failed else 1


def main(argv: list[str]) -> int:
    par = 16
    if "--parallel" in argv:
        i = argv.index("--parallel")
        par = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    ids = argv if argv and argv[0].startswith("w") else slice_ids(argv[0] if argv else "ci")
    if os.getenv("GCP_PROJECT_ID") == "kavach-care":
        print("Refusing: paid world runs go to the sunny-ship test project, never kavach-care.")
        return 2
    return asyncio.run(run(ids, par))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
