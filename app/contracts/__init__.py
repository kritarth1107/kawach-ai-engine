"""Data contracts for what the backend, dashboard and admin console read from the engine (plan B1.5).

The shapes are learnt from the synthetic world's responses (tests/test_world_reads.py, 120 families in 30 languages)
and committed as JSON Schema in contracts/. Every push validates every world response against them, so a renamed
field, a changed type or a missing key fails CI before a page breaks. TypeScript types for the backend and the
dashboard are generated from the same files (scripts/contract_types.sh).

    CONTRACTS_WRITE=1 pytest tests/test_world_reads.py   # relearn the schemas after an intended change, then commit
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DIR = Path(__file__).resolve().parents[2] / "contracts"
_DYNAMIC_KEY = re.compile(r"^([0-9a-f]{8}-[0-9a-f]{4}-|\d{4}-\d{2}-\d{2}|w\d{4}|[a-z]+:[a-z0-9_]+$)")


def _type(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


def infer(values: list) -> dict:
    """A JSON Schema that every one of the sample values satisfies (keys present in every sample are required)."""
    types = {_type(v) for v in values}
    if types == {"null"}:
        return {"x-seen": "only null"}  # not seen with a value yet: any shape is allowed (the world should grow to cover it)
    if {"integer", "number"} <= types:
        types.discard("integer")
    schema: dict = {"type": sorted(types)[0] if len(types) == 1 else sorted(types)} if types else {}
    objs = [v for v in values if isinstance(v, dict)]
    arrs = [v for v in values if isinstance(v, list)]
    if objs:
        keys = sorted({k for o in objs for k in o})
        dynamic = keys and (sum(1 for k in keys if _DYNAMIC_KEY.match(k)) >= max(1, len(keys) // 2) or len(keys) > 60)
        if dynamic:
            schema["additionalProperties"] = infer([x for o in objs for x in o.values()])
        else:
            schema["properties"] = {k: infer([o[k] for o in objs if k in o]) for k in keys}
            schema["required"] = [k for k in keys if all(k in o for o in objs)]
    if arrs:
        items = [x for a in arrs for x in a]
        schema["items"] = infer(items) if items else {}
    return schema


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Problems with a value against a schema from infer (types, required keys, items); extra keys are allowed."""
    if not schema or "type" not in schema and not schema.get("properties") and not schema.get("items") and not schema.get("additionalProperties"):
        return []
    want = schema.get("type")
    allowed = set(want if isinstance(want, list) else [want]) if want else set()
    t = _type(value)
    if allowed and t not in allowed and not (t == "integer" and "number" in allowed):
        return [f"{path}: {t}, expected {'/'.join(sorted(allowed))}"]
    errs: list[str] = []
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}.{k}: missing")
        props = schema.get("properties") or {}
        extra = schema.get("additionalProperties")
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif isinstance(extra, dict):
                errs += validate(v, extra, f"{path}[{k!r}]")
    elif isinstance(value, list) and schema.get("items"):
        for i, x in enumerate(value[:200]):
            errs += validate(x, schema["items"], f"{path}[{i}]")
            if len(errs) > 20:
                break
    return errs


def load(name: str) -> dict | None:
    p = DIR / f"{name}.schema.json"
    return json.loads(p.read_text()) if p.exists() else None


def write(name: str, samples: list) -> Path:
    DIR.mkdir(parents=True, exist_ok=True)
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": name, **infer(samples)}
    p = DIR / f"{name}.schema.json"
    p.write_text(json.dumps(schema, indent=1, sort_keys=True) + "\n")
    return p
