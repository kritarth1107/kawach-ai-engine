"""A tool call outside its schema is refused with the allowed values, never a crash (live 2026-10-14: a Bedrock model
called stop with domain "outcome" eight times and every call raised)."""

import json

from app.brain import tools


def test_enum_value_outside_the_list_is_named_with_the_choices():
    schema = {"properties": {"domain": {"type": "string", "enum": ["allergy", "medicine"]}, "name": {"type": "string"}},
              "required": ["domain", "name"]}
    problems = tools.arg_problems(schema, {"domain": "outcome", "name": "chest wound"})
    assert problems == ["domain must be one of allergy, medicine (got 'outcome')"]


def test_missing_required_field_and_bad_array_item():
    schema = {"properties": {"to": {"type": "string"}, "days": {"type": "array", "items": {"enum": ["mon", "tue"]}}},
              "required": ["to"]}
    assert tools.arg_problems(schema, {"days": ["mon", "sun"]}) == ["to is required", "days items must be from mon, tue (got ['sun'])"]


def test_a_good_call_and_an_extra_field_pass():
    schema = {"properties": {"to": {"type": "string"}}, "required": ["to"], "additionalProperties": False}
    assert tools.arg_problems(schema, {"to": "p1", "note": "extra"}) == []
    assert tools.arg_problems(schema, "not a dict") == ["arguments must be an object"]


def test_every_registered_tool_schema_checks_cleanly_against_itself():
    # an enum default must be valid for its own list, and no schema may make the checker raise
    for spec in tools.specs():
        props = spec.schema.get("properties") or {}
        sample = {k: (v["enum"][0] if v.get("enum") else None) for k, v in props.items()}
        tools.arg_problems(spec.schema, {k: v for k, v in sample.items() if v is not None})


async def test_stop_with_an_unknown_domain_is_refused_not_raised():
    class Ctx:
        actions: list = []
    ctx = Ctx()
    ctx.actions = []
    out, is_error = await tools.run(ctx, "stop", {"domain": "outcome", "name": "chest wound", "reason": "it was a joke"})
    assert is_error
    body = json.loads(out)
    assert "domain must be one of" in body["refused"] and "medicine" in body["refused"]
    assert ctx.actions[-1]["ok"] is False and "refused" in ctx.actions[-1]
