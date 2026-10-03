"""Release gate for the shopping, pharmacy and rides agents: every bench scenario must pass (no model calls)."""

import pytest

from app.specialists.agents import SPECIALISTS, specialist_for
from app.specialists.bench import SCENARIOS, gate, run_scenario


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.name for s in SCENARIOS])
async def test_bench_scenario(db, at, sc):
    at("2026-10-02 10:00")
    r = await run_scenario(db, sc)
    assert r["passed"], f"{sc.name}: {r['problems']}"


def test_every_service_has_one_specialist():
    from app.tasks.skills import SKILLS

    for service in SKILLS:
        assert specialist_for(service).name in SPECIALISTS
    assert sum(len(s.services) for s in SPECIALISTS.values()) == len(SKILLS)


def test_gate_blocks_any_safety_failure():
    ok = [{"name": f"s{i}", "safety": False, "passed": True} for i in range(40)]
    assert gate(ok)[0]
    assert not gate(ok + [{"name": "no_cod", "safety": True, "passed": False}])[0]


def test_bench_covers_each_agent_and_channel():
    names = " ".join(s.name for s in SCENARIOS)
    for word in ("shopping", "pharmacy", "rides", "connector"):
        assert word in names
    assert sum(s.safety for s in SCENARIOS) >= 15


async def test_metrics_rollup_counts_success_cost_and_channels(db, at):
    from app.specialists import channels, metrics
    from app.specialists.bench import SCENARIOS

    at("2026-10-02 10:00")
    by = {s.name: s for s in SCENARIOS}
    for name in ("shopping_happy_browser", "shopping_no_cod", "connector_happy", "pharmacy_happy"):
        assert (await run_scenario(db, by[name]))["passed"]
    out = await metrics.rollup(db, days=7)
    rows = {(r["agent"], r["service"]): r for r in out["rows"]}
    inst = rows[("shopping", "instamart")]
    assert inst["tasks"] == 3 and inst["placed"] == 2 and inst["failed"] == 1
    assert inst["channels"].get("connector") and inst["channels"].get("browser")
    assert inst["costInr"] > 0 and inst["medianSecondsToReady"] is not None
    assert rows[("pharmacy", "apollo")]["placed"] == 1
    assert out["totals"]["tasks"] == 4
    health = {(h["service"], h["channel"]): h for h in await channels.health_table(db)}
    assert health[("instamart", "connector")]["successes"] >= 2
