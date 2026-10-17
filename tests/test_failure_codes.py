"""Every failed task gets one code for why (the admin counts codes, not free text). From data only."""

from types import SimpleNamespace

from app.specialists.metrics import failure_code
from app.tasks.skills import PROBLEM_KINDS, schema_for


def task(result=None, details=None):
    return SimpleNamespace(result=result or {}, details=details or {})


def test_kavach_code_wins_then_agent_report_then_fields():
    assert failure_code(task({"problem_kind": "no_cod"}, {"fail_code": "no_reply"})) == "no_reply"
    assert failure_code(task({"problem_kind": "strength_unavailable"})) == "strength_unavailable"
    assert failure_code(task({"blocked": True})) == "site_blocked"
    assert failure_code(task({"deliverable": False})) == "not_deliverable"
    assert failure_code(task({"cod_available": False})) == "no_cod"
    assert failure_code(task({"needs_prescription": True})) == "needs_prescription"
    assert failure_code(task({"items": [{"name": "Cipcal 60K", "available": False}]})) == "out_of_stock"
    assert failure_code(task({"error": "Session not found"})) == "session_lost"
    assert failure_code(task({})) == "other"


def test_every_phase_asks_the_agent_for_its_kind():
    for kind, phase in (("order", "browse"), ("order", "prepare"), ("order", "place"), ("ride", "prepare")):
        prop = schema_for(kind, phase)["properties"]["problem_kind"]
        assert prop["enum"] == list(PROBLEM_KINDS) and "none" in prop["enum"]
