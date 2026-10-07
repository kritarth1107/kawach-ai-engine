"""Engine /v2/admin: switched off without its secret and caller; only the admin API's identity with its own secret."""

import base64
import json
import time

import pytest
from fastapi import HTTPException

from app.api import admin


def _tok(payload: dict, signed: bool = True) -> str:
    b = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{b({'alg': 'RS256'})}.{b(payload)}.{'sig' if signed else ''}"


SA = "kavach-admin-api@kavach-care.iam.gserviceaccount.com"
AUD = "https://engine.test"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ENGINE_ADMIN_SECRET", "s" * 40)
    monkeypatch.setenv("ADMIN_API_SA", SA)
    monkeypatch.setenv("ENGINE_AUDIENCES", AUD)


def test_caller_verified_token(env):
    verify = lambda t: {"email": SA, "aud": AUD, "email_verified": True}  # noqa: E731
    assert admin.caller_email("Bearer " + _tok({}), verify=verify) == SA


def test_caller_wrong_audience(env):
    verify = lambda t: {"email": SA, "aud": "https://other"}  # noqa: E731
    with pytest.raises(HTTPException) as e:
        admin.caller_email("Bearer " + _tok({}), verify=verify)
    assert e.value.status_code == 401


def test_unsigned_token_only_on_cloud_run(env):
    tok = _tok({"email": SA, "aud": AUD, "exp": time.time() + 60}, signed=False)
    assert admin.caller_email("Bearer " + tok, on_cloud_run=True) == SA
    with pytest.raises(HTTPException):
        admin.caller_email("Bearer " + tok, on_cloud_run=False)
    expired = _tok({"email": SA, "aud": AUD, "exp": time.time() - 5}, signed=False)
    with pytest.raises(HTTPException):
        admin.caller_email("Bearer " + expired, on_cloud_run=True)


def test_google_marker_counts_as_stripped(env):
    tok = _tok({"email": SA, "aud": AUD, "exp": time.time() + 60}, signed=False) + "SIGNATURE_REMOVED_BY_GOOGLE"
    assert admin.caller_email("Bearer " + tok, on_cloud_run=True) == SA


def test_no_audiences_configured_is_closed(env, monkeypatch):
    monkeypatch.setenv("ENGINE_AUDIENCES", "")
    verify = lambda t: {"email": SA, "aud": AUD}  # noqa: E731
    with pytest.raises(HTTPException):
        admin.caller_email("Bearer " + _tok({}), verify=verify)


def test_no_token():
    with pytest.raises(HTTPException) as e:
        admin.caller_email(None)
    assert e.value.status_code == 401


class _Req:
    def __init__(self, headers):
        self.headers = headers


async def test_switched_off_without_config(monkeypatch):
    monkeypatch.delenv("ENGINE_ADMIN_SECRET", raising=False)
    monkeypatch.delenv("ADMIN_API_SA", raising=False)
    with pytest.raises(HTTPException) as e:
        await admin.verify_admin_caller(_Req({}))
    assert e.value.status_code == 503


async def test_wrong_secret_refused(env):
    with pytest.raises(HTTPException) as e:
        await admin.verify_admin_caller(_Req({"x-admin-secret": "wrong", "authorization": "Bearer x"}))
    assert e.value.status_code == 401


async def test_backend_identity_refused(env, monkeypatch):
    monkeypatch.setattr(admin, "caller_email", lambda a: "303943038694-compute@developer.gserviceaccount.com")
    with pytest.raises(HTTPException) as e:
        await admin.verify_admin_caller(_Req({"x-admin-secret": "s" * 40, "authorization": "Bearer x"}))
    assert e.value.status_code == 403


async def test_admin_api_identity_passes(env, monkeypatch):
    monkeypatch.setattr(admin, "caller_email", lambda a: SA)
    assert await admin.verify_admin_caller(_Req({"x-admin-secret": "s" * 40, "authorization": "Bearer x"})) is None


def test_routes_all_behind_the_admin_check():
    import app.main  # noqa: F401  (the router is mounted there)

    routes = [r for r in admin.router.routes if r.path.startswith("/v2/admin")]
    assert len(routes) >= 9
    for r in routes:
        assert admin.verify_admin_caller in [d.call for d in r.dependant.dependencies], r.path


async def test_conversation_and_care_record(db, monkeypatch):
    from datetime import datetime, timezone

    from app.care.models import CareFact, Turn

    now = datetime.now(timezone.utc)
    db.add(Turn(family_id="f1", thread_id="u1", role="user", text="dawai le li", at=now, meta={}))
    db.add(Turn(family_id="f1", thread_id="u2", role="user", text="other person", at=now, meta={}))
    db.add(CareFact(family_id="f1", subject_id="u1", domain="medicine", key="med:metformin", value={}, text="Metformin 500",
                    source_kind="dashboard", valid_from=now, recorded_at=now))
    await db.flush()
    conv = await admin.admin_conversation("f1", "u1", db)
    assert [t["text"] for t in conv["turns"]] == ["dawai le li"]
    rec = await admin.admin_care_record("f1", "u1", db)
    assert rec["facts"][0]["key"] == "med:metformin"
    exp = await admin.admin_export("f1", "u1", db)
    assert exp["careRecord"][0]["text"] == "Metformin 500" and exp["conversation"][0]["text"] == "dawai le li"


async def test_models_lists_roles_with_prices(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTES", '{"brain": ["gemini:gemini-3.5-flash@asia-south1", "gemini:gemini-3.1-pro-preview"]}')
    from app.llm import spend

    spend.reset()
    out = await admin.admin_models()
    brain = out["roles"]["brain"]
    assert brain["configured"] is True and brain["routes"][0]["model"] == "gemini-3.5-flash"
    assert brain["routes"][0]["location"] == "asia-south1" and brain["routes"][0]["priceInr"][1] > 0
    assert out["roles"]["worker"]["configured"] is False  # default routes still listed
    assert out["softCap"] > 0 and out["hardCap"] >= out["softCap"] and out["usdInr"] > 0
