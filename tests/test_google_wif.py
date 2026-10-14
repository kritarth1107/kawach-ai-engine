"""Gemini from the AWS engine through workload identity federation: no Google key, only the task role."""

import pytest

from app.llm import google_wif


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(google_wif, "_cached", None)


def test_without_an_audience_the_default_chain_is_used(monkeypatch):
    monkeypatch.delenv("GOOGLE_WIF_AUDIENCE", raising=False)
    assert google_wif.credentials() is None


def test_an_audience_without_a_service_account_fails_loudly(monkeypatch):
    monkeypatch.setenv("GOOGLE_WIF_AUDIENCE", "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/x")
    monkeypatch.delenv("GOOGLE_VERTEX_SA", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_VERTEX_SA"):
        google_wif.credentials()


def test_federated_credentials_act_as_the_vertex_account_and_are_made_once(monkeypatch):
    aud = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/kavach-aws/providers/kavach-ecs"
    monkeypatch.setenv("GOOGLE_WIF_AUDIENCE", aud)
    monkeypatch.setenv("GOOGLE_VERTEX_SA", "brain@proj.iam.gserviceaccount.com")
    c = google_wif.credentials()
    assert c is google_wif.credentials(), "one credentials object, refreshed by google-auth"
    assert c._audience == aud
    assert c._service_account_impersonation_url.endswith("brain@proj.iam.gserviceaccount.com:generateAccessToken")


def test_the_supplier_reads_the_task_roles_keys_from_botocore(monkeypatch):
    import boto3

    class Frozen:
        access_key, secret_key, token = "AKIA_TEST", "secret", "session"

    class Creds:
        def get_frozen_credentials(self):
            return Frozen()

    monkeypatch.setattr(boto3, "Session", lambda: type("S", (), {"get_credentials": lambda self: Creds()})())
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    sup = google_wif._EcsRoleSupplier()
    got = sup.get_aws_security_credentials(None, None)
    assert (got.access_key_id, got.secret_access_key, got.session_token) == ("AKIA_TEST", "secret", "session")
    assert sup.get_aws_region(None, None) == "ap-south-1"
