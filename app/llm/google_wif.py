"""Google credentials for the engine on AWS, with no Google key stored anywhere.

The ECS task role signs a request to AWS STS; Google's workload identity federation (pool kavach-aws, provider kavach-ecs,
which only accepts assumed-role/kavach-ecs-task) trades it for a short-lived token of one service account, which may
only call Vertex AI. The backend does the same for Chirp (voicePipeline.ts getAccessToken).

Env: GOOGLE_WIF_AUDIENCE (the provider's full name) and GOOGLE_VERTEX_SA (the service account to act as). Without
GOOGLE_WIF_AUDIENCE this returns None and google-auth's own default chain is used, as on GCP.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cached = None


class _EcsRoleSupplier:
    """AWS credentials of the running task (ECS container endpoint, env vars or the local SSO profile) via botocore."""

    def get_aws_security_credentials(self, context, request):  # noqa: ARG002 — google-auth's interface
        from google.auth.aws import AwsSecurityCredentials
        import boto3

        frozen = boto3.Session().get_credentials().get_frozen_credentials()
        return AwsSecurityCredentials(frozen.access_key, frozen.secret_key, frozen.token)

    def get_aws_region(self, context, request):  # noqa: ARG002
        return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1"


def config() -> tuple[str, str] | None:
    audience = os.getenv("GOOGLE_WIF_AUDIENCE", "").strip()
    if not audience:
        return None
    sa = os.getenv("GOOGLE_VERTEX_SA", "").strip()
    if not sa:
        raise RuntimeError("GOOGLE_WIF_AUDIENCE is set but GOOGLE_VERTEX_SA is not")
    return audience, sa


def credentials():
    """Federated Google credentials (refreshed by google-auth as they expire), or None to use the default chain."""
    global _cached
    cfg = config()
    if not cfg:
        return None
    with _lock:
        if _cached is None:
            from google.auth import aws

            audience, sa = cfg
            _cached = aws.Credentials(
                audience=audience,
                subject_token_type="urn:ietf:params:aws:token-type:aws4_request",
                aws_security_credentials_supplier=_EcsRoleSupplier(),
                service_account_impersonation_url=f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa}:generateAccessToken",
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            logger.info("google credentials: workload identity federation as %s", sa)
        return _cached
