"""One CloudWatch Embedded Metric Format line on stdout.

The log group extracts it. No extra IAM. A cancelled hedge is not a call.
"""
from __future__ import annotations

import json
import time


def note(namespace: str, role: str, *, errors: int = 0, throttles: int = 0, refusals: int = 0, latency_ms: float | None = None,
         region: str | None = None) -> None:
    metrics = [
        {"Name": "Calls", "Unit": "Count"},
        {"Name": "Errors", "Unit": "Count"},
        {"Name": "Throttles", "Unit": "Count"},
        {"Name": "Refusals", "Unit": "Count"},
    ]
    body: dict = {"Role": role, "Calls": 1, "Errors": errors, "Throttles": throttles, "Refusals": refusals}
    # Region is only set when the call left Mumbai. The Role-only series stays the one dashboards already use.
    dims: list = [["Role"], []]
    if region:
        body["Region"] = region
        dims = [["Role", "Region"], ["Role"], []]
    if latency_ms is not None:
        metrics.append({"Name": "LatencyMs", "Unit": "Milliseconds"})
        body["LatencyMs"] = round(latency_ms, 1)
    print(json.dumps({
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": namespace,
                "Dimensions": dims,
                "Metrics": metrics,
            }],
        },
        **body,
    }), flush=True)
