"""Shared parsing helpers used by both v1 and v2 parsers."""

from __future__ import annotations

from typing import Any

from nthlayer_common.manifest.models import Observability


def parse_observability(obs_data: dict[str, Any] | None) -> Observability | None:
    """Parse observability section (common to v1 and v2 formats)."""
    if not obs_data:
        return None

    return Observability(
        metrics_prefix=obs_data.get("metrics_prefix"),
        logs_label=obs_data.get("logs_label"),
        traces_service=obs_data.get("traces_service"),
        prometheus_job=obs_data.get("prometheus_job"),
        grafana_url=obs_data.get("grafana_url"),
        labels=obs_data.get("labels", {}),
    )
