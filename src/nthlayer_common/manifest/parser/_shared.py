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


def extract_declared_dependencies(
    *,
    from_manifests: dict[str, Any] | None = None,
    from_dicts: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Build a {service_name: [dep_name, ...]} map from either Manifest
    dataclass instances (CLI/YAML path) or raw HTTP wire dicts (worker
    path). Exactly one of ``from_manifests`` / ``from_dicts`` must be
    supplied.

    Used by ``nthlayer-workers/learn`` retrospective generation
    (opensrm-jmy.21 / opensrm-dpws) to populate
    ``declared_dependencies_by_service`` on the retrospective payload.
    Downstream consumers (``_add_dependency_recommendations``) treat
    this map as the ground-truth view of operator-declared deps.

    A manifest with ``dependencies = None`` produces an empty list for
    that service; the absence of declared deps is itself information
    downstream consumers want to record. Dict entries without a
    non-empty ``name`` are silently skipped (mirrors the
    ``_extract_service_slos`` precedent in observe/worker.py).
    """
    if (from_manifests is None) == (from_dicts is None):
        raise ValueError(
            "extract_declared_dependencies: supply exactly one of "
            "from_manifests= or from_dicts="
        )

    if from_manifests is not None:
        return {
            service_name: [dep.name for dep in (manifest.dependencies or [])]
            for service_name, manifest in from_manifests.items()
        }

    # Mirror the dataclass branch's shape: dict-comp keyed by service
    # name, value = list of dep names. The extra `if d.get("name")`
    # filter is the only asymmetry — Manifest.Dependency.name is
    # guaranteed by the dataclass; HTTP wire dicts can omit it.
    return {
        m["name"]: [
            d["name"] for d in (m.get("dependencies") or []) if d.get("name")
        ]
        for m in (from_dicts or []) if m.get("name")
    }
