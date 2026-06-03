"""CloudEvents v1.0 envelope helpers for NthLayer events.

Wraps verdicts and assessments in CloudEvents v1.0 envelopes.
The envelope format is stable across v1.5 and v2 — type taxonomy,
attribute naming, and structure are frozen from v1.5 onwards.

Canonical import: ``from nthlayer_common.cloudevents import wrap_verdict, wrap_assessment``

Spec: NTHLAYER-TELEMETRY-ENVELOPE-v1 §3
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from nthlayer_common.verdicts.models import VALID_VERDICT_TYPES

# Deployment identity — configurable via env or passed explicitly
_DEFAULT_DEPLOYMENT_ID = os.environ.get("NTHLAYER_DEPLOYMENT_ID", "default")

# -- Type taxonomy --
# Pattern: io.nthlayer.<category>.<subtype>.v1
# Categories: verdict, assessment, change, audit, alert

# Wire-format validation reuses VALID_VERDICT_TYPES; ASSESSMENT_KINDS is
# the parallel canonical set for observations. Verdicts are decisions,
# assessments are observations — adding a new kind happens here for
# observations, in verdicts.models for decisions.
# See docs/superpowers/decisions/verdict-assessment-taxonomy-boundary.md
_VERDICT_TYPES = VALID_VERDICT_TYPES

# Public constant — canonical set of assessment kinds.
ASSESSMENT_KINDS = frozenset({
    "slo_status",
    "judgment_slo_evaluation",
    "burn_rate",
    "drift_signal",
    "portfolio_status",
    "deploy_gate",
    "dependency_graph",
    "correlation_snapshot",
    "topology_drift",
    "contract_divergence",
    "retrospective",
    "calibration_signal",
})



def _ce_type_for_verdict(verdict_type: str) -> str:
    """Compute CloudEvents type attribute for a verdict."""
    subtype = verdict_type if verdict_type in _VERDICT_TYPES else "unknown"
    return f"io.nthlayer.verdict.{subtype}.v1"


def _ce_type_for_assessment(kind: str) -> str:
    """Compute CloudEvents type attribute for an assessment."""
    subtype = kind if kind in ASSESSMENT_KINDS else "unknown"
    return f"io.nthlayer.assessment.{subtype}.v1"


def _ce_source(component: str, deployment_id: str | None = None) -> str:
    """Compute CloudEvents source attribute. Component is e.g. 'observe', 'respond', 'correlate'."""
    did = deployment_id or _DEFAULT_DEPLOYMENT_ID
    return f"urn:nthlayer:{component}:{did}"


def wrap_verdict(
    verdict: dict[str, Any],
    *,
    component: str = "unknown",
    deployment_id: str | None = None,
) -> dict[str, Any]:
    """Wrap a verdict dict in a CloudEvents v1.0 envelope.

    Args:
        verdict: The verdict dict. Must have 'id', 'type', and optionally
                 'created_at' and 'service'.
        component: The originating component ('observe', 'measure', 'correlate', 'respond', 'learn').
        deployment_id: Override deployment ID (default from NTHLAYER_DEPLOYMENT_ID env).

    Returns:
        CloudEvents v1.0 JSON-serializable dict.
    """
    if "id" not in verdict:
        raise ValueError("Verdict must have an 'id' field")
    verdict_id = verdict["id"]
    verdict_type = verdict.get("type", "unknown")
    created_at = verdict.get("created_at", datetime.now(UTC).isoformat())
    service = verdict.get("service")

    envelope: dict[str, Any] = {
        "specversion": "1.0",
        "type": _ce_type_for_verdict(verdict_type),
        "source": _ce_source(component, deployment_id),
        "id": verdict_id,
        "time": created_at,
        "datacontenttype": "application/json",
        "data": verdict,
    }

    if service:
        envelope["subject"] = f"component:default/{service}"

    return envelope


def wrap_assessment(
    assessment: dict[str, Any],
    *,
    component: str = "unknown",
    deployment_id: str | None = None,
) -> dict[str, Any]:
    """Wrap an assessment dict in a CloudEvents v1.0 envelope.

    Args:
        assessment: The assessment dict. Must have 'id', 'kind', 'service',
                    and optionally 'created_at'.
        component: The originating component.
        deployment_id: Override deployment ID.

    Returns:
        CloudEvents v1.0 JSON-serializable dict.
    """
    if "id" not in assessment:
        raise ValueError("Assessment must have an 'id' field")
    assessment_id = assessment["id"]
    kind = assessment.get("kind", "unknown")
    created_at = assessment.get("created_at", datetime.now(UTC).isoformat())
    service = assessment.get("service")

    envelope: dict[str, Any] = {
        "specversion": "1.0",
        "type": _ce_type_for_assessment(kind),
        "source": _ce_source(component, deployment_id),
        "id": assessment_id,
        "time": created_at,
        "datacontenttype": "application/json",
        "data": assessment,
    }

    if service:
        envelope["subject"] = f"component:default/{service}"

    return envelope


def parse_cloudevent(envelope: dict[str, Any]) -> dict[str, Any]:
    """Extract the data payload from a CloudEvents envelope.

    Validates required CloudEvents v1.0 attributes.

    Args:
        envelope: CloudEvents v1.0 dict.

    Returns:
        The 'data' payload dict.

    Raises:
        ValueError: If required CloudEvents attributes are missing.
    """
    if not isinstance(envelope, dict):
        raise ValueError(f"Expected dict, got {type(envelope).__name__}")
    required = ("specversion", "type", "source", "id")
    missing = [attr for attr in required if attr not in envelope]
    if missing:
        raise ValueError(f"Invalid CloudEvent: missing required attributes: {missing}")

    if envelope["specversion"] != "1.0":
        raise ValueError(f"Unsupported CloudEvents specversion: {envelope['specversion']}")

    return envelope.get("data", {})


def validate_cloudevent(envelope: dict[str, Any]) -> list[str]:
    """Validate a CloudEvents envelope. Returns list of issues (empty = valid).

    Non-raising alternative to parse_cloudevent for batch validation.
    """
    if not isinstance(envelope, dict):
        return [f"Expected dict, got {type(envelope).__name__}"]

    issues: list[str] = []

    required = ("specversion", "type", "source", "id")
    for attr in required:
        if attr not in envelope:
            issues.append(f"Missing required attribute: {attr}")

    if envelope.get("specversion") and envelope["specversion"] != "1.0":
        issues.append(f"Unsupported specversion: {envelope['specversion']}")

    ce_type = envelope.get("type", "")
    if ce_type and not ce_type.startswith("io.nthlayer."):
        issues.append(f"Type does not follow io.nthlayer.* pattern: {ce_type}")

    source = envelope.get("source", "")
    if source and not source.startswith("urn:nthlayer:"):
        issues.append(f"Source does not follow urn:nthlayer:* pattern: {source}")

    if "datacontenttype" in envelope and envelope["datacontenttype"] != "application/json":
        issues.append(f"Unexpected datacontenttype: {envelope['datacontenttype']}")

    return issues
