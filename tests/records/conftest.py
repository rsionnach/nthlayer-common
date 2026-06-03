"""Shared test builders for decision record tests."""

from datetime import datetime, UTC

from nthlayer_common.records.hashing import canonical_json, compute_hash
from nthlayer_common.records.models import (
    ZERO_HASH,
    Assessment,
    AssessmentType,
    Evaluation,
    EvaluationMethod,
    EvaluationOutcome,
    Severity,
    Summaries,
    Verdict,
    VerdictOutcome,
)

NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
LATER = datetime(2026, 4, 11, 12, 5, 0, tzinfo=UTC)
T1 = datetime(2026, 4, 11, 12, 1, 0, tzinfo=UTC)
T2 = datetime(2026, 4, 11, 12, 2, 0, tzinfo=UTC)


def build_test_assessment(previous_hash: str = ZERO_HASH, **overrides) -> Assessment:
    """Build an Assessment with a correctly computed content hash."""
    a = Assessment(
        hash="placeholder",
        previous_hash=previous_hash,
        schema_version="assessment/v1",
        timestamp=overrides.pop("timestamp", NOW),
        stream=overrides.pop("stream", "sli:checkout:latency-p99"),
        incident_id=overrides.pop("incident_id", None),
        type=overrides.pop("type", AssessmentType.THRESHOLD_BREACH),
        severity=overrides.pop("severity", Severity.CRITICAL),
        payload=overrides.pop("payload", {"current_value": 1247}),
        summaries=overrides.pop("summaries", Summaries(technical="t", plain="p", executive="e")),
    )
    h = compute_hash(canonical_json(a))
    return Assessment(
        hash=h, previous_hash=a.previous_hash, schema_version=a.schema_version,
        timestamp=a.timestamp, stream=a.stream, incident_id=a.incident_id,
        type=a.type, severity=a.severity, payload=a.payload, summaries=a.summaries,
    )


def build_test_verdict(previous_hash: str = ZERO_HASH, **overrides) -> Verdict:
    """Build a Verdict with a correctly computed content hash."""
    v = Verdict(
        hash="placeholder",
        previous_hash=previous_hash,
        schema_version="verdict/v1",
        timestamp=overrides.pop("timestamp", NOW),
        agent=overrides.pop("agent", "triage"),
        incident_id=overrides.pop("incident_id", "inc-001"),
        input_hashes=overrides.pop("input_hashes", []),
        prompt_hash=overrides.pop("prompt_hash", "d" * 64),
        response_hash=overrides.pop("response_hash", "e" * 64),
        model=overrides.pop("model", "test-model"),
        reasoning=overrides.pop("reasoning", "High severity"),
        action=overrides.pop("action", {"type": "escalate"}),
        outcome=overrides.pop("outcome", VerdictOutcome.RECOMMENDED),
        summaries=overrides.pop("summaries", Summaries(technical="t", plain="p", executive="e")),
    )
    h = compute_hash(canonical_json(v))
    return Verdict(
        hash=h, previous_hash=v.previous_hash, schema_version=v.schema_version,
        timestamp=v.timestamp, agent=v.agent, incident_id=v.incident_id,
        input_hashes=v.input_hashes, prompt_hash=v.prompt_hash,
        response_hash=v.response_hash, model=v.model, reasoning=v.reasoning,
        action=v.action, outcome=v.outcome, summaries=v.summaries,
    )


def build_test_evaluation(previous_hash: str = ZERO_HASH, **overrides) -> Evaluation:
    """Build an Evaluation with a correctly computed content hash."""
    e = Evaluation(
        hash="placeholder",
        previous_hash=previous_hash,
        schema_version="evaluation/v1",
        timestamp=overrides.pop("timestamp", NOW),
        incident_id=overrides.pop("incident_id", "inc-001"),
        verdict_hash=overrides.pop("verdict_hash", "b" * 64),
        method=overrides.pop("method", EvaluationMethod.METRIC_RECOVERY),
        outcome=overrides.pop("outcome", EvaluationOutcome.EFFECTIVE),
        evidence_hashes=overrides.pop("evidence_hashes", []),
        payload=overrides.pop("payload", {"recovery_time_seconds": 120}),
        summaries=overrides.pop("summaries", Summaries(technical="t", plain="p")),
    )
    h = compute_hash(canonical_json(e))
    return Evaluation(
        hash=h, previous_hash=e.previous_hash, schema_version=e.schema_version,
        timestamp=e.timestamp, incident_id=e.incident_id, verdict_hash=e.verdict_hash,
        method=e.method, outcome=e.outcome, evidence_hashes=e.evidence_hashes,
        payload=e.payload, summaries=e.summaries,
    )
