"""Tests for content-addressed decision record models."""

from datetime import datetime, UTC

from nthlayer_common.records.models import (
    Assessment,
    AssessmentType,
    Evaluation,
    EvaluationMethod,
    EvaluationOutcome,
    Incident,
    IncidentStatus,
    Severity,
    Summaries,
    Verdict,
    VerdictOutcome,
)

ZERO_HASH = "0" * 64
FAKE_HASH = "a" * 64
NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)


# --- Enums ---


class TestEnums:
    def test_assessment_type_values(self):
        assert AssessmentType.THRESHOLD_BREACH.value == "threshold_breach"
        assert AssessmentType.CORRELATION.value == "correlation"
        assert AssessmentType.DRIFT.value == "drift"
        assert AssessmentType.CHANGE_EVENT.value == "change_event"

    def test_severity_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"

    def test_verdict_outcome_values(self):
        assert VerdictOutcome.RECOMMENDED.value == "recommended"
        assert VerdictOutcome.APPROVED.value == "approved"
        assert VerdictOutcome.EXECUTED.value == "executed"
        assert VerdictOutcome.REJECTED.value == "rejected"
        assert VerdictOutcome.EXPIRED.value == "expired"

    def test_evaluation_method_values(self):
        assert EvaluationMethod.METRIC_RECOVERY.value == "metric_recovery"
        assert EvaluationMethod.HUMAN_REVIEW.value == "human_review"
        assert EvaluationMethod.TIMEOUT.value == "timeout"
        assert EvaluationMethod.SLO_RESTORATION.value == "slo_restoration"

    def test_evaluation_outcome_values(self):
        assert EvaluationOutcome.EFFECTIVE.value == "effective"
        assert EvaluationOutcome.INEFFECTIVE.value == "ineffective"
        assert EvaluationOutcome.INCONCLUSIVE.value == "inconclusive"
        assert EvaluationOutcome.PARTIAL.value == "partial"

    def test_incident_status_values(self):
        assert IncidentStatus.OPEN.value == "open"
        assert IncidentStatus.MITIGATED.value == "mitigated"
        assert IncidentStatus.RESOLVED.value == "resolved"
        assert IncidentStatus.LEARNING.value == "learning"
        assert IncidentStatus.CLOSED.value == "closed"


# --- Summaries ---


class TestSummaries:
    def test_create_summaries(self):
        s = Summaries(
            technical="Latency p99 breached 500ms",
            plain="Response times are slow",
            executive="SLO breach detected",
        )
        assert s.technical == "Latency p99 breached 500ms"
        assert s.plain == "Response times are slow"
        assert s.executive == "SLO breach detected"

    def test_summaries_executive_optional(self):
        s = Summaries(technical="tech", plain="plain")
        assert s.executive is None

    def test_summaries_truncate(self):
        long_tech = "x" * 300
        s = Summaries(technical=long_tech, plain="ok")
        truncated = s.truncated()
        assert len(truncated.technical) == 280
        assert truncated.plain == "ok"
        assert truncated.executive is None

    def test_summaries_truncate_executive_limit(self):
        s = Summaries(technical="ok", plain="ok", executive="y" * 200)
        truncated = s.truncated()
        assert len(truncated.executive) == 140

    def test_summaries_truncate_noop_when_within_limits(self):
        s = Summaries(technical="ok", plain="ok", executive="ok")
        truncated = s.truncated()
        assert truncated.technical == "ok"
        assert truncated.plain == "ok"
        assert truncated.executive == "ok"


# --- Assessment ---


class TestAssessment:
    def test_create_assessment(self):
        a = Assessment(
            hash=FAKE_HASH,
            previous_hash=ZERO_HASH,
            schema_version="assessment/v1",
            timestamp=NOW,
            stream="sli:checkout-service:latency-p99",
            incident_id=None,
            type=AssessmentType.THRESHOLD_BREACH,
            severity=Severity.CRITICAL,
            payload={"current_value": 1247, "threshold": 500},
            summaries=Summaries(technical="breach", plain="slow", executive="bad"),
        )
        assert a.hash == FAKE_HASH
        assert a.previous_hash == ZERO_HASH
        assert a.schema_version == "assessment/v1"
        assert a.stream == "sli:checkout-service:latency-p99"
        assert a.incident_id is None
        assert a.type == AssessmentType.THRESHOLD_BREACH
        assert a.severity == Severity.CRITICAL
        assert a.payload == {"current_value": 1247, "threshold": 500}

    def test_assessment_incident_id_optional(self):
        a = Assessment(
            hash=FAKE_HASH,
            previous_hash=ZERO_HASH,
            schema_version="assessment/v1",
            timestamp=NOW,
            stream="sli:test:latency",
            incident_id=None,
            type=AssessmentType.DRIFT,
            severity=Severity.INFO,
            payload={},
            summaries=Summaries(technical="t", plain="p"),
        )
        assert a.incident_id is None

    def test_assessment_with_incident_id(self):
        a = Assessment(
            hash=FAKE_HASH,
            previous_hash=ZERO_HASH,
            schema_version="assessment/v1",
            timestamp=NOW,
            stream="sli:test:latency",
            incident_id="inc-4821",
            type=AssessmentType.THRESHOLD_BREACH,
            severity=Severity.CRITICAL,
            payload={},
            summaries=Summaries(technical="t", plain="p"),
        )
        assert a.incident_id == "inc-4821"


# --- Verdict ---


class TestVerdict:
    def test_create_verdict(self):
        v = Verdict(
            hash=FAKE_HASH,
            previous_hash=ZERO_HASH,
            schema_version="verdict/v1",
            timestamp=NOW,
            agent="triage",
            incident_id="inc-4821",
            input_hashes=["b" * 64, "c" * 64],
            prompt_hash="d" * 64,
            response_hash="e" * 64,
            model="claude-sonnet-4-20250514",
            reasoning="High severity due to customer impact",
            action={"type": "escalate", "target": "payments-team"},
            outcome=VerdictOutcome.RECOMMENDED,
            summaries=Summaries(technical="t", plain="p", executive="e"),
        )
        assert v.hash == FAKE_HASH
        assert v.agent == "triage"
        assert v.incident_id == "inc-4821"
        assert len(v.input_hashes) == 2
        assert v.model == "claude-sonnet-4-20250514"
        assert v.outcome == VerdictOutcome.RECOMMENDED

    def test_verdict_requires_incident_id(self):
        """Verdicts always have an incident_id — they're produced during a respond cycle."""
        v = Verdict(
            hash=FAKE_HASH,
            previous_hash=ZERO_HASH,
            schema_version="verdict/v1",
            timestamp=NOW,
            agent="remediation",
            incident_id="inc-001",
            input_hashes=[],
            prompt_hash="d" * 64,
            response_hash="e" * 64,
            model="gpt-4o",
            reasoning="rollback recommended",
            action={"type": "rollback"},
            outcome=VerdictOutcome.RECOMMENDED,
            summaries=Summaries(technical="t", plain="p", executive="e"),
        )
        assert v.incident_id == "inc-001"


# --- Evaluation ---


class TestEvaluation:
    def test_create_evaluation(self):
        e = Evaluation(
            hash=FAKE_HASH,
            previous_hash=ZERO_HASH,
            schema_version="evaluation/v1",
            timestamp=NOW,
            incident_id="inc-4821",
            verdict_hash="b" * 64,
            method=EvaluationMethod.METRIC_RECOVERY,
            outcome=EvaluationOutcome.EFFECTIVE,
            evidence_hashes=["c" * 64],
            payload={"recovery_time_seconds": 120},
            summaries=Summaries(technical="recovered in 2m", plain="service recovered"),
        )
        assert e.hash == FAKE_HASH
        assert e.incident_id == "inc-4821"
        assert e.verdict_hash == "b" * 64
        assert e.method == EvaluationMethod.METRIC_RECOVERY
        assert e.outcome == EvaluationOutcome.EFFECTIVE
        assert len(e.evidence_hashes) == 1

    def test_evaluation_no_executive_summary(self):
        """Evaluation summaries only have technical and plain per spec."""
        e = Evaluation(
            hash=FAKE_HASH,
            previous_hash=ZERO_HASH,
            schema_version="evaluation/v1",
            timestamp=NOW,
            incident_id="inc-001",
            verdict_hash="b" * 64,
            method=EvaluationMethod.TIMEOUT,
            outcome=EvaluationOutcome.INCONCLUSIVE,
            evidence_hashes=[],
            payload={},
            summaries=Summaries(technical="no change observed", plain="inconclusive"),
        )
        assert e.summaries.executive is None


# --- Incident ---


class TestIncident:
    def test_create_incident(self):
        i = Incident(
            id="inc-4821",
            created_at=NOW,
            trigger_hash=FAKE_HASH,
            stream="sli:checkout-service:latency-p99",
            status=IncidentStatus.OPEN,
        )
        assert i.id == "inc-4821"
        assert i.trigger_hash == FAKE_HASH
        assert i.status == IncidentStatus.OPEN

    def test_incident_default_status_is_open(self):
        i = Incident(
            id="inc-001",
            created_at=NOW,
            trigger_hash=FAKE_HASH,
            stream="sli:test:latency",
        )
        assert i.status == IncidentStatus.OPEN
