"""Tests for hash chain verification."""

from datetime import datetime, UTC

import pytest

from nthlayer_common.records.hashing import canonical_json, compute_hash
from nthlayer_common.records.models import (
    ZERO_HASH,
    Assessment,
    AssessmentType,
    Evaluation,
    EvaluationMethod,
    EvaluationOutcome,
    Incident,
    Severity,
    Summaries,
    Verdict,
    VerdictOutcome,
)
from nthlayer_common.records.sqlite_store import SQLiteDecisionRecordStore
from nthlayer_common.records.verification import (
    verify_chain,
    verify_incident,
)

NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 4, 11, 12, 1, 0, tzinfo=UTC)
T2 = datetime(2026, 4, 11, 12, 2, 0, tzinfo=UTC)


def _hash_assessment(previous_hash: str = ZERO_HASH, **overrides) -> Assessment:
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
        hash=h,
        previous_hash=a.previous_hash,
        schema_version=a.schema_version,
        timestamp=a.timestamp,
        stream=a.stream,
        incident_id=a.incident_id,
        type=a.type,
        severity=a.severity,
        payload=a.payload,
        summaries=a.summaries,
    )


def _hash_verdict(previous_hash: str = ZERO_HASH, **overrides) -> Verdict:
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
        model=overrides.pop("model", "claude-sonnet-4-20250514"),
        reasoning=overrides.pop("reasoning", "High severity"),
        action=overrides.pop("action", {"type": "escalate"}),
        outcome=overrides.pop("outcome", VerdictOutcome.RECOMMENDED),
        summaries=overrides.pop("summaries", Summaries(technical="t", plain="p", executive="e")),
    )
    h = compute_hash(canonical_json(v))
    return Verdict(
        hash=h,
        previous_hash=v.previous_hash,
        schema_version=v.schema_version,
        timestamp=v.timestamp,
        agent=v.agent,
        incident_id=v.incident_id,
        input_hashes=v.input_hashes,
        prompt_hash=v.prompt_hash,
        response_hash=v.response_hash,
        model=v.model,
        reasoning=v.reasoning,
        action=v.action,
        outcome=v.outcome,
        summaries=v.summaries,
    )


def _hash_evaluation(previous_hash: str = ZERO_HASH, **overrides) -> Evaluation:
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
        hash=h,
        previous_hash=e.previous_hash,
        schema_version=e.schema_version,
        timestamp=e.timestamp,
        incident_id=e.incident_id,
        verdict_hash=e.verdict_hash,
        method=e.method,
        outcome=e.outcome,
        evidence_hashes=e.evidence_hashes,
        payload=e.payload,
        summaries=e.summaries,
    )


@pytest.fixture
def store(tmp_path):
    return SQLiteDecisionRecordStore(str(tmp_path / "decisions.db"))


# --- Chain Verification ---


class TestVerifyChain:
    def test_valid_single_record_chain(self, store):
        a = _hash_assessment()
        store.put_assessment(a)
        result = verify_chain(store, "assessment", "sli:checkout:latency-p99")
        assert result.verified is True
        assert result.record_count == 1
        assert result.errors == []

    def test_valid_multi_record_chain(self, store):
        a1 = _hash_assessment(timestamp=NOW)
        a2 = _hash_assessment(previous_hash=a1.hash, timestamp=T1, payload={"current_value": 999})
        a3 = _hash_assessment(previous_hash=a2.hash, timestamp=T2, payload={"current_value": 500})
        store.put_assessment(a1)
        store.put_assessment(a2)
        store.put_assessment(a3)
        result = verify_chain(store, "assessment", "sli:checkout:latency-p99")
        assert result.verified is True
        assert result.record_count == 3

    def test_tampered_hash_detected(self, store):
        a = _hash_assessment()
        # Tamper: replace hash with wrong value
        tampered = Assessment(
            hash="bad" + "0" * 61,
            previous_hash=a.previous_hash,
            schema_version=a.schema_version,
            timestamp=a.timestamp,
            stream=a.stream,
            incident_id=a.incident_id,
            type=a.type,
            severity=a.severity,
            payload=a.payload,
            summaries=a.summaries,
        )
        store.put_assessment(tampered)
        result = verify_chain(store, "assessment", "sli:checkout:latency-p99")
        assert result.verified is False
        assert any("hash mismatch" in e for e in result.errors)

    def test_broken_chain_link_rejected_at_store_level(self, store):
        """Two records with same previous_hash on same stream = fork, rejected by store."""
        import pytest

        a1 = _hash_assessment(timestamp=NOW)
        # a2 claims ZERO_HASH as previous — same as a1 = fork on same stream
        a2 = _hash_assessment(previous_hash=ZERO_HASH, timestamp=T1, payload={"current_value": 999})
        store.put_assessment(a1)
        from nthlayer_common.records.errors import ChainForkError
        with pytest.raises(ChainForkError, match="Chain fork"):
            store.put_assessment(a2)

    def test_empty_chain_is_valid(self, store):
        result = verify_chain(store, "assessment", "sli:nonexistent:stream")
        assert result.verified is True
        assert result.record_count == 0

    def test_first_record_must_link_to_zero_hash(self, store):
        a = _hash_assessment(previous_hash="bad" + "0" * 61)
        store.put_assessment(a)
        result = verify_chain(store, "assessment", "sli:checkout:latency-p99")
        assert result.verified is False
        assert any("genesis" in e for e in result.errors)

    def test_verdict_chain_verification(self, store):
        v1 = _hash_verdict(agent="triage", timestamp=NOW)
        v2 = _hash_verdict(agent="triage", previous_hash=v1.hash, timestamp=T1, reasoning="updated")
        store.put_verdict(v1)
        store.put_verdict(v2)
        result = verify_chain(store, "verdict", "triage")
        assert result.verified is True
        assert result.record_count == 2

    def test_evaluation_chain_verification(self, store):
        e = _hash_evaluation(incident_id="inc-001")
        store.put_evaluation(e)
        result = verify_chain(store, "evaluation", "inc-001")
        assert result.verified is True
        assert result.record_count == 1

    def test_backdated_record_breaks_chain(self, store):
        """A record with an earlier timestamp than its predecessor breaks chain linkage
        when the store returns by timestamp ASC."""
        a1 = _hash_assessment(timestamp=T1)
        a2 = _hash_assessment(previous_hash=a1.hash, timestamp=NOW, payload={"current_value": 999})
        store.put_assessment(a1)
        store.put_assessment(a2)
        result = verify_chain(store, "assessment", "sli:checkout:latency-p99")
        # Store returns [a2(NOW), a1(T1)] by timestamp — chain link is broken
        assert result.verified is False
        assert len(result.errors) > 0


# --- Incident Verification ---


class TestVerifyIncident:
    def test_valid_incident_with_all_record_types(self, store):
        a = _hash_assessment(incident_id="inc-001")
        store.put_assessment(a)

        v = _hash_verdict(incident_id="inc-001", input_hashes=[a.hash])
        store.put_verdict(v)
        store.put_prompt(v.prompt_hash, "prompt content")
        store.put_response(v.response_hash, "response content")

        e = _hash_evaluation(incident_id="inc-001", verdict_hash=v.hash, evidence_hashes=[a.hash])
        store.put_evaluation(e)

        inc = Incident(id="inc-001", created_at=NOW, trigger_hash=a.hash, stream="sli:checkout:latency-p99")
        store.create_incident(inc)

        result = verify_incident(store, "inc-001")
        assert result.verified is True
        assert result.assessment_count == 1
        assert result.verdict_count == 1
        assert result.evaluation_count == 1

    def test_missing_input_hash_detected(self, store):
        v = _hash_verdict(incident_id="inc-002", input_hashes=["nonexistent" + "0" * 53])
        store.put_verdict(v)
        store.put_prompt(v.prompt_hash, "prompt")
        store.put_response(v.response_hash, "response")

        inc = Incident(id="inc-002", created_at=NOW, trigger_hash="x" * 64, stream="sli:test")
        store.create_incident(inc)

        result = verify_incident(store, "inc-002")
        assert result.verified is False
        assert any("input_hash" in e for e in result.errors)

    def test_missing_prompt_detected(self, store):
        v = _hash_verdict(incident_id="inc-003")
        store.put_verdict(v)
        # Don't store prompt or response

        inc = Incident(id="inc-003", created_at=NOW, trigger_hash="x" * 64, stream="sli:test")
        store.create_incident(inc)

        result = verify_incident(store, "inc-003")
        assert result.verified is False
        assert any("prompt" in e for e in result.errors)

    def test_missing_evidence_hash_detected(self, store):
        e = _hash_evaluation(incident_id="inc-004", evidence_hashes=["nonexistent" + "0" * 53])
        store.put_evaluation(e)

        inc = Incident(id="inc-004", created_at=NOW, trigger_hash="x" * 64, stream="sli:test")
        store.create_incident(inc)

        result = verify_incident(store, "inc-004")
        assert result.verified is False
        assert any("evidence_hash" in e for e in result.errors)

    def test_missing_verdict_hash_on_evaluation_detected(self, store):
        e = _hash_evaluation(incident_id="inc-005", verdict_hash="nonexistent" + "0" * 53)
        store.put_evaluation(e)

        inc = Incident(id="inc-005", created_at=NOW, trigger_hash="x" * 64, stream="sli:test")
        store.create_incident(inc)

        result = verify_incident(store, "inc-005")
        assert result.verified is False
        assert any("verdict_hash" in e for e in result.errors)

    def test_nonexistent_incident(self, store):
        result = verify_incident(store, "nonexistent")
        assert result.verified is False
        assert any("not found" in e for e in result.errors)
