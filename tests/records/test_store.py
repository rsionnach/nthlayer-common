"""Tests for the DecisionRecordStore protocol and SQLite implementation."""

import threading
from datetime import UTC, datetime

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
    IncidentStatus,
    Severity,
    Summaries,
    Verdict,
    VerdictOutcome,
)
from nthlayer_common.records.sqlite_store import SQLiteDecisionRecordStore

NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
LATER = datetime(2026, 4, 11, 12, 5, 0, tzinfo=UTC)


# --- Helpers ---


def _build_assessment(previous_hash: str = ZERO_HASH, **overrides) -> Assessment:
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
    canonical = canonical_json(a)
    return Assessment(
        hash=compute_hash(canonical),
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


def _build_verdict(previous_hash: str = ZERO_HASH, **overrides) -> Verdict:
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
    canonical = canonical_json(v)
    return Verdict(
        hash=compute_hash(canonical),
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


def _build_evaluation(previous_hash: str = ZERO_HASH, **overrides) -> Evaluation:
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
    canonical = canonical_json(e)
    return Evaluation(
        hash=compute_hash(canonical),
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
    db_path = tmp_path / "decisions.db"
    return SQLiteDecisionRecordStore(str(db_path))


# --- Assessment CRUD ---


class TestAssessmentStore:
    def test_put_and_get_assessment(self, store):
        a = _build_assessment()
        store.put_assessment(a)
        retrieved = store.get_by_hash(a.hash)
        assert retrieved is not None
        assert isinstance(retrieved, Assessment)
        assert retrieved.hash == a.hash
        assert retrieved.stream == a.stream
        assert retrieved.payload == a.payload

    def test_duplicate_hash_is_idempotent(self, store):
        a = _build_assessment()
        store.put_assessment(a)
        store.put_assessment(a)  # should not raise
        retrieved = store.get_by_hash(a.hash)
        assert retrieved.hash == a.hash

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_by_hash("nonexistent") is None

    def test_get_chain_single_record(self, store):
        a = _build_assessment()
        store.put_assessment(a)
        chain = store.get_chain("assessment", "sli:checkout:latency-p99")
        assert len(chain) == 1
        assert chain[0].hash == a.hash

    def test_get_chain_ordered_by_timestamp(self, store):
        a1 = _build_assessment(timestamp=NOW)
        a2 = _build_assessment(previous_hash=a1.hash, timestamp=LATER, payload={"current_value": 999})
        store.put_assessment(a1)
        store.put_assessment(a2)
        chain = store.get_chain("assessment", "sli:checkout:latency-p99")
        assert len(chain) == 2
        assert chain[0].hash == a1.hash
        assert chain[1].hash == a2.hash

    def test_get_chain_filters_by_stream(self, store):
        a1 = _build_assessment(stream="sli:checkout:latency-p99")
        a2 = _build_assessment(stream="sli:other:error-rate", payload={"value": 0.05})
        store.put_assessment(a1)
        store.put_assessment(a2)
        chain = store.get_chain("assessment", "sli:checkout:latency-p99")
        assert len(chain) == 1
        assert chain[0].hash == a1.hash


# --- Verdict CRUD ---


class TestVerdictStore:
    def test_put_and_get_verdict(self, store):
        v = _build_verdict()
        store.put_verdict(v)
        retrieved = store.get_by_hash(v.hash)
        assert retrieved is not None
        assert isinstance(retrieved, Verdict)
        assert retrieved.agent == "triage"
        assert retrieved.input_hashes == []

    def test_get_chain_by_agent(self, store):
        v1 = _build_verdict(agent="triage")
        v2 = _build_verdict(agent="triage", previous_hash=v1.hash, timestamp=LATER, reasoning="updated")
        store.put_verdict(v1)
        store.put_verdict(v2)
        chain = store.get_chain("verdict", "triage")
        assert len(chain) == 2
        assert chain[0].hash == v1.hash
        assert chain[1].hash == v2.hash


# --- get_chain_tail ---


class TestGetChainTail:
    def test_returns_latest_assessment(self, store):
        a1 = _build_assessment(timestamp=NOW)
        a2 = _build_assessment(previous_hash=a1.hash, timestamp=LATER, payload={"current_value": 999})
        store.put_assessment(a1)
        store.put_assessment(a2)
        tail = store.get_chain_tail("assessment", "sli:checkout:latency-p99")
        assert tail is not None
        assert tail.hash == a2.hash

    def test_returns_none_for_empty_chain(self, store):
        assert store.get_chain_tail("assessment", "nonexistent") is None

    def test_get_chain_with_limit(self, store):
        a1 = _build_assessment(timestamp=NOW)
        a2 = _build_assessment(previous_hash=a1.hash, timestamp=LATER, payload={"current_value": 999})
        store.put_assessment(a1)
        store.put_assessment(a2)
        chain = store.get_chain("assessment", "sli:checkout:latency-p99", limit=1)
        assert len(chain) == 1
        assert chain[0].hash == a1.hash  # oldest first (ASC), limited to 1


# --- Close / Context Manager ---


class TestCloseAndContextManager:
    def test_context_manager(self, tmp_path):
        db = str(tmp_path / "cm.db")
        with SQLiteDecisionRecordStore(db) as store:
            a = _build_assessment()
            store.put_assessment(a)
        # After exiting context, store is closed but data is persisted
        store2 = SQLiteDecisionRecordStore(db)
        assert store2.get_by_hash(a.hash) is not None

    def test_close_method(self, tmp_path):
        db = str(tmp_path / "close.db")
        store = SQLiteDecisionRecordStore(db)
        a = _build_assessment()
        store.put_assessment(a)
        store.close()
        # Re-open and verify data
        store2 = SQLiteDecisionRecordStore(db)
        assert store2.get_by_hash(a.hash) is not None


# --- Evaluation CRUD ---


class TestEvaluationStore:
    def test_put_and_get_evaluation(self, store):
        e = _build_evaluation()
        store.put_evaluation(e)
        retrieved = store.get_by_hash(e.hash)
        assert retrieved is not None
        assert isinstance(retrieved, Evaluation)
        assert retrieved.method == EvaluationMethod.METRIC_RECOVERY

    def test_get_chain_by_incident(self, store):
        e = _build_evaluation(incident_id="inc-001")
        store.put_evaluation(e)
        chain = store.get_chain("evaluation", "inc-001")
        assert len(chain) == 1
        assert chain[0].hash == e.hash


# --- Incident CRUD ---


class TestIncidentStore:
    def test_create_and_get_incident(self, store):
        inc = Incident(
            id="inc-001",
            created_at=NOW,
            trigger_hash="a" * 64,
            stream="sli:checkout:latency-p99",
        )
        store.create_incident(inc)
        retrieved = store.get_incident("inc-001")
        assert retrieved is not None
        assert retrieved.id == "inc-001"
        assert retrieved.status == IncidentStatus.OPEN
        assert retrieved.trigger_hash == "a" * 64

    def test_update_incident_status(self, store):
        inc = Incident(
            id="inc-002",
            created_at=NOW,
            trigger_hash="a" * 64,
            stream="sli:checkout:latency-p99",
        )
        store.create_incident(inc)
        store.update_incident_status("inc-002", IncidentStatus.MITIGATED)
        retrieved = store.get_incident("inc-002")
        assert retrieved.status == IncidentStatus.MITIGATED

    def test_get_nonexistent_incident_returns_none(self, store):
        assert store.get_incident("nonexistent") is None

    def test_update_nonexistent_incident_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.update_incident_status("nonexistent", IncidentStatus.MITIGATED)

    def test_invalid_transition_raises(self, store):
        from nthlayer_common.records.errors import InvalidTransitionError
        inc = Incident(id="inc-trans", created_at=NOW, trigger_hash="a" * 64, stream="sli:test")
        store.create_incident(inc)
        store.update_incident_status("inc-trans", IncidentStatus.MITIGATED)
        # Cannot go back to OPEN from MITIGATED
        with pytest.raises(InvalidTransitionError):
            store.update_incident_status("inc-trans", IncidentStatus.OPEN)

    def test_valid_transition_chain(self, store):
        inc = Incident(id="inc-chain", created_at=NOW, trigger_hash="a" * 64, stream="sli:test")
        store.create_incident(inc)
        store.update_incident_status("inc-chain", IncidentStatus.MITIGATED)
        store.update_incident_status("inc-chain", IncidentStatus.RESOLVED)
        store.update_incident_status("inc-chain", IncidentStatus.LEARNING)
        store.update_incident_status("inc-chain", IncidentStatus.CLOSED)
        assert store.get_incident("inc-chain").status == IncidentStatus.CLOSED

    def test_get_incident_records(self, store):
        inc = Incident(
            id="inc-003",
            created_at=NOW,
            trigger_hash="trigger" + "0" * 58,
            stream="sli:checkout:latency-p99",
        )
        store.create_incident(inc)

        a = _build_assessment(incident_id="inc-003")
        v = _build_verdict(incident_id="inc-003")
        e = _build_evaluation(incident_id="inc-003")
        store.put_assessment(a)
        store.put_verdict(v)
        store.put_evaluation(e)

        records = store.get_incident_records("inc-003")
        assert len(records["assessments"]) == 1
        assert len(records["verdicts"]) == 1
        assert len(records["evaluations"]) == 1
        assert records["assessments"][0].hash == a.hash
        assert records["verdicts"][0].hash == v.hash
        assert records["evaluations"][0].hash == e.hash


# --- Prompt/Response Store ---


class TestPromptResponseStore:
    def test_put_and_get_prompt(self, store):
        store.put_prompt("p" * 64, "You are a triage agent...")
        content = store.get_prompt("p" * 64)
        assert content == "You are a triage agent..."

    def test_put_and_get_response(self, store):
        store.put_response("r" * 64, '{"action": "escalate"}')
        content = store.get_response("r" * 64)
        assert content == '{"action": "escalate"}'

    def test_get_nonexistent_prompt_returns_none(self, store):
        assert store.get_prompt("nonexistent") is None

    def test_get_nonexistent_response_returns_none(self, store):
        assert store.get_response("nonexistent") is None

    def test_duplicate_prompt_is_idempotent(self, store):
        store.put_prompt("p" * 64, "content")
        store.put_prompt("p" * 64, "content")  # should not raise
        assert store.get_prompt("p" * 64) == "content"


# --- Concurrency ---


class TestConcurrency:
    def test_concurrent_writes_to_different_streams(self, store):
        """Concurrent writes to separate streams should not interfere."""
        errors = []

        def write_assessments(stream_idx: int):
            try:
                prev = ZERO_HASH
                for i in range(5):
                    a = _build_assessment(
                        stream=f"sli:stream-{stream_idx}:metric",
                        previous_hash=prev,
                        payload={"idx": stream_idx * 100 + i},
                        timestamp=datetime(2026, 4, 11, 12, i, 0, tzinfo=UTC),
                    )
                    store.put_assessment(a)
                    prev = a.hash
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_assessments, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent write errors: {errors}"

    def test_chain_fork_rejected(self, store):
        """Two different records claiming the same previous_hash in the same stream are rejected."""
        a1 = _build_assessment()
        store.put_assessment(a1)
        # Second record with same previous_hash but different payload = fork
        a2 = _build_assessment(payload={"different": True})
        from nthlayer_common.records.errors import ChainForkError
        with pytest.raises(ChainForkError, match="Chain fork"):
            store.put_assessment(a2)
