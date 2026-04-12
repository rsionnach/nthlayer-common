"""Tests for canonical serialization and content-addressed hashing."""

import hashlib
import json
from datetime import datetime, timezone


from nthlayer_common.records.hashing import (
    canonical_json,
    compute_hash,
    verify_hash,
)
from nthlayer_common.records.models import (
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

ZERO_HASH = "0" * 64
NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)


def _make_assessment(**overrides) -> Assessment:
    defaults = dict(
        hash="placeholder",
        previous_hash=ZERO_HASH,
        schema_version="assessment/v1",
        timestamp=NOW,
        stream="sli:checkout:latency-p99",
        incident_id=None,
        type=AssessmentType.THRESHOLD_BREACH,
        severity=Severity.CRITICAL,
        payload={"current_value": 1247, "threshold": 500},
        summaries=Summaries(technical="breach", plain="slow", executive="bad"),
    )
    defaults.update(overrides)
    return Assessment(**defaults)


def _make_verdict(**overrides) -> Verdict:
    defaults = dict(
        hash="placeholder",
        previous_hash=ZERO_HASH,
        schema_version="verdict/v1",
        timestamp=NOW,
        agent="triage",
        incident_id="inc-4821",
        input_hashes=["b" * 64],
        prompt_hash="d" * 64,
        response_hash="e" * 64,
        model="claude-sonnet-4-20250514",
        reasoning="High severity",
        action={"type": "escalate"},
        outcome=VerdictOutcome.RECOMMENDED,
        summaries=Summaries(technical="t", plain="p", executive="e"),
    )
    defaults.update(overrides)
    return Verdict(**defaults)


def _make_evaluation(**overrides) -> Evaluation:
    defaults = dict(
        hash="placeholder",
        previous_hash=ZERO_HASH,
        schema_version="evaluation/v1",
        timestamp=NOW,
        incident_id="inc-4821",
        verdict_hash="b" * 64,
        method=EvaluationMethod.METRIC_RECOVERY,
        outcome=EvaluationOutcome.EFFECTIVE,
        evidence_hashes=["c" * 64],
        payload={"recovery_time_seconds": 120},
        summaries=Summaries(technical="recovered", plain="ok"),
    )
    defaults.update(overrides)
    return Evaluation(**defaults)


# --- canonical_json ---


class TestCanonicalJson:
    def test_excludes_hash_and_previous_hash_from_assessment(self):
        a = _make_assessment()
        canonical = canonical_json(a)
        parsed = json.loads(canonical)
        assert "hash" not in parsed
        assert "previous_hash" not in parsed

    def test_excludes_hash_and_previous_hash_from_verdict(self):
        v = _make_verdict()
        canonical = canonical_json(v)
        parsed = json.loads(canonical)
        assert "hash" not in parsed
        assert "previous_hash" not in parsed

    def test_excludes_hash_and_previous_hash_from_evaluation(self):
        e = _make_evaluation()
        canonical = canonical_json(e)
        parsed = json.loads(canonical)
        assert "hash" not in parsed
        assert "previous_hash" not in parsed

    def test_sorted_keys(self):
        a = _make_assessment()
        canonical = canonical_json(a)
        parsed = json.loads(canonical)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_no_formatting_whitespace(self):
        """Verify separators are compact (no spaces after : or ,) even with space-containing values."""
        a = _make_assessment(
            summaries=Summaries(technical="latency p99 breached 500ms", plain="service is slow", executive="SLO breach"),
        )
        canonical = canonical_json(a)
        # Re-serialize with same settings — should be byte-identical
        parsed = json.loads(canonical)
        reserialized = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        assert canonical == reserialized

    def test_returns_bytes(self):
        a = _make_assessment()
        canonical = canonical_json(a)
        assert isinstance(canonical, bytes)

    def test_deterministic_across_calls(self):
        a = _make_assessment()
        assert canonical_json(a) == canonical_json(a)

    def test_different_hash_values_produce_same_canonical(self):
        """hash/previous_hash are excluded, so changing them shouldn't affect canonical."""
        a1 = _make_assessment(hash="a" * 64, previous_hash="b" * 64)
        a2 = _make_assessment(hash="c" * 64, previous_hash="d" * 64)
        assert canonical_json(a1) == canonical_json(a2)

    def test_different_payload_produces_different_canonical(self):
        a1 = _make_assessment(payload={"value": 1})
        a2 = _make_assessment(payload={"value": 2})
        assert canonical_json(a1) != canonical_json(a2)

    def test_nested_payload_keys_sorted(self):
        a = _make_assessment(payload={"z_key": 1, "a_key": 2})
        canonical = canonical_json(a)
        parsed = json.loads(canonical)
        payload_keys = list(parsed["payload"].keys())
        assert payload_keys == sorted(payload_keys)

    def test_datetime_serialized_as_iso8601(self):
        a = _make_assessment()
        canonical = canonical_json(a)
        parsed = json.loads(canonical)
        assert parsed["timestamp"] == "2026-04-11T12:00:00+00:00"

    def test_enum_serialized_as_value(self):
        a = _make_assessment()
        canonical = canonical_json(a)
        parsed = json.loads(canonical)
        assert parsed["type"] == "threshold_breach"
        assert parsed["severity"] == "critical"

    def test_summaries_serialized_as_dict(self):
        a = _make_assessment()
        canonical = canonical_json(a)
        parsed = json.loads(canonical)
        assert isinstance(parsed["summaries"], dict)
        assert parsed["summaries"]["technical"] == "breach"

    def test_none_incident_id_included(self):
        a = _make_assessment(incident_id=None)
        canonical = canonical_json(a)
        parsed = json.loads(canonical)
        assert "incident_id" in parsed
        assert parsed["incident_id"] is None

    def test_list_fields_preserved(self):
        v = _make_verdict(input_hashes=["a" * 64, "b" * 64])
        canonical = canonical_json(v)
        parsed = json.loads(canonical)
        assert parsed["input_hashes"] == ["a" * 64, "b" * 64]


# --- compute_hash ---


class TestComputeHash:
    def test_returns_64_char_hex_string(self):
        a = _make_assessment()
        canonical = canonical_json(a)
        h = compute_hash(canonical)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_matches_manual_sha256(self):
        a = _make_assessment()
        canonical = canonical_json(a)
        expected = hashlib.sha256(canonical).hexdigest()
        assert compute_hash(canonical) == expected

    def test_different_content_different_hash(self):
        a1 = _make_assessment(payload={"value": 1})
        a2 = _make_assessment(payload={"value": 2})
        h1 = compute_hash(canonical_json(a1))
        h2 = compute_hash(canonical_json(a2))
        assert h1 != h2

    def test_same_content_same_hash(self):
        a = _make_assessment()
        h1 = compute_hash(canonical_json(a))
        h2 = compute_hash(canonical_json(a))
        assert h1 == h2


# --- verify_hash ---


class TestVerifyHash:
    def test_valid_assessment_hash(self):
        a = _make_assessment()
        canonical = canonical_json(a)
        correct_hash = compute_hash(canonical)
        a_with_hash = _make_assessment(hash=correct_hash)
        assert verify_hash(a_with_hash) is True

    def test_invalid_assessment_hash(self):
        a = _make_assessment(hash="bad" + "0" * 61)
        assert verify_hash(a) is False

    def test_valid_verdict_hash(self):
        v = _make_verdict()
        canonical = canonical_json(v)
        correct_hash = compute_hash(canonical)
        v_with_hash = _make_verdict(hash=correct_hash)
        assert verify_hash(v_with_hash) is True

    def test_valid_evaluation_hash(self):
        e = _make_evaluation()
        canonical = canonical_json(e)
        correct_hash = compute_hash(canonical)
        e_with_hash = _make_evaluation(hash=correct_hash)
        assert verify_hash(e_with_hash) is True

    def test_tampered_payload_fails_verification(self):
        a = _make_assessment(payload={"value": 1})
        canonical = canonical_json(a)
        correct_hash = compute_hash(canonical)
        tampered = _make_assessment(hash=correct_hash, payload={"value": 999})
        assert verify_hash(tampered) is False
