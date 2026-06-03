"""Tests for verdict serialisation."""

import json

import pytest

from nthlayer_common.verdicts.core import create, resolve
from nthlayer_common.verdicts.serialise import from_dict, from_json, to_dict, to_json


def _make_verdict(**overrides):
    kwargs = dict(
        subject={"type": "review", "ref": "git:abc123", "summary": "Test change"},
        judgment={"action": "approve", "confidence": 0.8, "score": 0.75},
        producer={"system": "test-producer", "model": "test-model"},
    )
    kwargs.update(overrides)
    return create(**kwargs)


class TestRoundTrip:
    def test_json_roundtrip_pending(self):
        v = _make_verdict()
        json_str = to_json(v)
        v2 = from_json(json_str)
        assert v2.id == v.id
        assert v2.version == v.version
        assert v2.producer.system == v.producer.system
        assert v2.subject.type == v.subject.type
        assert v2.judgment.action == v.judgment.action
        assert v2.judgment.confidence == v.judgment.confidence
        assert v2.judgment.score == v.judgment.score
        assert v2.outcome.status == "pending"

    def test_json_roundtrip_resolved(self):
        v = _make_verdict()
        v = resolve(v, status="overridden",
                    override={"by": "human:rob", "action": "reject"},
                    ground_truth={"signal": "test_failure", "value": "reject"},
                    resolution="Bug found in prod")
        json_str = to_json(v)
        v2 = from_json(json_str)
        assert v2.outcome.status == "overridden"
        assert v2.outcome.override.by == "human:rob"
        assert v2.outcome.ground_truth.signal == "test_failure"
        assert v2.outcome.resolution == "Bug found in prod"
        assert v2.outcome.closed_at is not None

    def test_dict_roundtrip(self):
        v = _make_verdict()
        d = to_dict(v)
        v2 = from_dict(d)
        assert v2.id == v.id
        assert v2.metadata.ttl == v.metadata.ttl

    def test_roundtrip_with_lineage(self):
        v = _make_verdict()
        v.lineage.parent = "vrd-parent-001"
        v.lineage.children = ["vrd-child-001"]
        v.lineage.context = ["vrd-ctx-001"]
        v2 = from_json(to_json(v))
        assert v2.lineage.parent == "vrd-parent-001"
        assert v2.lineage.children == ["vrd-child-001"]
        assert v2.lineage.context == ["vrd-ctx-001"]

    def test_roundtrip_with_metadata(self):
        v = _make_verdict(metadata={"cost_tokens": 500, "latency_ms": 100, "custom": {"env": "prod"}})
        v2 = from_json(to_json(v))
        assert v2.metadata.cost_tokens == 500
        assert v2.metadata.custom == {"env": "prod"}


class TestFromDictValidation:
    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError, match="Missing required field: 'id'"):
            from_dict({})

    def test_missing_timestamp_raises(self):
        # Error message references the canonical "created_at" name.
        # from_dict still accepts legacy "timestamp" payloads (see
        # test_legacy_timestamp_field_accepted) — the message is just
        # forward-looking.
        with pytest.raises(ValueError, match="Missing required field: 'created_at'"):
            from_dict({"id": "vrd-1", "version": 1})

    def test_legacy_timestamp_field_accepted(self):
        # Pre-opensrm-saun.1.2 stored data used "timestamp" (matching the
        # internal Verdict dataclass field name). Round-trip compat is
        # required: from_dict must still accept it.
        v = from_dict({
            "id": "vrd-legacy-1",
            "version": 1,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "producer": {"system": "test"},
            "subject": {"type": "review", "ref": "r", "summary": "s"},
            "judgment": {"action": "approve", "confidence": 0.5},
        })
        assert v.timestamp.year == 2026

    def test_explicit_null_created_at_falls_back_to_legacy_timestamp(self):
        # opensrm-saun.1.2 round-trip safety: a payload with both fields
        # where created_at is explicitly null (some serialisers write
        # null instead of omitting the key) should fall back to the
        # legacy timestamp rather than treating null as authoritative.
        v = from_dict({
            "id": "vrd-mixed-1",
            "version": 1,
            "created_at": None,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "producer": {"system": "test"},
            "subject": {"type": "review", "ref": "r", "summary": "s"},
            "judgment": {"action": "approve", "confidence": 0.5},
        })
        assert v.timestamp.year == 2026

    def test_legacy_verdict_type_field_accepted(self):
        # to_dict now emits "type"; from_dict accepts both for legacy data.
        v = from_dict({
            "id": "vrd-legacy-2",
            "version": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "verdict_type": "quality_breach",
            "producer": {"system": "test"},
            "subject": {"type": "review", "ref": "r", "summary": "s"},
            "judgment": {"action": "approve", "confidence": 0.5},
        })
        assert v.verdict_type == "quality_breach"

    def test_null_timestamp_raises(self):
        with pytest.raises(ValueError, match="must not be null"):
            from_dict({
                "id": "vrd-1", "version": 1, "timestamp": None,
                "producer": {"system": "test"},
                "subject": {"type": "review", "ref": "r", "summary": "s"},
                "judgment": {"action": "approve", "confidence": 0.5},
            })

    def test_unsupported_version_raises(self):
        with pytest.raises(ValueError, match="Unsupported schema version 99"):
            from_dict({
                "id": "vrd-1", "version": 99, "timestamp": "2026-01-01T00:00:00Z",
                "producer": {"system": "test"},
                "subject": {"type": "review", "ref": "r", "summary": "s"},
                "judgment": {"action": "approve", "confidence": 0.5},
            })

    def test_malformed_datetime_raises(self):
        with pytest.raises(ValueError, match="Invalid datetime in 'created_at'"):
            from_dict({
                "id": "vrd-1", "version": 1, "created_at": "not-a-date",
                "producer": {"system": "test"},
                "subject": {"type": "review", "ref": "r", "summary": "s"},
                "judgment": {"action": "approve", "confidence": 0.5},
            })

    def test_empty_string_datetime_raises(self):
        with pytest.raises(ValueError, match="Invalid datetime in 'created_at'"):
            from_dict({
                "id": "vrd-1", "version": 1, "created_at": "",
                "producer": {"system": "test"},
                "subject": {"type": "review", "ref": "r", "summary": "s"},
                "judgment": {"action": "approve", "confidence": 0.5},
            })

    def test_optional_sections_default_gracefully(self):
        v = from_dict({
            "id": "vrd-1", "version": 1, "timestamp": "2026-01-01T00:00:00+00:00",
            "producer": {"system": "test"},
            "subject": {"type": "review", "ref": "r", "summary": "s"},
            "judgment": {"action": "approve", "confidence": 0.5},
        })
        assert v.outcome.status == "pending"
        assert v.lineage.parent is None
        assert v.metadata.ttl == 90 * 24 * 60 * 60

    def test_from_json_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            from_json("")
