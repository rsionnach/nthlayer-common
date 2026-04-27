"""Tests for CloudEvents envelope helpers."""

import pytest

from nthlayer_common.cloudevents import (
    _ce_type_for_assessment,
    _ce_type_for_verdict,
    parse_cloudevent,
    validate_cloudevent,
    wrap_assessment,
    wrap_verdict,
)


def _verdict(**overrides):
    base = {
        "id": "vrd-test-001",
        "type": "action_request",
        "service": "payment-api",
        "created_at": "2026-04-23T00:00:00Z",
    }
    base.update(overrides)
    return base


def _assessment(**overrides):
    base = {
        "id": "asm-test-001",
        "kind": "slo_status",
        "service": "payment-api",
        "created_at": "2026-04-23T00:00:00Z",
    }
    base.update(overrides)
    return base


class TestTypeMapping:
    def test_known_verdict_type(self):
        assert _ce_type_for_verdict("action_request") == "io.nthlayer.verdict.action_request.v1"

    def test_all_verdict_types_produce_valid_pattern(self):
        """Every type in _VERDICT_TYPES maps to a known CloudEvents type."""
        from nthlayer_common.cloudevents import _VERDICT_TYPES

        for vtype in _VERDICT_TYPES:
            result = _ce_type_for_verdict(vtype)
            assert result.startswith("io.nthlayer.verdict.")
            assert result.endswith(".v1")
            assert "unknown" not in result

    def test_unknown_verdict_type_maps_to_unknown(self):
        assert _ce_type_for_verdict("bogus") == "io.nthlayer.verdict.unknown.v1"

    def test_known_assessment_kind(self):
        assert _ce_type_for_assessment("slo_status") == "io.nthlayer.assessment.slo_status.v1"

    def test_unknown_assessment_kind_maps_to_unknown(self):
        assert _ce_type_for_assessment("bogus") == "io.nthlayer.assessment.unknown.v1"

    def test_correlate_types_are_assessment_kinds(self):
        """correlation_snapshot, topology_drift, contract_divergence are assessments."""
        for kind in ("correlation_snapshot", "topology_drift", "contract_divergence"):
            result = _ce_type_for_assessment(kind)
            assert result == f"io.nthlayer.assessment.{kind}.v1"
            # Not verdict types
            vresult = _ce_type_for_verdict(kind)
            assert "unknown" in vresult

    def test_assessment_kinds_public_constant(self):
        from nthlayer_common.cloudevents import ASSESSMENT_KINDS

        assert "slo_status" in ASSESSMENT_KINDS
        assert "correlation_snapshot" in ASSESSMENT_KINDS
        assert "topology_drift" in ASSESSMENT_KINDS
        assert "contract_divergence" in ASSESSMENT_KINDS


class TestWrapVerdict:
    def test_required_ce_attributes_present(self):
        ce = wrap_verdict(_verdict())
        assert ce["specversion"] == "1.0"
        assert ce["type"] == "io.nthlayer.verdict.action_request.v1"
        assert ce["source"].startswith("urn:nthlayer:")
        assert ce["id"] == "vrd-test-001"
        assert ce["time"] == "2026-04-23T00:00:00Z"
        assert ce["datacontenttype"] == "application/json"

    def test_data_contains_original_verdict(self):
        v = _verdict()
        ce = wrap_verdict(v)
        assert ce["data"] == v

    def test_subject_set_from_service(self):
        ce = wrap_verdict(_verdict(service="fraud-detect"))
        assert ce["subject"] == "component:default/fraud-detect"

    def test_no_subject_when_no_service(self):
        v = _verdict()
        del v["service"]
        ce = wrap_verdict(v)
        assert "subject" not in ce

    def test_custom_component(self):
        ce = wrap_verdict(_verdict(), component="respond")
        assert "urn:nthlayer:respond:" in ce["source"]

    def test_custom_deployment_id(self):
        ce = wrap_verdict(_verdict(), deployment_id="eu-west-prod")
        assert ce["source"].endswith("eu-west-prod")

    def test_envelope_is_json_serializable(self):
        import json
        ce = wrap_verdict(_verdict())
        serialized = json.dumps(ce)
        assert isinstance(serialized, str)


class TestWrapAssessment:
    def test_required_ce_attributes_present(self):
        ce = wrap_assessment(_assessment())
        assert ce["specversion"] == "1.0"
        assert ce["type"] == "io.nthlayer.assessment.slo_status.v1"
        assert ce["id"] == "asm-test-001"
        assert ce["datacontenttype"] == "application/json"

    def test_data_contains_original_assessment(self):
        a = _assessment()
        ce = wrap_assessment(a)
        assert ce["data"] == a


class TestParseCloudEvent:
    def test_extracts_data(self):
        ce = wrap_verdict(_verdict())
        data = parse_cloudevent(ce)
        assert data["id"] == "vrd-test-001"
        assert data["type"] == "action_request"

    def test_roundtrip_verdict(self):
        v = _verdict()
        ce = wrap_verdict(v)
        data = parse_cloudevent(ce)
        assert data == v

    def test_roundtrip_assessment(self):
        a = _assessment()
        ce = wrap_assessment(a)
        data = parse_cloudevent(ce)
        assert data == a

    def test_missing_required_raises(self):
        with pytest.raises(ValueError, match="missing required attributes"):
            parse_cloudevent({"data": {}})

    def test_wrong_specversion_raises(self):
        ce = wrap_verdict(_verdict())
        ce["specversion"] = "0.3"
        with pytest.raises(ValueError, match="Unsupported"):
            parse_cloudevent(ce)

    def test_missing_data_returns_empty_dict(self):
        ce = {"specversion": "1.0", "type": "io.nthlayer.verdict.test.v1", "source": "urn:nthlayer:test:x", "id": "1"}
        data = parse_cloudevent(ce)
        assert data == {}


class TestValidateCloudEvent:
    def test_valid_returns_empty(self):
        ce = wrap_verdict(_verdict())
        issues = validate_cloudevent(ce)
        assert issues == []

    def test_missing_attributes(self):
        issues = validate_cloudevent({"data": {}})
        assert len(issues) >= 4  # specversion, type, source, id

    def test_bad_type_pattern(self):
        ce = wrap_verdict(_verdict())
        ce["type"] = "com.example.custom"
        issues = validate_cloudevent(ce)
        assert any("io.nthlayer" in i for i in issues)

    def test_bad_source_pattern(self):
        ce = wrap_verdict(_verdict())
        ce["source"] = "https://example.com"
        issues = validate_cloudevent(ce)
        assert any("urn:nthlayer" in i for i in issues)

    def test_bad_content_type(self):
        ce = wrap_verdict(_verdict())
        ce["datacontenttype"] = "text/xml"
        issues = validate_cloudevent(ce)
        assert any("datacontenttype" in i for i in issues)

    def test_non_dict_returns_issue(self):
        issues = validate_cloudevent("not a dict")
        assert len(issues) == 1
        assert "Expected dict" in issues[0]

    def test_none_returns_issue(self):
        issues = validate_cloudevent(None)
        assert len(issues) == 1


class TestEdgeCases:
    def test_verdict_missing_id_raises(self):
        v = {"type": "action_request", "service": "test"}
        with pytest.raises(ValueError, match="must have an 'id'"):
            wrap_verdict(v)

    def test_assessment_missing_id_raises(self):
        a = {"kind": "slo_status", "service": "test"}
        with pytest.raises(ValueError, match="must have an 'id'"):
            wrap_assessment(a)

    def test_parse_non_dict_raises(self):
        with pytest.raises(ValueError, match="Expected dict"):
            parse_cloudevent("not a dict")

    def test_parse_none_raises(self):
        with pytest.raises(ValueError, match="Expected dict"):
            parse_cloudevent(None)
