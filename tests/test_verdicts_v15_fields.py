"""Tests for v1.5 transitional verdict fields.

Verifies pipeline_latency_ms, chain_depth, parent_ids, verdict_type, service
on the Verdict dataclass, and that they round-trip through serialisation.
"""

from datetime import datetime, UTC

from nthlayer_common.verdicts import (
    VALID_VERDICT_TYPES,
    Verdict,
    create,
    from_json,
    to_json,
)


class TestValidVerdictTypes:
    def test_all_rbac_types_present(self):
        rbac_types = {
            "action_request", "approval", "capability", "denial",
            "execution", "operator_note",
        }
        assert rbac_types.issubset(VALID_VERDICT_TYPES)

    def test_measure_types_present(self):
        assert "autonomy_change" in VALID_VERDICT_TYPES
        assert "quality_breach" in VALID_VERDICT_TYPES

    def test_correlate_types_are_assessments_not_verdicts(self):
        """correlation_snapshot, topology_drift, contract_divergence are observations,
        not judgment-bearing decisions. They live in ASSESSMENT_KINDS, not VALID_VERDICT_TYPES."""
        from nthlayer_common.cloudevents import ASSESSMENT_KINDS

        assert "topology_drift" in ASSESSMENT_KINDS
        assert "contract_divergence" in ASSESSMENT_KINDS
        assert "correlation_snapshot" in ASSESSMENT_KINDS
        assert "topology_drift" not in VALID_VERDICT_TYPES
        assert "contract_divergence" not in VALID_VERDICT_TYPES
        assert "correlation_snapshot" not in VALID_VERDICT_TYPES

    def test_outcome_resolution_present(self):
        assert "outcome_resolution" in VALID_VERDICT_TYPES

    def test_assessment_type_absent(self):
        # Removed in opensrm-saun.1.2 — verdicts are decisions, assessments
        # are continuous observations; the entry violated this distinction.
        # Things that are continuous and not decisions belong in
        # ASSESSMENT_KINDS, not VALID_VERDICT_TYPES.
        assert "assessment" not in VALID_VERDICT_TYPES

    def test_respond_agent_role_types_present(self):
        # Added in opensrm-saun.1.2 so respond agents can set
        # verdict_type=self.role.value, matching subject.type.
        for role in ("triage", "investigation", "communication", "remediation"):
            assert role in VALID_VERDICT_TYPES


class TestV15Fields:
    def test_create_verdict_with_v15_fields(self):
        v = create(
            subject={"type": "evaluation", "ref": "test", "summary": "test"},
            judgment={"action": "approve", "confidence": 0.9},
            producer={"system": "test"},
        )
        # v1.5 fields have defaults
        assert v.pipeline_latency_ms is None
        assert v.chain_depth == 0
        assert v.parent_ids == []
        assert v.verdict_type is None
        assert v.service is None

    def test_verdict_with_all_v15_fields(self):
        v = Verdict(
            id="vrd-test",
            version=1,
            timestamp=datetime.now(UTC),
            producer=type("P", (), {"system": "test", "instance": None, "model": None, "prompt_version": None})(),
            subject=type("S", (), {"type": "evaluation", "ref": "t", "summary": "t", "agent": None, "service": None, "environment": None, "content_hash": None})(),
            judgment=type("J", (), {"action": "approve", "confidence": 0.9, "score": None, "dimensions": None, "reasoning": None, "tags": None})(),
            verdict_type="action_request",
            pipeline_latency_ms=250,
            chain_depth=3,
            parent_ids=["vrd-parent-1", "vrd-parent-2"],
            service="payment-api",
        )
        assert v.verdict_type == "action_request"
        assert v.pipeline_latency_ms == 250
        assert v.chain_depth == 3
        assert v.parent_ids == ["vrd-parent-1", "vrd-parent-2"]
        assert v.service == "payment-api"


class TestV15Serialisation:
    def test_roundtrip_with_v15_fields(self):
        v = create(
            subject={"type": "evaluation", "ref": "test", "summary": "test"},
            judgment={"action": "approve", "confidence": 0.9},
            producer={"system": "test"},
        )
        v.verdict_type = "quality_breach"
        v.pipeline_latency_ms = 150
        v.chain_depth = 2
        v.parent_ids = ["vrd-parent"]
        v.service = "fraud-detect"

        json_str = to_json(v)
        v2 = from_json(json_str)

        assert v2.verdict_type == "quality_breach"
        assert v2.pipeline_latency_ms == 150
        assert v2.chain_depth == 2
        assert v2.parent_ids == ["vrd-parent"]
        assert v2.service == "fraud-detect"

    def test_backward_compat_without_v15_fields(self):
        """Old verdicts without v1.5 fields still deserialise correctly."""
        v = create(
            subject={"type": "evaluation", "ref": "test", "summary": "test"},
            judgment={"action": "approve", "confidence": 0.9},
            producer={"system": "test"},
        )
        json_str = to_json(v)
        v2 = from_json(json_str)

        # v1.5 fields fall back to defaults
        assert v2.pipeline_latency_ms is None
        assert v2.chain_depth == 0
        assert v2.parent_ids == []
        assert v2.verdict_type is None
        assert v2.service is None

    def test_each_verdict_type_serialises(self):
        """Every VALID_VERDICT_TYPES value can be set and round-tripped."""
        for vtype in VALID_VERDICT_TYPES:
            v = create(
                subject={"type": "evaluation", "ref": "test", "summary": "test"},
                judgment={"action": "approve", "confidence": 0.9},
                producer={"system": "test"},
            )
            v.verdict_type = vtype
            v2 = from_json(to_json(v))
            assert v2.verdict_type == vtype, f"Failed for type {vtype}"
