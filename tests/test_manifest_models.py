"""Tests for nthlayer_common.manifest.models."""

import pytest

from nthlayer_common.manifest.models import (
    JUDGMENT_SLO_TYPES,
    BreachAction,
    ContractPromise,
    Dependency,
    DependencySLO,
    FallbackDeclaration,
    JudgmentMeasurement,
    JudgmentPromise,
    Ownership,
    ReliabilityContract,
    ReliabilityManifest,
    SLODefinition,
    SourceFormat,
    StatisticalRequirements,
)

# =============================================================================
# SLODefinition
# =============================================================================


class TestSLODefinition:
    def test_classical_slo(self):
        slo = SLODefinition(name="availability", target=99.95, slo_type="availability")
        assert not slo.is_judgment_slo()
        assert slo.judgment_type is None

    def test_judgment_slo(self):
        slo = SLODefinition(
            name="rev_rate",
            target=0.05,
            slo_type="availability",
            judgment_type="reversal_rate",
        )
        assert slo.is_judgment_slo()
        assert slo.judgment_type == "reversal_rate"

    def test_judgment_slo_with_measurement(self):
        slo = SLODefinition(
            name="cal",
            target=0.05,
            slo_type="availability",
            judgment_type="calibration",
            measurement=JudgmentMeasurement(method="brier_score", bins=10),
            statistical_requirements=StatisticalRequirements(
                confidence_interval_pct=95.0, method="brier_score"
            ),
        )
        assert slo.is_judgment_slo()
        assert slo.measurement.method == "brier_score"
        assert slo.statistical_requirements.confidence_interval_pct == 95.0

    def test_judgment_slo_with_breach_actions(self):
        slo = SLODefinition(
            name="rev",
            target=0.05,
            slo_type="availability",
            judgment_type="reversal_rate",
            breach_actions=[
                BreachAction(action_type="notify", target="team:sre"),
                BreachAction(action_type="create_case", parameters={"priority": "P2"}),
            ],
        )
        assert len(slo.breach_actions) == 2
        assert slo.breach_actions[0].action_type == "notify"

    def test_openslo_provenance(self):
        slo = SLODefinition(
            name="latency",
            target=0.99,
            slo_type="latency",
            source_ref="./slos/latency.yaml",
        )
        assert slo.source_ref == "./slos/latency.yaml"

    def test_ratio_metric_queries(self):
        slo = SLODefinition(
            name="avail",
            target=0.999,
            slo_type="availability",
            total_query='rate(http_requests_total[5m])',
            good_query='rate(http_requests_total{status=~"2.."}[5m])',
            indicator_query='(rate(http_requests_total{status=~"2.."}[5m])) / (rate(http_requests_total[5m]))',
        )
        assert slo.total_query is not None
        assert slo.good_query is not None


class TestJudgmentSLOTypes:
    def test_eight_standard_types(self):
        assert len(JUDGMENT_SLO_TYPES) == 8
        expected = {
            "reversal_rate", "high_confidence_failure", "audit_sampling",
            "outcomes", "escalation", "segments", "stability", "calibration",
        }
        assert JUDGMENT_SLO_TYPES == expected


# =============================================================================
# ReliabilityManifest
# =============================================================================


class TestReliabilityManifest:
    def test_minimal_manifest(self):
        m = ReliabilityManifest(name="svc", team="t", tier="critical", type="api")
        assert m.name == "svc"
        assert m.namespace == "default"
        assert m.source_format == SourceFormat.SRM_V1
        assert m.contracts == []

    def test_type_alias_normalisation(self):
        m = ReliabilityManifest(name="svc", team="t", tier="critical", type="background-job")
        assert m.type == "worker"

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="Invalid tier"):
            ReliabilityManifest(name="svc", team="t", tier="invalid", type="api")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid type"):
            ReliabilityManifest(name="svc", team="t", tier="critical", type="invalid")

    def test_is_ai_gate(self):
        m = ReliabilityManifest(name="svc", team="t", tier="critical", type="ai-gate")
        assert m.is_ai_gate()

    def test_get_judgment_slos(self):
        m = ReliabilityManifest(
            name="svc", team="t", tier="critical", type="ai-gate",
            slos=[
                SLODefinition(name="avail", target=99.9, slo_type="availability"),
                SLODefinition(name="rev", target=0.05, slo_type="availability", judgment_type="reversal_rate"),
            ],
        )
        assert len(m.get_judgment_slos()) == 1
        assert len(m.get_standard_slos()) == 1

    def test_validate_contracts_availability(self):
        m = ReliabilityManifest(
            name="svc", team="t", tier="critical", type="api",
            slos=[SLODefinition(name="availability", target=99.5, slo_type="availability")],
            contracts=[ReliabilityContract(
                name="svc-api",
                promise=ContractPromise(availability=0.999),
            )],
        )
        errors = m.validate_contracts()
        assert len(errors) == 1
        assert "looser" in errors[0]

    def test_validate_contracts_judgment(self):
        m = ReliabilityManifest(
            name="svc", team="t", tier="critical", type="ai-gate",
            slos=[SLODefinition(
                name="rev", target=0.10, slo_type="availability",
                judgment_type="reversal_rate",
            )],
            contracts=[ReliabilityContract(
                name="svc-api",
                promise=ContractPromise(judgment=[
                    JudgmentPromise(judgment_type="reversal_rate", threshold=0.05, direction="below"),
                ]),
            )],
        )
        errors = m.validate_contracts()
        assert len(errors) == 1

    def test_source_format_enum_values(self):
        assert SourceFormat.SRM_V1.value == "srm_v1"
        assert SourceFormat.OPENSRM_V2.value == "opensrm_v2"
        assert SourceFormat.LEGACY.value == "legacy"


# =============================================================================
# Contract Models
# =============================================================================


class TestReliabilityContract:
    def test_contract_with_judgment_promises(self):
        c = ReliabilityContract(
            name="fraud-api",
            promise=ContractPromise(
                availability=0.999,
                judgment=[
                    JudgmentPromise(judgment_type="reversal_rate", threshold=0.05, direction="below"),
                ],
            ),
        )
        assert len(c.promise.judgment) == 1
        assert c.promise.judgment[0].direction == "below"


# =============================================================================
# Dependency Models
# =============================================================================


class TestDependency:
    def test_dependency_with_slo(self):
        d = Dependency(
            name="user-svc",
            type="api",
            slo=DependencySLO(availability=0.999, latency_p99="50ms"),
        )
        assert d.slo.availability == 0.999

    def test_dependency_with_fallback(self):
        d = Dependency(
            name="cache",
            type="cache",
            fallback=FallbackDeclaration(kind="graceful_degradation", description="skip cache"),
        )
        assert d.fallback.kind == "graceful_degradation"


# =============================================================================
# Ownership
# =============================================================================


class TestOwnership:
    def test_ownership_with_backstage_refs(self):
        o = Ownership(
            team="payments-ml",
            group_ref="group:default/payments-ml",
            escalation_ref="group:default/sre-on-call",
            technical_contact_ref="user:default/jane.doe",
        )
        assert o.group_ref is not None
        assert o.team == "payments-ml"
