"""Tests for manifest parsers (v1, v2, OpenSLO, loader)."""

from pathlib import Path

import pytest
import yaml

from nthlayer_common.manifest import (
    Dependency,
    ManifestLoadError,
    OpenSRMParseError,
    OpenSRMV2ParseError,
    ReliabilityManifest,
    extract_declared_dependencies,
    load_manifest,
)
from nthlayer_common.manifest.models import SourceFormat
from nthlayer_common.manifest.openslo.parser import OpenSLOParseError as OpenSLOError
from nthlayer_common.manifest.openslo.parser import parse_openslo_slos
from nthlayer_common.manifest.parser.v1 import parse_srm_v1
from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def v1_manifest_data():
    return {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {
            "name": "payment-api",
            "team": "payments",
            "tier": "critical",
        },
        "spec": {
            "type": "api",
            "slos": {
                "availability": {
                    "target": 99.99,
                    "window": "30d",
                    "indicator": {
                        "type": "availability",
                        "query": 'sum(rate(http_requests_total{status=~"2.."}[5m])) / sum(rate(http_requests_total[5m]))',
                    },
                },
            },
            "dependencies": [
                {"name": "fraud-detect", "type": "ai-gate", "critical": True},
            ],
        },
    }


@pytest.fixture
def v1_ai_gate_data():
    return {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {
            "name": "fraud-detect",
            "team": "payments-ml",
            "tier": "critical",
        },
        "spec": {
            "type": "ai-gate",
            "slos": {
                "reversal_rate": {"target": 0.015, "window": "2m"},
                "availability": {"target": 99.9, "window": "30d"},
            },
        },
    }


@pytest.fixture
def v2_manifest_data():
    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {
            "name": "payment-service",
            "namespace": "payments",
            "labels": {"tier": "critical", "type": "api"},
        },
        "spec": {
            "owner": {"group": "group:default/sre-payments"},
            "service": {"name": "payment-service", "type": "api", "description": "Payments"},
            "slo": [
                {
                    "apiVersion": "openslo/v1",
                    "kind": "SLO",
                    "metadata": {"name": "payment-availability"},
                    "spec": {
                        "indicator": {
                            "metadata": {"name": "availability"},
                            "spec": {
                                "ratioMetric": {
                                    "total": {"metricSource": {"type": "Prometheus", "spec": {"query": "rate(http_requests_total[5m])"}}},
                                    "good": {"metricSource": {"type": "Prometheus", "spec": {"query": 'rate(http_requests_total{status=~"2.."}[5m])'}}},
                                }
                            },
                        },
                        "objectives": [{"target": 0.9999}],
                    },
                }
            ],
            "contracts": [
                {
                    "name": "payment-api",
                    "promise": {"availability": 0.999},
                }
            ],
            "dependencies": [
                {
                    "service": "component:default/fraud-detect",
                    "expected_availability": 0.995,
                    "expected_latency_p99": "100ms",
                    "fallback": {
                        "type": "graceful_degradation",
                        "description": "Proceed with reduced fraud confidence",
                    },
                }
            ],
        },
    }


@pytest.fixture
def v2_ai_gate_data():
    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {
            "name": "fraud-detect",
            "labels": {"tier": "critical"},
        },
        "spec": {
            "owner": {"group": "group:default/payments-ml"},
            "service": {"name": "fraud-detect", "type": "ai-gate"},
            "judgment_slo": [
                {
                    "metadata": {"name": "fraud-reversal-rate"},
                    "spec": {
                        "judgment_type": "reversal_rate",
                        "measurement": {"window": "7d", "source": "lineage"},
                        "target": {"maximum_reversal_rate": 0.05},
                        "breach_actions": [
                            {"notify": "group:default/sre-payments"},
                            {"create_case": {"priority": "P2"}},
                            {"reduce_autonomy": {"agent": "fraud-agent", "new_autonomy_level": "advisor"}},
                        ],
                    },
                }
            ],
        },
    }


# =============================================================================
# V1 Parser
# =============================================================================


class TestV1Parser:
    def test_parse_basic(self, v1_manifest_data):
        m = parse_srm_v1(v1_manifest_data)
        assert m.name == "payment-api"
        assert m.team == "payments"
        assert m.tier == "critical"
        assert m.type == "api"
        assert m.source_format == SourceFormat.SRM_V1

    def test_slo_type_inferred(self, v1_manifest_data):
        m = parse_srm_v1(v1_manifest_data)
        assert m.slos[0].slo_type == "availability"

    def test_judgment_slo_detection(self, v1_ai_gate_data):
        m = parse_srm_v1(v1_ai_gate_data)
        judgment = m.get_judgment_slos()
        assert len(judgment) == 1
        assert judgment[0].judgment_type == "reversal_rate"
        assert judgment[0].measurement is not None
        assert judgment[0].measurement.source == "lineage"
        assert judgment[0].statistical_requirements is not None
        assert judgment[0].statistical_requirements.confidence_interval_pct == 95.0

    def test_dependencies_parsed(self, v1_manifest_data):
        m = parse_srm_v1(v1_manifest_data)
        assert len(m.dependencies) == 1
        assert m.dependencies[0].name == "fraud-detect"
        assert m.dependencies[0].critical is True

    def test_contract_conversion(self):
        data = {
            "apiVersion": "srm/v1",
            "kind": "ServiceReliabilityManifest",
            "metadata": {"name": "svc", "team": "t", "tier": "critical"},
            "spec": {
                "type": "api",
                "slos": {"availability": {"target": 99.9}},
                "contract": {
                    "availability": 0.999,
                    "latency": {"p99": "500ms"},
                    "judgment": {"reversal_rate": 0.05},
                },
            },
        }
        m = parse_srm_v1(data)
        assert len(m.contracts) == 1
        assert m.contracts[0].name == "svc-api"
        assert m.contracts[0].promise.availability == 0.999
        assert m.contracts[0].promise.latency_p99 == "500ms"
        assert len(m.contracts[0].promise.judgment) == 1
        assert m.contracts[0].promise.judgment[0].direction == "below"

    def test_missing_name_raises(self):
        data = {
            "apiVersion": "srm/v1",
            "kind": "ServiceReliabilityManifest",
            "metadata": {"team": "t", "tier": "critical"},
            "spec": {"type": "api"},
        }
        with pytest.raises(OpenSRMParseError, match="metadata.name"):
            parse_srm_v1(data)

    def test_slo_type_inference_fails_loudly(self):
        data = {
            "apiVersion": "srm/v1",
            "kind": "ServiceReliabilityManifest",
            "metadata": {"name": "svc", "team": "t", "tier": "critical"},
            "spec": {
                "type": "api",
                "slos": {"mysterious_metric": {"target": 42.0}},
            },
        }
        with pytest.raises(OpenSRMParseError, match="Cannot infer slo_type"):
            parse_srm_v1(data)


# =============================================================================
# V2 Parser
# =============================================================================


class TestV2Parser:
    def test_parse_basic(self, v2_manifest_data):
        m = parse_opensrm_v2(v2_manifest_data)
        assert m.name == "payment-service"
        assert m.namespace == "payments"
        assert m.team == "sre-payments"
        assert m.tier == "critical"
        assert m.type == "api"
        assert m.source_format == SourceFormat.OPENSRM_V2

    def test_ownership_backstage_refs(self, v2_manifest_data):
        m = parse_opensrm_v2(v2_manifest_data)
        assert m.ownership.group_ref == "group:default/sre-payments"

    def test_openslo_translation(self, v2_manifest_data):
        m = parse_opensrm_v2(v2_manifest_data)
        assert len(m.slos) == 1
        slo = m.slos[0]
        assert slo.name == "payment-availability"
        assert slo.slo_type == "availability"
        assert slo.target == 0.9999
        assert slo.total_query is not None
        assert slo.good_query is not None

    def test_judgment_slo_parsing(self, v2_ai_gate_data):
        m = parse_opensrm_v2(v2_ai_gate_data)
        judgment = m.get_judgment_slos()
        assert len(judgment) == 1
        j = judgment[0]
        assert j.judgment_type == "reversal_rate"
        assert j.target == 0.05
        assert j.measurement.source == "lineage"
        assert j.measurement.window == "7d"
        assert len(j.breach_actions) == 3
        assert j.breach_actions[0].action_type == "notify"
        assert j.breach_actions[1].action_type == "create_case"
        assert j.breach_actions[2].action_type == "reduce_autonomy"

    def test_service_type_inferred_from_judgment_slo(self, v2_ai_gate_data):
        m = parse_opensrm_v2(v2_ai_gate_data)
        assert m.type == "ai-gate"

    def test_contracts_parsed(self, v2_manifest_data):
        m = parse_opensrm_v2(v2_manifest_data)
        assert len(m.contracts) == 1
        assert m.contracts[0].name == "payment-api"
        assert m.contracts[0].promise.availability == 0.999

    def test_dependencies_with_fallback(self, v2_manifest_data):
        m = parse_opensrm_v2(v2_manifest_data)
        assert len(m.dependencies) == 1
        d = m.dependencies[0]
        assert d.name == "fraud-detect"
        assert d.service_ref == "component:default/fraud-detect"
        assert d.slo.availability == 0.995
        assert d.slo.latency_p99 == "100ms"
        assert d.fallback.kind == "graceful_degradation"

    def test_missing_owner_raises(self):
        data = {
            "apiVersion": "opensrm.nthlayer.io/v2",
            "kind": "ServiceManifest",
            "metadata": {"name": "svc", "labels": {"tier": "critical", "type": "api"}},
            "spec": {"service": {"name": "svc", "type": "api"}},
        }
        with pytest.raises(OpenSRMV2ParseError, match="spec.owner"):
            parse_opensrm_v2(data)

    def test_missing_tier_raises(self):
        data = {
            "apiVersion": "opensrm.nthlayer.io/v2",
            "kind": "ServiceManifest",
            "metadata": {"name": "svc", "labels": {}},
            "spec": {
                "owner": {"group": "group:default/team"},
                "service": {"name": "svc", "type": "api"},
            },
        }
        with pytest.raises(OpenSRMV2ParseError, match="metadata.labels.tier"):
            parse_opensrm_v2(data)

    def test_missing_service_type_raises(self):
        """Was ``test_type_inference_fails_without_signals`` pre-opensrm-ih0v.

        The parser no longer infers a type from any signal, so the "without
        signals" framing no longer means anything: the field is required
        outright, and its absence is the only case left.
        """
        data = {
            "apiVersion": "opensrm.nthlayer.io/v2",
            "kind": "ServiceManifest",
            "metadata": {"name": "svc", "labels": {"tier": "critical"}},
            "spec": {
                "owner": {"group": "group:default/team"},
                "service": {"name": "svc"},
            },
        }
        with pytest.raises(OpenSRMV2ParseError, match=r"spec\.service\.type is required"):
            parse_opensrm_v2(data)

    def test_invalid_judgment_type_raises(self):
        data = {
            "apiVersion": "opensrm.nthlayer.io/v2",
            "kind": "ServiceManifest",
            "metadata": {"name": "svc", "labels": {"tier": "critical"}},
            "spec": {
                "owner": {"group": "group:default/team"},
                "service": {"name": "svc", "type": "ai-gate"},
                "judgment_slo": [
                    {
                        "metadata": {"name": "bad"},
                        "spec": {
                            "judgment_type": "nonexistent_type",
                            "target": {"something": 0.1},
                        },
                    }
                ],
            },
        }
        with pytest.raises(OpenSRMV2ParseError, match="Unknown judgment_type"):
            parse_opensrm_v2(data)


# All 8 judgment SLO types declared in OPENSRM-CORE-v2 §5.2 (opensrm-b22.1
# acceptance criterion: "All 8 judgment SLO types parseable"). Each type
# has a distinct target field name; the v2 parser maps them via
# _extract_judgment_target's target_fields dict.
_JUDGMENT_TYPE_TARGETS = [
    ("reversal_rate", "maximum_reversal_rate", 0.05),
    ("high_confidence_failure", "maximum_failure_rate", 0.01),
    ("audit_sampling", "audit_completion_rate", 0.95),
    ("outcomes", "desired_outcome_rate", 0.90),
    ("escalation", "maximum_escalation_rate", 0.10),
    ("segments", "maximum_variance_from_overall", 0.15),
    ("stability", "maximum_drift", 0.05),
    ("calibration", "maximum_brier_score", 0.20),
]


@pytest.mark.parametrize("judgment_type,target_field,target_value", _JUDGMENT_TYPE_TARGETS)
def test_v2_parser_handles_each_judgment_slo_type(
    judgment_type: str, target_field: str, target_value: float
) -> None:
    """Every judgment_type in OPENSRM-CORE-v2 §5.2 parses via the v2 parser.

    Pins opensrm-b22.1 acceptance: "All 8 judgment SLO types parseable".
    Each type carries its own target field name; verifies the type
    survives the round-trip with the input target value.
    """
    data = {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {"name": "svc", "labels": {"tier": "critical"}},
        "spec": {
            "owner": {"group": "group:default/team"},
            "service": {"name": "svc", "type": "ai-gate"},
            "judgment_slo": [
                {
                    "metadata": {"name": f"svc-{judgment_type}"},
                    "spec": {
                        "judgment_type": judgment_type,
                        "measurement": {"window": "7d", "source": "lineage"},
                        "target": {target_field: target_value},
                    },
                }
            ],
        },
    }
    manifest = parse_opensrm_v2(data)
    judgment_slos = manifest.get_judgment_slos()
    assert len(judgment_slos) == 1
    slo = judgment_slos[0]
    assert slo.judgment_type == judgment_type
    assert slo.target == target_value
    assert slo.is_judgment_slo() is True


# =============================================================================
# OpenSLO Parser
# =============================================================================


class TestOpenSLOParser:
    def test_parse_inline_slo(self):
        slo_data = [
            {
                "apiVersion": "openslo/v1",
                "kind": "SLO",
                "metadata": {"name": "test-availability"},
                "spec": {
                    "indicator": {
                        "metadata": {"name": "avail"},
                        "spec": {
                            "ratioMetric": {
                                "total": {"metricSource": {"type": "Prometheus", "spec": {"query": "total"}}},
                                "good": {"metricSource": {"type": "Prometheus", "spec": {"query": "good"}}},
                            }
                        },
                    },
                    "objectives": [{"target": 0.999}],
                },
            }
        ]
        results = parse_openslo_slos(slo_data)
        assert len(results) == 1
        assert results[0].name == "test-availability"
        assert results[0].slo_type == "availability"
        assert results[0].total_query == "total"
        assert results[0].good_query == "good"

    def test_parse_ref_slo(self, tmp_path):
        slo_file = tmp_path / "avail.yaml"
        slo_file.write_text(yaml.dump({
            "apiVersion": "openslo/v1",
            "kind": "SLO",
            "metadata": {"name": "ref-availability"},
            "spec": {
                "indicator": {
                    "metadata": {"name": "avail"},
                    "spec": {
                        "ratioMetric": {
                            "total": {"metricSource": {"type": "Prometheus", "spec": {"query": "total"}}},
                            "good": {"metricSource": {"type": "Prometheus", "spec": {"query": "good"}}},
                        }
                    },
                },
                "objectives": [{"target": 0.999}],
            },
        }))

        results = parse_openslo_slos([{"$ref": "avail.yaml"}], base_dir=tmp_path)
        assert len(results) == 1
        assert results[0].name == "ref-availability"
        assert results[0].source_ref == "avail.yaml"

    def test_missing_ref_raises(self, tmp_path):
        with pytest.raises(OpenSLOError, match="not found"):
            parse_openslo_slos([{"$ref": "missing.yaml"}], base_dir=tmp_path)

    def test_missing_objectives_raises(self):
        with pytest.raises(OpenSLOError, match="no objectives"):
            parse_openslo_slos([{
                "apiVersion": "openslo/v1",
                "kind": "SLO",
                "metadata": {"name": "bad"},
                "spec": {"objectives": []},
            }])


# =============================================================================
# Loader (Auto-Detection)
# =============================================================================


class TestLoader:
    def test_load_v1_manifest(self, tmp_path):
        manifest_file = tmp_path / "service.yaml"
        manifest_file.write_text(yaml.dump({
            "apiVersion": "srm/v1",
            "kind": "ServiceReliabilityManifest",
            "metadata": {"name": "svc", "team": "t", "tier": "standard"},
            "spec": {
                "type": "api",
                "slos": {"availability": {"target": 99.9}},
            },
        }))
        m = load_manifest(manifest_file)
        assert m.source_format == SourceFormat.SRM_V1

    def test_load_v2_manifest(self, tmp_path):
        manifest_file = tmp_path / "service.yaml"
        manifest_file.write_text(yaml.dump({
            "apiVersion": "opensrm.nthlayer.io/v2",
            "kind": "ServiceManifest",
            "metadata": {"name": "svc", "labels": {"tier": "standard", "type": "api"}},
            "spec": {
                "owner": {"group": "group:default/team"},
                "service": {"name": "svc", "type": "api"},
            },
        }))
        m = load_manifest(manifest_file)
        assert m.source_format == SourceFormat.OPENSRM_V2

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/file.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{invalid yaml")
        with pytest.raises(ManifestLoadError, match="Invalid YAML"):
            load_manifest(bad_file)

    def test_load_invalid_type_raises_manifest_error(self, tmp_path):
        """ValueError from __post_init__ must be wrapped in ManifestLoadError."""
        manifest_file = tmp_path / "bad.yaml"
        manifest_file.write_text(yaml.dump({
            "apiVersion": "srm/v1",
            "kind": "ServiceReliabilityManifest",
            "metadata": {"name": "svc", "team": "t", "tier": "critical"},
            "spec": {"type": "nonexistent_type"},
        }))
        with pytest.raises(ManifestLoadError, match="Invalid type"):
            load_manifest(manifest_file)

    def test_load_v2_invalid_type_raises_manifest_error(self, tmp_path):
        """ValueError from v2 path must also be wrapped."""
        manifest_file = tmp_path / "bad.yaml"
        manifest_file.write_text(yaml.dump({
            "apiVersion": "opensrm.nthlayer.io/v2",
            "kind": "ServiceManifest",
            "metadata": {"name": "svc", "labels": {"tier": "critical"}},
            "spec": {
                "owner": {"group": "group:default/team"},
                # The invalid value now belongs on the field, not the label:
                # post-opensrm-ih0v labels.type is neither read nor written,
                # so putting it there would test nothing.
                "service": {"name": "svc", "type": "nonexistent_type"},
            },
        }))
        with pytest.raises(ManifestLoadError, match="Invalid type"):
            load_manifest(manifest_file)

    def test_load_v2_with_template_resolves(self, tmp_path):
        """Template resolution must happen via load_manifest, not just parse_opensrm_v2_file."""
        # Create template
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "base.yaml").write_text(yaml.dump({
            "apiVersion": "opensrm.nthlayer.io/v2",
            "kind": "ServiceManifestTemplate",
            "metadata": {"name": "base"},
            "spec": {
                "instrumentation": {
                    "required_metrics": [
                        {"name": "from_template", "type": "counter"},
                    ],
                },
            },
        }))

        # Create manifest that extends template
        manifest_file = tmp_path / "svc.yaml"
        manifest_file.write_text(yaml.dump({
            "apiVersion": "opensrm.nthlayer.io/v2",
            "kind": "ServiceManifest",
            "metadata": {"name": "svc", "labels": {"tier": "standard", "type": "api"}},
            "spec": {
                "owner": {"group": "group:default/team"},
                "service": {"name": "svc", "type": "api"},
                "template": {"extends": "template:default/base"},
            },
        }))

        m = load_manifest(manifest_file)
        assert m.instrumentation is not None
        assert len(m.instrumentation.required_metrics) == 1
        assert m.instrumentation.required_metrics[0].name == "from_template"

    def test_openslo_ref_path_traversal_blocked(self, tmp_path):
        """$ref with ../ outside manifest dir must be rejected."""
        from nthlayer_common.manifest.openslo.parser import OpenSLOParseError as OError
        from nthlayer_common.manifest.openslo.parser import parse_openslo_slos

        with pytest.raises(OError, match="outside manifest directory"):
            parse_openslo_slos([{"$ref": "../../etc/passwd"}], base_dir=tmp_path)

    def test_load_demo_specs(self):
        """Verify loader works with actual demo specs."""
        demo_dir = Path(__file__).parent.parent.parent / "demo" / "specs"
        if not demo_dir.exists():
            pytest.skip("demo/specs not found")

        for spec_file in demo_dir.glob("*.yaml"):
            m = load_manifest(spec_file)
            assert m.name
            assert m.source_format == SourceFormat.SRM_V1


class TestExtractDeclaredDependencies:
    """opensrm-dpws: shared declared-dep extraction across CLI (Manifest
    dataclasses) and worker (raw HTTP dicts) retrospective paths."""

    def test_extract_declared_dependencies_from_manifests(self):
        """Manifest-dataclass input → {service: [dep_name, ...]}."""
        m_a = ReliabilityManifest(
            name="svc-a", team="t", tier="standard", type="api",
            dependencies=[
                Dependency(name="svc-b", type="api"),
                Dependency(name="svc-c", type="api"),
            ],
        )
        m_b = ReliabilityManifest(
            name="svc-b", team="t", tier="standard", type="api",
            dependencies=None,
        )

        result = extract_declared_dependencies(
            from_manifests={"svc-a": m_a, "svc-b": m_b},
        )
        assert result == {"svc-a": ["svc-b", "svc-c"], "svc-b": []}

    def test_extract_declared_dependencies_from_dicts(self):
        """HTTP dict input (GET /manifests wire shape) → same output shape."""
        manifest_dicts = [
            {"name": "svc-a", "dependencies": [
                {"name": "svc-b", "type": "api"},
                {"name": "svc-c", "type": "api"},
            ]},
            {"name": "svc-b", "dependencies": []},
            {"name": "svc-c"},  # missing dependencies key entirely
        ]
        result = extract_declared_dependencies(from_dicts=manifest_dicts)
        assert result == {
            "svc-a": ["svc-b", "svc-c"],
            "svc-b": [],
            "svc-c": [],
        }

    def test_extract_declared_dependencies_requires_exactly_one_input(self):
        """Neither / both supplied → ValueError."""
        with pytest.raises(ValueError, match="exactly one"):
            extract_declared_dependencies()
        with pytest.raises(ValueError, match="exactly one"):
            extract_declared_dependencies(
                from_manifests={}, from_dicts=[],
            )

    def test_extract_declared_dependencies_skips_dict_with_no_name(self):
        """Dict entries without a name key are silently skipped (mirrors
        the _extract_service_slos precedent in observe/worker.py)."""
        manifest_dicts = [
            {"name": "svc-a", "dependencies": [{"name": "svc-b"}]},
            {"dependencies": [{"name": "svc-x"}]},  # no name → skip
            {"name": "", "dependencies": []},  # empty-string name → skip
        ]
        result = extract_declared_dependencies(from_dicts=manifest_dicts)
        assert result == {"svc-a": ["svc-b"]}
