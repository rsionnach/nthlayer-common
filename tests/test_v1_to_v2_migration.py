"""Tests for v1 → v2 manifest migration (opensrm-b22.2).

Acceptance: every demo spec converts and round-trips through the v2 parser
without losing information.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2
from nthlayer_common.manifest.target_validation import TargetConventionWarning
from nthlayer_common.manifest.v1_compat import convert_v1_to_v2


def _v1_classical():
    return {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {
            "name": "payment-api",
            "team": "payments-team",
            "tier": "critical",
        },
        "spec": {
            "type": "api",
            "slos": {
                "availability": {
                    "target": 99.9,
                    "window": "30d",
                    "indicator": {
                        "query": "sum(rate(http_requests_total[5m]))",
                    },
                },
            },
            "dependencies": [
                {"name": "fraud-detect", "type": "ai-gate", "critical": True},
            ],
        },
    }


def _v1_judgment():
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
                "reversal_rate": {"target": 98.5, "window": "2m"},
                "availability": {
                    "target": 99.9, "window": "30d",
                    "indicator": {"query": "rate(http[5m])"},
                },
            },
        },
    }


class TestConvertShape:
    """Output dict has the canonical v2 shape that parse_opensrm_v2 accepts."""

    def test_apiversion_and_kind(self):
        v2 = convert_v1_to_v2(_v1_classical())
        assert v2["apiVersion"] == "opensrm.nthlayer.io/v2"
        assert v2["kind"] == "ServiceManifest"

    def test_team_to_owner_group_ref(self):
        v2 = convert_v1_to_v2(_v1_classical())
        assert v2["spec"]["owner"] == {"group": "group:default/payments-team"}

    def test_tier_to_labels_and_type_to_service_field(self):
        """Post-opensrm-ih0v the two v1 fields land in different places.

        ``tier`` is still a label (see opensrm-9bil, which asks whether it
        should be promoted the way ``type`` just was); ``type`` is now the
        first-class ``spec.service.type`` and is deliberately NOT mirrored
        into labels, so nothing can read a stale copy of it.
        """
        v2 = convert_v1_to_v2(_v1_classical())

        assert v2["metadata"]["labels"]["tier"] == "critical"
        assert v2["spec"]["service"]["type"] == "api"
        assert "type" not in v2["metadata"]["labels"]

    def test_classical_slo_uses_threshold_metric_and_ratio_target(self):
        v2 = convert_v1_to_v2(_v1_classical())
        slos = v2["spec"]["slo"]
        assert len(slos) == 1
        slo = slos[0]
        assert slo["apiVersion"] == "openslo/v1"
        assert slo["kind"] == "SLO"
        # Target normalised from 99.9 → 0.999 for OpenSLO ratio convention.
        assert slo["spec"]["objectives"][0]["target"] == pytest.approx(0.999)
        # Single PromQL query in thresholdMetric shape.
        ind_spec = slo["spec"]["indicator"]["spec"]
        assert "thresholdMetric" in ind_spec
        assert ind_spec["thresholdMetric"]["metricSource"]["spec"]["query"] == \
            "sum(rate(http_requests_total[5m]))"

    def test_judgment_slo_emitted_separately(self):
        v2 = convert_v1_to_v2(_v1_judgment())
        # reversal_rate → judgment_slo; availability → spec.slo
        assert len(v2["spec"]["judgment_slo"]) == 1
        assert len(v2["spec"]["slo"]) == 1
        jslo = v2["spec"]["judgment_slo"][0]
        assert jslo["spec"]["judgment_type"] == "reversal_rate"
        assert jslo["spec"]["target"] == {"maximum_reversal_rate": 98.5}

    def test_dependency_critical_flag_carried(self):
        v2 = convert_v1_to_v2(_v1_classical())
        deps = v2["spec"]["dependencies"]
        assert deps[0]["service"] == "component:default/fraud-detect"
        assert deps[0]["criticality"] == "critical"

    def test_non_v1_input_raises(self):
        with pytest.raises(ValueError, match="srm/v1"):
            convert_v1_to_v2({"apiVersion": "opensrm.nthlayer.io/v2"})

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="metadata.name"):
            convert_v1_to_v2({"apiVersion": "srm/v1", "metadata": {}, "spec": {}})


class TestRoundTripThroughV2Parser:
    """Synthetic v1 inputs convert + parse via parse_opensrm_v2 without error."""

    def test_classical_round_trip(self):
        v2 = convert_v1_to_v2(_v1_classical())
        manifest = parse_opensrm_v2(v2)
        assert manifest.name == "payment-api"
        assert manifest.tier == "critical"
        assert manifest.type == "api"
        assert len(manifest.slos) == 1
        assert manifest.slos[0].name == "availability"

    def test_judgment_round_trip(self):
        v2 = convert_v1_to_v2(_v1_judgment())
        manifest = parse_opensrm_v2(v2)
        # parse_opensrm_v2 merges classical + judgment into manifest.slos.
        names = [s.name for s in manifest.slos]
        assert "reversal_rate" in names
        assert "availability" in names
        # The reversal_rate SLO retains its judgment_type.
        rr = next(s for s in manifest.slos if s.name == "reversal_rate")
        assert rr.judgment_type == "reversal_rate"


class TestDemoSpecsRoundTrip:
    """Acceptance: all four demo specs convert + round-trip through v2 parser."""

    DEMO_SPECS = ["payment-api.yaml", "fraud-detect.yaml", "order-service.yaml", "checkout-svc.yaml"]

    @pytest.fixture
    def demo_dir(self) -> Path:
        # Resolve relative to nthlayer-ecosystem layout: nthlayer/demo/specs/.
        candidate = Path(__file__).resolve().parents[2] / "nthlayer" / "demo" / "specs"
        if not candidate.exists():
            pytest.skip(f"Demo specs directory not found at {candidate}")
        return candidate

    @pytest.mark.parametrize("filename", DEMO_SPECS)
    def test_demo_spec_converts_and_parses(self, demo_dir: Path, filename: str):
        path = demo_dir / filename
        if not path.exists():
            pytest.skip(f"{filename} not present")

        with open(path) as f:
            v1_data = yaml.safe_load(f)

        v2_data = convert_v1_to_v2(v1_data)

        # Suppress the target-convention warnings the v2 parser may emit
        # for converted demo specs (judgment-target shape varies).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", TargetConventionWarning)
            manifest = parse_opensrm_v2(v2_data)

        # Manifest preserved canonical fields.
        assert manifest.name == v1_data["metadata"]["name"]
        assert manifest.tier == v1_data["metadata"]["tier"]
        assert manifest.type == v1_data["spec"]["type"]
        # Every v1 SLO survives in some form (either classical or judgment).
        v1_slo_names = set((v1_data["spec"].get("slos") or {}).keys())
        v2_slo_names = {s.name for s in manifest.slos}
        assert v1_slo_names <= v2_slo_names, (
            f"Lost SLOs in {filename}: {v1_slo_names - v2_slo_names}"
        )
