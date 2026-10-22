"""opensrm/v1 slos.judgment (opensrm-pnmv).

Shapes: spec/v1/schema.json JudgmentSLOs — calibration {target, window}
(FlatCalibrationSLO, both required), feedback_latency {p50, p90} durations,
reversal_rate / high_confidence_failure. Mapping: AUTHORING.md §7.6 (calibration
and feedback_latency → judgment_threshold) and §7.7 step 8 (judgment_slo is not
migration). v2 allows judgment blocks only on ai-gate (§3.1).
"""
import pytest

from nthlayer_common.manifest.opensrm_v1 import convert_opensrm_v1_to_v2
from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2


def _v1(judgment: dict, stype: str = "ai-gate") -> dict:
    return {"apiVersion": "opensrm/v1", "kind": "ServiceReliabilityManifest",
            "metadata": {"name": "pr-review-agent", "team": "platform", "tier": "critical"},
            "spec": {"type": stype, "slos": {"judgment": judgment}}}


def test_calibration_and_feedback_latency_become_thresholds():
    r = convert_opensrm_v1_to_v2(_v1({
        "calibration": {"target": 0.10, "window": "30d"},
        "feedback_latency": {"p50": "48h", "p90": "168h"},
    }))
    assert r.manifest["spec"]["judgment_threshold"] == [
        {"type": "calibration", "bounds": {"maximum_expected_calibration_error": 0.10},
         "measurement": {"bins": 10, "window": "30d"}},
        {"type": "feedback_latency", "bounds": {"maximum_p50": "48h", "maximum_p90": "168h"}},
    ]
    # bins is a value v1 never declares: reported so the author checks it.
    assert [f.path for f in r.unmapped] == ["spec.slos.judgment.calibration"]
    parse_opensrm_v2(r.manifest)


@pytest.mark.parametrize("key", ["reversal_rate", "high_confidence_failure"])
def test_judgment_slos_are_reported_unmapped(key):
    r = convert_opensrm_v1_to_v2(_v1({key: {"target": 0.03, "window": "30d"}}))
    assert "judgment_slo" not in r.manifest["spec"]
    assert [f.path for f in r.unmapped] == [f"spec.slos.judgment.{key}"]
    assert "new capability, not migration" in r.unmapped[0].reason


def test_a_non_ai_gate_manifest_reports_the_whole_judgment_block():
    r = convert_opensrm_v1_to_v2(_v1({"calibration": {"target": 0.1, "window": "30d"}}, stype="api"))
    assert "judgment_threshold" not in r.manifest["spec"]
    assert [f.path for f in r.unmapped] == ["spec.slos.judgment"]
    parse_opensrm_v2(r.manifest)


@pytest.mark.parametrize("cfg, match", [
    ({"calibration": {"window": "30d"}}, "target"),
    ({"calibration": {"target": 0.1}}, "window"),
    ({"calibration": {"target": 2, "window": "30d"}}, "target"),
    ({"feedback_latency": {"p50": "two days"}}, "p50"),
])
def test_malformed_converted_judgment_fields_raise(cfg, match):
    with pytest.raises(ValueError, match=match):
        convert_opensrm_v1_to_v2(_v1(cfg))


def test_feedback_latency_without_bounds_is_reported():
    r = convert_opensrm_v1_to_v2(_v1({"feedback_latency": {}}))
    assert [f.path for f in r.unmapped] == ["spec.slos.judgment.feedback_latency"]


def test_unknown_judgment_key_is_reported_not_dropped():
    r = convert_opensrm_v1_to_v2(_v1({"mystery": {"target": 1}}))
    assert [f.path for f in r.unmapped] == ["spec.slos.judgment.mystery"]
    assert "judgment_threshold" not in r.manifest["spec"]


def test_unknown_key_inside_calibration_is_reported_not_dropped():
    r = convert_opensrm_v1_to_v2(_v1({"calibration": {"target": 0.1, "window": "30d", "bogus": 1}}))
    assert "spec.slos.judgment.calibration.bogus" in [f.path for f in r.unmapped]
    assert r.manifest["spec"]["judgment_threshold"][0]["type"] == "calibration"


def test_unknown_key_inside_feedback_latency_is_reported_not_dropped():
    r = convert_opensrm_v1_to_v2(_v1({"feedback_latency": {"p50": "48h", "p99": "9h"}}))
    assert [f.path for f in r.unmapped] == ["spec.slos.judgment.feedback_latency.p99"]
    assert r.manifest["spec"]["judgment_threshold"][0]["bounds"] == {"maximum_p50": "48h"}
