"""srm/v1 calibration and feedback_latency become judgment thresholds.

Provenance: nthlayer-generate/examples/opensrm/ai-gate.reliability.yaml - the
only srm/v1 file in the workspace using either - declares

    calibration:      {target: 0.05, window: 30d}   # "Max 5% ECE"
    feedback_latency: {target: 3600, window: 30d}   # "within 1 hour"

ECE, not Brier, because opensrm/v1's FlatCalibrationSLO describes its target as
"Expected Calibration Error". feedback_latency maps to p50 by decision
(spec 2.3): a bare "within an hour" is the weaker, typical-case claim.
"""
from __future__ import annotations

from typing import Any

import pytest

from nthlayer_common.manifest.parser.v1 import OpenSRMParseError, parse_srm_v1


def _srm_v1(slos: dict[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "t", "tier": "critical"},
        "spec": {"type": "ai-gate", "slos": slos},
    }


def test_generate_example_shapes_become_thresholds():
    manifest = parse_srm_v1(_srm_v1({
        "calibration": {"target": 0.05, "window": "30d"},
        "feedback_latency": {"target": 3600, "window": "30d"},
    }))

    by_type = {t.type: t for t in manifest.judgment_thresholds}
    assert by_type["calibration"].bounds == {"maximum_expected_calibration_error": 0.05}
    assert by_type["calibration"].measurement.window == "30d"
    assert by_type["calibration"].measurement.bins == 10
    assert by_type["feedback_latency"].bounds == {"maximum_p50": "3600s"}
    # vrpa's observable defect: this used to be a ratio SLO with judgment_type=None.
    assert [s.name for s in manifest.slos] == []


def test_rate_types_stay_judgment_slos():
    manifest = parse_srm_v1(_srm_v1({"reversal_rate": {"target": 98.5}}))
    assert [(s.name, s.judgment_type) for s in manifest.slos] == [("reversal_rate", "reversal_rate")]
    assert manifest.judgment_thresholds == []


@pytest.mark.parametrize("name", ["segments", "stability", "audit_backlog"])
def test_types_the_flat_format_cannot_express_are_rejected(name):
    with pytest.raises(OpenSRMParseError, match="v2"):
        parse_srm_v1(_srm_v1({name: {"target": 0.05}}))


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"target": 5.0}, r"\[0, 1\]"),       # a percentage written for a ratio bound
        ({"target": True}, "number"),
        ({}, "target"),
    ],
)
def test_calibration_target_is_a_ratio(config, match):
    with pytest.raises(OpenSRMParseError, match=match):
        parse_srm_v1(_srm_v1({"calibration": config}))


@pytest.mark.parametrize("target", [-1, 1.5, -2.0, "1h", True, None])
def test_feedback_latency_target_is_whole_seconds(target):
    with pytest.raises(OpenSRMParseError, match="seconds"):
        parse_srm_v1(_srm_v1({"feedback_latency": {"target": target}}))


def test_scalar_shorthand_is_accepted_like_other_srm_v1_slos():
    """srm/v1 lets `name: value` stand for `name: {target: value}`
    (parser/v1.py _parse_slos); thresholds keep that."""
    manifest = parse_srm_v1(_srm_v1({"feedback_latency": 60}))
    assert manifest.judgment_thresholds[0].bounds == {"maximum_p50": "60s"}


def test_feedback_latency_accepts_an_integral_float_target():
    manifest = parse_srm_v1(_srm_v1({"feedback_latency": {"target": 3600.0}}))
    assert manifest.judgment_thresholds[0].bounds == {"maximum_p50": "3600s"}


@pytest.mark.parametrize(
    ("name", "config", "key"),
    [
        ("feedback_latency", {"target": 3600, "p90": "24h"}, "p90"),
        ("feedback_latency", {"p50": "1h", "p90": "24h"}, "p50"),  # no target: bound-like keys win
        ("feedback_latency", {"p50": "1h", "p90": "24h"}, "p90"),
        ("feedback_latency", {"target": 3600, "maximum_p90": "24h"}, "maximum_p90"),
        ("calibration", {"target": 0.05, "bins": 20}, "bins"),
        ("calibration", {"target": 0.05, "method": "ece"}, "method"),
        ("calibration", {"target": 0.05, "maximum_brier_score": 0.1}, "maximum_brier_score"),
    ],
)
def test_bound_like_keys_srm_v1_cannot_carry_are_rejected_not_dropped(name, config, key):
    with pytest.raises(OpenSRMParseError, match=key) as exc_info:
        parse_srm_v1(_srm_v1({name: config}))
    assert "judgment_threshold" in str(exc_info.value)


def test_keys_the_lift_ignores_harmlessly_stay_accepted():
    manifest = parse_srm_v1(_srm_v1({
        "feedback_latency": {"target": 60, "window": "30d", "description": "d", "labels": {"a": "b"}},
    }))
    assert manifest.judgment_thresholds[0].bounds == {"maximum_p50": "60s"}


def test_calibration_window_defaults_to_30d_when_absent():
    manifest = parse_srm_v1(_srm_v1({"calibration": {"target": 0.05}}))
    [threshold] = manifest.judgment_thresholds
    assert threshold.measurement.window == "30d"


@pytest.mark.parametrize("window", [30, "30 days"])
def test_calibration_rejects_a_non_duration_window(window):
    with pytest.raises(OpenSRMParseError, match="window"):
        parse_srm_v1(_srm_v1({"calibration": {"target": 0.05, "window": window}}))
