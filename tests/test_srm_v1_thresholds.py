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


# srm/v1's SLO key set, from what the srm/v1 parsers read
# (nthlayer-generate specs/opensrm_parser.py; parser/v1.py::_parse_slos):
# target | minimum, window, type, unit, percentile, query, description, labels.
# The lift accepts exactly target, window, unit, description and labels
# (opensrm-1rnr); the rest have no threshold meaning and are rejected rather
# than dropped.


def test_description_and_labels_are_accepted_and_dropped():
    """Prose and labels: a threshold has no home for them, and losing them
    changes nothing that is evaluated."""
    manifest = parse_srm_v1(_srm_v1({
        "feedback_latency": {"target": 60, "window": "30d", "description": "d", "labels": {"a": "b"}},
        "calibration": {"target": 0.05, "description": "d", "labels": {"a": "b"}},
    }))
    by_type = {t.type: t for t in manifest.judgment_thresholds}
    assert by_type["feedback_latency"].bounds == {"maximum_p50": "60s"}
    assert by_type["calibration"].bounds == {"maximum_expected_calibration_error": 0.05}


@pytest.mark.parametrize("name", ["feedback_latency", "calibration"])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("minimum", 1),
        ("type", "latency"),
        ("percentile", "p50"),
        ("query", "rate(x[5m])"),
        ("indicator", {"type": "ratio"}),
    ],
)
def test_srm_v1_keys_with_no_threshold_meaning_are_rejected_not_dropped(name, key, value):
    target = 3600 if name == "feedback_latency" else 0.05
    with pytest.raises(OpenSRMParseError, match=key):
        parse_srm_v1(_srm_v1({name: {"target": target, key: value}}))


@pytest.mark.parametrize(
    ("unit", "bound"),
    [(None, "90s"), ("s", "90s"), ("ms", "90ms"), ("m", "90m"), ("h", "90h"), ("d", "90d"), ("w", "90w")],
)
def test_feedback_latency_honours_unit(unit, bound):
    """The target is a count of `unit` (default seconds); the bound keeps it,
    in the Duration grammar the v2 bound uses."""
    config: dict[str, Any] = {"target": 90}
    if unit is not None:
        config["unit"] = unit
    [threshold] = parse_srm_v1(_srm_v1({"feedback_latency": config})).judgment_thresholds
    assert threshold.bounds == {"maximum_p50": bound}


@pytest.mark.parametrize("unit", ["hours", "sec", "H", "", 5, None], ids=repr)
def test_feedback_latency_rejects_a_unit_outside_the_duration_table(unit):
    with pytest.raises(OpenSRMParseError, match="unit"):
        parse_srm_v1(_srm_v1({"feedback_latency": {"target": 90, "unit": unit}}))


@pytest.mark.parametrize("window", ["7d", "1w", "720h", 30, "30 days"], ids=repr)
def test_feedback_latency_window_other_than_30d_is_rejected(window):
    """v2's feedback_latency window is fixed at 30d (CORE §5.5.5); a different
    srm/v1 window has nowhere to go, so it is an error, not dropped."""
    with pytest.raises(OpenSRMParseError, match="window"):
        parse_srm_v1(_srm_v1({"feedback_latency": {"target": 90, "window": window}}))


@pytest.mark.parametrize("config", [{"target": 90}, {"target": 90, "window": "30d"}])
def test_feedback_latency_window_absent_or_30d_is_accepted(config):
    [threshold] = parse_srm_v1(_srm_v1({"feedback_latency": config})).judgment_thresholds
    assert threshold.bounds == {"maximum_p50": "90s"}


def test_calibration_rejects_unit():
    """The calibration target is a ratio (ECE); a unit on it is a mistake the
    lift would otherwise drop silently."""
    with pytest.raises(OpenSRMParseError, match="unit"):
        parse_srm_v1(_srm_v1({"calibration": {"target": 0.05, "unit": "ratio"}}))


def test_unknown_keys_are_reported_through_convert_v1_to_v2_too():
    from nthlayer_common.manifest.v1_compat import convert_v1_to_v2

    with pytest.raises(ValueError, match="query"):
        convert_v1_to_v2(_srm_v1({"feedback_latency": {"target": 90, "query": "x"}}))


def test_calibration_window_defaults_to_30d_when_absent():
    manifest = parse_srm_v1(_srm_v1({"calibration": {"target": 0.05}}))
    [threshold] = manifest.judgment_thresholds
    assert threshold.measurement.window == "30d"


@pytest.mark.parametrize("window", [30, "30 days"])
def test_calibration_rejects_a_non_duration_window(window):
    with pytest.raises(OpenSRMParseError, match="window"):
        parse_srm_v1(_srm_v1({"calibration": {"target": 0.05, "window": window}}))
