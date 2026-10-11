"""parse_bounds / check_type_fields / is_duration, against the schema's rules.

Every rejected input below is something schema.json rejects (the comment names
the schema clause), so the parser and the schema give one verdict. Python's
bool is an int subclass and YAML 1.1 reads `yes` as True, so booleans are
listed explicitly: the schema's `type: number` rejects them, and a bare
isinstance(x, (int, float)) would not.
"""
from __future__ import annotations

import math

import pytest

from nthlayer_common.manifest.thresholds import (
    FORBIDDEN_THRESHOLD_KEYS,
    check_type_fields,
    is_duration,
    parse_bounds,
)


@pytest.mark.parametrize("value", ["1ms", "30s", "5m", "24h", "30d", "2w", "0s"])
def test_is_duration_accepts_the_schema_pattern(value):
    assert is_duration(value)


@pytest.mark.parametrize(
    "value",
    ["", "h", "24", "24H", "1.5h", "-1h", " 24h", "24h ", "²h", "24hh", 24, None, True],
)
def test_is_duration_rejects_what_the_schema_pattern_rejects(value):
    # schema: Duration = {type: string, pattern: ^[0-9]+(ms|s|m|h|d|w)$}
    assert not is_duration(value)


def test_parse_bounds_returns_declared_values_untouched():
    assert parse_bounds("stability", {"maximum_drift": 0.02}) == {"maximum_drift": 0.02}
    assert parse_bounds("audit_backlog", {"maximum_age": "24h"}) == {"maximum_age": "24h"}
    assert parse_bounds(
        "calibration",
        {"maximum_brier_score": 0.05, "maximum_expected_calibration_error": 0.03},
    ) == {"maximum_brier_score": 0.05, "maximum_expected_calibration_error": 0.03}


@pytest.mark.parametrize(
    ("threshold_type", "raw", "match"),
    [
        ("stability", None, "mapping"),                                   # bounds: type object
        ("stability", [0.02], "mapping"),
        ("stability", {}, "maximum_drift"),                               # required
        ("calibration", {}, "at least one"),                              # anyOf
        ("feedback_latency", {}, "at least one"),
        ("stability", {"maximum_drift": 0.02, "maximum_p50": "1h"}, "maximum_p50"),  # closed
        ("stability", {"maximum_drift": True}, "number"),                 # Ratio: number
        ("stability", {"maximum_drift": "0.02"}, "number"),
        ("stability", {"maximum_drift": None}, "number"),
        ("stability", {"maximum_drift": 1.5}, r"\[0, 1\]"),               # Ratio: max 1
        ("stability", {"maximum_drift": -0.1}, r"\[0, 1\]"),              # Ratio: min 0
        ("stability", {"maximum_drift": math.nan}, "finite"),
        ("stability", {"maximum_drift": math.inf}, "finite"),
        ("calibration", {"maximum_brier_score": 2}, r"\[0, 1\]"),         # inline 0-1
        ("audit_backlog", {"maximum_age": 24}, "duration"),               # Duration: string
        ("audit_backlog", {"maximum_age": "1 day"}, "duration"),
        ("feedback_latency", {"maximum_p50": 0.5}, "duration"),
    ],
)
def test_parse_bounds_rejects_what_the_schema_rejects(threshold_type, raw, match):
    with pytest.raises(ValueError, match=match):
        parse_bounds(threshold_type, raw)


def test_parse_bounds_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="threshold type"):
        parse_bounds("reversal_rate", {"maximum_reversal_rate": 0.05})


@pytest.mark.parametrize(
    ("threshold_type", "entry", "match"),
    [
        ("segments", {"segment_values": ["a"]}, "segment_dimension"),
        ("segments", {"segment_dimension": "tier", "segment_values": []}, "segment_values"),
        ("segments", {"segment_dimension": "tier", "segment_values": "a"}, "segment_values"),
        ("segments", {"segment_dimension": "tier", "segment_values": [1]}, "segment_values"),
        ("stability", {}, "probe"),
        ("stability", {"probe": {"frozen_input_set": "s3://x"}}, "frequency"),
        ("stability", {"probe": {"frozen_input_set": "s3://x", "frequency": "monthly"}}, "frequency"),
        ("calibration", {}, "measurement"),
        ("calibration", {"measurement": {"bins": 1, "window": "30d"}}, "bins"),
        ("calibration", {"measurement": {"bins": True, "window": "30d"}}, "bins"),
        # schema `type: integer` accepts 10.0 but not 2.5
        ("calibration", {"measurement": {"bins": 2.5, "window": "30d"}}, "bins"),
        ("calibration", {"measurement": {"bins": 10, "window": "30"}}, "window"),
        ("calibration", {"measurement": {"bins": 10, "window": "30d", "method": "brier_score"}}, "method"),
        ("segments", {"segment_dimension": "", "segment_values": ["a"]}, "segment_dimension"),
        ("segments", {"segment_dimension": "tier", "segment_values": ["a", ""]}, "segment_values"),
    ],
)
def test_check_type_fields_rejects_what_the_schema_rejects(threshold_type, entry, match):
    with pytest.raises(ValueError, match=match):
        check_type_fields(threshold_type, entry)


@pytest.mark.parametrize(
    ("threshold_type", "entry"),
    [
        ("segments", {"segment_dimension": "tier", "segment_values": ["a", "b"]}),
        ("stability", {"probe": {"frozen_input_set": "s3://x", "frequency": "daily"}}),
        ("calibration", {"measurement": {"bins": 10, "window": "30d"}}),
        # schema.json bins `type: integer`: draft-07 accepts an integral float
        ("calibration", {"measurement": {"bins": 10.0, "window": "30d"}}),
        ("audit_backlog", {}),
        ("feedback_latency", {}),
    ],
)
def test_check_type_fields_accepts_the_schema_shapes(threshold_type, entry):
    check_type_fields(threshold_type, entry)


@pytest.mark.parametrize("key", FORBIDDEN_THRESHOLD_KEYS)
@pytest.mark.parametrize(
    ("threshold_type", "entry"),
    [
        ("segments", {"segment_dimension": "tier", "segment_values": ["a"]}),
        ("stability", {"probe": {"frozen_input_set": "s3://x", "frequency": "daily"}}),
        ("calibration", {"measurement": {"bins": 10, "window": "30d"}}),
        ("audit_backlog", {}),
        ("feedback_latency", {}),
    ],
)
def test_check_type_fields_rejects_forbidden_keys_for_every_type(threshold_type, entry, key):
    with pytest.raises(ValueError, match=key):
        check_type_fields(threshold_type, {**entry, key: "x"})


def test_forbidden_keys_are_rejected_before_required_fields():
    with pytest.raises(ValueError, match="breach_actions"):
        check_type_fields("stability", {"breach_actions": []})


def test_integral_float_bins_is_normalised_to_int():
    entry = {"measurement": {"bins": 10.0, "window": "30d"}}
    check_type_fields("calibration", entry)
    assert entry["measurement"]["bins"] == 10
    assert isinstance(entry["measurement"]["bins"], int)
