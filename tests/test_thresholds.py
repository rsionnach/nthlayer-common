"""parse_bounds / check_type_fields / is_duration, against the schema's rules.

Every rejected input below is something schema.json rejects (the comment names
the schema clause), so the parser and the schema give one verdict. Python's
bool is an int subclass and YAML 1.1 reads `yes` as True, so booleans are
listed explicitly: the schema's `type: number` rejects them, and a bare
isinstance(x, (int, float)) would not.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from nthlayer_common.manifest.thresholds import (
    FORBIDDEN_THRESHOLD_KEYS,
    THRESHOLD_BOUNDS,
    check_type_fields,
    duration_seconds,
    is_duration,
    normalise_bins,
    parse_bounds,
    threshold_to_v2_entry,
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


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("1ms", 0.001),
        ("30s", 30),
        ("5m", 300),
        ("24h", 86400),
        ("30d", 2592000),
        ("2w", 1209600),
        ("0s", 0),
    ],
)
def test_duration_seconds_converts_each_suffix(value, seconds):
    result = duration_seconds(value)
    assert isinstance(result, float)
    assert result == pytest.approx(seconds)


@pytest.mark.parametrize("value", ["24", "1.5h", 24, True, "24H", "", None, "²h"], ids=repr)
def test_duration_seconds_rejects_what_is_duration_rejects(value):
    assert not is_duration(value)
    with pytest.raises(ValueError):
        duration_seconds(value)


def test_duration_seconds_is_exported_from_the_package():
    from nthlayer_common import manifest

    assert manifest.duration_seconds is duration_seconds
    assert "duration_seconds" in manifest.__all__


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


@pytest.mark.parametrize("raw", [{1: "1h", "x": "2h"}, {None: "1h", "x": "2h"}], ids=["int-key", "null-key"])
def test_parse_bounds_reports_mixed_type_keys_as_a_value_error(raw):
    """YAML allows int/null mapping keys; sorting them with str keys is a TypeError."""
    with pytest.raises(ValueError, match=r"do not belong to 'audit_backlog'"):
        parse_bounds("audit_backlog", raw)


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
        ("segments", {"segment_dimension": 5, "segment_values": ["a"]}, "segment_dimension"),
        ("stability", {"probe": {"frequency": "daily"}}, "frozen_input_set"),
        ("stability", {"probe": {"frozen_input_set": 5, "frequency": "daily"}}, "frozen_input_set"),
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


def test_check_type_fields_does_not_mutate_its_input():
    entry = {"measurement": {"bins": 10.0, "window": "30d"}}
    before = copy.deepcopy(entry)
    check_type_fields("calibration", entry)
    assert entry == before
    assert isinstance(entry["measurement"]["bins"], float)


def test_normalise_bins_converts_integral_float_to_int():
    result = normalise_bins(10.0)
    assert result == 10
    assert isinstance(result, int)


def test_normalise_bins_leaves_int_unchanged():
    result = normalise_bins(10)
    assert result == 10
    assert isinstance(result, int)


def test_a_huge_integer_bound_is_a_range_error_not_an_overflow():
    """math.isfinite(10**400) raises OverflowError, which no parser catches;
    an integer is always finite, so it must reach the [0, 1] check instead."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_bounds("stability", {"maximum_drift": 10**400})


# parents: [0]=tests/ [1]=nthlayer-common/ [2]=<ecosystem>/
_ALL_FIVE = (
    Path(__file__).resolve().parents[2]
    / "opensrm" / "spec" / "v2" / "examples" / "judgment-thresholds" / "all-five-types.yaml"
)


def test_threshold_to_v2_entry_round_trips_all_five_types():
    """The docstring claims it inverts the v2 parser for every type; callers
    only reach calibration/feedback_latency, so check the rest directly."""
    import yaml

    from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2

    assert _ALL_FIVE.is_file(), f"spec example missing: {_ALL_FIVE}"
    doc = yaml.safe_load(_ALL_FIVE.read_text(encoding="utf-8"))
    originals = parse_opensrm_v2(doc, base_dir=_ALL_FIVE.parent).judgment_thresholds
    assert {t.type for t in originals} == set(THRESHOLD_BOUNDS)

    doc["spec"]["judgment_threshold"] = [threshold_to_v2_entry(t) for t in originals]
    reparsed = parse_opensrm_v2(doc, base_dir=_ALL_FIVE.parent).judgment_thresholds
    assert reparsed == originals



def test_duration_seconds_reports_an_unrepresentable_duration_as_value_error():
    """'1' + 400 zeros + 'd' passes is_duration but overflows a float; it must
    surface as ValueError like every other rejected duration."""
    with pytest.raises(ValueError, match="too large"):
        duration_seconds("1" + "0" * 400 + "d")
