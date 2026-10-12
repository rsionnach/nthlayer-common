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
from typing import Any

import pytest

from nthlayer_common.manifest.thresholds import (
    FORBIDDEN_THRESHOLD_KEYS,
    SEGMENTS_DEFAULT_WINDOW,
    SEGMENTS_METRICS,
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
        ("segments", {"metric": "escalation", "segment_values": ["a"]}, "segment_dimension"),
        ("segments", {"metric": "escalation", "segment_dimension": "tier", "segment_values": []}, "segment_values"),
        ("segments", {"metric": "escalation", "segment_dimension": "tier", "segment_values": "a"}, "segment_values"),
        ("segments", {"metric": "escalation", "segment_dimension": "tier", "segment_values": [1]}, "segment_values"),
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
        ("segments", {"metric": "escalation", "segment_dimension": "", "segment_values": ["a"]}, "segment_dimension"),
        ("segments", {"metric": "escalation", "segment_dimension": 5, "segment_values": ["a"]}, "segment_dimension"),
        ("stability", {"probe": {"frequency": "daily"}}, "frozen_input_set"),
        ("stability", {"probe": {"frozen_input_set": 5, "frequency": "daily"}}, "frozen_input_set"),
        ("segments", {"metric": "escalation", "segment_dimension": "tier", "segment_values": ["a", ""]}, "segment_values"),
    ],
)
def test_check_type_fields_rejects_what_the_schema_rejects(threshold_type, entry, match):
    with pytest.raises(ValueError, match=match):
        check_type_fields(threshold_type, entry)


@pytest.mark.parametrize(
    ("threshold_type", "entry"),
    [
        ("segments", {"metric": "escalation", "segment_dimension": "tier", "segment_values": ["a", "b"]}),
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
        ("segments", {"metric": "escalation", "segment_dimension": "tier", "segment_values": ["a"]}),
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
_SPEC_V2 = Path(__file__).resolve().parents[2] / "opensrm" / "spec" / "v2"
_THRESHOLD_EXAMPLES = _SPEC_V2 / "examples" / "judgment-thresholds"
_AI_GATE = _SPEC_V2 / "examples" / "services" / "ai-gate.yaml"


def _threshold_sources() -> list[Path]:
    """The live spec files whose thresholds the round trip covers: every
    segments metric form the spec ships (opensrm-7zpz) and all five types."""
    return [
        _THRESHOLD_EXAMPLES / "all-five-types.yaml",
        _THRESHOLD_EXAMPLES / "segments-two-dimensions.yaml",
        _AI_GATE,
    ]


def _bare_ai_gate(entries: list[Any]) -> dict[str, Any]:
    """A minimal ai-gate manifest carrying *entries*. ai-gate.yaml itself does
    not parse whole yet (opensrm-a742: its flat judgment_slo items), so its
    thresholds are lifted into this shell, unchanged."""
    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {"name": "svc", "labels": {"tier": "critical"}},
        "spec": {
            "owner": {"group": "group:default/team"},
            "service": {"name": "svc", "type": "ai-gate"},
            "judgment_threshold": entries,
        },
    }


@pytest.mark.parametrize("source", _threshold_sources(), ids=lambda p: p.name)
def test_threshold_to_v2_entry_round_trips_the_spec_examples(source):
    """threshold_to_v2_entry inverts the v2 parser: emitting a parsed threshold
    and re-parsing it gives an equal JudgmentThreshold."""
    import yaml

    from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2

    assert source.is_file(), f"spec example missing: {source}"
    entries = yaml.safe_load(source.read_text(encoding="utf-8"))["spec"]["judgment_threshold"]
    assert entries, f"{source.name} declares no thresholds"
    originals = parse_opensrm_v2(_bare_ai_gate(entries)).judgment_thresholds
    assert len(originals) == len(entries)

    emitted = [threshold_to_v2_entry(t) for t in originals]
    reparsed = parse_opensrm_v2(_bare_ai_gate(emitted)).judgment_thresholds
    assert reparsed == originals


def test_the_round_trip_sources_cover_every_type_and_metric():
    """Pins the coverage the round trip claims, so a spec edit that drops a
    form from these files fails here instead of thinning the test."""
    import yaml

    entries = [
        e
        for path in _threshold_sources()
        for e in yaml.safe_load(path.read_text(encoding="utf-8"))["spec"]["judgment_threshold"]
    ]
    assert {e["type"] for e in entries} == set(THRESHOLD_BOUNDS)
    assert {e["metric"] for e in entries if e["type"] == "segments"} == set(SEGMENTS_METRICS)


def test_segments_emit_always_carries_the_stored_window():
    """The parser stores the 30d default, so the emit writes it out."""
    from nthlayer_common.manifest.models import JudgmentThreshold, ThresholdMeasurement

    threshold = JudgmentThreshold(
        type="segments",
        bounds={"maximum_variance_from_overall": 0.05},
        measurement=ThresholdMeasurement(
            metric="escalation", window=SEGMENTS_DEFAULT_WINDOW,
            segment_dimension="tier", segment_values=["a"],
        ),
    )
    entry = threshold_to_v2_entry(threshold)
    assert entry["metric"] == "escalation"
    assert entry["measurement"] == {"window": SEGMENTS_DEFAULT_WINDOW}
    assert "outcome_signal" not in entry


# =============================================================================
# segments metric (CORE §5.5.1, opensrm-7zpz) and closed entries (CORE §5.5).
# Each rejected shape is one the schema's segments branch rejects; the clause
# is named beside it. The spec's own negatives (n67-n92) are run whole in
# test_judgment_thresholds_against_spec.py.
# =============================================================================

_SEG = {"segment_dimension": "tier", "segment_values": ["a"]}


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({**_SEG}, "metric"),                                                     # required
        ({**_SEG, "metric": "audit_sampling"}, "audit_sampling"),                # enum, by name
        ({**_SEG, "metric": "accuracy"}, "metric"),                              # enum
        ({**_SEG, "metric": ["escalation"]}, "metric"),                          # enum, unhashable
        ({**_SEG, "metric": "reversal_rate"}, "source"),                         # then: required
        ({**_SEG, "metric": "reversal_rate", "measurement": {}}, "source"),
        ({**_SEG, "metric": "reversal_rate", "measurement": {"source": "audit"}}, "source"),
        ({**_SEG, "metric": "reversal_rate", "measurement": {"source": ["lineage"]}}, "source"),
        ({**_SEG, "metric": "high_confidence_failure"}, "confidence_threshold"),
        ({**_SEG, "metric": "high_confidence_failure",
          "measurement": {"window": "7d"}}, "confidence_threshold"),
        ({**_SEG, "metric": "escalation", "measurement": {"source": "lineage"}}, "source"),  # else: not
        ({**_SEG, "metric": "outcomes", "measurement": {"source": "lineage"},
          "outcome_signal": {"source": "s", "window": "1h"}}, "source"),
        ({**_SEG, "metric": "escalation",
          "measurement": {"confidence_threshold": 0.9}}, "confidence_threshold"),
        ({**_SEG, "metric": "outcomes"}, "outcome_signal"),
        ({**_SEG, "metric": "escalation",
          "outcome_signal": {"source": "s", "window": "1h"}}, "outcome_signal"),
        ({**_SEG, "metric": "escalation", "measurement": "7d"}, "measurement"),  # type object
        ({**_SEG, "metric": "escalation", "measurement": {"windw": "7d"}}, "windw"),  # closed
        ({**_SEG, "metric": "escalation", "measurement": {"window": "7 days"}}, "window"),
        ({**_SEG, "metric": "escalation", "measurement": {"window": 7}}, "window"),
        ({**_SEG, "metric": "escalation", "measurement": {"window": "0d"}}, "window"),  # not 0+
        ({**_SEG, "metric": "escalation", "measurement": {"window": "000ms"}}, "window"),
        ({**_SEG, "metric": "outcomes", "outcome_signal": "s"}, "outcome_signal"),
        ({**_SEG, "metric": "outcomes", "outcome_signal": {"source": "s"}}, "window"),
        ({**_SEG, "metric": "outcomes", "outcome_signal": {"window": "1h"}}, "source"),
        ({**_SEG, "metric": "outcomes",
          "outcome_signal": {"source": "s", "window": "1 hour"}}, "window"),
        ({**_SEG, "metric": "outcomes",
          "outcome_signal": {"source": 5, "window": "1h"}}, "source"),
        ({**_SEG, "metric": "outcomes",
          "outcome_signal": {"source": "s", "window": "1h", "wait": "2h"}}, "wait"),
    ],
)
def test_segments_metric_rules_reject_what_the_schema_rejects(entry, match):
    with pytest.raises(ValueError, match=match):
        check_type_fields("segments", entry)


# Ratio: number, 0..1. A string is not a number to the schema.
@pytest.mark.parametrize("value", [1.5, -0.1, True, "0.9", None, math.nan, math.inf, 10**400], ids=repr)
def test_segments_confidence_threshold_is_a_ratio(value):
    entry = {**_SEG, "metric": "high_confidence_failure", "measurement": {"confidence_threshold": value}}
    with pytest.raises(ValueError, match="confidence_threshold"):
        check_type_fields("segments", entry)


@pytest.mark.parametrize(
    "entry",
    [
        {**_SEG, "metric": "escalation"},
        {**_SEG, "metric": "escalation", "measurement": {"window": "7d"}},
        {**_SEG, "metric": "escalation", "measurement": {}},
        {**_SEG, "metric": "reversal_rate", "measurement": {"source": "lineage"}},
        {**_SEG, "metric": "reversal_rate", "measurement": {"source": "calibration_sample", "window": "1w"}},
        {**_SEG, "metric": "reversal_rate", "measurement": {"source": "downstream_signal"}},
        {**_SEG, "metric": "high_confidence_failure", "measurement": {"confidence_threshold": 0}},
        {**_SEG, "metric": "high_confidence_failure", "measurement": {"confidence_threshold": 1}},
        {**_SEG, "metric": "high_confidence_failure", "measurement": {"confidence_threshold": 0.9}},
        {**_SEG, "metric": "outcomes", "outcome_signal": {"source": "s", "window": "1h"}},
        # The outcome wait may be zero: it is the SLO's Duration, unguarded.
        {**_SEG, "metric": "outcomes", "outcome_signal": {"source": "s", "window": "0s"}},
        {**_SEG, "metric": "outcomes", "outcome_signal": {"source": "", "window": "1h"},
         "measurement": {"window": "10d"}},
    ],
    ids=lambda e: f"{e['metric']}-{sorted(e.get('measurement', {}))}-{sorted(e.get('outcome_signal', {}))}",
)
def test_segments_metric_rules_accept_the_schema_shapes(entry):
    check_type_fields("segments", entry)


@pytest.mark.parametrize(
    ("threshold_type", "entry", "key"),
    [
        # entry level: n74, n79, n85, n86, n87
        ("segments", {**_SEG, "metric": "escalation", "window": "7d"}, "window"),
        ("segments", {**_SEG, "metric": "escalation", "probe": {}}, "probe"),
        ("segments", {**_SEG, "metric": "escalation", "sampling": None}, "sampling"),
        ("stability", {"window": "7d", "probe": {"frozen_input_set": "s3://x", "frequency": "daily"}},
         "window"),
        ("stability", {"metric": "escalation", "probe": {"frozen_input_set": "s3://x", "frequency": "daily"}},
         "metric"),
        ("calibration", {"bins": 10, "measurement": {"bins": 10, "window": "30d"}}, "bins"),
        ("calibration", {"segment_values": ["a"], "measurement": {"bins": 10, "window": "30d"}},
         "segment_values"),
        ("audit_backlog", {"measurement": {"window": "7d"}}, "measurement"),
        ("audit_backlog", {"metric": "escalation"}, "metric"),
        ("feedback_latency", {"window": "7d"}, "window"),
        ("feedback_latency", {"outcome_signal": {"source": "s", "window": "1h"}}, "outcome_signal"),
        # nested objects: n77, n78
        ("stability", {"probe": {"frozen_input_set": "s3://x", "frequency": "daily", "frequncy": "hourly"}},
         "frequncy"),
        ("calibration", {"measurement": {"bins": 10, "window": "30d", "bin": 20}}, "bin"),
    ],
)
def test_threshold_entries_are_closed(threshold_type, entry, key):
    with pytest.raises(ValueError, match=rf"'{key}'.*§5\.5"):
        check_type_fields(threshold_type, entry)


@pytest.mark.parametrize("threshold_type", list(THRESHOLD_BOUNDS))
def test_type_and_bounds_are_always_allowed(threshold_type):
    """Closure admits the two keys every entry has; check_type_fields leaves
    their validation to the parser and parse_bounds."""
    entry = {
        "segments": {**_SEG, "metric": "escalation"},
        "stability": {"probe": {"frozen_input_set": "s3://x", "frequency": "daily"}},
        "calibration": {"measurement": {"bins": 10, "window": "30d"}},
    }.get(threshold_type, {})
    check_type_fields(threshold_type, {"type": threshold_type, "bounds": {}, **entry})


def test_mixed_type_entry_keys_are_a_value_error():
    """YAML allows int/null mapping keys; sorting them with str keys is a TypeError."""
    with pytest.raises(ValueError, match="audit_backlog"):
        check_type_fields("audit_backlog", {1: "x", None: "y", "z": 2})


def test_unknown_calibration_measurement_key_is_reported_after_method():
    """measurement.method keeps its own message (schema: method is `not`)."""
    with pytest.raises(ValueError, match="measurement.method"):
        check_type_fields(
            "calibration", {"measurement": {"bins": 10, "window": "30d", "method": "x", "bin": 1}}
        )


def test_duration_seconds_reports_an_unrepresentable_duration_as_value_error():
    """'1' + 400 zeros + 'd' passes is_duration but overflows a float; it must
    surface as ValueError like every other rejected duration."""
    with pytest.raises(ValueError, match="too large"):
        duration_seconds("1" + "0" * 400 + "d")


def test_duration_seconds_is_correctly_rounded_for_milliseconds():
    """n * 0.001 is not correctly rounded (e.g. 9ms); n / 1000 is. A bound
    compared at its edge must not be off by one ULP."""
    assert all(duration_seconds(f"{n}ms") == n / 1000 for n in range(1, 5000))


def test_duration_seconds_past_the_int_digit_limit_uses_the_same_message():
    with pytest.raises(ValueError, match="too large"):
        duration_seconds("1" + "0" * 5000 + "s")


def test_audit_sampling_gets_its_own_rejection_not_the_generic_enum_one():
    """CORE §5.5.1 says why audit_sampling is excluded; the parser must say so
    rather than list it as an unknown value. Compared structurally: the two
    messages differ beyond the value they echo."""
    def message(metric: str) -> str:
        with pytest.raises(ValueError) as exc_info:
            check_type_fields("segments", {**_SEG, "metric": metric})
        return str(exc_info.value)

    assert message("audit_sampling") != message("accuracy").replace("accuracy", "audit_sampling")
