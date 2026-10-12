"""Judgment THRESHOLDS: bounds on error magnitudes (decision 3c, opensrm-l33e).

Not SLOs. A threshold has no population of good events, no complement and no
error budget; a breach is the measured value exceeding a declared bound. So
the parsers never convert a v2 bound: bounds are stored exactly as declared,
which is the opposite of target_validation.py's job for judgment SLO targets.
The coercions are normalise_bins() (an integral float bins becomes an int) and
srm_v1_threshold(), which builds a float ECE bound and a "<n>s" duration from
srm/v1's bare numbers.

This module also holds the srm/v1 lift (srm_v1_threshold), the v2 emit
(threshold_to_v2_entry), and duration_seconds(), which turns a Duration bound
into seconds for evaluators.

Threshold entries are closed (CORE §5.5, opensrm-7zpz): check_type_fields
rejects any key a type does not define, from the one THRESHOLD_TYPE_FIELDS
table, and a segments entry names the rate it compares (SEGMENTS_METRICS).

The tables restate opensrm/spec/v2/schema.json's JudgmentThresholdSpec so the
parsers can enforce it without a JSON-Schema dependency. They are pinned against
the live schema in BOTH directions by
tests/test_judgment_thresholds_against_spec.py; edit the schema first, then
these, or that test fails.

Package-internal, cross-module (both parsers and v1_compat use it), and so not
re-exported from manifest/__init__.py (hard rule 3), the same as
target_validation; except duration_seconds(), which is public and re-exported.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from nthlayer_common.manifest.models import JudgmentThreshold, ThresholdMeasurement

RATIO = "ratio"
DURATION = "duration"

THRESHOLD_BOUNDS: dict[str, dict[str, str]] = {
    "segments": {"maximum_variance_from_overall": RATIO},
    "stability": {"maximum_drift": RATIO},
    "calibration": {
        "maximum_brier_score": RATIO,
        "maximum_expected_calibration_error": RATIO,
    },
    "audit_backlog": {"maximum_age": DURATION},
    "feedback_latency": {"maximum_p50": DURATION, "maximum_p90": DURATION},
}

THRESHOLD_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "segments": ("metric", "segment_dimension", "segment_values"),
    "stability": ("probe",),
    "calibration": ("measurement",),
    "audit_backlog": (),
    "feedback_latency": (),
}

# Keys the schema sets to {"not": {}} on a threshold item: stale carry-overs
# from judgment_slo entries, rejected so they fail instead of being ignored.
FORBIDDEN_THRESHOLD_KEYS: tuple[str, ...] = (
    "breach_actions",
    "target",
    "judgment_type",
    "service",
)

# Every key each type defines besides `type` and `bounds` (CORE §5.5: entries
# are closed, opensrm-7zpz). The value is the closed key set of an
# object-valued key, or None for a non-object key. Closure is enforced from
# this one table, and it is pinned against the schema's per-type branches in
# both directions. calibration's measurement.method is absent on purpose: the
# schema defines it only to forbid it.
THRESHOLD_TYPE_FIELDS: dict[str, dict[str, tuple[str, ...] | None]] = {
    "segments": {
        "metric": None,
        "segment_dimension": None,
        "segment_values": None,
        "measurement": ("window", "source", "confidence_threshold"),
        "outcome_signal": ("source", "window"),
    },
    "stability": {"probe": ("frozen_input_set", "frequency")},
    "calibration": {"measurement": ("bins", "window")},
    "audit_backlog": {},
    "feedback_latency": {},
}

# The rates a segments threshold may compare (CORE §5.5.1). audit_sampling is
# a judgment SLO type but not a segments metric: it measures sampling.
SEGMENTS_METRICS: tuple[str, ...] = (
    "reversal_rate",
    "high_confidence_failure",
    "outcomes",
    "escalation",
)
# measurement.source for metric reversal_rate, as on the reversal_rate SLO.
SEGMENTS_SOURCES: tuple[str, ...] = ("lineage", "calibration_sample", "downstream_signal")
# CORE §5.5.1: measurement.window is optional on segments; the parser stores
# this default so every parsed segments threshold carries its window.
SEGMENTS_DEFAULT_WINDOW = "30d"

_PROBE_FREQUENCIES = ("hourly", "daily", "weekly")

# Longest suffix first: "5ms" must not be read as "5m" + a stray "s".
# Values are seconds per unit; dict order is the match order.
# Seconds per unit as an exact (numerator, denominator) pair. Multiplying by the
# float 0.001 is not correctly rounded (671 of 1..4999 ms differ from n/1000), so
# ms is divided instead: int / int true division in Python is correctly rounded.
_DURATION_SECONDS: dict[str, tuple[int, int]] = {
    "ms": (1, 1000),
    "s": (1, 1),
    "m": (60, 1),
    "h": (3600, 1),
    "d": (86400, 1),
    "w": (604800, 1),
}
_DURATION_UNITS = tuple(_DURATION_SECONDS)


def is_duration(value: object) -> bool:
    """Whether *value* matches schema.json's Duration: ``^[0-9]+(ms|s|m|h|d|w)$``.

    A suffix lookup rather than a regex. ``isascii`` matters: ``str.isdigit``
    accepts superscripts such as ``²``, which the schema's ``[0-9]`` does not.
    """
    if not isinstance(value, str):
        return False
    for unit in _DURATION_UNITS:
        if value.endswith(unit):
            digits = value[: -len(unit)]
            return digits.isascii() and digits.isdigit()
    return False


def duration_seconds(value: str) -> float:
    """Convert a Duration string such as ``"24h"`` to seconds, as a float.

    Accepts exactly what is_duration accepts (``^[0-9]+(ms|s|m|h|d|w)$``) and
    raises ValueError for anything else, including non-strings and bools. For
    evaluators comparing a measured age or latency with a Duration bound.
    """
    if not is_duration(value):
        raise ValueError(
            f"not a duration such as '24h' (<integer><ms|s|m|h|d|w>): {value!r}"
        )
    for unit, (numerator, denominator) in _DURATION_SECONDS.items():
        if value.endswith(unit):
            try:
                return int(value[: -len(unit)]) * numerator / denominator
            # An absurdly long integer: OverflowError past float range, ValueError
            # past CPython's int-digit limit. One message for both.
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"duration too large to represent: {value!r}") from exc
    raise AssertionError(f"unreachable: is_duration accepted {value!r}")  # pragma: no cover


def _ratio(name: str, value: object, what: str = "bound") -> float:
    """Validate a schema Ratio (``type: number``, 0..1); return it as a float.

    A string spelling a number is rejected, unlike target_validation's
    require_number(): the schema's ``type: number`` rejects it too.
    """
    # bool first: it is an int subclass, and YAML 1.1 reads `yes` as True.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{what} '{name}' must be a number, got {type(value).__name__}: {value!r}"
        )
    # Only floats can be non-finite. math.isfinite on a huge YAML int raises
    # OverflowError, which is not a parse error, so ints go straight to range.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{what} '{name}' must be a finite number, got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{what} '{name}' must be in [0, 1], got {value!r}")
    return float(value)


def parse_bounds(threshold_type: str, raw: object) -> dict[str, float | str]:
    """Validate a ``bounds`` mapping for *threshold_type*; return it as declared.

    Raises ValueError naming the offending bound. Callers re-raise as their own
    parser's error type.
    """
    kinds = THRESHOLD_BOUNDS.get(threshold_type)
    if kinds is None:
        raise ValueError(
            f"'{threshold_type}' is not a judgment threshold type; expected one "
            f"of {', '.join(THRESHOLD_BOUNDS)}"
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"bounds must be a mapping, got {type(raw).__name__}: {raw!r}"
        )
    unknown = sorted(set(raw) - set(kinds), key=repr)
    if unknown:
        raise ValueError(
            f"bound(s) {', '.join(map(str, unknown))} do not belong to "
            f"'{threshold_type}'; its bounds are {', '.join(kinds)}"
        )
    if not raw:
        if len(kinds) == 1:
            raise ValueError(f"'{threshold_type}' requires bound '{next(iter(kinds))}'")
        raise ValueError(
            f"'{threshold_type}' requires at least one of: {', '.join(kinds)}"
        )
    bounds: dict[str, float | str] = {}
    for name, value in raw.items():
        if kinds[name] == RATIO:
            bounds[name] = _ratio(name, value)
        elif is_duration(value):
            bounds[name] = value  # type: ignore[assignment]  # is_duration narrows to str
        else:
            raise ValueError(
                f"bound '{name}' must be a duration such as '24h' "
                f"(<integer><ms|s|m|h|d|w>), got {value!r}"
            )
    return bounds


def normalise_bins(value: int | float) -> int:
    """Return a validated ``measurement.bins`` as an int.

    ``check_type_fields`` accepts an integral float (10.0) because the schema's
    ``type: integer`` does, but it never rewrites its input. Callers building a
    ThresholdMeasurement from a validated entry must pass bins through this.
    """
    return int(value)


def _reject_undefined_keys(where: str, keys: Iterable[Any], allowed: tuple[str, ...]) -> None:
    """Closure (CORE §5.5): raise naming every key of *keys* not in *allowed*.

    Sorted by repr: YAML admits int and null keys beside strings.
    """
    unknown = sorted((k for k in keys if k not in allowed), key=repr)
    if unknown:
        raise ValueError(
            f"{', '.join(repr(k) for k in unknown)} {'is' if len(unknown) == 1 else 'are'} "
            f"not defined on {where} (OPENSRM-CORE-v2 §5.5: threshold entries are "
            f"closed); its keys are {', '.join(allowed) or 'none'}"
        )


def _is_zero_duration(value: str) -> bool:
    """Whether a valid Duration is zero: schema ``^0+(ms|s|m|h|d|w)$``."""
    return set(value.rstrip("mshdw")) == {"0"}


def _check_closed_objects(threshold_type: str, entry: dict[str, Any]) -> None:
    """Every object-valued key present is a mapping holding only its keys.

    calibration's measurement.method is reported first with its own message:
    it is the one key a 3.x manifest is known to carry there.
    """
    for key, allowed in THRESHOLD_TYPE_FIELDS[threshold_type].items():
        if allowed is None or key not in entry:
            continue
        value = entry[key]
        if not isinstance(value, dict):
            raise ValueError(f"'{key}' must be a mapping, got {type(value).__name__}: {value!r}")
        if threshold_type == "calibration" and key == "measurement" and "method" in value:
            raise ValueError(
                "'measurement.method' is not part of a calibration threshold in "
                "OpenSRM v2 (OPENSRM-CORE-v2 §5.5); the declared "
                "bounds decide what is evaluated"
            )
        _reject_undefined_keys(f"a '{threshold_type}' threshold's '{key}'", value, allowed)


# Each metric's parameter (CORE §5.5.1): required with that metric, rejected
# with any other. (location, key) - location None is the entry itself.
_SEGMENTS_METRIC_PARAMETER: dict[str, tuple[str | None, str]] = {
    "reversal_rate": ("measurement", "source"),
    "high_confidence_failure": ("measurement", "confidence_threshold"),
    "outcomes": (None, "outcome_signal"),
}


def _check_segments(entry: dict[str, Any]) -> None:
    """The segments-specific rules (CORE §5.5.1). Closure is already checked."""
    # Emptiness is enforced by the required-field guard.
    if not isinstance(entry["segment_dimension"], str):
        raise ValueError("'segment_dimension' must be a non-empty string")
    values = entry["segment_values"]
    if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
        raise ValueError("'segment_values' must be a non-empty list of non-empty strings")

    metric = entry["metric"]
    if metric == "audit_sampling":
        raise ValueError(
            "'metric' cannot be 'audit_sampling': it measures how decisions are "
            "sampled, not how they perform (OPENSRM-CORE-v2 §5.5.1); use one of "
            f"{', '.join(SEGMENTS_METRICS)}"
        )
    if not isinstance(metric, str) or metric not in SEGMENTS_METRICS:
        raise ValueError(
            f"'metric' must be one of {', '.join(SEGMENTS_METRICS)} "
            f"(OPENSRM-CORE-v2 §5.5.1), got {metric!r}"
        )

    measurement: dict[str, Any] = entry.get("measurement", {})
    for owner, (location, key) in _SEGMENTS_METRIC_PARAMETER.items():
        holder = entry if location is None else measurement
        name = key if location is None else f"{location}.{key}"
        if owner == metric and key not in holder:
            raise ValueError(f"metric '{metric}' requires '{name}' (OPENSRM-CORE-v2 §5.5.1)")
        if owner != metric and key in holder:
            raise ValueError(
                f"'{name}' parameterises metric '{owner}' only, not '{metric}' "
                f"(OPENSRM-CORE-v2 §5.5.1)"
            )

    if "window" in measurement:
        window = measurement["window"]
        if not is_duration(window):
            raise ValueError(
                f"'measurement.window' must be a duration such as '30d', got {window!r}"
            )
        if _is_zero_duration(window):
            raise ValueError(
                f"'measurement.window' must be greater than zero: an empty population "
                f"window could never be evaluated (OPENSRM-CORE-v2 §5.5.1), got {window!r}"
            )
    if "source" in measurement:
        source = measurement["source"]
        if not isinstance(source, str) or source not in SEGMENTS_SOURCES:
            raise ValueError(
                f"'measurement.source' must be one of {', '.join(SEGMENTS_SOURCES)}, "
                f"got {source!r}"
            )
    if "confidence_threshold" in measurement:
        _ratio("measurement.confidence_threshold", measurement["confidence_threshold"], "field")
    if "outcome_signal" in entry:
        signal = entry["outcome_signal"]
        for key in ("source", "window"):
            if key not in signal:
                raise ValueError(f"'outcome_signal' requires '{key}' (OPENSRM-CORE-v2 §5.5.1)")
        if not isinstance(signal["source"], str):
            raise ValueError(
                f"'outcome_signal.source' must be a string, got {signal['source']!r}"
            )
        # Unlike the population window, the outcome wait may be zero (as on
        # the outcomes judgment SLO, whose Duration it shares).
        if not is_duration(signal["window"]):
            raise ValueError(
                f"'outcome_signal.window' must be a duration such as '1h', got "
                f"{signal['window']!r}"
            )


def check_type_fields(threshold_type: str, entry: dict[str, Any]) -> None:
    """Enforce the non-``bounds`` fields schema.json defines per type.

    Order: forbidden keys, then entry closure, then required fields, then the
    closure of each object-valued key, then type-specific shape. ``type`` and
    ``bounds`` are admitted but not validated here: the parser checks
    ``type`` and ``parse_bounds`` checks ``bounds``.
    """
    for key in FORBIDDEN_THRESHOLD_KEYS:
        if key in entry:
            raise ValueError(
                f"'{key}' is not allowed on a '{threshold_type}' threshold; "
                f"thresholds do not carry it (OPENSRM-CORE-v2 §5.5)"
            )
    fields = THRESHOLD_TYPE_FIELDS[threshold_type]
    _reject_undefined_keys(
        f"a '{threshold_type}' threshold", entry, ("type", "bounds", *fields)
    )
    for name in THRESHOLD_REQUIRED_FIELDS.get(threshold_type, ()):
        if entry.get(name) in (None, "", [], {}):
            raise ValueError(f"'{threshold_type}' requires '{name}'")
    _check_closed_objects(threshold_type, entry)

    if threshold_type == "segments":
        _check_segments(entry)
    elif threshold_type == "stability":
        probe = entry["probe"]
        if not isinstance(probe.get("frozen_input_set"), str):
            raise ValueError("'probe' must be a mapping with a 'frozen_input_set' string")
        if probe.get("frequency") not in _PROBE_FREQUENCIES:
            raise ValueError(
                f"'probe.frequency' must be one of {', '.join(_PROBE_FREQUENCIES)}, "
                f"got {probe.get('frequency')!r}"
            )
    elif threshold_type == "calibration":
        measurement = entry["measurement"]
        bins = measurement.get("bins")
        # schema.json: bins is `type: integer`, which draft-07 satisfies with an
        # integral float such as 10.0, so the parser accepts it too. This is a
        # pure check; callers convert with normalise_bins().
        valid_float = isinstance(bins, float) and bins.is_integer()
        if isinstance(bins, bool) or not (isinstance(bins, int) or valid_float) or bins < 2:
            raise ValueError(f"'measurement.bins' must be an integer >= 2, got {bins!r}")
        if not is_duration(measurement.get("window")):
            raise ValueError(
                f"'measurement.window' must be a duration such as '30d', got "
                f"{measurement.get('window')!r}"
            )


# srm/v1 defaulted calibration to 10 bins (before 3c, v1_compat.default_measurement);
# v2's calibration threshold requires bins, so the lift keeps that default.
_SRM_V1_CALIBRATION_BINS = 10


# Bound-like keys an srm/v1 entry may carry that the lift has no home for. Checked
# before anything else so they cannot be dropped, nor hidden behind a
# "target must be..." message when target is absent.
_SRM_V1_UNCARRIABLE_KEYS: dict[str, tuple[str, ...]] = {
    "feedback_latency": ("p50", "p90"),
    "calibration": ("bins", "method"),
}


def srm_v1_threshold(name: str, config: dict[str, Any]) -> JudgmentThreshold:
    """Lift an srm/v1 ``spec.slos.<name>`` entry into a JudgmentThreshold.

    Shared by parser/v1.py and v1_compat.convert_v1_to_v2 so the two cannot
    map one srm/v1 entry to different thresholds. Raises ValueError.
    """
    if name == "audit_backlog":
        raise ValueError(
            "srm/v1 cannot declare 'audit_backlog': it has no srm/v1 form. "
            "Declare it in an opensrm.nthlayer.io/v2 manifest under "
            "spec.judgment_threshold."
        )
    if name in ("segments", "stability"):
        raise ValueError(
            f"srm/v1 cannot declare '{name}': its threshold form needs "
            f"{' and '.join(THRESHOLD_REQUIRED_FIELDS[name])}, which the flat "
            f"srm/v1 format cannot express. Declare it in an "
            f"opensrm.nthlayer.io/v2 manifest under spec.judgment_threshold."
        )
    if name in _SRM_V1_UNCARRIABLE_KEYS:
        lost = sorted(
            (
                k for k in config
                if k in _SRM_V1_UNCARRIABLE_KEYS[name]
                or (isinstance(k, str) and k.startswith("maximum_"))
            ),
            key=repr,
        )
        if lost:
            raise ValueError(
                f"srm/v1 '{name}' cannot express {', '.join(map(str, lost))}: the "
                f"lift would drop it silently. Declare the threshold in an "
                f"opensrm.nthlayer.io/v2 manifest under spec.judgment_threshold "
                f"(OPENSRM-CORE-v2 §5.5)."
            )
    target = config.get("target")
    if name == "calibration":
        if target is None:
            raise ValueError("srm/v1 'calibration' requires a target (maximum ECE)")
        window = config.get("window", "30d")
        if not is_duration(window):
            raise ValueError(f"srm/v1 'calibration' window must be a duration, got {window!r}")
        bound = _ratio("maximum_expected_calibration_error", target)
        return JudgmentThreshold(
            type="calibration",
            bounds={"maximum_expected_calibration_error": bound},
            measurement=ThresholdMeasurement(window=window, bins=_SRM_V1_CALIBRATION_BINS),
        )
    if name == "feedback_latency":
        # An integral float (3600.0) is a whole number of seconds, as for bins.
        whole = isinstance(target, int) or (isinstance(target, float) and target.is_integer())
        if isinstance(target, bool) or not whole or target < 0:
            raise ValueError(
                f"srm/v1 'feedback_latency' target must be a non-negative whole "
                f"number of seconds, got {target!r}"
            )
        return JudgmentThreshold(type="feedback_latency", bounds={"maximum_p50": f"{int(target)}s"})
    raise ValueError(f"'{name}' is not an srm/v1 judgment threshold")


def threshold_to_v2_entry(threshold: JudgmentThreshold) -> dict[str, Any]:
    """The flat ``spec.judgment_threshold`` entry for *threshold*.

    Inverse of parser/v2.py::_parse_judgment_threshold for every type, so that
    convert_v1_to_v2's output re-parses to an equal JudgmentThreshold.
    """
    entry: dict[str, Any] = {"type": threshold.type, "bounds": dict(threshold.bounds)}
    m = threshold.measurement
    if threshold.type == "calibration":
        entry["measurement"] = {"bins": m.bins, "window": m.window}
    elif threshold.type == "stability" and m.probe is not None:
        entry["probe"] = {"frozen_input_set": m.probe.frozen_input_set, "frequency": m.probe.frequency}
    elif threshold.type == "segments":
        entry["metric"] = m.metric
        # window always: the parser stores the 30d default when it is absent.
        measurement: dict[str, Any] = {"window": m.window}
        if m.source is not None:
            measurement["source"] = m.source
        if m.confidence_threshold is not None:
            measurement["confidence_threshold"] = m.confidence_threshold
        entry["measurement"] = measurement
        if m.outcome_signal is not None:
            entry["outcome_signal"] = {
                "source": m.outcome_signal.source,
                "window": m.outcome_signal.window,
            }
        entry["segment_dimension"] = m.segment_dimension
        entry["segment_values"] = list(m.segment_values)
    return entry
