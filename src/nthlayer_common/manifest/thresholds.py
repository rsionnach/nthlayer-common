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
    "segments": ("segment_dimension", "segment_values"),
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

_PROBE_FREQUENCIES = ("hourly", "daily", "weekly")

# Longest suffix first: "5ms" must not be read as "5m" + a stray "s".
# Values are seconds per unit; dict order is the match order.
_DURATION_SECONDS: dict[str, float] = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 604800.0,
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
    for unit, seconds in _DURATION_SECONDS.items():
        if value.endswith(unit):
            return int(value[: -len(unit)]) * seconds
    raise AssertionError(f"unreachable: is_duration accepted {value!r}")  # pragma: no cover


def _ratio(name: str, value: object) -> float:
    # bool first: it is an int subclass, and YAML 1.1 reads `yes` as True.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"bound '{name}' must be a number, got {type(value).__name__}: {value!r}"
        )
    # Only floats can be non-finite. math.isfinite on a huge YAML int raises
    # OverflowError, which is not a parse error, so ints go straight to range.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"bound '{name}' must be a finite number, got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"bound '{name}' must be in [0, 1], got {value!r}")
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


def check_type_fields(threshold_type: str, entry: dict[str, Any]) -> None:
    """Enforce the non-``bounds`` fields schema.json requires per type.

    Order: forbidden keys, then required fields, then type-specific shape.
    """
    for key in FORBIDDEN_THRESHOLD_KEYS:
        if key in entry:
            raise ValueError(
                f"'{key}' is not allowed on a '{threshold_type}' threshold; "
                f"thresholds do not carry it (OPENSRM-CORE-v2 §5.5)"
            )
    for name in THRESHOLD_REQUIRED_FIELDS.get(threshold_type, ()):
        if entry.get(name) in (None, "", [], {}):
            raise ValueError(f"'{threshold_type}' requires '{name}'")

    if threshold_type == "segments":
        # Emptiness is enforced by the required-field guard above.
        if not isinstance(entry["segment_dimension"], str):
            raise ValueError("'segment_dimension' must be a non-empty string")
        values = entry["segment_values"]
        if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
            raise ValueError("'segment_values' must be a non-empty list of non-empty strings")
    elif threshold_type == "stability":
        probe = entry["probe"]
        if not isinstance(probe, dict) or not isinstance(probe.get("frozen_input_set"), str):
            raise ValueError("'probe' must be a mapping with a 'frozen_input_set' string")
        if probe.get("frequency") not in _PROBE_FREQUENCIES:
            raise ValueError(
                f"'probe.frequency' must be one of {', '.join(_PROBE_FREQUENCIES)}, "
                f"got {probe.get('frequency')!r}"
            )
    elif threshold_type == "calibration":
        measurement = entry["measurement"]
        if not isinstance(measurement, dict):
            raise ValueError("'measurement' must be a mapping")
        if "method" in measurement:
            raise ValueError(
                "'measurement.method' is not part of a calibration threshold in "
                "OpenSRM v2 (OPENSRM-CORE-v2 §5.5); the declared "
                "bounds decide what is evaluated"
            )
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
        entry["segment_dimension"] = m.segment_dimension
        entry["segment_values"] = list(m.segment_values)
    return entry
