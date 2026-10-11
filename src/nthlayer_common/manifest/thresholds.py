"""Judgment THRESHOLDS: bounds on error magnitudes (decision 3c, opensrm-l33e).

Not SLOs. A threshold has no population of good events, no complement and no
error budget; a breach is the measured value exceeding a declared bound. So
nothing here converts a value: bounds are stored exactly as declared, which is
the opposite of target_validation.py's job for judgment SLO targets.

The tables restate opensrm/spec/v2/schema.json's JudgmentThresholdSpec so the
parsers can enforce it without a JSON-Schema dependency. They are pinned against
the live schema in BOTH directions by
tests/test_judgment_thresholds_against_spec.py; edit the schema first, then
these, or that test fails.

Package-internal, cross-module (both parsers and v1_compat use it), and so not
re-exported from manifest/__init__.py (hard rule 3), the same as
target_validation.
"""

from __future__ import annotations

import math
from typing import Any

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
_DURATION_UNITS = ("ms", "s", "m", "h", "d", "w")


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


def _ratio(name: str, value: object) -> float:
    # bool first: it is an int subclass, and YAML 1.1 reads `yes` as True.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"bound '{name}' must be a number, got {type(value).__name__}: {value!r}"
        )
    if not math.isfinite(value):
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
    unknown = sorted(set(raw) - set(kinds))
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


def check_type_fields(threshold_type: str, entry: dict[str, Any]) -> None:
    """Enforce the non-``bounds`` fields schema.json requires per type.

    Order: forbidden keys, then required fields, then type-specific shape.
    """
    for key in FORBIDDEN_THRESHOLD_KEYS:
        if key in entry:
            raise ValueError(
                f"'{key}' is not allowed on a '{threshold_type}' threshold; "
                f"thresholds do not carry it (opensrm-l33e)"
            )
    for name in THRESHOLD_REQUIRED_FIELDS.get(threshold_type, ()):
        if entry.get(name) in (None, "", [], {}):
            raise ValueError(f"'{threshold_type}' requires '{name}'")

    if threshold_type == "segments":
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
                "'measurement.method' was removed in opensrm-l33e; the declared "
                "bounds decide what is evaluated"
            )
        bins = measurement.get("bins")
        # schema.json: bins is `type: integer`, which draft-07 satisfies with an
        # integral float such as 10.0, so the parser accepts it too.
        if isinstance(bins, float) and bins.is_integer():
            bins = int(bins)
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError(f"'measurement.bins' must be an integer >= 2, got {bins!r}")
        measurement["bins"] = bins
        if not is_duration(measurement.get("window")):
            raise ValueError(
                f"'measurement.window' must be a duration such as '30d', got "
                f"{measurement.get('window')!r}"
            )
