"""Judgment-threshold evaluation per OPENSRM-CORE-v2 §5.5.6 (opensrm-g32d).

Pure and deterministic: no I/O. A sampler (nthlayer-workers) supplies the raw
measurements the spec defines; evaluate_threshold applies E1-E4 and returns a
ThresholdOutcome, whose to_dict() is the judgment_threshold_evaluation
assessment payload.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from nthlayer_common.manifest.models import JudgmentThreshold
from nthlayer_common.manifest.thresholds import DURATION, THRESHOLD_BOUNDS, duration_seconds

Status = Literal["within", "breach", "not_evaluable"]
MissingReason = Literal["no_data_source", "no_data", "insufficient_window"]
ResultReason = MissingReason | Literal["lower_bound"]

_STATUSES: tuple[str, ...] = get_args(Status)
_REASONS: tuple[str, ...] = get_args(MissingReason)
# E4: breach > not_evaluable > within.
_PRECEDENCE = {"within": 0, "not_evaluable": 1, "breach": 2}


@dataclass(frozen=True)
class Missing:
    """A measurement that could not be taken (E3). ``reason`` says why:

    - ``no_data_source``: no series is defined for this threshold type/bound.
    - ``no_data``: the series is absent, its labels do not match, the query
      failed, or the value is non-finite.
    - ``insufficient_window``: Prometheus retention is shorter than the window.
    """

    reason: MissingReason
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in _REASONS:
            raise ValueError(f"Missing.reason must be one of {_REASONS}, got {self.reason!r}")


@dataclass(frozen=True)
class AtLeast:
    """A lower bound: the true value is strictly greater than ``value``.

    What a classic Prometheus histogram yields when a quantile falls in the
    ``+Inf`` bucket: the value lies beyond the largest finite boundary, which
    histogram_quantile would instead return the largest finite boundary,
    silently capping the value."""

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"AtLeast.value must be finite, got {self.value!r}")


@dataclass(frozen=True)
class Samples:
    """The raw measurements for one threshold.

    A bound omitted from ``bounds`` is reported as ``no_data``. A bool is not
    data (True == 1.0 would otherwise pass as a rate) and is treated the same.
    Segments carry rates; the evaluator computes |segment - overall| itself.

    ``bounds`` maps a bound name to its measured value for every type except
    segments; a value may be a lower bound (``AtLeast``). For segments,
    ``overall`` is the overall rate and ``segments`` maps each listed segment
    value to its rate; the evaluator computes the variance itself (CORE §5.5.1).
    """

    bounds: Mapping[str, float | Missing | AtLeast] = field(default_factory=dict)
    overall: float | Missing | None = None
    segments: Mapping[str, float | Missing] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundResult:
    """The outcome for one bound. ``limit`` is as declared (a ratio float or a
    Duration string). ``reason`` is a MissingReason when not_evaluable,
    ``"lower_bound"`` when a breach was decided from an AtLeast, else None.
    For a segments threshold, ``measured`` is the largest evaluated variance
    and ``reason`` the first segment reason."""

    bound: str
    status: Status
    measured: float | None
    limit: float | str
    reason: ResultReason | None


@dataclass(frozen=True)
class SegmentResult:
    """The outcome for one segment value. ``measured`` is |segment rate -
    overall rate|, rounded to 12 places. ``reason`` is a MissingReason when
    not_evaluable, else None."""

    segment: str
    status: Status
    measured: float | None
    reason: ResultReason | None


def combine(statuses: Iterable[str]) -> Status:
    """E4: breach > not_evaluable > within. Nothing decided -> not_evaluable."""
    best: Status = "within"
    seen = False
    for s in statuses:
        if s not in _PRECEDENCE:
            raise ValueError(f"unknown status {s!r}")
        seen = True
        if _PRECEDENCE[s] > _PRECEDENCE[best]:
            best = s  # type: ignore[assignment]
    return best if seen else "not_evaluable"


@dataclass(frozen=True)
class ThresholdOutcome:
    """The evaluation of one threshold: ``status`` is the E4 combination of
    its ``bounds`` (one BoundResult per bound, each ``limit`` as declared: a
    ratio float or a Duration string) and, for segments, ``segments``.
    ``to_dict()`` is the judgment_threshold_evaluation assessment payload."""

    type: str
    status: Status
    bounds: tuple[BoundResult, ...]
    segments: tuple[SegmentResult, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "status": self.status,
            "bounds": [
                {
                    "bound": b.bound,
                    "status": b.status,
                    "measured": b.measured,
                    "limit": b.limit,
                    "reason": b.reason,
                }
                for b in self.bounds
            ],
            "segments": [
                {
                    "segment": s.segment,
                    "status": s.status,
                    "measured": s.measured,
                    "reason": s.reason,
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ThresholdOutcome:
        return cls(
            type=d["type"],
            status=d["status"],
            bounds=tuple(BoundResult(**b) for b in d["bounds"]),
            segments=tuple(SegmentResult(**s) for s in d.get("segments", ())),
        )


def _limit_value(threshold_type: str, bound: str, limit: float | str) -> float:
    kinds = THRESHOLD_BOUNDS[threshold_type]
    if bound not in kinds:
        # Parsed thresholds never get here (parse_bounds rejects it); a
        # hand-built one gets a ValueError, not a KeyError.
        raise ValueError(f"{threshold_type} has no bound {bound!r}; expected one of {sorted(kinds)}")
    if kinds[bound] == DURATION:
        return duration_seconds(limit)  # type: ignore[arg-type]
    return float(limit)


def _decide(
    measured: float | Missing | AtLeast | None, limit: float
) -> tuple[Status, float | None, str | None]:
    """E1 for one measured value; E3 for a missing or non-finite one.

    A lower bound (AtLeast) breaches when it already reaches the limit: the
    true value is strictly greater than it, so strictly greater than the
    limit. AtLeast below the limit is reported not_evaluable/no_data: the true
    value is above a boundary that is itself within the bound, so neither
    breach nor within can be decided."""
    if measured is None or isinstance(measured, bool):  # a bool is not a measurement
        return "not_evaluable", None, "no_data"
    if isinstance(measured, Missing):
        return "not_evaluable", None, measured.reason
    if isinstance(measured, AtLeast):
        if measured.value >= limit:
            return "breach", measured.value, "lower_bound"
        return "not_evaluable", None, "no_data"
    if not math.isfinite(measured):
        return "not_evaluable", None, "no_data"
    return ("breach" if measured > limit else "within"), measured, None


def evaluate_threshold(threshold: JudgmentThreshold, samples: Samples) -> ThresholdOutcome:
    """Apply CORE §5.5.6 E1-E4 to one threshold. Pure; raises ValueError for
    a type that is not a judgment threshold type, a bound that type does not
    define, or a segments threshold without exactly one bound."""
    t = threshold.type
    if t not in THRESHOLD_BOUNDS:
        raise ValueError(f"not a judgment threshold type: {t!r}")
    if t == "segments":
        return _evaluate_segments(threshold, samples)
    results = []
    for bound, limit in threshold.bounds.items():
        status, measured, reason = _decide(samples.bounds.get(bound), _limit_value(t, bound, limit))
        results.append(BoundResult(bound, status, measured, limit, reason))
    return ThresholdOutcome(t, combine(r.status for r in results), tuple(results))


def _evaluate_segments(threshold: JudgmentThreshold, samples: Samples) -> ThresholdOutcome:
    """CORE §5.5.1: each listed segment's rate is compared with the overall
    rate; the bound caps |segment - overall| in either direction."""
    if len(threshold.bounds) != 1:
        raise ValueError(f"a segments threshold has exactly one bound, got {sorted(threshold.bounds)}")
    ((bound, limit),) = threshold.bounds.items()
    limit_v = _limit_value("segments", bound, limit)
    overall = samples.overall
    results: list[SegmentResult] = []
    for value in dict.fromkeys(threshold.measurement.segment_values):  # distinct, in order
        if overall is None or isinstance(overall, (Missing, bool)) or not math.isfinite(overall):
            reason = overall.reason if isinstance(overall, Missing) else "no_data"
            results.append(SegmentResult(value, "not_evaluable", None, reason))
            continue
        rate = samples.segments.get(value)
        if rate is None or isinstance(rate, (Missing, bool)):
            variance = rate
        else:
            # `measured` is this rounded variance. Bounds are decimal and binary
            # subtraction is noisy (0.9 - 0.7 > 0.2), which would breach an exact
            # equality (E1). Differences under 1e-12 are below any rate
            # resolution, so a variance within 1e-12 of the bound counts as equal.
            variance = round(abs(rate - overall), 12)
        status, measured, reason = _decide(variance, limit_v)
        results.append(SegmentResult(value, status, measured, reason))
    status = combine(s.status for s in results)
    evaluated = [s.measured for s in results if s.measured is not None]
    reason = None if status != "not_evaluable" else next(
        (x.reason for x in results if x.reason), "no_data"
    )
    bound_result = BoundResult(bound, status, max(evaluated) if evaluated else None, limit, reason)
    return ThresholdOutcome("segments", status, (bound_result,), tuple(results))
