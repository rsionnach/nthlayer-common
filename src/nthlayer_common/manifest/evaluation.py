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

_STATUSES: tuple[str, ...] = get_args(Status)
_REASONS: tuple[str, ...] = get_args(MissingReason)
# E4: breach > not_evaluable > within.
_PRECEDENCE = {"within": 0, "not_evaluable": 1, "breach": 2}


@dataclass(frozen=True)
class Missing:
    """A measurement that could not be taken (E3)."""

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
    PromQL's histogram_quantile would report as the value itself."""

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"AtLeast.value must be finite, got {self.value!r}")


@dataclass(frozen=True)
class Samples:
    """The raw measurements for one threshold.

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
    bound: str
    status: Status
    measured: float | None
    limit: float | str  # as declared: a ratio, or a Duration string
    reason: str | None


@dataclass(frozen=True)
class SegmentResult:
    segment: str
    status: Status
    measured: float | None  # |segment rate - overall rate|
    reason: str | None


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
    limit. Below the limit it decides nothing, so it is not evaluable."""
    if measured is None:
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
    raise NotImplementedError("segments: Task 5")
