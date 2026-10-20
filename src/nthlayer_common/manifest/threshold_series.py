"""The judgment-threshold metric contract (opensrm-g32d).

Which Prometheus series measures each threshold bound. This table is the
source of truth: nthlayer-workers builds its queries from it and the front
door's docs/metrics-contract.md documents it. Pinned against THRESHOLD_BOUNDS
in tests/test_threshold_series.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# CORE §5.5.5: feedback latency is computed over the last 30 days, anchored on
# ground-truth arrival; a manifest cannot change it.
FEEDBACK_LATENCY_WINDOW = "30d"


@dataclass(frozen=True)
class ThresholdSeries:
    """One bound's series. ``labels`` are the labels a sampler must match
    (``bins``/``window`` take the manifest's measurement values); ``quantile``
    is set only for a ``histogram_quantile`` series: a quantile of a classic
    histogram, which the sampler computes from the ``_bucket`` counts (it
    does not send PromQL's histogram_quantile, which hides a quantile in the
    +Inf bucket)."""

    metric: str
    kind: Literal["gauge", "histogram_quantile"]
    labels: tuple[str, ...]
    quantile: float | None = None


_CALIBRATION_LABELS = ("service", "bins", "window")

THRESHOLD_SERIES: dict[str, dict[str, ThresholdSeries]] = {
    "calibration": {
        "maximum_expected_calibration_error": ThresholdSeries(
            "gen_ai_calibration_ece", "gauge", _CALIBRATION_LABELS
        ),
        "maximum_brier_score": ThresholdSeries(
            "gen_ai_calibration_brier", "gauge", _CALIBRATION_LABELS
        ),
    },
    "feedback_latency": {
        "maximum_p50": ThresholdSeries(
            "gen_ai_feedback_latency_seconds", "histogram_quantile", ("service",), 0.5
        ),
        "maximum_p90": ThresholdSeries(
            "gen_ai_feedback_latency_seconds", "histogram_quantile", ("service",), 0.9
        ),
    },
    # No data source yet (opensrm-g32d follow-up beads): evaluated as
    # not evaluable, reason no_data_source.
    "segments": {},
    "stability": {},
    "audit_backlog": {},
}
