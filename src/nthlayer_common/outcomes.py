"""Financial impact calculation primitive (spec § 1).

Pure function over an ``Outcomes`` block plus an observed decision count.
Both nthlayer-workers retrospectives and correlate blast-radius assessments
call into this; downstream consumers receive the same shape regardless of
where the figure was produced.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from nthlayer_common.manifest.models import Outcomes


# Spec § 1 field-value enums.
VolumeSource = Literal["metric", "spec_estimate"]
FailureMode = Literal["false_positive", "false_negative"]


@dataclass
class FinancialImpact:
    """Per-service financial impact figure derived from an Outcomes block.

    ``estimated`` is the headline number; ``decisions_affected`` and
    ``volume_source`` carry provenance so retrospective readers can tell
    metric-derived figures from spec-fallback ones.
    """

    estimated: float
    currency: str
    decisions_affected: int
    failure_mode: FailureMode
    volume_source: VolumeSource


def compute_financial_impact(
    outcomes: Outcomes,
    *,
    decisions_affected: int,
    failure_mode: FailureMode,
    volume_source: VolumeSource,
) -> FinancialImpact | None:
    """Multiply per-failure cost by impacted decision count.

    Returns ``None`` when ``outcomes`` lacks a cost figure for the
    requested failure mode (a service can declare ``decision_value``
    without enumerating both failure costs). Callers treat None as
    "no financial signal available" rather than zero.
    """
    if decisions_affected < 0:
        raise ValueError(
            f"decisions_affected must be >= 0; got {decisions_affected}"
        )

    cost_field = (
        outcomes.decision_value.false_positive
        if failure_mode == "false_positive"
        else outcomes.decision_value.false_negative
    )
    if cost_field is None:
        return None

    # Cast to float defensively — YAML loaders that emit Decimal/int
    # would otherwise propagate mixed types into downstream rounding.
    return FinancialImpact(
        estimated=float(decisions_affected) * float(cost_field.cost),
        currency=outcomes.decision_value.currency,
        decisions_affected=decisions_affected,
        failure_mode=failure_mode,
        volume_source=volume_source,
    )


def estimate_decisions_in_window(
    outcomes: Outcomes, *, window: timedelta,
) -> int | None:
    """Spec-fallback decision count for an incident window.

    Used when real-time ``gen_ai_decision_total`` metrics are absent.
    Returns ``None`` when the spec lacks a daily estimate or the
    window is non-positive (a zero-duration window is "we don't know
    how long the incident lasted," not "no impact" — callers omit
    the financial figure rather than report 0).
    """
    volume = outcomes.volume
    if volume is None or volume.estimated_daily_decisions is None:
        return None
    seconds = window.total_seconds()
    if seconds <= 0:
        return None
    daily_seconds = 86400.0
    raw = volume.estimated_daily_decisions * (seconds / daily_seconds)
    # Floor to 1 when there's a non-zero daily estimate but the window
    # is short enough that integer truncation would silently report
    # "no impact" — a 30-second incident on a 1000/day service still
    # has at least one decision exposed.
    if 0 < raw < 1 and volume.estimated_daily_decisions > 0:
        return 1
    return int(raw)
