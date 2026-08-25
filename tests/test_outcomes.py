"""Tests for Outcomes (business outcome binding) — opensrm-jmy.1.

Covers spec § 1: schema validation, v2 parser integration, the
``compute_financial_impact`` primitive, and the spec-fallback volume
estimator.

Spec scenarios covered: 2 (spec-estimate fallback path), 3 (no outcomes
→ no error), 6 (currency rejection). Scenarios 1 (real-time metric path)
and 4 (correlate blast-radius sum) and 5 (Grafana panel generation) are
deferred to opensrm-jmy.1.2/.3 since they touch worker and generate
repos respectively.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from nthlayer_common.manifest import (
    DecisionValue,
    FailureCost,
    Outcomes,
    RevenueAttribution,
    VolumeEstimate,
)
from nthlayer_common.manifest.parser.v2 import (
    OpenSRMV2ParseError,
    parse_opensrm_v2,
)
from nthlayer_common.outcomes import (
    FinancialImpact,
    compute_financial_impact,
    estimate_decisions_in_window,
)

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestOutcomesModel:
    def test_decision_value_currency_must_be_iso_4217(self):
        with pytest.raises(ValueError, match="ISO 4217"):
            DecisionValue(correct=100.0, currency="EURO")
        with pytest.raises(ValueError, match="ISO 4217"):
            DecisionValue(correct=100.0, currency="eur")
        with pytest.raises(ValueError, match="ISO 4217"):
            DecisionValue(correct=100.0, currency="€")

    def test_decision_value_accepts_valid_currency(self):
        dv = DecisionValue(correct=138.50, currency="EUR")
        assert dv.currency == "EUR"

    def test_failure_cost_rejects_negative_cost(self):
        with pytest.raises(ValueError, match=">= 0"):
            FailureCost(cost=-1.0)

    def test_failure_cost_rejects_unknown_category(self):
        with pytest.raises(ValueError, match="category"):
            FailureCost(cost=2400.0, category="unknown_bucket")

    def test_failure_cost_accepts_known_categories(self):
        for cat in (
            "financial_loss", "friction", "compliance",
            "reputational", "operational",
        ):
            assert FailureCost(cost=10.0, category=cat).category == cat

    def test_revenue_attribution_enum_validated(self):
        with pytest.raises(ValueError, match="attribution"):
            RevenueAttribution(attribution="random")

    def test_revenue_signal_must_be_identifier_shape(self):
        # nthlayer-generate maps the signal to a backend metric — reject
        # shapes the metric plane could not represent.
        with pytest.raises(ValueError, match="identifier"):
            RevenueAttribution(signal="approved transactions value")
        with pytest.raises(ValueError, match="identifier"):
            RevenueAttribution(signal="approved-transactions")

    def test_volume_estimate_rejects_negative(self):
        with pytest.raises(ValueError, match=">= 0"):
            VolumeEstimate(estimated_daily_decisions=-1)
        with pytest.raises(ValueError, match=">= 0"):
            VolumeEstimate(peak_multiplier=-0.5)


# ---------------------------------------------------------------------------
# v2 parser integration
# ---------------------------------------------------------------------------


def _v2_skeleton(spec_extras: dict) -> dict:
    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {
            "name": "fraud-detection",
            "labels": {"tier": "critical", "type": "ai-gate"},
        },
        "spec": {
            "owner": {"group": "group:default/payments"},
            "service": {"name": "fraud-detection", "type": "ai-gate"},
            **spec_extras,
        },
    }


class TestOutcomesParser:
    def test_no_outcomes_block_yields_none(self):
        # Spec §1 test scenario 3: no outcomes → no errors, no fields.
        manifest = parse_opensrm_v2(_v2_skeleton({}))
        assert manifest.outcomes is None

    def test_full_outcomes_round_trip(self):
        outcomes = {
            "decision_value": {
                "correct": 138.50,
                "currency": "EUR",
                "false_positive": {"cost": 0, "category": "friction"},
                "false_negative": {"cost": 2400, "category": "financial_loss"},
            },
            "revenue": {
                "attribution": "direct",
                "signal": "approved_transaction_value",
            },
            "volume": {
                "estimated_daily_decisions": 12000,
                "peak_multiplier": 2.3,
            },
        }
        manifest = parse_opensrm_v2(_v2_skeleton({"outcomes": outcomes}))
        assert manifest.outcomes is not None
        dv = manifest.outcomes.decision_value
        assert dv.correct == 138.50
        assert dv.currency == "EUR"
        assert dv.false_negative is not None
        assert dv.false_negative.cost == 2400
        assert dv.false_negative.category == "financial_loss"
        assert manifest.outcomes.revenue is not None
        assert manifest.outcomes.revenue.signal == "approved_transaction_value"
        assert manifest.outcomes.volume is not None
        assert manifest.outcomes.volume.estimated_daily_decisions == 12000

    def test_outcomes_missing_decision_value_rejected(self):
        with pytest.raises(OpenSRMV2ParseError, match="decision_value is required"):
            parse_opensrm_v2(_v2_skeleton({"outcomes": {"revenue": {}}}))

    def test_outcomes_invalid_currency_rejected(self):
        # Spec §1 test scenario 6.
        bad = {
            "decision_value": {"correct": 1.0, "currency": "EUROS"},
        }
        with pytest.raises(OpenSRMV2ParseError, match="ISO 4217"):
            parse_opensrm_v2(_v2_skeleton({"outcomes": bad}))


# ---------------------------------------------------------------------------
# Financial impact primitive
# ---------------------------------------------------------------------------


def _outcomes_with_costs(
    *, fp_cost: float | None = 0.0, fn_cost: float | None = 2400.0,
) -> Outcomes:
    return Outcomes(
        decision_value=DecisionValue(
            correct=138.50,
            currency="EUR",
            false_positive=(
                FailureCost(cost=fp_cost, category="friction")
                if fp_cost is not None else None
            ),
            false_negative=(
                FailureCost(cost=fn_cost, category="financial_loss")
                if fn_cost is not None else None
            ),
        ),
        volume=VolumeEstimate(estimated_daily_decisions=12000),
    )


class TestComputeFinancialImpact:
    def test_false_negative_metric_path(self):
        # Spec §1 test scenario 1 (metric-derived decisions_affected).
        impact = compute_financial_impact(
            _outcomes_with_costs(),
            decisions_affected=340,
            failure_mode="false_negative",
            volume_source="metric",
        )
        assert isinstance(impact, FinancialImpact)
        assert impact.estimated == pytest.approx(816000.0, abs=0.01)
        assert impact.currency == "EUR"
        assert impact.decisions_affected == 340
        assert impact.failure_mode == "false_negative"
        assert impact.volume_source == "metric"

    def test_false_positive_path(self):
        impact = compute_financial_impact(
            _outcomes_with_costs(fp_cost=12.0),
            decisions_affected=200,
            failure_mode="false_positive",
            volume_source="metric",
        )
        assert impact is not None
        assert impact.estimated == pytest.approx(2400.0, abs=0.01)

    def test_returns_none_when_failure_mode_undeclared(self):
        # decision_value declared but no false_positive cost → None.
        outcomes = _outcomes_with_costs(fp_cost=None)
        result = compute_financial_impact(
            outcomes,
            decisions_affected=10,
            failure_mode="false_positive",
            volume_source="metric",
        )
        assert result is None

    def test_negative_decisions_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            compute_financial_impact(
                _outcomes_with_costs(),
                decisions_affected=-1,
                failure_mode="false_negative",
                volume_source="metric",
            )

    def test_zero_decisions_yields_zero(self):
        impact = compute_financial_impact(
            _outcomes_with_costs(),
            decisions_affected=0,
            failure_mode="false_negative",
            volume_source="metric",
        )
        assert impact is not None
        assert impact.estimated == 0.0


class TestEstimateDecisionsInWindow:
    def test_one_hour_of_12000_daily(self):
        outcomes = _outcomes_with_costs()
        # 12000 / 24 = 500 per hour.
        n = estimate_decisions_in_window(outcomes, window=timedelta(hours=1))
        assert n == 500

    def test_returns_none_when_volume_absent(self):
        outcomes = Outcomes(
            decision_value=DecisionValue(correct=1.0, currency="EUR"),
        )
        assert estimate_decisions_in_window(
            outcomes, window=timedelta(hours=1),
        ) is None

    def test_returns_none_when_estimate_absent(self):
        outcomes = Outcomes(
            decision_value=DecisionValue(correct=1.0, currency="EUR"),
            volume=VolumeEstimate(peak_multiplier=2.0),
        )
        assert estimate_decisions_in_window(
            outcomes, window=timedelta(hours=1),
        ) is None

    def test_zero_window_yields_none(self):
        # A zero-duration window is "no signal", not "no impact" —
        # the caller should omit the financial figure rather than 0.
        outcomes = _outcomes_with_costs()
        assert estimate_decisions_in_window(
            outcomes, window=timedelta(seconds=0),
        ) is None

    def test_short_window_floors_to_one_not_zero(self):
        # A 30-second incident on a 1000/day service still has at
        # least one decision exposed; integer truncation of the raw
        # 0.347 must not silently report "no impact".
        outcomes = Outcomes(
            decision_value=DecisionValue(correct=1.0, currency="EUR"),
            volume=VolumeEstimate(estimated_daily_decisions=1000),
        )
        n = estimate_decisions_in_window(outcomes, window=timedelta(seconds=30))
        assert n == 1
