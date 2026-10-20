"""evaluate_threshold: E1-E4 for calibration, stability, audit_backlog,
feedback_latency (opensrm-g32d). Expected outcomes follow CORE §5.5.6."""

import math

import pytest

from nthlayer_common.manifest.evaluation import AtLeast, Missing, Samples, evaluate_threshold
from nthlayer_common.manifest.models import JudgmentThreshold, ThresholdMeasurement


def _cal(**bounds):
    return JudgmentThreshold("calibration", dict(bounds), ThresholdMeasurement(bins=10, window="30d"))


def _lat(**bounds):
    return JudgmentThreshold("feedback_latency", dict(bounds))


# E1 ---------------------------------------------------------------------

@pytest.mark.parametrize("measured,expected", [(0.05, "within"), (0.0500001, "breach"), (0.01, "within")])
def test_e1_ratio_bound_is_strict(measured, expected):
    o = evaluate_threshold(_cal(maximum_expected_calibration_error=0.05),
                           Samples(bounds={"maximum_expected_calibration_error": measured}))
    assert o.status == expected
    assert o.bounds[0].measured == measured
    assert o.bounds[0].limit == 0.05


@pytest.mark.parametrize("measured,expected", [(0.0, "within"), (1e-9, "breach")])
def test_e1_limit_zero(measured, expected):
    o = evaluate_threshold(_cal(maximum_brier_score=0.0), Samples(bounds={"maximum_brier_score": measured}))
    assert o.status == expected


# E2 ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "limit,measured,expected",
    [("1h", 3600.0, "within"), ("1h", 3600.001, "breach"), ("60m", 3600.0, "within"),
     ("1d", 86400.0, "within"), ("1w", 604800.5, "breach"), ("500ms", 0.5, "within")],
)
def test_e2_duration_bounds_compare_in_seconds(limit, measured, expected):
    o = evaluate_threshold(_lat(maximum_p50=limit), Samples(bounds={"maximum_p50": measured}))
    assert o.status == expected
    assert o.bounds[0].limit == limit  # kept as declared


# E3 ---------------------------------------------------------------------

@pytest.mark.parametrize("reason", ["no_data_source", "no_data", "insufficient_window"])
def test_e3_missing_sample_is_not_evaluable_with_its_reason(reason):
    o = evaluate_threshold(_lat(maximum_p50="1h"), Samples(bounds={"maximum_p50": Missing(reason)}))
    assert o.status == "not_evaluable"
    assert o.bounds[0].reason == reason
    assert o.bounds[0].measured is None


def test_e3_absent_sample_is_no_data():
    o = evaluate_threshold(_lat(maximum_p50="1h"), Samples())
    assert (o.status, o.bounds[0].reason) == ("not_evaluable", "no_data")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_sample_is_not_evaluable(bad):
    o = evaluate_threshold(_cal(maximum_brier_score=0.1), Samples(bounds={"maximum_brier_score": bad}))
    assert (o.status, o.bounds[0].reason) == ("not_evaluable", "no_data")


# E4 ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "p50,p90,expected",
    [(10.0, 10.0, "within"), (10.0, Missing("no_data"), "not_evaluable"),
     (99999.0, Missing("no_data"), "breach"), (10.0, 99999.0, "breach"),
     (Missing("no_data"), Missing("no_data"), "not_evaluable")],
)
def test_e4_combines_bounds(p50, p90, expected):
    o = evaluate_threshold(_lat(maximum_p50="1h", maximum_p90="1h"),
                           Samples(bounds={"maximum_p50": p50, "maximum_p90": p90}))
    assert o.status == expected
    assert [b.bound for b in o.bounds] == ["maximum_p50", "maximum_p90"]  # declared order


def test_every_declared_bound_is_evaluated_and_only_those():
    o = evaluate_threshold(_cal(maximum_brier_score=0.1),
                           Samples(bounds={"maximum_brier_score": 0.05,
                                           "maximum_expected_calibration_error": 0.9}))
    assert [b.bound for b in o.bounds] == ["maximum_brier_score"]
    assert o.status == "within"


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        evaluate_threshold(JudgmentThreshold("vibes", {"x": 1.0}), Samples())


def test_unknown_bound_for_the_type_raises_value_error():
    # Hand-built (the parsers reject it): a ValueError, not a KeyError.
    with pytest.raises(ValueError):
        evaluate_threshold(_cal(maximum_p50="1h"), Samples())


# A lower bound: what a histogram quantile in the +Inf bucket yields --------

@pytest.mark.parametrize("lower,limit,expected", [
    # The true value is > 86400 = 1d: a breach.
    (86400.0, "1d", ("breach", 86400.0, "lower_bound")),
    # > 3600 = 1h, equal at the boundary: still strictly above the limit.
    (3600.0, "1h", ("breach", 3600.0, "lower_bound")),
    # > 10000 (OTel's default top bucket) says nothing about a 1d limit.
    (10000.0, "1d", ("not_evaluable", None, "no_data")),
])
def test_lower_bound_breaches_only_when_it_reaches_the_limit(lower, limit, expected):
    o = evaluate_threshold(_lat(maximum_p90=limit), Samples(bounds={"maximum_p90": AtLeast(lower)}))
    b = o.bounds[0]
    assert (b.status, b.measured, b.reason) == expected
    assert o.status == expected[0]
