"""THRESHOLD_SERIES: the bound -> metric contract (opensrm-g32d)."""

import pytest

from nthlayer_common.manifest.threshold_series import (
    FEEDBACK_LATENCY_WINDOW,
    THRESHOLD_SERIES,
    ThresholdSeries,
)
from nthlayer_common.manifest.thresholds import THRESHOLD_BOUNDS

# The types with no data source yet (spec §2). Pinned: when a source is
# added, this set and the metrics contract change together.
_UNSOURCED = {"segments", "stability", "audit_backlog"}


def test_covers_every_threshold_type():
    assert set(THRESHOLD_SERIES) == set(THRESHOLD_BOUNDS)


def test_every_series_names_a_real_bound():
    for t, series in THRESHOLD_SERIES.items():
        assert set(series) <= set(THRESHOLD_BOUNDS[t]), t


def test_a_sourced_type_covers_all_its_bounds():
    for t, series in THRESHOLD_SERIES.items():
        if t not in _UNSOURCED:
            assert set(series) == set(THRESHOLD_BOUNDS[t]), t


def test_unsourced_types_are_exactly_these():
    assert {t for t, s in THRESHOLD_SERIES.items() if not s} == _UNSOURCED


def test_calibration_series_carry_bins_and_window():
    for s in THRESHOLD_SERIES["calibration"].values():
        assert s.kind == "gauge"
        assert s.labels == ("service", "bins", "window")


def test_feedback_latency_is_one_histogram_two_quantiles():
    p50 = THRESHOLD_SERIES["feedback_latency"]["maximum_p50"]
    p90 = THRESHOLD_SERIES["feedback_latency"]["maximum_p90"]
    assert p50.metric == p90.metric == "gen_ai_feedback_latency_seconds"
    assert (p50.kind, p50.quantile) == ("histogram_quantile", 0.5)
    assert (p90.kind, p90.quantile) == ("histogram_quantile", 0.9)
    assert FEEDBACK_LATENCY_WINDOW == "30d"


def test_quantile_set_only_for_histograms():
    for series in THRESHOLD_SERIES.values():
        for s in series.values():
            assert isinstance(s, ThresholdSeries)
            assert (s.quantile is not None) == (s.kind == "histogram_quantile")


def test_threshold_series_is_read_only():
    with pytest.raises(TypeError):
        THRESHOLD_SERIES["x"] = {}
    with pytest.raises(TypeError):
        THRESHOLD_SERIES["calibration"]["x"] = None
    with pytest.raises(TypeError):
        THRESHOLD_SERIES["segments"]["x"] = None


# Transcribed from nthlayer/docs/metrics-contract.md, "Judgment threshold
# metrics (opensrm-g32d)" (not read at test time). The histogram is stored by
# base name; the contract's query appends _bucket.
_CONTRACT = {
    ("calibration", "maximum_expected_calibration_error"):
        ("gen_ai_calibration_ece", "gauge", ("service", "bins", "window"), None),
    ("calibration", "maximum_brier_score"):
        ("gen_ai_calibration_brier", "gauge", ("service", "bins", "window"), None),
    ("feedback_latency", "maximum_p50"):
        ("gen_ai_feedback_latency_seconds", "histogram_quantile", ("service",), 0.5),
    ("feedback_latency", "maximum_p90"):
        ("gen_ai_feedback_latency_seconds", "histogram_quantile", ("service",), 0.9),
}


def test_series_match_the_front_door_metrics_contract():
    actual = {
        (t, b): (s.metric, s.kind, s.labels, s.quantile)
        for t, series in THRESHOLD_SERIES.items() for b, s in series.items()
    }
    assert actual == _CONTRACT
    for t in ("segments", "stability", "audit_backlog"):
        assert not THRESHOLD_SERIES[t]
