"""THRESHOLD_SERIES: the bound -> metric contract (opensrm-g32d)."""

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
