"""Segments evaluation (CORE §5.5.1 + §5.5.6) and every spec example."""

from pathlib import Path

import pytest
import yaml

from nthlayer_common.manifest import threshold_from_v2_entry
from nthlayer_common.manifest.evaluation import Missing, Samples, evaluate_threshold
from nthlayer_common.manifest.models import JudgmentThreshold, ThresholdMeasurement


def _seg(limit=0.05, values=("eu", "us")):
    return JudgmentThreshold(
        "segments", {"maximum_variance_from_overall": limit},
        ThresholdMeasurement(metric="escalation", segment_dimension="region",
                             segment_values=list(values), window="30d"),
    )


@pytest.mark.parametrize("eu,expected", [(0.15, "within"), (0.150001, "breach"),
                                         (0.05, "within"), (0.049999, "breach")])
def test_variance_is_absolute_in_both_directions(eu, expected):
    o = evaluate_threshold(_seg(values=("eu",)), Samples(overall=0.1, segments={"eu": eu}))
    assert o.status == expected
    assert o.segments[0].measured == pytest.approx(abs(eu - 0.1))


def test_each_segment_decided_independently():
    o = evaluate_threshold(_seg(), Samples(overall=0.1, segments={"eu": 0.3, "us": Missing("no_data")}))
    assert [(s.segment, s.status) for s in o.segments] == [("eu", "breach"), ("us", "not_evaluable")]
    assert o.status == "breach"
    assert o.bounds[0].measured == pytest.approx(0.2)


def test_missing_overall_makes_every_segment_not_evaluable():
    o = evaluate_threshold(_seg(), Samples(overall=Missing("no_data_source"),
                                           segments={"eu": 0.3, "us": 0.1}))
    assert {s.status for s in o.segments} == {"not_evaluable"}
    assert {s.reason for s in o.segments} == {"no_data_source"}
    assert o.status == "not_evaluable"
    assert o.bounds[0].measured is None


def test_unsampled_segment_is_no_data():
    o = evaluate_threshold(_seg(), Samples(overall=0.1, segments={"eu": 0.1}))
    assert [(s.segment, s.reason) for s in o.segments] == [("eu", None), ("us", "no_data")]


def test_duplicate_segment_value_evaluated_once():
    o = evaluate_threshold(_seg(values=("eu", "eu", "us")),
                           Samples(overall=0.1, segments={"eu": 0.1, "us": 0.1}))
    assert [s.segment for s in o.segments] == ["eu", "us"]


@pytest.mark.parametrize("bounds", [
    {},
    {"maximum_variance_from_overall": 0.05, "maximum_drift": 0.1},
    {"maximum_drift": 0.1},  # one bound, but not a segments bound
])
def test_segments_without_exactly_its_one_bound_raises_value_error(bounds):
    # Hand-built (the parsers reject all three).
    t = JudgmentThreshold("segments", bounds,
                          ThresholdMeasurement(metric="escalation", segment_dimension="region",
                                               segment_values=["eu"], window="30d"))
    with pytest.raises(ValueError):
        evaluate_threshold(t, Samples(overall=0.1, segments={"eu": 0.1}))


_SPEC_V2 = Path(__file__).resolve().parents[2] / "opensrm" / "spec" / "v2"
# Every shipped example that declares judgment thresholds (spec §9: every one).
_EXAMPLES = sorted(
    p for p in (_SPEC_V2 / "examples").rglob("*.yaml")
    if "judgment_threshold" in (yaml.safe_load(p.read_text()) or {}).get("spec", {})
) if _SPEC_V2.is_dir() else []


def test_the_threshold_examples_are_present():
    # 8 at the time of writing (opensrm main, 2026-10-12): the six in
    # examples/judgment-thresholds/, service-manifest-full.yaml and
    # services/ai-gate.yaml. A floor, not an equality: new examples are welcome,
    # but a glob or filter that silently drops some must fail here.
    names = {p.relative_to(_SPEC_V2 / "examples").as_posix() for p in _EXAMPLES}
    assert {"judgment-thresholds/all-five-types.yaml",
            "judgment-thresholds/segments-two-dimensions.yaml",
            "services/ai-gate.yaml"} <= names, names
    assert len(_EXAMPLES) >= 8, names


@pytest.mark.parametrize("source", _EXAMPLES, ids=lambda p: p.name)
def test_every_spec_example_threshold_evaluates(source):
    """Every threshold the spec ships is evaluable: all-within when each
    measured value equals its bound (E1: equal is within), all-breach just
    above it, all-not_evaluable when nothing is measured. Entries are parsed
    one by one (threshold_from_v2_entry), so a manifest that does not parse
    whole for unrelated reasons (ai-gate.yaml, opensrm-a742) still counts."""
    entries = yaml.safe_load(source.read_text())["spec"]["judgment_threshold"]
    for t in (threshold_from_v2_entry(e) for e in entries):
        at, above, none = _samples_at(t, 0.0), _samples_at(t, 1e-6), Samples()
        assert evaluate_threshold(t, at).status == "within", t
        assert evaluate_threshold(t, above).status == "breach", t
        assert evaluate_threshold(t, none).status == "not_evaluable", t


def _samples_at(t, delta):
    from nthlayer_common.manifest.thresholds import DURATION, THRESHOLD_BOUNDS, duration_seconds

    if t.type == "segments":
        limit = float(t.bounds["maximum_variance_from_overall"])
        return Samples(overall=0.0, segments={v: limit + delta for v in t.measurement.segment_values})
    vals = {}
    for bound, limit in t.bounds.items():
        base = duration_seconds(limit) if THRESHOLD_BOUNDS[t.type][bound] == DURATION else float(limit)
        vals[bound] = base + delta
    return Samples(bounds=vals)


@pytest.mark.parametrize("overall,segment,bound", [
    (0.7, 0.9, 0.2), (0.9, 0.7, 0.2), (0.1, 0.3, 0.2), (0.33, 0.38, 0.05), (0.6, 0.65, 0.05),
])
def test_decimal_equality_with_the_bound_is_within(overall, segment, bound):
    # E1: equal is within. The bounds are decimal; binary subtraction noise must not breach.
    o = evaluate_threshold(_seg(limit=bound, values=("eu",)),
                           Samples(overall=overall, segments={"eu": segment}))
    assert o.status == "within"
    beyond = segment + 1e-9 if segment > overall else segment - 1e-9
    o = evaluate_threshold(_seg(limit=bound, values=("eu",)),
                           Samples(overall=overall, segments={"eu": beyond}))
    assert o.status == "breach"


@pytest.mark.parametrize("bad", [True, False])
def test_bool_samples_are_not_data(bad):
    o = evaluate_threshold(_seg(values=("eu",)), Samples(overall=0.1, segments={"eu": bad}))
    assert o.segments[0].status == "not_evaluable" and o.segments[0].reason == "no_data"
    o = evaluate_threshold(_seg(values=("eu",)), Samples(overall=bad, segments={"eu": 0.1}))
    assert o.status == "not_evaluable"


@pytest.mark.parametrize("bad", [True, False])
def test_bool_bound_sample_is_not_data(bad):
    t = JudgmentThreshold("stability", {"maximum_drift": 1.0},
                          ThresholdMeasurement(metric="m", window="30d"))
    o = evaluate_threshold(t, Samples(bounds={"maximum_drift": bad}))
    assert (o.status, o.bounds[0].reason) == ("not_evaluable", "no_data")


@pytest.mark.parametrize("form", ["overall", "bounds"])
def test_no_data_source_propagates_from_overall_or_bounds_level_missing(form):
    m = Missing("no_data_source")
    s = (Samples(overall=m) if form == "overall"
         else Samples(bounds={"maximum_variance_from_overall": m}))
    o = evaluate_threshold(_seg(), s)
    assert {x.reason for x in o.segments} == {"no_data_source"}
    assert o.bounds[0].reason == "no_data_source"
    assert o.status == "not_evaluable"


@pytest.mark.parametrize("bad", ["0.1", object(), [0.1], {"a": 1}])
def test_non_numeric_segment_samples_are_no_data_not_type_error(bad):
    o = evaluate_threshold(_seg(values=("eu",)), Samples(overall=0.1, segments={"eu": bad}))
    assert (o.segments[0].status, o.segments[0].reason) == ("not_evaluable", "no_data")
    o = evaluate_threshold(_seg(values=("eu",)), Samples(overall=bad, segments={"eu": 0.1}))
    assert (o.segments[0].status, o.segments[0].reason) == ("not_evaluable", "no_data")


@pytest.mark.parametrize("bad", ["0.1", object(), None, float("nan")])
def test_non_numeric_bound_samples_are_no_data_not_type_error(bad):
    t = JudgmentThreshold("stability", {"maximum_drift": 1.0},
                          ThresholdMeasurement(metric="m", window="30d"))
    o = evaluate_threshold(t, Samples(bounds={"maximum_drift": bad}))
    assert (o.status, o.bounds[0].reason) == ("not_evaluable", "no_data")


@pytest.mark.parametrize("bad_limit", [True, None, "0.05", float("nan"), float("inf"), -0.1, 1.5])
def test_bad_ratio_limit_is_value_error(bad_limit):
    t = JudgmentThreshold("stability", {"maximum_drift": bad_limit},
                          ThresholdMeasurement(metric="m", window="30d"))
    with pytest.raises(ValueError):
        evaluate_threshold(t, Samples(bounds={"maximum_drift": 0.1}))
    with pytest.raises(ValueError):
        evaluate_threshold(_seg(limit=bad_limit), Samples(overall=0.1, segments={"eu": 0.1}))


@pytest.mark.parametrize("bad_limit", [24.0, 3600, True, None, "24"])
def test_bad_duration_limit_is_value_error(bad_limit):
    t = JudgmentThreshold("audit_backlog", {"maximum_age": bad_limit})
    with pytest.raises(ValueError):
        evaluate_threshold(t, Samples(bounds={"maximum_age": 1.0}))
