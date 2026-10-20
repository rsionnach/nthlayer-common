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
