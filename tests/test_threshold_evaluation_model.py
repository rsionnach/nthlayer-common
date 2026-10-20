"""Outcome model for threshold evaluation (opensrm-g32d)."""

import pytest

from nthlayer_common.manifest.evaluation import (
    AtLeast,
    BoundResult,
    Missing,
    SegmentResult,
    ThresholdOutcome,
    combine,
)


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["within"], "within"),
        (["within", "within"], "within"),
        (["within", "not_evaluable"], "not_evaluable"),
        (["not_evaluable", "within"], "not_evaluable"),
        (["within", "breach"], "breach"),
        (["not_evaluable", "breach"], "breach"),
        (["breach", "not_evaluable", "within"], "breach"),
        (["not_evaluable"], "not_evaluable"),
        ([], "not_evaluable"),
    ],
)
def test_combine_is_e4_precedence(statuses, expected):
    assert combine(statuses) == expected


def test_combine_rejects_an_unknown_status():
    with pytest.raises(ValueError):
        combine(["fine"])


def _outcome():
    return ThresholdOutcome(
        type="segments",
        status="breach",
        bounds=(BoundResult("maximum_variance_from_overall", "breach", 0.2, 0.05, None),),
        segments=(
            SegmentResult("eu", "breach", 0.2, None),
            SegmentResult("us", "not_evaluable", None, "no_data"),
        ),
    )


def test_to_dict_from_dict_round_trip():
    o = _outcome()
    assert ThresholdOutcome.from_dict(o.to_dict()) == o


def test_to_dict_is_json_serialisable_and_shaped():
    import json

    d = _outcome().to_dict()
    json.dumps(d)
    assert d["status"] == "breach"
    assert d["bounds"][0] == {
        "bound": "maximum_variance_from_overall",
        "status": "breach",
        "measured": 0.2,
        "limit": 0.05,
        "reason": None,
    }
    assert d["segments"][1]["reason"] == "no_data"


def test_missing_rejects_an_unknown_reason():
    with pytest.raises(ValueError):
        Missing("because")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_lower_bound_must_be_finite(bad):
    with pytest.raises(ValueError):
        AtLeast(bad)


def test_from_dict_ignores_unknown_nested_keys_and_missing_segments():
    d = _outcome().to_dict()
    d["bounds"][0]["future_field"] = 1
    d["segments"][0]["future_field"] = 2
    assert ThresholdOutcome.from_dict(d) == _outcome()
    for absent in (None, "missing"):
        d = _outcome().to_dict()
        d["segments"] = None
        if absent == "missing":
            del d["segments"]
        assert ThresholdOutcome.from_dict(d).segments == ()


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(status="fine"),
    lambda d: d["bounds"][0].update(status="fine"),
    lambda d: d["segments"][0].update(status="fine"),
    lambda d: d["bounds"][0].update(reason=7),
    lambda d: d["segments"][1].update(reason=["no_data"]),
])
def test_from_dict_rejects_invalid_status_or_reason(mutate):
    d = _outcome().to_dict()
    mutate(d)
    with pytest.raises(ValueError):
        ThresholdOutcome.from_dict(d)


def test_from_dict_accepts_every_documented_reason():
    for reason in ("no_data_source", "no_data", "insufficient_window", "lower_bound", None):
        d = _outcome().to_dict()
        d["bounds"][0]["reason"] = reason
        assert ThresholdOutcome.from_dict(d).bounds[0].reason == reason


@pytest.mark.parametrize("bad", [True, "1", None])
def test_at_least_rejects_non_numbers(bad):
    with pytest.raises(ValueError):
        AtLeast(bad)


def test_from_dict_missing_or_none_bounds_is_empty():
    for absent in (None, "missing"):
        d = _outcome().to_dict()
        d["bounds"] = None
        if absent == "missing":
            del d["bounds"]
        assert ThresholdOutcome.from_dict(d).bounds == ()


@pytest.mark.parametrize("mutate", [
    lambda d: d["bounds"].__setitem__(0, 5),
    lambda d: d["segments"].__setitem__(0, "eu"),
    lambda d: d["bounds"][0].pop("limit"),
    lambda d: d["bounds"][0].pop("measured"),
    lambda d: d["segments"][0].pop("segment"),
    lambda d: d["segments"][0].pop("reason"),
    lambda d: d.pop("type"),
    lambda d: d.pop("status"),
])
def test_from_dict_malformed_payload_is_value_error_never_type_error(mutate):
    d = _outcome().to_dict()
    mutate(d)
    with pytest.raises(ValueError):
        ThresholdOutcome.from_dict(d)


def test_from_dict_names_the_missing_key():
    d = _outcome().to_dict()
    del d["bounds"][0]["limit"]
    with pytest.raises(ValueError, match="limit"):
        ThresholdOutcome.from_dict(d)


def test_from_dict_preserves_unknown_reason_strings():
    d = _outcome().to_dict()
    d["bounds"][0]["reason"] = "reason_from_a_newer_writer"
    d["segments"][1]["reason"] = "another_new_one"
    o = ThresholdOutcome.from_dict(d)
    assert o.bounds[0].reason == "reason_from_a_newer_writer"
    assert o.segments[1].reason == "another_new_one"
    assert ThresholdOutcome.from_dict(o.to_dict()) == o


def test_from_dict_accepts_beyond_top_bucket_and_none_stays_none():
    d = _outcome().to_dict()
    d["bounds"][0]["reason"] = "beyond_top_bucket"
    assert ThresholdOutcome.from_dict(d).bounds[0].reason == "beyond_top_bucket"
    d["bounds"][0]["reason"] = None
    assert ThresholdOutcome.from_dict(d).bounds[0].reason is None


@pytest.mark.parametrize("bad", [True, "", "  "])
def test_from_dict_rejects_bool_empty_and_blank_reasons(bad):
    for path in (("bounds", 0), ("segments", 1)):
        d = _outcome().to_dict()
        d[path[0]][path[1]]["reason"] = bad
        with pytest.raises(ValueError):
            ThresholdOutcome.from_dict(d)


def test_from_dict_normalises_a_str_subclass_to_str():
    class Tagged(str):
        pass

    d = _outcome().to_dict()
    d["bounds"][0]["reason"] = Tagged("future_reason")
    d["segments"][1]["reason"] = Tagged("future_reason")
    o = ThresholdOutcome.from_dict(d)
    assert type(o.bounds[0].reason) is str and o.bounds[0].reason == "future_reason"
    assert type(o.segments[1].reason) is str
