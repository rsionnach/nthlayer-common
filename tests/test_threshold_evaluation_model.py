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
