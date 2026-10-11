"""Manifests written for nthlayer-common 3.x, loaded by 4.0.0.

Each pre-3c placement is rejected with an error that shows its replacement.
The fixtures are the spec's own retired examples, both wrapped (a JudgmentSLO
document inline) and flat (the schema's inline item) shapes, because the
parser must reject either. The flat shape is otherwise unparsed (opensrm-a742),
so these tests are what keeps the rejection from depending on that bead.
"""
from __future__ import annotations

import warnings
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from nthlayer_common.manifest import load_manifest
from nthlayer_common.manifest.parser.v2 import OpenSRMV2ParseError, parse_opensrm_v2
from nthlayer_common.manifest.target_validation import (
    TargetConventionWarning,
    judgment_promise_direction,
    judgment_target_percent,
    judgment_target_ratio,
)
from nthlayer_common.manifest.v1_compat import convert_v1_contract

INVALID_DIR = Path(__file__).resolve().parents[2] / "opensrm" / "spec" / "v2" / "tests" / "invalid"


def _manifest(**spec: Any) -> dict[str, Any]:
    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {"name": "svc", "labels": {"tier": "critical"}},
        "spec": {
            "owner": {"group": "group:default/team"},
            "service": {"name": "svc", "type": "ai-gate"},
            **spec,
        },
    }


@pytest.mark.parametrize(
    "fixture",
    [
        "n20-segments-as-judgment-slo.yaml",
        "n21-stability-as-judgment-slo.yaml",
        "n22-calibration-as-judgment-slo.yaml",
    ],
)
@pytest.mark.parametrize("shape", ["wrapped", "flat"])
def test_departing_type_under_judgment_slo_shows_the_threshold_form(fixture, shape):
    doc = yaml.safe_load((INVALID_DIR / fixture).read_text(encoding="utf-8"))
    item = {"metadata": doc["metadata"], "spec": doc["spec"]} if shape == "wrapped" else doc["spec"]
    judgment_type = doc["spec"]["judgment_type"]

    with pytest.raises(OpenSRMV2ParseError) as exc_info:
        parse_opensrm_v2(_manifest(judgment_slo=[item]))

    message = str(exc_info.value)
    assert "judgment_threshold" in message
    assert f"type: {judgment_type}" in message
    # target -> bounds, with every declared bound name carried over
    for bound in doc["spec"]["target"]:
        assert bound in message


@pytest.mark.parametrize("shape", ["wrapped", "flat"])
def test_audit_backlog_in_audit_sampling_target_shows_the_threshold_form(shape):
    doc = yaml.safe_load(
        (INVALID_DIR / "n23-audit-backlog-in-audit-sampling-target.yaml").read_text(encoding="utf-8")
    )
    item = {"metadata": doc["metadata"], "spec": doc["spec"]} if shape == "wrapped" else doc["spec"]

    with pytest.raises(
        OpenSRMV2ParseError,
        match=r"audit_completion_rate stays in the audit_sampling target(?s:.*)type: audit_backlog",
    ) as exc_info:
        parse_opensrm_v2(_manifest(judgment_slo=[item]))
    assert "maximum_age: 24h" in str(exc_info.value)


def test_audit_backlog_suggestion_is_safe_dumped_not_interpolated():
    item = {
        "judgment_type": "audit_sampling",
        "target": {"audit_backlog_maximum_age": "24h\nx"},
    }
    with pytest.raises(OpenSRMV2ParseError) as exc_info:
        parse_opensrm_v2(_manifest(judgment_slo=[item]))
    entry = _suggested_entry(str(exc_info.value))
    assert entry == {"type": "audit_backlog", "bounds": {"maximum_age": "24h\nx"}}


@pytest.mark.parametrize(
    "measurement",
    [None, {"method": "expected_calibration_error"}, {"window": "30d"}, {"bins": 10}],
    ids=["absent", "method-only", "no-bins", "no-window"],
)
def test_calibration_without_measurement_says_one_is_required(measurement):
    item = {"judgment_type": "calibration", "target": {"maximum_brier_score": 0.1}}
    if measurement is not None:
        item["measurement"] = measurement
    with pytest.raises(
        OpenSRMV2ParseError,
        match=r"calibration threshold requires `measurement` with `bins` and `window`",
    ):
        parse_opensrm_v2(_manifest(judgment_slo=[item]))


def _suggested_entry(message: str) -> dict[str, Any]:
    """The YAML block the error message offers, parsed back."""
    block = message.split(":\n\n", 1)[1].split("\n\n", 1)[0]
    return yaml.safe_load(block)["judgment_threshold"][0]


@pytest.mark.parametrize(
    "fixture",
    [
        "n20-segments-as-judgment-slo.yaml",
        "n21-stability-as-judgment-slo.yaml",
        "n22-calibration-as-judgment-slo.yaml",
    ],
)
@pytest.mark.parametrize("shape", ["wrapped", "flat"])
def test_suggested_threshold_entry_loads_unchanged(fixture, shape):
    """Round trip: copying the fix-it block into judgment_threshold must load."""
    doc = yaml.safe_load((INVALID_DIR / fixture).read_text(encoding="utf-8"))
    item = {"metadata": doc["metadata"], "spec": doc["spec"]} if shape == "wrapped" else doc["spec"]
    with pytest.raises(OpenSRMV2ParseError) as exc_info:
        parse_opensrm_v2(_manifest(judgment_slo=[item]))

    entry = _suggested_entry(str(exc_info.value))
    manifest = parse_opensrm_v2(_manifest(judgment_threshold=[entry]))

    assert [t.type for t in manifest.judgment_thresholds] == [doc["spec"]["judgment_type"]]
    assert manifest.judgment_thresholds[0].bounds == doc["spec"]["target"]


def test_suggested_calibration_entry_drops_measurement_method():
    doc = yaml.safe_load(
        (INVALID_DIR / "n22-calibration-as-judgment-slo.yaml").read_text(encoding="utf-8")
    )
    assert "method" in doc["spec"]["measurement"]
    with pytest.raises(OpenSRMV2ParseError) as exc_info:
        parse_opensrm_v2(_manifest(judgment_slo=[doc["spec"]]))
    entry = _suggested_entry(str(exc_info.value))
    assert "method" not in entry["measurement"]
    assert entry["measurement"] == {"bins": 10, "window": "30d"}


def test_message_says_breach_actions_and_service_do_not_carry_over():
    doc = yaml.safe_load(
        (INVALID_DIR / "n21-stability-as-judgment-slo.yaml").read_text(encoding="utf-8")
    )
    assert "service" in doc["spec"]
    with pytest.raises(
        OpenSRMV2ParseError,
        match="`breach_actions` and `service` do not carry over: thresholds do not declare them",
    ):
        parse_opensrm_v2(_manifest(judgment_slo=[doc["spec"]]))


def test_non_yaml_native_target_still_raises_only_the_parse_error():
    """The dict API can pass values safe_dump refuses (RepresenterError)."""
    item = {"judgment_type": "stability", "target": {"maximum_drift": Decimal("0.02")}}
    with pytest.raises(OpenSRMV2ParseError, match="judgment_threshold"):
        parse_opensrm_v2(_manifest(judgment_slo=[item]))


@pytest.mark.parametrize(
    "item",
    [
        # judgment_type of a non-string shape: unhashable values must not
        # reach a frozenset membership test (TypeError).
        {"judgment_type": ["stability"], "target": {"maximum_drift": 0.02}},
        {"judgment_type": {"stability": 1}, "target": {"maximum_drift": 0.02}},
        {"metadata": {"name": "x"}, "spec": {"judgment_type": ["calibration"]}},
        # target a scalar, on both a departing type and audit_sampling
        {"judgment_type": "audit_sampling", "target": "24h"},
        {"metadata": {"name": "x"}, "spec": {"judgment_type": "audit_sampling", "target": 5}},
        {"metadata": {"name": "x"}, "spec": {"judgment_type": "stability", "target": "0.02"}},
        # spec a scalar
        {"metadata": {"name": "x"}, "spec": "stability"},
        {"metadata": {"name": "x"}, "spec": None},
    ],
    ids=[
        "judgment_type-list-flat",
        "judgment_type-dict-flat",
        "judgment_type-list-wrapped",
        "target-scalar-audit_sampling-flat",
        "target-int-audit_sampling-wrapped",
        "target-scalar-stability-wrapped",
        "spec-scalar",
        "spec-null",
    ],
)
def test_odd_judgment_slo_shapes_raise_only_the_parse_error(item):
    """The pre-3c check reads untrusted shapes; it must never leak a
    TypeError/AttributeError past the loader's except clauses."""
    with pytest.raises(OpenSRMV2ParseError):
        parse_opensrm_v2(_manifest(judgment_slo=[item]))


def test_departing_type_with_scalar_target_still_shows_the_threshold_form():
    item = {"metadata": {"name": "x"}, "spec": {"judgment_type": "stability", "target": "0.02"}}
    with pytest.raises(OpenSRMV2ParseError, match="type: stability"):
        parse_opensrm_v2(_manifest(judgment_slo=[item]))


@pytest.mark.parametrize("threshold_type", ["calibration", "segments", "stability", "audit_backlog", "feedback_latency"])
def test_contract_promise_on_a_threshold_type_is_rejected(threshold_type):
    contract = {"name": "c", "promise": {"judgment": {threshold_type: 0.1}}}
    with pytest.raises(OpenSRMV2ParseError, match="threshold"):
        parse_opensrm_v2(_manifest(contracts=[contract]))


def test_srm_v1_contract_promise_on_a_threshold_type_is_rejected():
    """The same spec 2.4 rule, reached through the v1 path."""
    with pytest.raises(ValueError, match="threshold"):
        convert_v1_contract("svc", None, None, {"calibration": 0.1})


def test_parse_srm_v1_reports_a_contract_promise_on_a_threshold_as_a_parse_error():
    from nthlayer_common.manifest.parser.v1 import OpenSRMParseError, parse_srm_v1

    doc = {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "t", "tier": "critical"},
        "spec": {
            "type": "ai-gate",
            "slos": {},
            "contract": {"judgment": {"calibration": 0.1}},
        },
    }
    with pytest.raises(OpenSRMParseError, match="threshold"):
        parse_srm_v1(doc)


def test_contract_promise_on_an_unknown_type_is_rejected():
    """Was silently inert: a promise that matched no SLO, so
    validate_contracts() could never report it (spec 2.4, Review Focus 5)."""
    contract = {"name": "c", "promise": {"judgment": {"reversal_rat": 0.05}}}
    with pytest.raises(OpenSRMV2ParseError, match="reversal_rat"):
        parse_opensrm_v2(_manifest(contracts=[contract]))


# The parser rejects magnitudes before these are reached, so only a direct call
# exercises the removed passthrough. Field names are schema.json's
# JudgmentThresholdSpec bound names, which were never SLO target fields.
@pytest.mark.parametrize(
    "field", ["maximum_drift", "maximum_variance_from_overall", "maximum_brier_score"]
)
@pytest.mark.parametrize(
    "convert",
    [
        lambda f: judgment_target_percent(f, 0.05),
        lambda f: judgment_target_ratio(f, 5.0),
        judgment_promise_direction,
    ],
    ids=["percent", "ratio", "direction"],
)
def test_magnitude_fields_are_no_longer_passed_through(convert, field):
    with pytest.raises(ValueError, match="not a judgment SLO target field"):
        convert(field)


def test_thresholds_never_trip_the_target_convention_warning(tmp_path):
    """l33e acceptance. Before 3c, `maximum_drift: 0.05` warned 'write 5.0'."""
    path = tmp_path / "svc.yaml"
    path.write_text(yaml.safe_dump(_manifest(judgment_threshold=[
        {"type": "stability", "probe": {"frozen_input_set": "s3://p", "frequency": "daily"},
         "bounds": {"maximum_drift": 0.05}},
        {"type": "calibration", "measurement": {"bins": 10, "window": "30d"},
         "bounds": {"maximum_brier_score": 0.05}},
    ])), encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", TargetConventionWarning)
        manifest = load_manifest(path, suppress_deprecation_warning=True)

    assert len(manifest.judgment_thresholds) == 2
