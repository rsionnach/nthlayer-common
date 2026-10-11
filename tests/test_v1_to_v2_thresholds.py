"""srm/v1 -> v2 conversion of thresholds: schema-valid output and a lossless
round trip through the v2 parser.

Checked against the schema with jsonschema directly (a dev dependency), not via
importorskip: a skip would let this go silently untested.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from nthlayer_common.manifest.parser.v1 import parse_srm_v1
from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2
from nthlayer_common.manifest.v1_compat import convert_v1_to_v2

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "opensrm" / "spec" / "v2" / "schema.json"

# The generate example's two threshold entries, verbatim (see test_srm_v1_thresholds.py).
# No judgment SLO here, deliberately: convert_v1_to_v2 emits judgment_slo items
# in the wrapped metadata/spec shape, which the schema's flat JudgmentSLOSpec
# rejects - pre-existing, opensrm-a742 - and that would fail the schema check
# below for a reason unrelated to thresholds.
SRM_V1 = {
    "apiVersion": "srm/v1",
    "kind": "ServiceReliabilityManifest",
    "metadata": {"name": "svc", "team": "t", "tier": "critical"},
    "spec": {
        "type": "ai-gate",
        "slos": {
            "calibration": {"target": 0.05, "window": "30d"},
            "feedback_latency": {"target": 3600, "window": "30d"},
        },
    },
}


def test_converted_document_validates_against_the_v2_schema():
    v2 = convert_v1_to_v2(SRM_V1)
    jsonschema.validate(v2, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    assert {e["type"] for e in v2["spec"]["judgment_threshold"]} == {"calibration", "feedback_latency"}
    assert "judgment_slo" not in v2["spec"]


def test_rate_types_still_route_to_judgment_slo():
    doc = {**SRM_V1, "spec": {"type": "ai-gate", "slos": {
        **SRM_V1["spec"]["slos"], "reversal_rate": {"target": 98.5, "window": "30d"},
    }}}
    v2 = convert_v1_to_v2(doc)
    assert [e["spec"]["judgment_type"] for e in v2["spec"]["judgment_slo"]] == ["reversal_rate"]
    assert len(v2["spec"]["judgment_threshold"]) == 2


def test_round_trip_preserves_every_threshold():
    direct = parse_srm_v1(SRM_V1).judgment_thresholds
    via_v2 = parse_opensrm_v2(convert_v1_to_v2(SRM_V1)).judgment_thresholds
    assert via_v2 == direct


def test_thresholds_alone_on_a_non_ai_gate_are_rejected():
    doc = {**SRM_V1, "spec": {"type": "api", "slos": {"calibration": {"target": 0.05}}}}
    with pytest.raises(ValueError, match="calibration"):
        convert_v1_to_v2(doc)


@pytest.mark.parametrize("name", ["segments", "stability"])
def test_types_srm_v1_cannot_express_raise_value_error(name):
    doc = {**SRM_V1, "spec": {"type": "ai-gate", "slos": {name: {"target": 0.05}}}}
    with pytest.raises(ValueError, match="v2"):
        convert_v1_to_v2(doc)


def test_bound_like_key_raises_value_error_rather_than_being_dropped():
    doc = {**SRM_V1, "spec": {"type": "ai-gate", "slos": {"feedback_latency": {"target": 3600, "p90": "24h"}}}}
    with pytest.raises(ValueError, match="p90"):
        convert_v1_to_v2(doc)


def test_scalar_shorthand_threshold_is_converted_not_dropped():
    doc = {**SRM_V1, "spec": {"type": "ai-gate", "slos": {"feedback_latency": 60}}}
    direct = parse_srm_v1(doc).judgment_thresholds
    via_v2 = parse_opensrm_v2(convert_v1_to_v2(doc)).judgment_thresholds
    assert len(direct) == 1
    assert via_v2 == direct


def test_calibration_bins_are_emitted_as_int():
    [entry] = [
        e for e in convert_v1_to_v2(SRM_V1)["spec"]["judgment_threshold"]
        if e["type"] == "calibration"
    ]
    assert isinstance(entry["measurement"]["bins"], int)
    assert not isinstance(entry["measurement"]["bins"], bool)
