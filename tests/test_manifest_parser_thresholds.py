"""v2 judgment_threshold parsing: the shape, the guards and the ai-gate rule.

Every rejection mirrors a schema clause; the matching negative fixture is named
where one exists in opensrm/spec/v2/tests/invalid/. Template resolution follows
OPENSRM-CORE-v2 §5.5: the resolved list is the template's entries followed by
the manifest's, and an override may append but never replace.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from nthlayer_common.manifest.parser.v2 import (
    OpenSRMV2ParseError,
    parse_opensrm_v2,
    parse_opensrm_v2_file,
)

# parents: [0]=tests/ [1]=nthlayer-common/ [2]=<ecosystem>/
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "opensrm" / "spec" / "v2" / "schema.json"


def _forbidden_threshold_keys() -> list[str]:
    """Properties of JudgmentThresholdSpec that no value satisfies, read from
    the schema rather than restated. Reading at import time is deliberate: a
    missing spec is a collection error, not an empty (silently passing)
    parametrize list."""
    props = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["definitions"][
        "JudgmentThresholdSpec"
    ]["properties"]
    return sorted(
        name
        for name, prop in props.items()
        if prop is False
        or (isinstance(prop, dict) and prop.get("not") == {} and set(prop) <= {"not", "description"})
    )


FORBIDDEN_KEYS = _forbidden_threshold_keys()


def _manifest(service_type: str = "ai-gate", **spec: Any) -> dict[str, Any]:
    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {"name": "svc", "labels": {"tier": "critical"}},
        "spec": {
            "owner": {"group": "group:default/team"},
            "service": {"name": "svc", "type": service_type},
            **spec,
        },
    }


STABILITY = {
    "type": "stability",
    "probe": {"frozen_input_set": "s3://p.jsonl", "frequency": "daily"},
    "bounds": {"maximum_drift": 0.02},
}
AUDIT_BACKLOG = {"type": "audit_backlog", "bounds": {"maximum_age": "24h"}}
FEEDBACK_LATENCY = {"type": "feedback_latency", "bounds": {"maximum_p50": "1h"}}


def test_threshold_measurement_fields_are_carried():
    manifest = parse_opensrm_v2(_manifest(judgment_threshold=[STABILITY]))

    [threshold] = manifest.judgment_thresholds
    assert threshold.type == "stability"
    assert threshold.bounds == {"maximum_drift": 0.02}
    assert threshold.measurement.probe is not None
    assert threshold.measurement.probe.frozen_input_set == "s3://p.jsonl"
    assert threshold.measurement.probe.frequency == "daily"


def test_segment_fields_are_carried():
    entry = {
        "type": "segments",
        "segment_dimension": "customer_tier",
        "segment_values": ["enterprise", "smb"],
        "bounds": {"maximum_variance_from_overall": 0.05},
    }
    [threshold] = parse_opensrm_v2(_manifest(judgment_threshold=[entry])).judgment_thresholds
    assert threshold.measurement.segment_dimension == "customer_tier"
    assert threshold.measurement.segment_values == ["enterprise", "smb"]


def test_integral_float_bins_become_an_int():
    """The schema's `type: integer` accepts 10.0; the model carries an int."""
    entry = {
        "type": "calibration",
        "measurement": {"bins": 10.0, "window": "30d"},
        "bounds": {"maximum_brier_score": 0.05},
    }
    [threshold] = parse_opensrm_v2(_manifest(judgment_threshold=[entry])).judgment_thresholds
    assert threshold.measurement.bins == 10
    assert isinstance(threshold.measurement.bins, int)
    assert threshold.measurement.window == "30d"


def test_absent_block_means_no_thresholds():
    assert parse_opensrm_v2(_manifest()).judgment_thresholds == []


def test_thresholds_are_not_slos():
    manifest = parse_opensrm_v2(_manifest(judgment_threshold=[STABILITY]))
    assert manifest.get_judgment_slos() == []
    assert manifest.slos == []


@pytest.mark.parametrize("value", [None, "stability", 3, {"type": "stability"}])
def test_block_must_be_a_list(value):
    with pytest.raises(OpenSRMV2ParseError, match="must be a list"):
        parse_opensrm_v2(_manifest(judgment_threshold=value))


@pytest.mark.parametrize("item", [None, "stability", ["stability"]])
def test_elements_must_be_mappings(item):
    with pytest.raises(OpenSRMV2ParseError, match=r"judgment_threshold\[0\] must be a mapping"):
        parse_opensrm_v2(_manifest(judgment_threshold=[item]))


def test_rejected_on_a_non_ai_gate():
    # opensrm spec/v2/tests/invalid/n24-threshold-on-non-ai-gate.yaml
    with pytest.raises(OpenSRMV2ParseError, match="only permitted on an 'ai-gate'"):
        parse_opensrm_v2(_manifest("api", judgment_threshold=[STABILITY]))


def test_an_empty_block_on_a_non_ai_gate_is_still_rejected():
    """Presence, not truthiness, as for judgment_slo: the schema's `false`
    rejects any value, including []."""
    with pytest.raises(OpenSRMV2ParseError, match="only permitted on an 'ai-gate'"):
        parse_opensrm_v2(_manifest("api", judgment_threshold=[]))


def test_missing_type():
    # n29-threshold-missing-type.yaml
    with pytest.raises(OpenSRMV2ParseError, match=r"judgment_threshold\[0\].*type"):
        parse_opensrm_v2(_manifest(judgment_threshold=[{"bounds": {"maximum_drift": 0.02}}]))


def test_an_slo_type_points_back_to_judgment_slo():
    entry = {"type": "reversal_rate", "bounds": {"maximum_reversal_rate": 0.05}}
    with pytest.raises(OpenSRMV2ParseError, match="judgment_slo"):
        parse_opensrm_v2(_manifest(judgment_threshold=[entry]))


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({**STABILITY, "bounds": {"maximum_drift": 0.02, "maximum_variance_from_overall": 0.05}},
         "maximum_variance_from_overall"),                                        # n25
        ({"type": "calibration", "measurement": {"bins": 10, "window": "30d"}, "bounds": {}},
         "at least one"),                                                          # n26
        ({"type": "feedback_latency", "bounds": {"maximum_p50": 0.5}}, "duration"),  # n27
        ({"type": "calibration",
          "measurement": {"bins": 10, "window": "30d", "method": "brier_score"},
          "bounds": {"maximum_brier_score": 0.05}}, "method"),                     # n28
        ({"type": "segments", "segment_values": ["a"],
          "bounds": {"maximum_variance_from_overall": 0.05}}, "segment_dimension"),  # n30
        ({"type": "calibration", "measurement": {"bins": 1, "window": "30d"},
          "bounds": {"maximum_brier_score": 0.05}}, "bins"),                       # n12
        ({**STABILITY, "bounds": None}, "mapping"),
        # schema.json: `type` is an enum of strings; an unhashable or null
        # value is an unknown type, not a TypeError.
        ({"type": ["x"], "bounds": {}}, r"'type' must be one of"),
        ({"type": {"a": 1}, "bounds": {}}, r"'type' must be one of"),
        ({"type": None, "bounds": {}}, r"'type' must be one of"),
    ],
)
def test_invalid_entries_raise_the_parser_error(entry, match):
    with pytest.raises(OpenSRMV2ParseError, match=match):
        parse_opensrm_v2(_manifest(judgment_threshold=[entry]))


# JudgmentThresholdSpec is open: `measurement`, `probe`, `sampling` and the
# segment_* keys are constrained only inside the allOf branch of the type that
# uses them, so on any other type the schema accepts every value for them.
_STRAY_KEYS = ("measurement", "probe", "sampling", "segment_values")
_STRAY_VALUES = (None, "x", [1])


def _assert_empty_measurement(threshold: Any) -> None:
    m = threshold.measurement
    assert (m.source, m.confidence_threshold, m.bins, m.method, m.window, m.frequency) == (
        None, None, None, None, None, None,
    )
    assert m.probe is None
    assert m.sampling is None
    assert m.segment_dimension is None
    assert m.segment_values == []


@pytest.mark.parametrize("value", _STRAY_VALUES, ids=repr)
@pytest.mark.parametrize("key", _STRAY_KEYS)
def test_keys_irrelevant_to_the_type_are_ignored(key, value):
    entry = {**AUDIT_BACKLOG, key: value}
    [threshold] = parse_opensrm_v2(_manifest(judgment_threshold=[entry])).judgment_thresholds
    _assert_empty_measurement(threshold)


def test_keys_irrelevant_to_feedback_latency_are_ignored():
    entry = {**FEEDBACK_LATENCY, "probe": "x", "measurement": [1], "segment_values": "abc"}
    [threshold] = parse_opensrm_v2(_manifest(judgment_threshold=[entry])).judgment_thresholds
    _assert_empty_measurement(threshold)


def test_the_schema_accepts_stray_keys_on_a_type_that_does_not_use_them():
    """Pins the premise of the two tests above: if the schema ever closes
    JudgmentThresholdSpec, the parser must start rejecting these too."""
    entry = {**AUDIT_BACKLOG, "measurement": "x", "probe": [1], "sampling": None,
             "segment_values": "abc"}
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(_manifest(judgment_threshold=[entry]), schema)
    parse_opensrm_v2(_manifest(judgment_threshold=[entry]))


def test_stability_carries_only_its_probe():
    entry = {**STABILITY, "measurement": "x", "sampling": "x", "segment_values": "abc"}
    [threshold] = parse_opensrm_v2(_manifest(judgment_threshold=[entry])).judgment_thresholds
    assert threshold.measurement.probe is not None
    assert threshold.measurement.sampling is None
    assert threshold.measurement.segment_values == []


def test_forbidden_keys_were_derived():
    assert FORBIDDEN_KEYS, "no forbidden keys derived from schema.json"


@pytest.mark.parametrize("key", FORBIDDEN_KEYS)
def test_forbidden_keys_surface_as_the_parser_error(key):
    # schema.json JudgmentThresholdSpec sets each of these to {"not": {}}.
    entry = {**AUDIT_BACKLOG, key: "anything"}
    with pytest.raises(OpenSRMV2ParseError, match=rf"judgment_threshold\[0\].*'{key}'"):
        parse_opensrm_v2(_manifest(judgment_threshold=[entry]))


@pytest.mark.parametrize(
    "entry",
    [
        {"segment_dimension": "", "segment_values": ["a"]},
        {"segment_dimension": "tier", "segment_values": ["a", ""]},
    ],
    ids=["empty-dimension", "empty-value"],
)
def test_segment_strings_must_be_non_empty(entry):
    # schema.json segments branch: minLength 1 on both.
    full = {"type": "segments", **entry, "bounds": {"maximum_variance_from_overall": 0.05}}
    with pytest.raises(OpenSRMV2ParseError, match="segment_"):
        parse_opensrm_v2(_manifest(judgment_threshold=[full]))


# =============================================================================
# Template resolution (CORE §5.5), through parse_opensrm_v2_file, which
# resolves the template exactly as load_manifest does.
# =============================================================================


def _write_pair(
    tmp_path: Path,
    template_spec: dict[str, Any],
    service_type: str = "ai-gate",
    overrides: dict[str, Any] | None = None,
    **manifest_spec: Any,
) -> Path:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "base.yaml").write_text(yaml.safe_dump({
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifestTemplate",
        "metadata": {"name": "base"},
        "spec": template_spec,
    }))
    template_block: dict[str, Any] = {"extends": "template:default/base"}
    if overrides is not None:
        template_block["overrides"] = overrides
    doc = _manifest(service_type, template=template_block, **manifest_spec)
    path = tmp_path / "svc.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _types(path: Path) -> list[str]:
    return [t.type for t in parse_opensrm_v2_file(path).judgment_thresholds]


def test_template_only_thresholds_survive(tmp_path):
    path = _write_pair(tmp_path, {"judgment_threshold": [STABILITY]})
    [threshold] = parse_opensrm_v2_file(path).judgment_thresholds
    assert threshold.type == "stability"
    assert threshold.bounds == {"maximum_drift": 0.02}


def test_template_and_manifest_thresholds_concatenate_in_that_order(tmp_path):
    """A manifest cannot remove a template's threshold: unlike every other
    list, judgment_threshold is not replaced by the manifest's."""
    path = _write_pair(
        tmp_path,
        {"judgment_threshold": [STABILITY]},
        judgment_threshold=[AUDIT_BACKLOG],
    )
    assert _types(path) == ["stability", "audit_backlog"]


def test_override_append_lands_after_the_concatenation(tmp_path):
    path = _write_pair(
        tmp_path,
        {"judgment_threshold": [STABILITY]},
        overrides={"judgment_threshold": {"append": [FEEDBACK_LATENCY]}},
        judgment_threshold=[AUDIT_BACKLOG],
    )
    assert _types(path) == ["stability", "audit_backlog", "feedback_latency"]


def test_override_replace_is_rejected(tmp_path):
    """CORE §5.5: overrides MUST NOT remove or replace a template's entries."""
    path = _write_pair(
        tmp_path,
        {"judgment_threshold": [STABILITY]},
        overrides={"judgment_threshold": {"replace": [AUDIT_BACKLOG]}},
    )
    with pytest.raises(OpenSRMV2ParseError, match=r"judgment_threshold.*replace"):
        parse_opensrm_v2_file(path)


def test_override_append_must_be_a_list(tmp_path):
    """Any other list key ignores a non-list append; here that would drop a
    declared threshold without a word."""
    path = _write_pair(
        tmp_path,
        {"judgment_threshold": [STABILITY]},
        overrides={"judgment_threshold": {"append": AUDIT_BACKLOG}},
    )
    with pytest.raises(OpenSRMV2ParseError, match=r"judgment_threshold.*append.*list"):
        parse_opensrm_v2_file(path)


def test_a_non_list_manifest_block_does_not_mask_the_template(tmp_path):
    """Without a guard the manifest's null would replace the template's list
    and the parser would see a valid-looking manifest-side value."""
    path = _write_pair(
        tmp_path,
        {"judgment_threshold": [STABILITY]},
        judgment_threshold=None,
    )
    with pytest.raises(OpenSRMV2ParseError, match="must be a list"):
        parse_opensrm_v2_file(path)


def test_template_thresholds_on_a_non_ai_gate_are_rejected(tmp_path):
    """CORE §5.5: the ai-gate requirement applies to the resolved manifest."""
    path = _write_pair(tmp_path, {"judgment_threshold": [STABILITY]}, service_type="api")
    with pytest.raises(OpenSRMV2ParseError, match="only permitted on an 'ai-gate'"):
        parse_opensrm_v2_file(path)


def test_other_lists_still_replace(tmp_path):
    """Only judgment_threshold concatenates; every other list keeps the
    manifest-replaces-template default."""
    path = _write_pair(
        tmp_path,
        {"contracts": [{"name": "from-template", "promise": {"availability": 0.999}}]},
        contracts=[{"name": "from-manifest", "promise": {"availability": 0.99}}],
    )
    assert [c.name for c in parse_opensrm_v2_file(path).contracts] == ["from-manifest"]


@pytest.mark.parametrize(
    ("directive", "match"),
    [
        ([{"append": [AUDIT_BACKLOG]}], r"must be a mapping.*\{append: \[\.\.\.\]\}.*CORE §9"),
        ({"prepend": [AUDIT_BACKLOG]}, "only 'append'"),
        ({}, "'append' must be a list"),
    ],
    ids=["list-form", "unknown-directive", "empty-directive"],
)
def test_threshold_override_directives_the_merger_would_ignore_are_rejected(tmp_path, directive, match):
    """Each of these is inert for any other key; for judgment_threshold an
    inert directive silently drops thresholds the author declared."""
    path = _write_pair(
        tmp_path,
        {"judgment_threshold": [STABILITY]},
        overrides={"judgment_threshold": directive},
    )
    with pytest.raises(OpenSRMV2ParseError, match=match):
        parse_opensrm_v2_file(path)
