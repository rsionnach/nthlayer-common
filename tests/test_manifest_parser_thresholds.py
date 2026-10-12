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

from nthlayer_common.manifest.models import OutcomeSignal, ThresholdMeasurement
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


def test_non_default_probe_frequency_and_calibration_window_are_carried():
    """Every other fixture uses the defaults (daily / 30d), so a hardcoded
    value would pass them. weekly and 7d are allowed by the schema's enum and
    Duration pattern."""
    stability = {**STABILITY, "probe": {"frozen_input_set": "s3://p.jsonl", "frequency": "weekly"}}
    calibration = {
        "type": "calibration",
        "measurement": {"bins": 10, "window": "7d"},
        "bounds": {"maximum_brier_score": 0.05},
    }
    parsed = parse_opensrm_v2(_manifest(judgment_threshold=[stability, calibration]))

    by_type = {t.type: t for t in parsed.judgment_thresholds}
    assert by_type["stability"].measurement.probe.frequency == "weekly"
    assert by_type["calibration"].measurement.window == "7d"


def test_scalar_judgment_slo_metadata_is_a_parse_error_not_an_attribute_error():
    item = {
        "metadata": "x",
        "spec": {
            "judgment_type": "reversal_rate",
            "measurement": {"window": "7d"},
            "target": {"maximum_reversal_rate": 0.05},
        },
    }
    with pytest.raises(OpenSRMV2ParseError, match="metadata must be a mapping"):
        parse_opensrm_v2(_manifest(judgment_slo=[item]))


def test_segment_fields_are_carried():
    entry = {
        "type": "segments",
        "metric": "escalation",
        "segment_dimension": "customer_tier",
        "segment_values": ["enterprise", "smb"],
        "bounds": {"maximum_variance_from_overall": 0.05},
    }
    [threshold] = parse_opensrm_v2(_manifest(judgment_threshold=[entry])).judgment_thresholds
    assert threshold.measurement.segment_dimension == "customer_tier"
    assert threshold.measurement.segment_values == ["enterprise", "smb"]
    assert threshold.measurement.metric == "escalation"
    # CORE §5.5.1: measurement.window defaults to 30d, stored as such.
    assert threshold.measurement.window == "30d"
    assert threshold.measurement.source is None
    assert threshold.measurement.confidence_threshold is None
    assert threshold.measurement.outcome_signal is None


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"metric": "reversal_rate", "measurement": {"source": "downstream_signal", "window": "14d"}},
         {"metric": "reversal_rate", "source": "downstream_signal", "window": "14d"}),
        ({"metric": "high_confidence_failure", "measurement": {"confidence_threshold": 0.8}},
         {"metric": "high_confidence_failure", "confidence_threshold": 0.8, "window": "30d"}),
        ({"metric": "outcomes", "outcome_signal": {"source": "retro", "window": "2h"},
          "measurement": {"window": "60d"}},
         {"metric": "outcomes", "outcome_signal": OutcomeSignal(source="retro", window="2h"),
          "window": "60d"}),
    ],
    ids=["reversal_rate", "high_confidence_failure", "outcomes"],
)
def test_segments_metric_parameters_are_carried(extra, expected):
    """Non-default values throughout (14d/60d windows, a 0.8 threshold, a
    downstream_signal source), so a hardcoded default cannot pass."""
    entry = {
        "type": "segments",
        "segment_dimension": "tier",
        "segment_values": ["a"],
        "bounds": {"maximum_variance_from_overall": 0.05},
        **extra,
    }
    [threshold] = parse_opensrm_v2(_manifest(judgment_threshold=[entry])).judgment_thresholds
    defaults = {"source": None, "confidence_threshold": None, "outcome_signal": None}
    m = threshold.measurement
    for name, value in {**defaults, **expected}.items():
        assert getattr(m, name) == value, name


def test_segments_confidence_threshold_is_carried_as_a_float():
    entry = {
        "type": "segments", "metric": "high_confidence_failure",
        "measurement": {"confidence_threshold": 1},
        "segment_dimension": "tier", "segment_values": ["a"],
        "bounds": {"maximum_variance_from_overall": 0.05},
    }
    [threshold] = parse_opensrm_v2(_manifest(judgment_threshold=[entry])).judgment_thresholds
    assert threshold.measurement.confidence_threshold == 1.0
    assert isinstance(threshold.measurement.confidence_threshold, float)


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
        ({"type": "segments", "metric": "escalation", "segment_values": ["a"],
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


@pytest.mark.parametrize("bounds", [{1: "1h", "x": "2h"}, {None: "1h", "x": "2h"}], ids=["int-key", "null-key"])
def test_mixed_type_bound_keys_raise_the_parser_error(bounds):
    entry = {"type": "audit_backlog", "bounds": bounds}
    with pytest.raises(OpenSRMV2ParseError):
        parse_opensrm_v2(_manifest(judgment_threshold=[entry]))


# CORE §5.5 (opensrm-7zpz): each entry is closed. Keys that belong to another
# type, or to no type, are errors on every type that does not define them,
# whatever their value. Until 5.0.0 these were ignored.
_STRAY_KEYS = ("measurement", "probe", "sampling", "segment_values", "metric", "window",
               "outcome_signal")
_STRAY_VALUES = (None, "x", [1])


@pytest.mark.parametrize("value", _STRAY_VALUES, ids=repr)
@pytest.mark.parametrize("key", _STRAY_KEYS)
def test_keys_irrelevant_to_the_type_are_rejected(key, value):
    entry = {**AUDIT_BACKLOG, key: value}
    with pytest.raises(OpenSRMV2ParseError, match=rf"judgment_threshold\[0\].*'{key}'"):
        parse_opensrm_v2(_manifest(judgment_threshold=[entry]))


@pytest.mark.parametrize("key", ["probe", "measurement", "segment_values", "window"])
def test_keys_irrelevant_to_feedback_latency_are_rejected(key):
    entry = {**FEEDBACK_LATENCY, key: "x"}
    with pytest.raises(OpenSRMV2ParseError, match=rf"'{key}'"):
        parse_opensrm_v2(_manifest(judgment_threshold=[entry]))


def test_the_schema_rejects_stray_keys_on_a_type_that_does_not_use_them():
    """Pins the premise of the tests above against the live schema: it closed
    JudgmentThresholdSpec in opensrm-7zpz, so the parser rejects these too."""
    entry = {**AUDIT_BACKLOG, "measurement": "x"}
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_manifest(judgment_threshold=[entry]), schema)
    with pytest.raises(OpenSRMV2ParseError, match="'measurement'"):
        parse_opensrm_v2(_manifest(judgment_threshold=[entry]))


@pytest.mark.parametrize("key", ["measurement", "sampling", "segment_values"])
def test_stability_rejects_keys_beside_its_probe(key):
    entry = {**STABILITY, key: "x"}
    with pytest.raises(OpenSRMV2ParseError, match=rf"'{key}'"):
        parse_opensrm_v2(_manifest(judgment_threshold=[entry]))


def test_stability_measurement_has_no_slo_only_fields():
    [threshold] = parse_opensrm_v2(_manifest(judgment_threshold=[STABILITY])).judgment_thresholds
    m = threshold.measurement
    assert type(m) is ThresholdMeasurement
    assert m.probe is not None
    assert not hasattr(m, "sampling")
    assert (m.bins, m.window, m.segment_dimension, m.metric) == (None, None, None, None)
    assert m.segment_values == []


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
    full = {"type": "segments", "metric": "escalation", **entry, "bounds": {"maximum_variance_from_overall": 0.05}}
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


@pytest.mark.parametrize("via", ["template", "manifest", "override"])
def test_closure_applies_to_every_source_of_an_entry(tmp_path, via):
    """The schema checks closure only on judgment_threshold itself, not on
    template.overrides (AUTHORING §9, opensrm-0myj); the parser checks the
    resolved list, so an appended entry is closed too."""
    stray = {**AUDIT_BACKLOG, "window": "7d"}
    kwargs: dict[str, Any] = {}
    template_spec: dict[str, Any] = {"judgment_threshold": [STABILITY]}
    if via == "template":
        template_spec = {"judgment_threshold": [STABILITY, stray]}
    elif via == "manifest":
        kwargs["judgment_threshold"] = [stray]
    else:
        kwargs["overrides"] = {"judgment_threshold": {"append": [stray]}}
    path = _write_pair(tmp_path, template_spec, **kwargs)
    with pytest.raises(OpenSRMV2ParseError, match=r"judgment_threshold\[1\].*'window'"):
        parse_opensrm_v2_file(path)


def test_override_replace_is_rejected(tmp_path):
    """CORE §5.5: overrides MUST NOT remove or replace a template's entries."""
    path = _write_pair(
        tmp_path,
        {"judgment_threshold": [STABILITY]},
        overrides={"judgment_threshold": {"replace": [AUDIT_BACKLOG]}},
    )
    with pytest.raises(OpenSRMV2ParseError, match=r"must not 'replace'.*MUST NOT remove or replace.*§5\.5"):
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


def test_mixed_type_override_keys_raise_the_parser_error(tmp_path):
    """YAML allows int mapping keys; sorting them with str keys is a TypeError."""
    path = _write_pair(
        tmp_path,
        {"judgment_threshold": [STABILITY]},
        overrides={"judgment_threshold": {"append": [], 1: 2, "b": 3}},
    )
    with pytest.raises(OpenSRMV2ParseError, match="only 'append'"):
        parse_opensrm_v2_file(path)


ALL_FIVE_TYPES = SCHEMA_PATH.parent / "examples" / "judgment-thresholds" / "all-five-types.yaml"


def test_every_parsed_threshold_carries_a_threshold_measurement():
    """Read live from the spec: every type, not only those whose builder
    branch a hand-written fixture happens to hit."""
    declared = yaml.safe_load(ALL_FIVE_TYPES.read_text(encoding="utf-8"))
    manifest = parse_opensrm_v2_file(ALL_FIVE_TYPES)

    types = [t.type for t in manifest.judgment_thresholds]
    assert types == [e["type"] for e in declared["spec"]["judgment_threshold"]]
    assert len(set(types)) == 5, f"expected all five types, got {types}"
    for threshold in manifest.judgment_thresholds:
        assert type(threshold.measurement) is ThresholdMeasurement, threshold.type
