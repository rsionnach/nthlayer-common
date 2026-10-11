"""The threshold bound tables, derived from opensrm's schema.json at runtime.

thresholds.py restates the schema's bound names, kinds and required fields so
the parser can enforce them without a JSON-Schema dependency at runtime. A
restatement is a second copy, and a second copy drifts - that is how vrpa and
ih0v happened. These tests read the copy that is authoritative and assert the
restatement agrees in BOTH directions, so a bound added to either side alone
fails here.

Fixtures are the live spec files, never copies (the convention
test_manifest_v2_archetypes.py sets). The presence check is an assertion, not a
skip: CLAUDE.md records this module shape silently skipping under a misplaced
worktree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from nthlayer_common.manifest import (
    V2_JUDGMENT_SLO_TYPES,
    V2_JUDGMENT_THRESHOLD_TYPES,
)
from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2, parse_opensrm_v2_file
from nthlayer_common.manifest.thresholds import (
    DURATION,
    FORBIDDEN_THRESHOLD_KEYS,
    RATIO,
    THRESHOLD_BOUNDS,
    THRESHOLD_REQUIRED_FIELDS,
    check_type_fields,
)

# parents: [0]=tests/ [1]=nthlayer-common/ [2]=<ecosystem>/
SPEC_V2_DIR = Path(__file__).resolve().parents[2] / "opensrm" / "spec" / "v2"
SCHEMA_PATH = SPEC_V2_DIR / "schema.json"
THRESHOLD_EXAMPLES_DIR = SPEC_V2_DIR / "examples" / "judgment-thresholds"


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _branches() -> dict[str, dict[str, Any]]:
    spec = _schema()["definitions"]["JudgmentThresholdSpec"]
    return {
        branch["if"]["properties"]["type"]["const"]: branch["then"]
        for branch in spec["allOf"]
    }


def _kind(prop: dict[str, Any]) -> str:
    ref = prop.get("$ref", "")
    if ref == "#/definitions/Duration":
        return DURATION
    if ref == "#/definitions/Ratio":
        return RATIO
    # maximum_brier_score is bounded inline ("not a Ratio semantically"), but
    # it has exactly Ratio's domain, which is all the parser enforces.
    if prop.get("type") == "number" and prop.get("minimum") == 0 and prop.get("maximum") == 1:
        return RATIO
    raise AssertionError(f"bound shape not classified by this test: {prop!r}")


def test_the_spec_is_present():
    assert SCHEMA_PATH.is_file(), f"missing: {SCHEMA_PATH}"
    assert "JudgmentThresholdSpec" in _schema()["definitions"]
    assert sorted(THRESHOLD_EXAMPLES_DIR.glob("*.yaml")), f"no examples in {THRESHOLD_EXAMPLES_DIR}"


def test_threshold_types_match_the_schema_enum_and_branches():
    spec = _schema()["definitions"]["JudgmentThresholdSpec"]
    enum = set(spec["properties"]["type"]["enum"])
    assert set(THRESHOLD_BOUNDS) == enum
    assert set(_branches()) == enum


def test_bound_names_and_kinds_match_the_schema():
    derived = {
        t: {name: _kind(prop) for name, prop in then["properties"]["bounds"]["properties"].items()}
        for t, then in _branches().items()
    }
    assert derived == THRESHOLD_BOUNDS


def test_every_bounds_object_is_closed():
    for t, then in _branches().items():
        assert then["properties"]["bounds"].get("additionalProperties") is False, t


def test_single_bound_types_require_it_and_multi_bound_types_need_one():
    """parse_bounds encodes ONE rule for this: a lone bound is required, and
    several are an anyOf. Pin that the schema still says exactly that."""
    for t, then in _branches().items():
        bounds = then["properties"]["bounds"]
        names = list(bounds["properties"])
        if len(names) == 1:
            assert bounds.get("required") == names, t
            assert "anyOf" not in bounds, t
        else:
            assert "required" not in bounds, t
            assert bounds["anyOf"] == [{"required": [n]} for n in names], t


def test_required_type_fields_match_the_schema():
    derived = {
        t: tuple(f for f in then.get("required", []) if f != "bounds")
        for t, then in _branches().items()
    }
    assert derived == THRESHOLD_REQUIRED_FIELDS


@pytest.mark.parametrize(
    "example",
    sorted(THRESHOLD_EXAMPLES_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_threshold_examples_validate_against_the_schema(example: Path):
    """Half one of the seam; the parser half is
    test_threshold_examples_parse_to_the_declared_bounds, over the same files."""
    jsonschema.validate(yaml.safe_load(example.read_text(encoding="utf-8")), _schema())


def _is_forbidden(prop: object) -> bool:
    """A schema property that no value satisfies: `false`, or `{"not": {}}`
    optionally carrying a `description`."""
    if prop is False:
        return True
    return (
        isinstance(prop, dict)
        and prop.get("not") == {}
        and set(prop) <= {"not", "description"}
    )


def test_forbidden_keys_match_the_schema():
    props = _schema()["definitions"]["JudgmentThresholdSpec"]["properties"]
    derived = {name for name, prop in props.items() if _is_forbidden(prop)}
    assert derived, "no forbidden keys derived; the derivation is broken"
    assert derived == set(FORBIDDEN_THRESHOLD_KEYS)

    method = _branches()["calibration"]["properties"]["measurement"]["properties"]["method"]
    assert _is_forbidden(method)
    with pytest.raises(ValueError, match="method"):
        check_type_fields(
            "calibration",
            {"measurement": {"bins": 10, "window": "30d", "method": "brier_score"}},
        )


def test_segment_strings_must_be_non_empty_per_schema():
    props = _branches()["segments"]["properties"]
    assert props["segment_dimension"]["minLength"] == 1
    assert props["segment_values"]["items"]["minLength"] == 1


def test_v2_judgment_vocabularies_match_the_schema_both_ways():
    """vrpa's acceptance test, on the half that has a schema the parser
    implements. Read from schema.json, never restated: a hard-coded copy
    would be a third list to drift."""
    defs = _schema()["definitions"]
    assert set(
        defs["JudgmentSLOSpec"]["properties"]["judgment_type"]["enum"]
    ) == V2_JUDGMENT_SLO_TYPES
    assert set(
        defs["JudgmentThresholdSpec"]["properties"]["type"]["enum"]
    ) == V2_JUDGMENT_THRESHOLD_TYPES
    assert not V2_JUDGMENT_SLO_TYPES & V2_JUDGMENT_THRESHOLD_TYPES


def _parse_example(example: Path, doc: dict[str, Any], tmp_path: Path):
    """Parse an example the way its `kind` says it is consumed.

    A ServiceManifest is parsed directly. A ServiceManifestTemplate is never a
    manifest: it is copied, byte for byte, to where template resolution looks
    for it and extended by a minimal ai-gate manifest that declares no
    thresholds of its own, so the resolved list is exactly the template's.
    """
    kind = doc.get("kind")
    if kind == "ServiceManifest":
        return parse_opensrm_v2(doc, base_dir=example.parent)
    if kind == "ServiceManifestTemplate":
        name = doc["metadata"]["name"]
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / f"{name}.yaml").write_bytes(example.read_bytes())
        manifest = tmp_path / "extending.yaml"
        manifest.write_text(yaml.safe_dump({
            "apiVersion": "opensrm.nthlayer.io/v2",
            "kind": "ServiceManifest",
            "metadata": {"name": "extending", "labels": {"tier": "critical"}},
            "spec": {
                "owner": {"group": "group:default/team"},
                "service": {"name": "extending", "type": "ai-gate"},
                "template": {"extends": f"template:default/{name}"},
            },
        }))
        return parse_opensrm_v2_file(manifest)
    raise AssertionError(f"{example.name}: unhandled kind {kind!r}")


@pytest.mark.parametrize(
    "example",
    sorted(THRESHOLD_EXAMPLES_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_threshold_examples_parse_to_the_declared_bounds(example: Path, tmp_path: Path):
    """Half two: the parser accepts the same bytes, and every declared bound
    survives unchanged, compared against the file rather than a literal."""
    doc = yaml.safe_load(example.read_text(encoding="utf-8"))

    manifest = _parse_example(example, doc, tmp_path)

    declared = [(t["type"], t["bounds"]) for t in doc["spec"]["judgment_threshold"]]
    assert declared, f"{example.name} declares no thresholds"
    parsed = [(t.type, t.bounds) for t in manifest.judgment_thresholds]
    assert parsed == declared
    assert manifest.get_judgment_slos() == []

    # The measurement fields too, against the file: bounds-only comparison let
    # a hardcoded frequency/window survive, because every example uses the
    # defaults.
    for entry, threshold in zip(doc["spec"]["judgment_threshold"], manifest.judgment_thresholds, strict=True):
        m = threshold.measurement
        if entry["type"] == "stability":
            assert m.probe is not None
            assert m.probe.frozen_input_set == entry["probe"]["frozen_input_set"]
            assert m.probe.frequency == entry["probe"]["frequency"]
        elif entry["type"] == "calibration":
            assert m.bins == entry["measurement"]["bins"]
            assert m.window == entry["measurement"]["window"]
        elif entry["type"] == "segments":
            assert m.segment_dimension == entry["segment_dimension"]
            assert m.segment_values == entry["segment_values"]
