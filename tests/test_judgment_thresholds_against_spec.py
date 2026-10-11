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
    """Half one of the seam; the parser half is in test_manifest_parser.py
    (Task B3), over the same files."""
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
