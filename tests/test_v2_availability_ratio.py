"""v2 availability ratios are checked at parse time (opensrm-f0co).

``contracts[].promise.availability`` and
``dependencies[].expected_availability`` are schema.json's ``Ratio``: a
number from 0 to 1. The parser stored whatever was written, so ``"high"``,
a quoted ``"0.9995"``, a v1-style ``99.95``, ``true`` and NaN all loaded,
and consumers then crashed comparing a string with a float or reported
9995.0%.

The bounds and the type come from the schema (the vendored snapshot in
fixtures/load_shape_matrix), not from the parser: the cases below are
built from ``Ratio``'s ``minimum`` and ``maximum``. The manifest is the
spec's own migration example, which declares both fields.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from nthlayer_common.manifest import ManifestLoadError, load_manifest

FIXTURES = Path(__file__).parent / "fixtures" / "load_shape_matrix"
SCHEMA = json.loads((FIXTURES / "v2-schema.json").read_text())
EXAMPLE = yaml.safe_load(
    (FIXTURES / "v2-spec-examples" / "migration" / "api-full-v2.yaml").read_text()
)

RATIO = SCHEMA["definitions"]["Ratio"]
LOW, HIGH = RATIO["minimum"], RATIO["maximum"]
RATIO_REF = {"$ref": "#/definitions/Ratio"}


def test_the_schema_declares_both_fields_as_its_ratio():
    # What the cases below are derived from. If the spec changes either
    # field's type or the Ratio's bounds, this fails before they mislead.
    contract = SCHEMA["definitions"]["Contract"]["properties"]["promise"]["properties"]
    dependency = SCHEMA["definitions"]["Dependency"]["properties"]

    assert contract["availability"] == RATIO_REF
    assert dependency["expected_availability"] == RATIO_REF
    assert {k: RATIO[k] for k in ("type", "minimum", "maximum")} == {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
    }


def test_the_example_declares_both_fields():
    assert EXAMPLE["spec"]["contracts"][0]["promise"]["availability"] == 0.9995
    assert EXAMPLE["spec"]["dependencies"][0]["expected_availability"] == 0.9995


# Inside the Ratio: its two bounds (as the integers the schema writes and
# as floats), and values between them.
RATIOS: dict[str, Any] = {
    "minimum": LOW,
    "maximum": HIGH,
    "minimum-float": float(LOW),
    "maximum-float": float(HIGH),
    "midpoint": (LOW + HIGH) / 2,
    "three-nines": 0.999,
    "just-above-minimum": math.nextafter(LOW, HIGH),
    "just-below-maximum": math.nextafter(HIGH, LOW),
}

# Numbers outside it, and everything YAML can write that is not a number.
NOT_RATIOS: dict[str, Any] = {
    "just-below-minimum": math.nextafter(LOW, -math.inf),
    "just-above-maximum": math.nextafter(HIGH, math.inf),
    "negative": -0.1,
    "percent": 99.95,
    "one-and-a-half": 1.5,
    "nan": float("nan"),
    "infinity": float("inf"),
    "negative-infinity": float("-inf"),
    "huge-integer": 10**400,
    "quoted-ratio": "0.9995",
    "word": "high",
    "empty-string": "",
    "true": True,
    "false": False,
    "list": [0.9995],
    "mapping": {"value": 0.9995},
    "date": datetime.date(2026, 1, 1),
    "binary": b"0.9995",
}

RATIO_IDS = sorted(RATIOS)
NOT_RATIO_IDS = sorted(NOT_RATIOS)
assert len(RATIO_IDS) == 8
assert len(NOT_RATIO_IDS) == 18


def _set_promise(document: dict[str, Any], value: Any) -> None:
    document["spec"]["contracts"][0]["promise"]["availability"] = value


def _set_expected(document: dict[str, Any], value: Any) -> None:
    document["spec"]["dependencies"][0]["expected_availability"] = value


def _promise(manifest: Any) -> Any:
    return manifest.contracts[0].promise.availability


def _expected(manifest: Any) -> Any:
    return manifest.dependencies[0].slo.availability


# field id -> (how to write it, how to read it back, what an error names:
# the position, and the contract or dependency it is in).
FIELDS = {
    "promise": (_set_promise, _promise, ("promise.availability", "payment-api")),
    "expected": (_set_expected, _expected, ("expected_availability", "postgresql")),
}
FIELD_IDS = sorted(FIELDS)
assert len(FIELD_IDS) == 2


def _load(tmp_path: Path, field: str, value: Any) -> Any:
    document = copy.deepcopy(EXAMPLE)
    FIELDS[field][0](document, value)
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_manifest(path)


@pytest.mark.parametrize("field", FIELD_IDS)
@pytest.mark.parametrize("case", RATIO_IDS)
def test_a_ratio_loads_as_written(field, case, tmp_path):
    value = RATIOS[case]

    manifest = _load(tmp_path, field, value)

    read = FIELDS[field][1](manifest)
    assert read == value
    assert type(read) is type(value)


@pytest.mark.parametrize("field", FIELD_IDS)
@pytest.mark.parametrize("case", NOT_RATIO_IDS)
def test_anything_else_is_refused_naming_the_field(field, case, tmp_path):
    with pytest.raises(ManifestLoadError) as raised:
        _load(tmp_path, field, NOT_RATIOS[case])

    position, owner = FIELDS[field][2]
    assert raised.value.backstop is False
    assert position in str(raised.value)
    assert owner in str(raised.value)


@pytest.mark.parametrize("field", FIELD_IDS)
def test_a_null_is_undeclared(field, tmp_path):
    # `availability:` with no value has always loaded as no promise, as an
    # absent key does; the model's field is optional.
    manifest = _load(tmp_path, field, None)

    assert FIELDS[field][1](manifest) is None


def test_the_other_field_does_not_mask_a_bad_one(tmp_path):
    # Each field is checked on its own: a good promise beside a bad
    # expectation is still refused, for the expectation.
    document = copy.deepcopy(EXAMPLE)
    _set_promise(document, 0.999)
    _set_expected(document, 99.95)
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    with pytest.raises(ManifestLoadError) as raised:
        load_manifest(path)

    assert "expected_availability" in str(raised.value)


def test_every_contract_and_dependency_is_checked(tmp_path):
    # Not only the first of each: the example's last dependency.
    document = copy.deepcopy(EXAMPLE)
    last = len(document["spec"]["dependencies"]) - 1
    assert last >= 2
    document["spec"]["dependencies"][last]["expected_availability"] = 99.99
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    with pytest.raises(ManifestLoadError) as raised:
        load_manifest(path)

    assert "stripe-api" in str(raised.value)
