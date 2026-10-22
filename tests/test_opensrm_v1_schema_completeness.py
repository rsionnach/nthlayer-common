"""Every opensrm/v1 schema property is converted or reported (opensrm-pnmv).

Read from opensrm/spec/v1/schema.json, so a field the schema gains is caught
here rather than dropped silently by the converter. Both ways: the converter
may not claim a key the schema does not define.

A ``_SECTIONS`` key is a schema definition name, optionally followed by a
dotted path of inline properties (``"Instrumentation.events"`` is the
``events`` property of the ``Instrumentation`` definition).
"""
import json
from pathlib import Path

import pytest

from nthlayer_common.manifest.opensrm_v1 import _SECTIONS

V1_SCHEMA = Path(__file__).resolve().parents[2] / "opensrm" / "spec" / "v1" / "schema.json"
DEFINITIONS = json.loads(V1_SCHEMA.read_text())["definitions"] if V1_SCHEMA.is_file() else {}


def _deref(node: dict) -> dict:
    while "$ref" in node:
        node = DEFINITIONS[node["$ref"].removeprefix("#/definitions/")]
    return node


def _schema_properties(section: str) -> set[str]:
    """The property names the schema defines at *section*; KeyError if it does not resolve."""
    head, *rest = section.split(".")
    node = _deref(DEFINITIONS[head])
    for key in rest:
        node = _deref(node["properties"][key])
    return set(node.get("properties", {}))


def test_the_v1_schema_is_present_and_every_section_resolves_in_it():
    assert V1_SCHEMA.is_file()
    for section in _SECTIONS:
        _schema_properties(section)
    assert len(_SECTIONS) >= 20


@pytest.mark.parametrize("section", sorted(_SECTIONS))
def test_section_covers_the_schema_both_ways(section):
    converted, unmapped = _SECTIONS[section]
    assert not converted & set(unmapped)
    assert converted | set(unmapped) == _schema_properties(section), section


@pytest.mark.parametrize("definition", ["Spec", "Metadata", "SLOs", "JudgmentSLOs", "Contract",
                                        "Dependency", "Ownership", "Observability", "Instrumentation"])
def test_the_sections_the_spec_names_are_covered(definition):
    assert definition in _SECTIONS
