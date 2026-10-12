"""threshold_from_v2_entry: the public inverse of threshold_to_v2_entry
(opensrm-908x). API consumers (nthlayer-workers reading core's /manifests)
turn a served v2 entry back into a JudgmentThreshold with the parser's own
rules, instead of hand-parsing the dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from nthlayer_common.manifest import (
    JudgmentThreshold,
    OpenSRMV2ParseError,
    threshold_from_v2_entry,
    threshold_to_v2_entry,
)
from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2

_SPEC_V2 = Path(__file__).resolve().parents[2] / "opensrm" / "spec" / "v2"
_SOURCES = [
    _SPEC_V2 / "examples" / "judgment-thresholds" / "all-five-types.yaml",
    _SPEC_V2 / "examples" / "judgment-thresholds" / "segments-two-dimensions.yaml",
    _SPEC_V2 / "examples" / "services" / "ai-gate.yaml",
]
_INVALID = _SPEC_V2 / "tests" / "invalid"


def _entries(source: Path) -> list[dict[str, Any]]:
    assert source.is_file(), f"spec example missing: {source}"
    entries = yaml.safe_load(source.read_text(encoding="utf-8"))["spec"]["judgment_threshold"]
    assert entries, f"{source.name} declares no thresholds"
    return entries


def _shell(entries: list[Any]) -> dict[str, Any]:
    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {"name": "svc", "labels": {"tier": "critical"}},
        "spec": {
            "owner": {"group": "group:default/team"},
            "service": {"name": "svc", "type": "ai-gate"},
            "judgment_threshold": entries,
        },
    }


def test_is_public():
    """Listed in the package's __all__ (the public-API rule)."""
    import nthlayer_common.manifest as manifest

    assert "threshold_from_v2_entry" in manifest.__all__


@pytest.mark.parametrize("source", _SOURCES, ids=lambda p: p.name)
def test_matches_the_manifest_parser_entry_for_entry(source):
    """Same result as parsing the entry inside a manifest: one set of rules."""
    entries = _entries(source)
    parsed = parse_opensrm_v2(_shell(entries)).judgment_thresholds
    assert [threshold_from_v2_entry(e) for e in entries] == list(parsed)


@pytest.mark.parametrize("source", _SOURCES, ids=lambda p: p.name)
def test_inverts_threshold_to_v2_entry(source):
    """from(to(t)) == t and to(from(e)) is the served entry (what core 3.0
    serves on /manifests), for every threshold the spec ships."""
    for threshold in parse_opensrm_v2(_shell(_entries(source))).judgment_thresholds:
        served = threshold_to_v2_entry(threshold)
        back = threshold_from_v2_entry(served)
        assert isinstance(back, JudgmentThreshold)
        assert back == threshold
        assert threshold_to_v2_entry(back) == served


# The spec negatives whose only defect is inside a judgment_threshold entry:
# opensrm-7zpz added n67-n92. Presence is asserted below, never skipped.
_FIRST_THRESHOLD_NEGATIVE = 67
_LAST_THRESHOLD_NEGATIVE = 92
_THRESHOLD_NEGATIVE_COUNT = _LAST_THRESHOLD_NEGATIVE - _FIRST_THRESHOLD_NEGATIVE + 1


def _threshold_negatives() -> list[Path]:
    return sorted(
        p for p in _INVALID.glob("n*.yaml")
        if _FIRST_THRESHOLD_NEGATIVE
        <= int(p.name[1:].split("-", 1)[0])
        <= _LAST_THRESHOLD_NEGATIVE
    )


def test_the_threshold_negatives_are_present():
    assert len(_threshold_negatives()) == _THRESHOLD_NEGATIVE_COUNT


@pytest.mark.parametrize("path", _threshold_negatives(), ids=lambda p: p.name)
def test_rejects_every_spec_threshold_negative(path):
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))["spec"]["judgment_threshold"][0]
    with pytest.raises(OpenSRMV2ParseError):
        threshold_from_v2_entry(entry)


@pytest.mark.parametrize("bad", [None, [], "segments", 3], ids=["none", "list", "str", "int"])
def test_rejects_a_non_mapping(bad):
    with pytest.raises(OpenSRMV2ParseError):
        threshold_from_v2_entry(bad)


def test_error_names_the_caller_location():
    with pytest.raises(OpenSRMV2ParseError) as err:
        threshold_from_v2_entry({"type": "segments"}, where="/manifests[3].judgment_thresholds[1]")
    assert str(err.value).startswith("/manifests[3].judgment_thresholds[1]")


def test_does_not_mutate_the_entry():
    entry = _entries(_SOURCES[0])[0]
    before = yaml.safe_dump(entry)
    threshold_from_v2_entry(entry)
    assert yaml.safe_dump(entry) == before


def test_json_decoded_entries_equal_yaml_parsed_ones():
    """The consumer reads JSON (core's /manifests); ints and floats decode
    differently from YAML but must give an equal threshold."""
    import json

    for source in _SOURCES:
        for entry in _entries(source):
            assert threshold_from_v2_entry(json.loads(json.dumps(entry))) == threshold_from_v2_entry(entry)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_rejects_json_non_finite_bounds(token):
    import json

    entry = json.loads(
        '{"type": "stability", "probe": {"frozen_input_set": "s3://p", '
        '"frequency": "daily"}, "bounds": {"maximum_drift": ' + token + "}}"
    )
    with pytest.raises(OpenSRMV2ParseError):
        threshold_from_v2_entry(entry)


def test_accepts_a_dict_subclass():
    from collections import OrderedDict

    entry = _entries(_SOURCES[0])[0]
    assert threshold_from_v2_entry(OrderedDict(entry)) == threshold_from_v2_entry(entry)


def test_a_key_from_a_newer_common_is_an_error_not_ignored():
    """Forward compatibility is deliberately strict (closed entries, CORE
    §5.5): consumers upgrade nthlayer-common no later than the producer."""
    entry = dict(_entries(_SOURCES[0])[0], added_in_a_later_version=1)
    with pytest.raises(OpenSRMV2ParseError):
        threshold_from_v2_entry(entry)
