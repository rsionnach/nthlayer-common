"""Legacy-format judgment SLOs carry judgment_type like srm/v1 (opensrm-wo62.1).

nthlayer-generate dispatches judgment recording rules on
``SLODefinition.judgment_type``. The srm/v1 parser has always set it for SLO
names in ``V2_JUDGMENT_SLO_TYPES``; the legacy parser did not, so a
legacy-format ai-gate manifest loaded with ``judgment_type=None`` on its
reversal_rate / high_confidence_failure SLOs and generate silently dropped
their judgment rules.

Fixture provenance:

- The legacy document follows the shape of
  ``nthlayer-generate/examples/uat/payment-api.yaml`` — top-level
  ``service: {name, team, tier, type}`` (all four required by
  ``_parse_legacy_to_manifest``) and ``resources:`` entries of
  ``kind: SLO`` with ``spec.objective`` / ``spec.window`` / ``spec.indicator``
  (the keys ``_extract_slos_from_resources`` reads). Only ``service.type`` and
  the SLO resource names differ: ``ai-gate`` and the judgment names.
- The srm/v1 comparison document is built independently from the srm/v1 shape
  used by ``tests/test_manifest_parser.py::v1_ai_gate_data`` — flat
  ``spec.slos`` mapping keyed by SLO name with ``target`` / ``window`` — not
  derived from the legacy parse result.
"""

from __future__ import annotations

import warnings

import pytest
import yaml

from nthlayer_common.manifest import load_manifest
from nthlayer_common.manifest.models import SourceFormat
from nthlayer_common.manifest.parser.v1 import parse_srm_v1

JUDGMENT_NAMES = ("reversal_rate", "high_confidence_failure")


def _legacy_doc() -> dict:
    return {
        "service": {
            "name": "fraud-detect",
            "team": "payments-ml",
            "tier": "critical",
            "type": "ai-gate",
        },
        "resources": [
            {
                "kind": "SLO",
                "name": "reversal_rate",
                "spec": {"objective": 95.0, "window": "30d"},
            },
            {
                "kind": "SLO",
                "name": "high_confidence_failure",
                "spec": {"objective": 98.0, "window": "7d"},
            },
            {
                "kind": "SLO",
                "name": "availability",
                "spec": {
                    "objective": 99.9,
                    "window": "30d",
                    "indicator": {
                        "type": "availability",
                        "query": 'sum(rate(http_requests_total{status=~"2.."}[5m]))'
                        " / sum(rate(http_requests_total[5m]))",
                    },
                },
            },
        ],
    }


def _srm_v1_doc() -> dict:
    return {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "fraud-detect", "team": "payments-ml", "tier": "critical"},
        "spec": {
            "type": "ai-gate",
            "slos": {
                "reversal_rate": {"target": 95.0, "window": "30d"},
                "high_confidence_failure": {"target": 98.0, "window": "7d"},
            },
        },
    }


@pytest.fixture
def legacy_manifest(tmp_path):
    path = tmp_path / "fraud-detect.yaml"
    path.write_text(yaml.safe_dump(_legacy_doc(), sort_keys=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manifest = load_manifest(path)
    assert manifest.source_format == SourceFormat.LEGACY
    return manifest


def _by_name(manifest):
    return {s.name: s for s in manifest.slos}


@pytest.mark.parametrize("name", JUDGMENT_NAMES)
def test_legacy_judgment_slo_sets_judgment_type(legacy_manifest, name):
    assert _by_name(legacy_manifest)[name].judgment_type == name


def test_legacy_get_judgment_slos_returns_judgment_slos(legacy_manifest):
    assert sorted(s.name for s in legacy_manifest.get_judgment_slos()) == sorted(JUDGMENT_NAMES)
    assert [s.name for s in legacy_manifest.get_standard_slos()] == ["availability"]


@pytest.mark.parametrize("name", JUDGMENT_NAMES)
def test_legacy_judgment_fields_match_srm_v1(legacy_manifest, name):
    legacy = _by_name(legacy_manifest)[name]
    v1 = _by_name(parse_srm_v1(_srm_v1_doc()))[name]

    assert v1.judgment_type == name  # the reference really is a judgment SLO
    assert legacy.judgment_type == v1.judgment_type
    assert legacy.slo_type == v1.slo_type
    assert legacy.window == v1.window
    assert legacy.measurement is not None
    assert legacy.measurement == v1.measurement
    assert legacy.statistical_requirements is not None
    assert legacy.statistical_requirements == v1.statistical_requirements


def test_legacy_judgment_measurement_uses_resource_window(legacy_manifest):
    # Different windows on the two judgment resources, so a hard-coded
    # window in the legacy path would fail one of them.
    slos = _by_name(legacy_manifest)
    assert slos["reversal_rate"].measurement.window == "30d"
    assert slos["high_confidence_failure"].measurement.window == "7d"


def test_legacy_non_judgment_slo_unchanged(legacy_manifest):
    slo = _by_name(legacy_manifest)["availability"]
    assert slo.judgment_type is None
    assert slo.measurement is None
    assert slo.statistical_requirements is None
    assert slo.is_judgment_slo() is False
    assert slo.slo_type == "availability"
    assert slo.target == 99.9
    assert slo.window == "30d"
    assert slo.indicator_query.startswith("sum(rate(http_requests_total")
