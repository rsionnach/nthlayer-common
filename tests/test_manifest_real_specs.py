"""Regression tests that load real ecosystem manifests.

The bug that motivated this file (discovered via the P5.1 three-tier
integration test, opensrm-saun.1): the v1 parser was reading top-level
``config["query"]``, but every real manifest in the demo uses the
canonical nested ``indicator.query`` form. Unit tests passed because their
fixtures matched the buggy parser; the divergence stayed silent until the
integration test exercised the worker → core /manifests path.

These tests close that gap by parsing each demo manifest and asserting the
canonical fields populate. If the parser ever drifts from the canonical
schema again, this fails immediately at unit-test speed.

The fixtures are not written for this parser: they are the demo's own
manifests, snapshotted in ``tests/fixtures/demo-specs/`` (its README names
the source commit). They used to be read live from a sibling checkout of
the front door, which skipped the whole module wherever that checkout was
absent (CI included) and let a demo change turn this suite red
(opensrm-t93o).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nthlayer_common.manifest import load_manifest

DEMO_SPECS_DIR = Path(__file__).resolve().parent / "fixtures" / "demo-specs"

# Listed, not globbed: a glob that matched nothing would collect no tests
# and pass. test_every_snapshotted_spec_is_listed keeps the list complete.
DEMO_SPECS = [
    "checkout-svc.yaml",
    "fraud-detect.yaml",
    "order-service.yaml",
    "payment-api.yaml",
]


def _declared_queries(spec_path: Path) -> dict[str, str]:
    """SLO name -> the ``indicator.query`` the file declares, read from the
    YAML without the parser. srm/v1 writes ``spec.slos`` as a mapping."""
    slos = yaml.safe_load(spec_path.read_text())["spec"]["slos"]
    assert isinstance(slos, dict), f"{spec_path.name}: spec.slos is not a mapping"
    return {
        name: slo["indicator"]["query"]
        for name, slo in slos.items()
        if isinstance(slo, dict) and "indicator" in slo
    }


def test_every_snapshotted_spec_is_listed():
    on_disk = sorted(p.name for p in DEMO_SPECS_DIR.iterdir() if p.suffix in (".yaml", ".yml"))
    assert on_disk == DEMO_SPECS


@pytest.mark.parametrize("spec_name", DEMO_SPECS)
def test_demo_spec_parses_with_indicator_query_populated(spec_name: str):
    """Every demo spec must parse, and each SLO that declares an
    ``indicator`` block must carry that block's query, unchanged, as
    ``indicator_query`` on the SLODefinition.

    This is the regression guard: before the fix, the parser silently
    dropped ``indicator.query`` and ``indicator_query`` came back as None,
    causing the worker SLO collector to emit NO_DATA for every SLO.
    """
    spec_path = DEMO_SPECS_DIR / spec_name
    declared = _declared_queries(spec_path)
    # Every demo spec declares at least one; without this the equality
    # below would pass on two empty dicts.
    assert declared, f"{spec_name}: no SLO declares an indicator block"
    assert all(q.strip() for q in declared.values()), f"{spec_name}: an empty declared query"

    manifest = load_manifest(spec_path)
    assert manifest.name, f"{spec_name}: manifest.name not populated"
    parsed = {slo.name: slo.indicator_query for slo in manifest.slos}

    assert {name: parsed.get(name) for name in declared} == declared


def test_fraud_detect_judgment_slo_round_trips():
    """The opensrm-saun.1 trigger spec — fraud-detect's reversal_rate
    judgment SLO — must round-trip with judgment_type set and
    indicator_query equal to the ``indicator.query`` the file declares. This
    SLO is the entry point for the whole measure → correlate → respond →
    learn chain in the integration test.

    The query is compared with the file's own, not with a metric name: the
    series it reads are the demo's to name.
    """
    spec_path = DEMO_SPECS_DIR / "fraud-detect.yaml"
    declared_query = _declared_queries(spec_path)["reversal_rate"]

    manifest = load_manifest(spec_path)
    by_name = {slo.name: slo for slo in manifest.slos}
    reversal = by_name["reversal_rate"]

    assert reversal.judgment_type == "reversal_rate"
    assert reversal.indicator_query == declared_query
