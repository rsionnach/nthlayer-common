"""Regression tests that load real ecosystem manifests from disk.

The bug that motivated this file (discovered via the P5.1 three-tier
integration test, opensrm-saun.1): the v1 parser was reading top-level
``config["query"]``, but every real manifest in ``demo/specs/`` uses the
canonical nested ``indicator.query`` form. Unit tests passed because their
fixtures matched the buggy parser; the divergence stayed silent until the
integration test exercised the worker → core /manifests path.

These tests close that gap by parsing each real manifest from disk and
asserting the canonical fields populate. If the parser ever drifts from
the canonical schema again, this fails immediately at unit-test speed.

The fixtures here are not synthetic — they are the live demo manifests.
That's load-bearing: the value of this file is precisely that it cannot
co-evolve with parser bugs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nthlayer_common.manifest import load_manifest

# File path: <ecosystem>/nthlayer-common/tests/test_manifest_real_specs.py
# parents: [0]=tests/ [1]=nthlayer-common/ [2]=<ecosystem>/
# Post 2026-04-26 ecosystem-to-front-door consolidation, demo/specs/ lives
# inside the front-door (nthlayer/) repo at <ecosystem>/nthlayer/demo/specs.
# The legacy <ecosystem>/demo/specs path is kept as a fallback so this test
# also works in older ecosystem checkouts.
ECOSYSTEM_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATE_DIRS = (
    ECOSYSTEM_ROOT / "nthlayer" / "demo" / "specs",  # post-consolidation
    ECOSYSTEM_ROOT / "demo" / "specs",  # pre-consolidation fallback
)
DEMO_SPECS_DIR = next(
    (p for p in _CANDIDATE_DIRS if p.is_dir()),
    _CANDIDATE_DIRS[0],  # for error messages — neither dir exists
)


def _demo_spec_paths() -> list[Path]:
    if not DEMO_SPECS_DIR.is_dir():
        return []
    return sorted(p for p in DEMO_SPECS_DIR.iterdir() if p.suffix in (".yaml", ".yml"))


# Skip the entire module when the ecosystem checkout layout doesn't
# include demo/specs/ (e.g. nthlayer-common is being built standalone in a
# CI matrix that doesn't sibling-checkout the ecosystem). The integration
# tests cover the same ground when the layout is present.
pytestmark = pytest.mark.skipif(
    not _demo_spec_paths(),
    reason="demo/specs/ not present in this checkout layout",
)


@pytest.mark.parametrize(
    "spec_path",
    _demo_spec_paths(),
    ids=lambda p: p.name,
)
def test_demo_spec_parses_with_indicator_query_populated(spec_path: Path):
    """Every demo spec must parse, and SLOs that declare an ``indicator``
    block must surface a non-empty ``indicator_query`` on the SLODefinition.

    This is the regression guard: before the fix, the parser silently
    dropped ``indicator.query`` and ``indicator_query`` came back as None,
    causing the worker SLO collector to emit NO_DATA for every SLO.
    """
    manifest = load_manifest(spec_path)
    assert manifest.name, f"{spec_path.name}: manifest.name not populated"
    assert manifest.slos, f"{spec_path.name}: no SLOs parsed"

    # Re-read the raw YAML to know which SLOs declared an indicator block.
    # An SLO that never declared one is allowed to have indicator_query=None.
    import yaml

    raw = yaml.safe_load(spec_path.read_text())
    raw_slos = (raw.get("spec") or {}).get("slos") or {}

    for slo in manifest.slos:
        raw_slo = raw_slos.get(slo.name) or {}
        if isinstance(raw_slo, dict) and "indicator" in raw_slo:
            assert slo.indicator_query, (
                f"{spec_path.name}: SLO {slo.name!r} declares an indicator "
                f"block but parser produced indicator_query={slo.indicator_query!r}"
            )


def test_fraud_detect_judgment_slo_round_trips():
    """The opensrm-saun.1 trigger spec — fraud-detect's reversal_rate
    judgment SLO — must round-trip with both judgment_type and
    indicator_query populated. This SLO is the entry point for the whole
    measure → correlate → respond → learn chain in the integration test.
    """
    spec_path = DEMO_SPECS_DIR / "fraud-detect.yaml"
    if not spec_path.exists():
        pytest.skip("fraud-detect.yaml not present")

    manifest = load_manifest(spec_path)
    by_name = {slo.name: slo for slo in manifest.slos}
    reversal = by_name.get("reversal_rate")
    assert reversal is not None, "reversal_rate SLO missing from fraud-detect.yaml"
    assert reversal.judgment_type == "reversal_rate", (
        f"reversal_rate must be tagged as a judgment SLO; got {reversal.judgment_type!r}"
    )
    assert reversal.indicator_query and "gen_ai_overrides_total" in reversal.indicator_query, (
        f"reversal_rate.indicator_query must contain the canonical PromQL; "
        f"got {reversal.indicator_query!r}"
    )
