"""Tests for the SLO target convention validator.

The validator (opensrm-pa2w mitigation, opensrm-5fff.1 canonical convention)
flags SLO targets that look like ratios (0.0-1.0). The canonical convention
is 0-100 percentage; ratios are flagged as likely author errors. The
OpenSLO surface uses ratio with explicit boundary conversion in
``nthlayer-generate.slos.pipeline``.

These tests pin the heuristic at the boundaries (``target == 1.0``
ambiguous, out-of-range ignored) and confirm the real demo specs no
longer produce warnings now that the canonical convention is in place.
"""
from __future__ import annotations

import warnings

import pytest

from nthlayer_common.manifest.models import (
    JudgmentMeasurement,
    ReliabilityManifest,
    SLODefinition,
)
from nthlayer_common.manifest.target_validation import (
    TargetConventionWarning,
    warn_target_convention_mismatches,
)


def _manifest(slos: list[SLODefinition]) -> ReliabilityManifest:
    return ReliabilityManifest(
        name="svc",
        team="team",
        tier="standard",
        type="api",
        slos=slos,
    )


def _classical(target: float, slo_type: str = "availability") -> SLODefinition:
    return SLODefinition(name="avail", target=target, slo_type=slo_type)


def _judgment(target: float) -> SLODefinition:
    """A judgment SLO has slo_type='availability' (rate-shaped indicator) +
    a judgment_type discriminator. See SLODefinition class docstring.
    """
    return SLODefinition(
        name="reversal_rate",
        target=target,
        slo_type="availability",
        judgment_type="reversal_rate",
        measurement=JudgmentMeasurement(source="ai_decisions", method="rate", window="2m"),
    )


def _collect(manifest: ReliabilityManifest) -> list[str]:
    """Run the validator and return warning messages (as strings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_target_convention_mismatches(manifest)
        return [str(w.message) for w in caught if issubclass(w.category, TargetConventionWarning)]


class TestPercentageRange:
    """Targets in the canonical 0-100 percentage range pass silently."""

    def test_classical_99_9_clean(self):
        assert _collect(_manifest([_classical(99.9)])) == []

    def test_classical_99_99_clean(self):
        assert _collect(_manifest([_classical(99.99)])) == []

    def test_classical_50_clean(self):
        # Low percentage is unusual but valid — no warning.
        assert _collect(_manifest([_classical(50.0)])) == []

    def test_judgment_98_5_clean(self):
        # Post-xte fraud-detect reversal_rate target — canonical now.
        assert _collect(_manifest([_judgment(98.5)])) == []

    def test_judgment_99_clean(self):
        assert _collect(_manifest([_judgment(99.0)])) == []


class TestRatioWarns:
    """Targets in the (0, 1) range are flagged as ratio author errors."""

    def test_classical_0_999_warns(self):
        msgs = _collect(_manifest([_classical(0.999)]))
        assert len(msgs) == 1
        assert "target=0.999" in msgs[0]
        assert "ratio" in msgs[0]
        assert "99.9" in msgs[0]  # suggested correction

    def test_classical_0_5_warns(self):
        msgs = _collect(_manifest([_classical(0.5)]))
        assert len(msgs) == 1
        assert "target=0.5" in msgs[0]

    def test_judgment_0_985_warns(self):
        # Pre-migration convention; now flagged.
        msgs = _collect(_manifest([_judgment(0.985)]))
        assert len(msgs) == 1
        assert "target=0.985" in msgs[0]
        assert "98.5" in msgs[0]
        assert "judgment SLO" in msgs[0]

    def test_judgment_0_015_warns(self):
        # The old "1.5% reversal rate threshold" expressed as ratio.
        msgs = _collect(_manifest([_judgment(0.015)]))
        assert len(msgs) == 1
        assert "ratio" in msgs[0]


class TestEdgeCases:
    """Boundary values that should NOT warn."""

    def test_target_1_0_classical_no_warning(self):
        # 1.0 is ambiguous: 100% percentage or full-ratio. Pass silently.
        assert _collect(_manifest([_classical(1.0)])) == []

    def test_target_1_0_judgment_no_warning(self):
        assert _collect(_manifest([_judgment(1.0)])) == []

    def test_target_0_skipped(self):
        # Zero is invalid range but not a convention mismatch.
        assert _collect(_manifest([_classical(0.0)])) == []

    def test_target_negative_skipped(self):
        assert _collect(_manifest([_classical(-1.0)])) == []

    def test_target_above_100_skipped(self):
        # Above-100 is invalid range — different concern.
        assert _collect(_manifest([_classical(101.0)])) == []


class TestMixedManifests:
    """Manifests with multiple SLOs warn per-mismatch."""

    def test_one_clean_one_dirty(self):
        manifest = _manifest([
            _classical(99.9),    # canonical percentage
            _judgment(0.985),    # ratio — flagged
        ])
        msgs = _collect(manifest)
        assert len(msgs) == 1
        assert "reversal_rate" in msgs[0]
        assert "0.985" in msgs[0]

    def test_warning_includes_service_name(self):
        manifest = ReliabilityManifest(
            name="my-special-service",
            team="t",
            tier="standard",
            type="ai-gate",
            slos=[_judgment(0.985)],
        )
        msgs = _collect(manifest)
        assert len(msgs) == 1
        assert "'my-special-service'" in msgs[0]


class TestRealDemoSpecs:
    """The post-migration demo specs use canonical percentage and produce no warnings."""

    def test_fraud_detect_loads_clean(self):
        # Pre-migration this manifest emitted a warning on reversal_rate
        # (target=98.5, judgment SLO, measure expected ratio). Post
        # opensrm-5fff.1 the canonical convention is percentage and the
        # spec is correct as-is.
        from pathlib import Path

        from nthlayer_common.manifest import load_manifest

        spec_path = Path(__file__).resolve().parents[2] / "nthlayer" / "demo" / "specs" / "fraud-detect.yaml"
        if not spec_path.exists():
            pytest.skip(f"fraud-detect.yaml not present at {spec_path}")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_manifest(spec_path, suppress_deprecation_warning=True)
            target_warnings = [w for w in caught if issubclass(w.category, TargetConventionWarning)]
        assert target_warnings == []
