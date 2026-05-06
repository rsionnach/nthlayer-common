"""Tests for the SLO target convention validator (opensrm-pa2w).

The validator is a load-time heuristic that flags likely unit errors when
an SLO's ``target`` value's shape doesn't match the convention expected by
its consumer subsystem. It never rejects manifests — manifest authors
who know what they're doing can ignore the warnings or filter them via
Python's ``warnings`` machinery.

These tests pin the heuristic's behaviour at convention boundaries
(``target == 1.0`` ambiguous, out-of-range ignored) and confirm that
real-world demo manifests produce the warnings opensrm-pa2w expects.
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


class TestClassicalSLOs:
    """Classical SLOs (consumed by observe) expect 0-100 percentage convention."""

    def test_target_99_9_is_clean(self):
        assert _collect(_manifest([_classical(99.9)])) == []

    def test_target_99_99_is_clean(self):
        assert _collect(_manifest([_classical(99.99)])) == []

    def test_target_50_is_clean(self):
        # 50% is unusual but valid — no warning, range looks like percentage
        assert _collect(_manifest([_classical(50.0)])) == []

    def test_target_0_999_warns(self):
        # Looks like ratio (0.0-1.0); classical expects percentage
        msgs = _collect(_manifest([_classical(0.999)]))
        assert len(msgs) == 1
        assert "target=0.999" in msgs[0]
        assert "ratio convention" in msgs[0]
        assert "observe" in msgs[0]

    def test_target_0_5_warns(self):
        # Ratio shape on a classical SLO — likely unit error
        msgs = _collect(_manifest([_classical(0.5)]))
        assert len(msgs) == 1
        assert "ratio convention" in msgs[0]

    def test_per_slo_type_in_message(self):
        # The warning names the slo_type so authors know which convention
        # the consumer expected.
        msgs = _collect(_manifest([_classical(0.99, slo_type="latency")]))
        assert len(msgs) == 1
        assert "latency SLO" in msgs[0]


class TestJudgmentSLOs:
    """Judgment SLOs (consumed by measure) expect 0.0-1.0 ratio convention."""

    def test_target_0_985_is_clean(self):
        assert _collect(_manifest([_judgment(0.985)])) == []

    def test_target_0_015_is_clean(self):
        # Low-target ratio (e.g. ≤1.5% reversal rate threshold) — clean
        assert _collect(_manifest([_judgment(0.015)])) == []

    def test_target_98_5_warns(self):
        # The opensrm-xte symptom: post-xte fraud-detect.yaml has
        # reversal_rate target=98.5, but reversal_rate is a judgment SLO
        # consumed by measure (which expects ratio). The validator catches
        # this exact mismatch.
        msgs = _collect(_manifest([_judgment(98.5)]))
        assert len(msgs) == 1
        assert "target=98.5" in msgs[0]
        assert "percentage convention" in msgs[0]
        assert "measure" in msgs[0]
        assert "judgment SLO" in msgs[0]


class TestEdgeCases:
    """Boundary values that should NOT warn."""

    def test_target_1_0_classical_no_warning(self):
        # 1.0 is ambiguous: 100% percentage or full-ratio. Pass through silently.
        assert _collect(_manifest([_classical(1.0)])) == []

    def test_target_1_0_judgment_no_warning(self):
        assert _collect(_manifest([_judgment(1.0)])) == []

    def test_target_0_skipped(self):
        # Zero is invalid range but not a convention mismatch — different concern.
        assert _collect(_manifest([_classical(0.0)])) == []

    def test_target_negative_skipped(self):
        assert _collect(_manifest([_classical(-1.0)])) == []

    def test_target_above_100_skipped(self):
        # Above-100 is invalid range — different concern, not a convention mismatch.
        assert _collect(_manifest([_classical(101.0)])) == []


class TestMixedManifests:
    """Manifests with both classical + judgment SLOs warn correctly per-SLO."""

    def test_manifest_with_one_clean_one_dirty(self):
        # The post-xte fraud-detect shape: availability (classical, percentage,
        # clean) + reversal_rate (judgment, percentage, dirty).
        manifest = _manifest([
            _classical(99.9),       # clean — classical with percentage shape
            _judgment(98.5),        # dirty — judgment with percentage shape
        ])
        msgs = _collect(manifest)
        assert len(msgs) == 1
        assert "reversal_rate" in msgs[0]
        assert "judgment SLO" in msgs[0]

    def test_warning_includes_service_name(self):
        # Multi-service portfolios benefit from naming which manifest the
        # mismatch is in.
        manifest = ReliabilityManifest(
            name="my-special-service",
            team="t",
            tier="standard",
            type="ai-gate",
            slos=[_judgment(98.5)],
        )
        msgs = _collect(manifest)
        assert len(msgs) == 1
        assert "'my-special-service'" in msgs[0]


class TestRealDemoSpecs:
    """Pin behaviour against the real demo specs (regression guard)."""

    def test_post_xte_fraud_detect_warns_on_reversal_rate(self):
        # If the fraud-detect manifest gets re-converted to ratio for the
        # reversal_rate SLO (closing pa2w.followup), this test changes.
        # Until then it's a load-bearing assertion: today's spec produces
        # this specific warning.
        from pathlib import Path
        from nthlayer_common.manifest import load_manifest

        spec_path = Path(__file__).resolve().parents[2] / "nthlayer" / "demo" / "specs" / "fraud-detect.yaml"
        if not spec_path.exists():
            pytest.skip(f"fraud-detect.yaml not present at {spec_path}")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_manifest(spec_path, suppress_deprecation_warning=True)
            target_warnings = [w for w in caught if issubclass(w.category, TargetConventionWarning)]
        assert len(target_warnings) == 1
        msg = str(target_warnings[0].message)
        assert "fraud-detect" in msg
        assert "reversal_rate" in msg
