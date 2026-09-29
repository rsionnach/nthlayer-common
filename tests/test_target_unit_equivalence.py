"""v1 and v2 must produce the same SLODefinition.target for the same SLO.

The boundary no test crossed before opensrm-ocvu. Both format parsers had their
own green suites; nothing compared them, so they were free to disagree — and
they did, by a factor of 100.

THE RULE, from the decision record
(nthlayer/docs/superpowers/decisions/slo-target-units-and-judgment-semantics.md):

    SLODefinition.target is a 0-100 percentage, always, for classical AND
    judgment SLOs. It answers "how good must this be".

    classical           OpenSLO objectives[].target is a ratio -> x100
                        OpenSLO targetPercent is already 0-100 -> as-is
    judgment MAXIMA     reversal_rate, high_confidence_failure, escalation
                        v2 states "at most X reversed" -> (1 - X) * 100
    judgment FLOORS     outcomes (desired_outcome_rate),
                        audit_sampling (audit_completion_rate)
                        v2 already states "at least X" -> X * 100, NO complement
    error magnitudes    segments, stability, calibration — out of scope here,
                        they leave the SLO concept entirely under 3c

The maxima/floors distinction is the part that bites. Complementing a floor
yields a plausible number in the right range that INVERTS the constraint: no
exception, no warning, just a service judged against the opposite of what its
author wrote. An earlier draft of the decision record named only `outcomes` as
a floor and would have silently inverted `audit_sampling`.
"""
from __future__ import annotations

import warnings

import pytest
import yaml

from nthlayer_common.manifest.parser.v2 import (
    OpenSRMV2ParseError,
    parse_opensrm_v2,
)
from nthlayer_common.manifest.target_validation import TargetConventionWarning
from nthlayer_common.manifest.v1_compat import convert_v1_to_v2

V1_AVAILABILITY = """
apiVersion: srm/v1
kind: ServiceReliabilityManifest
metadata: {name: svc, team: t, tier: critical}
spec:
  type: api
  slos:
    availability: {target: 99.9, window: 30d}
"""


def _v2_classical(objective: str) -> dict:
    return yaml.safe_load(f"""
apiVersion: opensrm.nthlayer.io/v2
kind: ServiceManifest
metadata: {{name: svc, labels: {{tier: critical}}}}
spec:
  owner: {{group: 'group:default/t'}}
  service: {{name: svc, type: api}}
  slo:
    - metadata: {{name: availability}}
      spec:
        service: svc
        objectives: [{{{objective}}}]
""")


def _v2_judgment(judgment_type: str, target_block: str) -> dict:
    return yaml.safe_load(f"""
apiVersion: opensrm.nthlayer.io/v2
kind: ServiceManifest
metadata: {{name: svc, labels: {{tier: critical}}}}
spec:
  owner: {{group: 'group:default/t'}}
  service: {{name: svc, type: ai-gate}}
  judgment_slo:
    - metadata: {{name: guard}}
      spec:
        service: svc
        judgment_type: {judgment_type}
        target: {{{target_block}}}
""")


def _targets(manifest) -> list[float]:
    return [s.target for s in manifest.slos]


# --- classical -------------------------------------------------------------


def test_v1_and_v2_agree_on_a_classical_target(tmp_path):
    """The bead's headline acceptance criterion.

    Measured before the fix: v1 gave 99.9 and v2 gave 0.999 for the same SLO.
    """
    from nthlayer_common.manifest import load_manifest

    (tmp_path / "v1.yaml").write_text(V1_AVAILABILITY)
    v1_target = _targets(
        load_manifest(tmp_path / "v1.yaml", suppress_deprecation_warning=True)
    )[0]
    v2_target = _targets(
        parse_opensrm_v2(_v2_classical("target: 0.999"), base_dir=tmp_path)
    )[0]

    assert v1_target == pytest.approx(v2_target), (
        f"same SLO, different formats: v1={v1_target}, v2={v2_target}"
    )
    assert v2_target == pytest.approx(99.9)


def test_openslo_target_percent_is_accepted(tmp_path):
    """OpenSLO defines `targetPercent` alongside `target`; we ignored it.

    A spec-legal manifest using targetPercent parsed as if it had no target at
    all, or fell through to a default — either way silently wrong.
    """
    manifest = parse_opensrm_v2(
        _v2_classical("targetPercent: 99.9"), base_dir=tmp_path
    )
    assert _targets(manifest)[0] == pytest.approx(99.9)


def test_declaring_both_target_and_target_percent_is_rejected(tmp_path):
    """OpenSLO requires exactly one. Accepting both invites silent divergence.

    Asserts the error the CALLER sees: parse_opensrm_v2 wraps OpenSLOParseError
    in OpenSRMV2ParseError, so matching the inner type would pass only by
    reaching past the public surface.
    """
    with pytest.raises(OpenSRMV2ParseError, match="exactly one"):
        parse_opensrm_v2(
            _v2_classical("target: 0.999, targetPercent: 99.9"), base_dir=tmp_path
        )


def test_the_documented_v1_to_v2_migration_round_trips(tmp_path):
    """v1_compat's docstring promises this and it did not hold.

    `convert_v1_to_v2` divides a v1 percentage to a ratio, then
    `parse_opensrm_v2` read that ratio back unconverted — so a MIGRATION
    silently changed a service's SLO target by a factor of 100. Measured at
    0.999 for a 99.9 manifest before the fix.
    """
    v1 = yaml.safe_load(V1_AVAILABILITY)
    round_tripped = parse_opensrm_v2(convert_v1_to_v2(v1), base_dir=tmp_path)
    assert _targets(round_tripped)[0] == pytest.approx(99.9)


# --- judgment: maxima complement, floors do not ----------------------------


@pytest.mark.parametrize(
    ("judgment_type", "target_block", "expected"),
    [
        # MAXIMA — v2 says "at most X"; the SLI floor is its complement.
        ("reversal_rate", "maximum_reversal_rate: 0.05", 95.0),
        ("high_confidence_failure", "maximum_failure_rate: 0.02", 98.0),
        ("escalation", "maximum_escalation_rate: 0.10", 90.0),
        # FLOORS — v2 already says "at least X". Scale only.
        ("outcomes", "desired_outcome_rate: 0.95", 95.0),
        ("audit_sampling", "audit_completion_rate: 0.95", 95.0),
    ],
)
def test_judgment_rate_targets_become_an_sli_floor(
    judgment_type, target_block, expected, tmp_path
):
    manifest = parse_opensrm_v2(
        _v2_judgment(judgment_type, target_block), base_dir=tmp_path
    )
    assert _targets(manifest)[0] == pytest.approx(expected)


def test_a_floor_is_not_complemented(tmp_path):
    """The specific inversion a blanket complement would cause.

    `audit_completion_rate: 0.95` means "at least 95% of sampled decisions get
    audited". Complemented it becomes 5.0 — still a plausible percentage, and
    the exact opposite of the author's intent. This asserts the failing value
    explicitly so the mistake cannot reappear as a passing test.
    """
    manifest = parse_opensrm_v2(
        _v2_judgment("audit_sampling", "audit_completion_rate: 0.95"),
        base_dir=tmp_path,
    )
    target = _targets(manifest)[0]
    assert target == pytest.approx(95.0)
    assert target != pytest.approx(5.0), "a floor was complemented"


# --- the warning that flagged the parser's own output ----------------------


def test_no_target_convention_warning_on_a_valid_v2_manifest(tmp_path):
    """TargetConventionWarning fired on every ordinary v2 manifest.

    The validator was correctly flagging the parser's output. Nothing acted on
    it and the unconverted value flowed on, which is what made this silent.
    Scoped to classical and rate SLOs: the error-magnitude types still warn
    until 3c moves them out of the SLO concept entirely.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", TargetConventionWarning)
        parse_opensrm_v2(_v2_classical("target: 0.999"), base_dir=tmp_path)
        parse_opensrm_v2(
            _v2_judgment("reversal_rate", "maximum_reversal_rate: 0.05"),
            base_dir=tmp_path,
        )


# --- out of scope: error magnitudes are untouched until 3c -----------------


@pytest.mark.parametrize(
    ("judgment_type", "target_block"),
    [
        ("segments", "maximum_variance_from_overall: 0.15"),
        ("stability", "maximum_drift: 0.10"),
        ("calibration", "maximum_brier_score: 0.20"),
    ],
)
def test_error_magnitude_targets_are_left_alone(judgment_type, target_block, tmp_path):
    """Deliberately unchanged by sections 1 and 2.

    These leave the SLO concept under 3c and have no SLI-floor reading: there is
    no complement of a drift or a Brier score. Converting them here would be
    inventing semantics the spec does not define — the very thing that produced
    this bead. They keep parsing as they do today, and keep warning, until 3c
    is implemented.
    """
    manifest = parse_opensrm_v2(
        _v2_judgment(judgment_type, target_block), base_dir=tmp_path
    )
    raw = float(target_block.split(":")[1])
    assert _targets(manifest)[0] == pytest.approx(raw)
