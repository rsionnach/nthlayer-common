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

from nthlayer_common.manifest.parser.v1 import OpenSRMParseError, parse_srm_v1
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

    The validator was correctly flagging the parser's own output. Nothing acted
    on it and the unconverted value flowed on, which is what made this silent.

    MUST go through ``load_manifest``, not ``parse_opensrm_v2``. The warning is
    emitted by ``warn_target_convention_mismatches`` in ``parser/loader.py``,
    which only ``load_manifest`` calls — an earlier version of this test used
    ``parse_opensrm_v2`` and therefore passed against fully pre-fix code. It
    asserted nothing at all. Verified by re-running it on the unpatched tree:
    1 passed.

    Scoped to classical and rate SLOs: the error-magnitude types still warn
    until 3c moves them out of the SLO concept entirely.
    """
    from nthlayer_common.manifest import load_manifest

    cases = {
        "classical.yaml": _v2_classical("target: 0.999"),
        "judgment.yaml": _v2_judgment(
            "reversal_rate", "maximum_reversal_rate: 0.05"
        ),
    }
    for filename, doc in cases.items():
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(doc))
        with warnings.catch_warnings():
            warnings.simplefilter("error", TargetConventionWarning)
            load_manifest(path, suppress_deprecation_warning=True)


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


# --- contract validation must compare like with like -----------------------


def _v2_with_contract(judgment_type: str, slo_target: str, promise: float) -> dict:
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
        target: {{{slo_target}}}
  contracts:
    - name: caller-contract
      promise:
        judgment: {{{judgment_type}: {promise}}}
""")


@pytest.mark.parametrize(
    ("judgment_type", "slo_target", "promise", "expect_error"),
    [
        # MAXIMUM type. Contract promises "at most 5% reversed" -> SLI floor 95.
        # An SLO at 2% reversed is a floor of 98 — STRICTER, so no error.
        ("reversal_rate", "maximum_reversal_rate: 0.02", 0.05, False),
        # An SLO at 8% reversed is a floor of 92 — LOOSER than the contract.
        ("reversal_rate", "maximum_reversal_rate: 0.08", 0.05, True),
        # FLOOR type. Contract promises "at least 95% desired outcomes".
        # An SLO promising only 90% is LOOSER. This case passed silently before
        # opensrm-ocvu: direction was hardcoded "below", so the comparison ran
        # backwards for floor types and never fired.
        ("outcomes", "desired_outcome_rate: 0.90", 0.95, True),
        ("outcomes", "desired_outcome_rate: 0.98", 0.95, False),
        # ERROR MAGNITUDES — segments / stability / calibration. These are
        # deliberately NOT converted (decision 3c takes them out of the SLO
        # concept), so both sides stay in raw lower-is-better magnitude space
        # and the promise is a CEILING, not a floor.
        #
        # This block is the regression test for a defect the FIX for the round-1
        # CRITICAL introduced: making direction uniformly "above" read these
        # unconverted magnitudes as floors and inverted them BOTH ways —
        # measured, drift 0.02 against a 0.05 promise reported "looser" while
        # 0.08 against 0.05 reported clean. The parametrise then covered only
        # reversal_rate and outcomes, so nothing saw it.
        ("stability", "maximum_drift: 0.02", 0.05, False),
        ("stability", "maximum_drift: 0.08", 0.05, True),
        ("segments", "maximum_variance_from_overall: 0.01", 0.03, False),
        ("segments", "maximum_variance_from_overall: 0.04", 0.03, True),
        ("calibration", "maximum_brier_score: 0.05", 0.10, False),
        ("calibration", "maximum_brier_score: 0.15", 0.10, True),
    ],
)
def test_contract_validation_compares_in_sli_floor_space(
    judgment_type, slo_target, promise, expect_error, tmp_path
):
    """Both sides of the comparison must be in the same space.

    `slo.target` became a 0-100 SLI floor under opensrm-ocvu while
    `JudgmentPromise.threshold` was still parsed as a raw ratio with
    `direction="below"` hardcoded — so every judgment SLO paired with a contract
    reported as looser, because any 0-100 value exceeds any ratio. The
    availability branch beside it already converted its promise
    (`promise.availability * 100`); the judgment branch did not.

    Thresholds are now converted by the same polarity rule as targets, which
    makes every judgment promise a floor and `direction` uniformly "above".
    """
    manifest = parse_opensrm_v2(
        _v2_with_contract(judgment_type, slo_target, promise), base_dir=tmp_path
    )
    errors = [e for e in manifest.validate_contracts() if "looser" in e]
    if expect_error:
        assert errors, "expected a looser-than-contract error, got none"
    else:
        assert not errors, f"unexpected error: {errors}"


# =============================================================================
# v1 must agree with v2 on contract thresholds too [opensrm-ocvu]
# =============================================================================


def _v1_with_judgment_contract(slo_target: float, promise: float) -> dict:
    """A v1 manifest whose judgment SLO is paired with a judgment contract.

    v1 declares the SLO target as a 0-100 SLI floor and the contract threshold
    as a raw ratio naming a maximum acceptable rate — exactly as v2 does.
    """
    return {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "payments", "tier": "critical"},
        "spec": {
            "type": "ai-gate",
            "slos": {"reversal_rate": {"target": slo_target}},
            "contract": {"judgment": {"reversal_rate": promise}},
        },
    }


@pytest.mark.parametrize(
    ("slo_target", "promise", "expect_error"),
    [
        # 98.5 floor vs a 5%-max promise (a 95.0 floor) — STRICTER.
        (98.5, 0.05, False),
        # 92.0 floor vs the same promise — LOOSER.
        (92.0, 0.05, True),
    ],
)
def test_v1_contract_thresholds_share_the_v2_convention(
    slo_target, promise, expect_error
):
    """The v1 path must not keep its own convention.

    opensrm-ocvu moved v2 thresholds into SLI-floor space and left
    v1_compat.convert_v1_contract emitting a raw ratio with a hardcoded
    "below", so one shared model had two contradictory producers. Every v1
    manifest carrying a judgment contract then reported a strictly stricter SLO
    as looser — unconditionally. Measured before the fix:

        Judgment SLO 'reversal_rate' (98.5) is looser than
        contract 'svc-api' threshold (0.05)

    Both paths now derive threshold AND direction from the same helpers, so
    this is the test that fails if either one drifts again.
    """
    manifest = parse_srm_v1(_v1_with_judgment_contract(slo_target, promise))

    errors = [e for e in manifest.validate_contracts() if "looser" in e]
    if expect_error:
        assert errors, "expected a looser-than-contract error, got none"
    else:
        assert not errors, f"unexpected error: {errors}"


def test_v1_and_v2_produce_identical_judgment_promises():
    """The equivalence itself, asserted on structured values.

    The same declaration — a reversal_rate SLO plus a 5% contract promise — must
    yield the same JudgmentPromise threshold and direction whichever format it
    arrives in. Asserting the promise fields directly rather than the presence
    of an error message, so a future divergence cannot hide behind two
    independently-correct verdicts.
    """
    v1 = parse_srm_v1(_v1_with_judgment_contract(98.5, 0.05))
    v1_promise = v1.contracts[0].promise.judgment[0]

    v2 = parse_opensrm_v2(
        _v2_with_contract("reversal_rate", "maximum_reversal_rate: 0.015", 0.05),
        base_dir=None,
    )
    v2_promise = v2.contracts[0].promise.judgment[0]

    assert v1_promise.threshold == v2_promise.threshold == 95.0
    assert v1_promise.direction == v2_promise.direction == "above"


# =============================================================================
# Domain validation at the conversion boundary [opensrm-ocvu edge-cases pass]
# =============================================================================
#
# The complement LAUNDERS an out-of-domain value: it turns garbage into
# something inside the plausible range that both existing safety nets then miss.
# Measured before these guards existed:
#
#   maximum_reversal_rate: 5     -> target -400.0, no TargetConventionWarning
#                                   (_check_one returns None for target <= 0)
#   maximum_reversal_rate: .nan  -> NaN target, and validate_contracts()
#                                   returned [] against a real promise, because
#                                   every NaN comparison is False — a breach
#                                   reported CLEAN
#   maximum_reversal_rate: 100   -> -9900.0
#
# The domain is not invented here: opensrm/spec/v2/schema.json $refs every one
# of these fields to Ratio = {minimum: 0, maximum: 1}.


# YAML literals, not Python floats: `nan` in a YAML scalar is the STRING "nan",
# so only `.nan` / `.inf` / `-.inf` reach the parser as non-finite numbers. A
# Python float formatted into the document would have tested string handling.
@pytest.mark.parametrize(
    "declared",
    ["5", "100", "-0.1", ".nan", ".inf", "-.inf"],
)
def test_out_of_domain_judgment_target_is_rejected(declared, tmp_path):
    """...and rejected as the PARSER's declared error type, not a bare ValueError."""
    doc = _v2_judgment("reversal_rate", f"maximum_reversal_rate: {declared}")

    with pytest.raises(OpenSRMV2ParseError):
        parse_opensrm_v2(doc, base_dir=tmp_path)


@pytest.mark.parametrize(
    ("declared", "expected_floor"),
    [("0.02", 98.0), ("0", 100.0), ("1", 0.0)],
)
def test_domain_boundaries_still_accepted(declared, expected_floor, tmp_path):
    """0 and 1 are legal ratios. Degenerate, but the spec allows them, so the
    guard must not over-reject — the failure mode that would make this bead's
    fix worse than the bug."""
    doc = _v2_judgment("reversal_rate", f"maximum_reversal_rate: {declared}")

    manifest = parse_opensrm_v2(doc, base_dir=tmp_path)

    assert manifest.slos[0].target == pytest.approx(expected_floor)


def test_nan_target_cannot_pass_contract_validation(tmp_path):
    """The specific laundering that mattered most.

    A NaN target made validate_contracts() return [] against a real promise,
    so a contract breach reported clean. Asserted at the parse boundary,
    because that is where it is now stopped.
    """
    doc = _v2_with_contract(
        "reversal_rate", "maximum_reversal_rate: .nan", 0.05
    )

    with pytest.raises(OpenSRMV2ParseError):
        parse_opensrm_v2(doc, base_dir=tmp_path)


@pytest.mark.parametrize("bad", [None, [1, 2], {"a": 1}, "abc", True])
def test_non_numeric_promise_raises_a_declared_error(bad):
    """`judgment: {reversal_rate:}` is an ordinary YAML typo.

    It reached float() and raised TypeError — a type NEITHER parser declares,
    so it escaped callers' except clauses as a stack trace. New on the v1 path
    when this bead routed v1 through the shared factory, so it is this bead's
    regression to fix. `True` is included because bool is an int subclass and
    float(True) == 1.0 would otherwise be accepted silently.
    """
    doc = {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "t", "tier": "critical"},
        "spec": {
            "type": "ai-gate",
            "slos": {"reversal_rate": {"target": 98.5}},
            "contract": {"judgment": {"reversal_rate": bad}},
        },
    }

    with pytest.raises(ValueError):
        parse_srm_v1(doc)


def test_migration_rejects_a_v1_slo_with_no_target():
    """Emitting `target: {}` produced a v2 document that could not be re-parsed,
    surfacing at load time far from the manifest that caused it."""
    doc = {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "t", "tier": "critical"},
        "spec": {"type": "ai-gate", "slos": {"reversal_rate": {"window": "2m"}}},
    }

    with pytest.raises(ValueError, match="no target"):
        convert_v1_to_v2(doc)


def test_migration_warns_when_a_v1_target_looks_like_a_ratio():
    """`target: 0.985` in v1 complements to 0.99015 — a LEGAL Ratio — and
    re-parses to 0.985, a 0.985% SLI floor, wrong by ~100x and flagged by
    nothing.

    Warns rather than raises, reusing this repo's existing (0, 1) heuristic and
    its stated policy that the warning never rejects: 0.985% is a legal floor,
    just an implausible one.
    """
    doc = {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "t", "tier": "critical"},
        "spec": {"type": "ai-gate", "slos": {"reversal_rate": {"target": 0.985}}},
    }

    with pytest.warns(TargetConventionWarning, match="looks like a ratio"):
        convert_v1_to_v2(doc)


def test_migration_does_not_warn_for_a_normal_percentage():
    """The other half — otherwise the assertion above passes for any input."""
    doc = {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "t", "tier": "critical"},
        "spec": {"type": "ai-gate", "slos": {"reversal_rate": {"target": 98.5}}},
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", TargetConventionWarning)
        convert_v1_to_v2(doc)

    assert not [
        w for w in caught if issubclass(w.category, TargetConventionWarning)
    ]


def test_migration_does_not_warn_for_an_error_magnitude_in_zero_to_one():
    """An error MAGNITUDE below 1 is correct, not a ratio mistake.

    segments / stability / calibration pass through unconverted, and a Brier
    score of 0.2 is an ordinary value. The first version of the warning above
    fired on `calibration: {target: 0.2}` in this repo's own suite — the same
    converted-vs-unconverted distinction the whole bead turns on, got wrong in
    the guard written to protect it.
    """
    doc = {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "t", "tier": "critical"},
        "spec": {
            "type": "ai-gate",
            "slos": {"calibration": {"target": 0.2}},
        },
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", TargetConventionWarning)
        v2 = convert_v1_to_v2(doc)

    assert not [
        w for w in caught if issubclass(w.category, TargetConventionWarning)
    ]
    # and it is emitted unchanged, since magnitudes are not converted
    target = v2["spec"]["judgment_slo"][0]["spec"]["target"]
    assert target["maximum_brier_score"] == pytest.approx(0.2)


# Finiteness is checked for EVERY field; the [0, 1] range only for converted
# ones. The first version of the guard combined them and ran both AFTER the
# magnitude early-return, so the three error magnitudes kept laundering NaN:
# measured, a stability target of .nan against a real promise gave
# validate_contracts() -> [], a breach reporting CLEAN, on half the field set.
@pytest.mark.parametrize(
    ("judgment_type", "field"),
    [
        ("stability", "maximum_drift"),
        ("segments", "maximum_variance_from_overall"),
        ("calibration", "maximum_brier_score"),
    ],
)
@pytest.mark.parametrize("bad", [".nan", ".inf", "-.inf"])
def test_error_magnitudes_reject_non_finite_targets(judgment_type, field, bad, tmp_path):
    doc = _v2_judgment(judgment_type, f"{field}: {bad}")

    # match="finite", not a bare raises: _extract_judgment_target raises the
    # SAME type for a MISSING target, so a typo in the field name above would
    # otherwise go green while testing nothing.
    with pytest.raises(OpenSRMV2ParseError, match="finite"):
        parse_opensrm_v2(doc, base_dir=tmp_path)


@pytest.mark.parametrize("bad", [".nan", ".inf", "-.inf"])
def test_error_magnitude_promises_reject_non_finite(bad, tmp_path):
    """The promise side too — a NaN THRESHOLD silenced the comparison just as
    effectively as a NaN target."""
    doc = _v2_with_contract("stability", "maximum_drift: 0.02", bad)

    with pytest.raises(OpenSRMV2ParseError, match="finite"):
        parse_opensrm_v2(doc, base_dir=tmp_path)


@pytest.mark.parametrize("declared", ["0.2", "1.5", "12"])
def test_error_magnitudes_stay_unranged(declared, tmp_path):
    """The other half, and the reason finiteness had to be SEPARATED from range.

    A Brier score or a drift bound is not a ratio, so magnitudes are
    legitimately outside [0, 1]. Applying the range check to them would reject
    valid manifests, and would also encode a taxonomy decision 3c is about to
    change.
    """
    doc = _v2_judgment("stability", f"maximum_drift: {declared}")

    manifest = parse_opensrm_v2(doc, base_dir=tmp_path)

    assert manifest.slos[0].target == pytest.approx(float(declared))


# =============================================================================
# The CLASSICAL boundary needs the same guards [opensrm-ocvu edge-cases iter 3]
# =============================================================================
#
# _objective_target_percent is a function THIS bead added, and it shipped with
# the two defects the judgment side had already been fixed for. Measured before
# these guards:
#
#   objectives: [{target: .nan}] -> SLODefinition.target = nan, and with a
#                                   contract promise validate_contracts()
#                                   returned [] — a breach reporting CLEAN, the
#                                   identical outcome on the identical field
#   target: {}                   -> bare TypeError out of float(), a type this
#                                   parser does not declare
#
# RANGE stays unchecked and that is deliberate: plain `target` has been unranged
# since long before this bead, so rejecting out-of-range here would reject
# manifests that parse today. Scoped decision, not a fix.


def _v2_classical(objective: str, promise: str | None = None) -> dict:
    contracts = (
        f"\n  contracts:\n    - name: c\n      promise: {{availability: {promise}}}"
        if promise
        else ""
    )
    return yaml.safe_load(f"""
apiVersion: opensrm.nthlayer.io/v2
kind: ServiceManifest
metadata: {{name: svc, labels: {{tier: critical}}}}
spec:
  owner: {{group: 'group:default/t'}}
  service: {{name: svc, type: api}}
  slo:
    - apiVersion: openslo/v1
      kind: SLO
      metadata: {{name: availability}}
      spec:
        indicator:
          metadata: {{name: availability}}
          spec:
            thresholdMetric:
              metricSource: {{type: Prometheus, spec: {{query: up}}}}
        objectives: [{{{objective}}}]{contracts}
""")


@pytest.mark.parametrize("field", ["target", "targetPercent"])
@pytest.mark.parametrize("bad", [".nan", ".inf", "-.inf"])
def test_classical_target_rejects_non_finite(field, bad, tmp_path):
    doc = _v2_classical(f"{field}: {bad}")

    # OpenSRMV2ParseError, not OpenSLOParseError: parse_opensrm_v2 wraps the
    # inner error, and this file already establishes that asserting the inner
    # type passes only by reaching past the public surface.
    with pytest.raises(OpenSRMV2ParseError, match="finite"):
        parse_opensrm_v2(doc, base_dir=tmp_path)


@pytest.mark.parametrize("field", ["target", "targetPercent"])
@pytest.mark.parametrize("bad", ["{}", "[]"])
def test_classical_target_rejects_non_numeric(field, bad, tmp_path):
    """A bare TypeError is not a type this parser declares."""
    doc = _v2_classical(f"{field}: {bad}")

    with pytest.raises(OpenSRMV2ParseError, match="must be a number"):
        parse_opensrm_v2(doc, base_dir=tmp_path)


def test_classical_nan_target_cannot_pass_contract_validation(tmp_path):
    """The specific laundering, on the classical field."""
    doc = _v2_classical("target: .nan", promise="0.999")

    # OpenSRMV2ParseError, not OpenSLOParseError: parse_opensrm_v2 wraps the
    # inner error, and this file already establishes that asserting the inner
    # type passes only by reaching past the public surface.
    with pytest.raises(OpenSRMV2ParseError, match="finite"):
        parse_opensrm_v2(doc, base_dir=tmp_path)


@pytest.mark.parametrize(
    ("objective", "expected"),
    [("target: 0.999", 99.9), ("targetPercent: 99.9", 99.9), ("target: 0", 0.0)],
)
def test_classical_legitimate_targets_still_accepted(objective, expected, tmp_path):
    """The over-rejection guard: these must keep working."""
    manifest = parse_opensrm_v2(_v2_classical(objective), base_dir=tmp_path)

    assert manifest.slos[0].target == pytest.approx(expected)


# The LAST unguarded writer of SLODefinition.target [opensrm-ocvu iter 4].
# Leaving it would have recreated this bead's own subject one level down: v2
# classical rejecting a NaN target while v1 classical accepted it. Measured
# before the guard: target=nan with a contract gave validate_contracts() -> [],
# the fourth instance of breach-reports-clean found in this gate.


def _v1_classical_doc(target: object) -> dict:
    return {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "t", "tier": "critical"},
        "spec": {
            "type": "api",
            "slos": {"availability": {"target": target}},
            "contract": {"availability": 99.9},
        },
    }


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_v1_classical_target_rejects_non_finite(bad):
    with pytest.raises(OpenSRMParseError, match="finite"):
        parse_srm_v1(_v1_classical_doc(bad))


@pytest.mark.parametrize("bad", [{}, [], True])
def test_v1_classical_target_rejects_non_numeric(bad):
    """`True` included for the same reason as elsewhere: bool is an int
    subclass, so float(True) == 1.0 would be accepted silently.

    `None` is deliberately NOT in this list: an existing guard earlier in
    _parse_slos catches it first with a better message ("requires a target or
    minimum value"), and asserting "must be a number" for it would have forced
    my expectation over the code's actual, more specific behaviour. Pinned by
    the test below so that guard cannot quietly stop running.
    """
    with pytest.raises(OpenSRMParseError, match="must be a number"):
        parse_srm_v1(_v1_classical_doc(bad))


def test_v1_classical_absent_target_keeps_its_own_error():
    """The pre-existing missing-target guard runs BEFORE the numeric one, and
    says something more useful. Pinned so adding the numeric guard cannot have
    silently taken over its case."""
    with pytest.raises(OpenSRMParseError, match="requires a target or minimum"):
        parse_srm_v1(_v1_classical_doc(None))


@pytest.mark.parametrize("good", [99.9, 0, 100, -5])
def test_v1_classical_target_stays_unranged(good):
    """Over-rejection guard, and it pins the DELIBERATE range gap.

    Range is unchecked here exactly as on the v2 classical path — including -5,
    which is nonsense but has parsed in both formats since long before this
    bead. Pinning it means a future decision to range-check has to change this
    test deliberately rather than discover it.
    """
    manifest = parse_srm_v1(_v1_classical_doc(good))

    assert manifest.slos[0].target == pytest.approx(float(good))


def test_migration_rejects_a_v1_target_above_one_hundred():
    """judgment_target_ratio's OUTBOUND 0-100 guard [provenance IMPORTANT 2].

    The guard was live and reachable but invisible to the suite: disabling
    `if not 0.0 <= percent <= 100.0` left all 1150 tests green, and no test
    referenced judgment_target_ratio by name or its message. The (0, 1) case is
    covered via the warning path above; the out-of-range RAISE was not.

    150 is not ambiguous the way 0.985 is — there is no reading of hard rule 1
    under which it is a legal percentage — so this raises rather than warns.
    """
    doc = {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "t", "tier": "critical"},
        "spec": {"type": "ai-gate", "slos": {"reversal_rate": {"target": 150}}},
    }

    with pytest.raises(ValueError, match="0-100 percentage"):
        convert_v1_to_v2(doc)


@pytest.mark.parametrize("good", [0, 50, 98.5, 100])
def test_migration_accepts_the_full_percentage_range(good):
    """The over-rejection half, so the guard above cannot be over-tightened.

    0 and 100 are the boundaries and both are legal: a 0.0 floor and a 100.0
    floor are degenerate but declarable.
    """
    doc = {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "t", "tier": "critical"},
        "spec": {"type": "ai-gate", "slos": {"reversal_rate": {"target": good}}},
    }

    with warnings.catch_warnings():
        # 0 < good < 1 would warn, but no value here is in that range; the
        # filter keeps this test from depending on the warning's behaviour.
        warnings.simplefilter("ignore", TargetConventionWarning)
        v2 = convert_v1_to_v2(doc)

    emitted = v2["spec"]["judgment_slo"][0]["spec"]["target"]
    assert emitted["maximum_reversal_rate"] == pytest.approx(1.0 - good / 100.0)
