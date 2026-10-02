"""The complement rule, checked against the SPEC's own worked examples.

THE THIRD PARTY THIS BEAD LACKED. opensrm-ocvu exists because the v1 and v2
parsers disagreed about SLO target units by a factor of 100, and each had its
own green suite — there was no external authority either could be measured
against, so both were free to be wrong in their own direction.

``opensrm/spec/v2/examples/judgment-slos/*.yaml`` is that authority. Every one
of the eight judgment types has a worked example there with a declared target
value chosen by the spec, not by this implementation. Running the complement
rule over them answers the question the other fixtures cannot: does the rule
agree with something nobody here wrote?

WHY THIS FILE EXISTS SEPARATELY from tests/test_target_unit_equivalence.py.
That file is 700+ lines written alongside the code it tests, across nine
commits of a fix-loop. Its judgment expectations happen to be byte-identical to
the numbers below — but it was written by the author of the predicate, which is
precisely the opensrm-oh27 precondition: a suite that agrees with the
implementation, including its bugs. oh27 survived five gate iterations, a
44-fixture suite and two full R5 passes for exactly that reason. The fixtures
here are the LIVE shipped spec files, not copies, following the convention
test_manifest_v2_archetypes.py sets out: a copy could co-evolve with a parser
bug, and these cannot.

The fragments are JudgmentSLO documents rather than whole manifests, so they
are spliced into a minimal valid ServiceManifest envelope. The envelope is
scaffolding; every number asserted below comes from the spec file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2

# File path: <ecosystem>/nthlayer-common/tests/this_file.py
# parents: [0]=tests/ [1]=nthlayer-common/ [2]=<ecosystem>/
# The opensrm spec repo is a sibling checkout, exactly as
# test_manifest_v2_archetypes.py resolves it.
ECOSYSTEM_ROOT = Path(__file__).resolve().parents[2]
JUDGMENT_EXAMPLES_DIR = (
    ECOSYSTEM_ROOT / "opensrm" / "spec" / "v2" / "examples" / "judgment-slos"
)

# The expected 0-100 SLI floor for each example, derived from the DECLARED value
# in the spec file by applying decision section 2 BY HAND:
#
#   three maxima        complement -> (1 - declared) * 100
#   outcomes, audit     scale      -> declared * 100
#   error magnitudes    untouched  -> declared
#
# Written as literals rather than computed, so a change to the converter cannot
# quietly redefine what "correct" means here. If a spec example's declared value
# changes, this table must be updated deliberately — which is the point.
EXPECTED_FLOORS: dict[str, float] = {
    "01-reversal-rate.yaml": 95.0,              # maximum_reversal_rate 0.05
    "02-high-confidence-failure.yaml": 98.0,    # maximum_failure_rate 0.02
    "03-audit-sampling.yaml": 95.0,             # audit_completion_rate 0.95, SCALED
    "04-outcomes.yaml": 90.0,                   # desired_outcome_rate 0.9, SCALED
    "05-escalation.yaml": 85.0,                 # maximum_escalation_rate 0.15
    "06-segments.yaml": 0.05,                   # maximum_variance..., UNTOUCHED
    "07-stability.yaml": 0.02,                  # maximum_drift, UNTOUCHED
    "08-calibration.yaml": 0.05,                # maximum_brier_score, UNTOUCHED
}


def _envelope(judgment_spec: dict[str, Any]) -> dict[str, Any]:
    """A minimal valid v2 ServiceManifest carrying one judgment SLO."""
    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {"name": "svc", "labels": {"tier": "critical"}},
        "spec": {
            "owner": {"group": "group:default/t"},
            "service": {"name": "svc", "type": "ai-gate"},
            "judgment_slo": [
                {"metadata": {"name": "from-spec"}, "spec": judgment_spec}
            ],
        },
    }


def test_the_spec_examples_are_actually_present():
    """Asserted FIRST and separately, because the whole module is worthless if
    the sibling checkout is missing or the layout moved.

    CLAUDE.md records this exact module shape silently skipping: under one
    worktree layout test_manifest_v2_archetypes.py found 0 archetypes and
    skipped its whole module, and the suite stayed green while it had stopped
    testing the seam. A non-zero count assertion is the cheap guard against
    repeating that, and it is why this does not use pytest.skip.
    """
    assert JUDGMENT_EXAMPLES_DIR.is_dir(), f"missing: {JUDGMENT_EXAMPLES_DIR}"

    found = sorted(p.name for p in JUDGMENT_EXAMPLES_DIR.glob("*.yaml"))

    assert found == sorted(EXPECTED_FLOORS), (
        "the spec's judgment examples changed; update EXPECTED_FLOORS "
        f"deliberately. found={found}"
    )


@pytest.mark.parametrize("filename", sorted(EXPECTED_FLOORS))
def test_spec_example_target_converts_to_the_expected_sli_floor(filename):
    """The complement rule against a value the spec chose, not this code."""
    document = yaml.safe_load((JUDGMENT_EXAMPLES_DIR / filename).read_text())

    manifest = parse_opensrm_v2(_envelope(document["spec"]), base_dir=None)

    assert len(manifest.slos) == 1
    assert manifest.slos[0].target == pytest.approx(EXPECTED_FLOORS[filename])


def test_audit_sampling_is_scaled_and_not_complemented():
    """Called out on its own because it is the case the draft ruling got wrong.

    ``audit_completion_rate`` is a FLOOR — "95% of flagged items get audited" —
    while its judgment_type sits alongside the maxima. The draft of decision
    section 2 named only ``outcomes`` as no-complement, and a blanket
    complement would have turned 0.95 into 5.0: a plausible number with the
    constraint inverted and no error anywhere. Scoping that ruling is what
    caught it, per the bead's own notes.
    """
    document = yaml.safe_load(
        (JUDGMENT_EXAMPLES_DIR / "03-audit-sampling.yaml").read_text()
    )
    declared = document["spec"]["target"]["audit_completion_rate"]

    manifest = parse_opensrm_v2(_envelope(document["spec"]), base_dir=None)

    assert declared == 0.95, "spec example changed; re-check the ruling"
    assert manifest.slos[0].target == pytest.approx(95.0)
    # The complemented value, asserted as explicitly NOT the answer.
    assert manifest.slos[0].target != pytest.approx(5.0)


def test_escalation_human_agreement_rate_is_a_floor_per_the_spec():
    """The one polarity entry no call path reaches yet [provenance IMPORTANT 1].

    ``escalation``'s target field is ``maximum_escalation_rate``, so
    ``escalation_human_agreement_rate`` is registered in
    TARGET_FIELD_IS_CEILING but never looked up — and flipping its polarity
    left all 1150 tests green. It is spec-correct as a FLOOR:
    examples/judgment-slos/05-escalation.yaml sets it to 0.8, and
    OPENSRM-CORE-v2.md glosses it "humans agree with escalation 80% of the
    time".

    Tested directly against the converter rather than through a parse, because
    nothing parses it yet. That is the point: whichever change wires it into
    JUDGMENT_TARGET_FIELDS inherits a tested invariant instead of a comment.
    Flipped to a ceiling, 0.8 becomes 19.999... — a plausible percentage with
    the constraint inverted.
    """
    from nthlayer_common.manifest.target_validation import judgment_target_percent

    document = yaml.safe_load(
        (JUDGMENT_EXAMPLES_DIR / "05-escalation.yaml").read_text()
    )
    declared = document["spec"]["target"]["escalation_human_agreement_rate"]

    assert declared == 0.8, "spec example changed; re-check the polarity"
    assert judgment_target_percent(
        "escalation_human_agreement_rate", declared
    ) == pytest.approx(80.0)
