"""SLO target convention validation for parsed manifests.

The ``manifest.SLODefinition.target`` field uses 0-100 percentage convention
canonically (opensrm-5fff: decision recorded in
docs/superpowers/decisions/ and implemented in opensrm-5fff.1):

- ``observe.collector``: ``error_budget = (100 - target) / 100``
- ``measure.worker``: severity classifiers operate on percentage targets
- OpenSLO surface (``slo_models.SLO``): 0.0-1.0 ratio with explicit
  boundary conversion in ``nthlayer-generate.slos.pipeline``

This module emits a load-time warning when an SLO target value looks
like the wrong convention — typically because the author wrote a ratio
(``0.999``) instead of a percentage (``99.9``). The warning is loud
enough to catch contributor mistakes; it never rejects.

Heuristic:

- ``0 < target < 1.0``: looks like a ratio author error → warn.
- ``target == 1.0``: ambiguous (100% in either convention), pass silently.
- ``target <= 0`` or ``target > 100``: out of valid range, caller's concern.
- ``1.0 < target <= 100``: canonical percentage, pass.
"""

from __future__ import annotations

import warnings

from nthlayer_common.manifest.models import ReliabilityManifest, SLODefinition

# =============================================================================
# Judgment target polarity — the single source for BOTH parsers [opensrm-ocvu]
# =============================================================================
#
# Lives here, not in parser/v2.py, because the v1 and v2 paths must not be able
# to disagree. That divergence IS this bead: v2 thresholds were moved into
# SLI-floor space while v1_compat.convert_v1_contract kept emitting a raw ratio,
# so one shared model had two contradictory producers and every v1 manifest with
# a judgment contract reported a strictly stricter SLO as "looser".
#
# This module already owns the 0-100 convention, so it owns the polarity that
# converts into it. It imports only models, so both parsers can import it.

# The REQUIRED target field for each judgment type, per opensrm/spec/v2's
# if/then blocks. Contract promises are keyed by judgment TYPE while polarity is
# a property of the FIELD, so the contract parsers need this indirection.
#
# `calibration` requires TWO fields (maximum_brier_score and
# maximum_expected_calibration_error) and only the first is named here —
# pre-existing, and resolved by decision 3c, which removes calibration from the
# SLO concept altogether.
JUDGMENT_TARGET_FIELDS: dict[str, str] = {
    "reversal_rate": "maximum_reversal_rate",
    "high_confidence_failure": "maximum_failure_rate",
    "audit_sampling": "audit_completion_rate",
    "outcomes": "desired_outcome_rate",
    "escalation": "maximum_escalation_rate",
    "segments": "maximum_variance_from_overall",
    "stability": "maximum_drift",
    "calibration": "maximum_brier_score",
}

# True  = the declared value is a CEILING (a maximum bad rate) -> complement.
# False = the declared value is already a FLOOR                -> scale only.
# ABSENT = an error MAGNITUDE, left in its own space untouched (decision 3c).
#
# Per decision section 2, only the three maxima complement. `outcomes` and
# `audit_sampling` are floors: complementing them would invert the constraint
# while producing a perfectly plausible number.
TARGET_FIELD_IS_CEILING: dict[str, bool] = {
    "maximum_reversal_rate": True,
    "maximum_failure_rate": True,
    "maximum_escalation_rate": True,
    "escalation_human_agreement_rate": False,
    "desired_outcome_rate": False,
    "audit_completion_rate": False,
}


def judgment_target_percent(field_name: str, value: float) -> float:
    """A judgment target as the canonical 0-100 SLI floor.

    Fields absent from TARGET_FIELD_IS_CEILING pass through UNCHANGED rather
    than being guessed at: a wrong conversion is indistinguishable from a right
    one downstream, whereas an unconverted value still trips
    TargetConventionWarning.
    """
    is_ceiling = TARGET_FIELD_IS_CEILING.get(field_name)
    if is_ceiling is True:
        return (1.0 - value) * 100.0
    if is_ceiling is False:
        return value * 100.0
    return value


def judgment_promise_direction(field_name: str) -> str:
    """Which way a contract promise for *field_name* must be compared.

    MUST be derived from the same lookup as judgment_target_percent(), because
    the two answers have to describe the same space. Hardcoding either one is
    how this bead produced a defect in each direction:

    - hardcoded "below" put a 0-100 floor against a raw ratio, so any converted
      target exceeded any threshold and every judgment SLO read as looser.
    - hardcoded "above" then inverted the three error MAGNITUDES, which
      judgment_target_percent deliberately leaves unconverted — measured, a
      stability SLO at 0.02 drift against a 0.05 promise reported "looser"
      while 0.08 against 0.05 reported clean.

    A field this module converts is in SLI-floor space, where higher is better,
    so the promise is a FLOOR and the comparison is "above". A field it leaves
    alone is a raw error magnitude, where lower is better, so the promise is a
    CEILING and the comparison is "below".
    """
    return "above" if field_name in TARGET_FIELD_IS_CEILING else "below"


class TargetConventionWarning(UserWarning):
    """Warn when an SLO's target value looks like a ratio (likely author error).

    The canonical convention for ``manifest.SLODefinition.target`` is
    0-100 percentage. Targets in the (0, 1) range are flagged as likely
    ratio-convention author errors; the OpenSLO surface uses ratio with
    explicit boundary conversion.

    Filterable via ``warnings.filterwarnings(..., category=TargetConventionWarning)``.
    """


def warn_target_convention_mismatches(manifest: ReliabilityManifest) -> None:
    """Inspect every SLO in ``manifest`` and emit a warning per likely mismatch.

    Called by ``load_manifest`` after each successful parse. Side-effecting
    only: emits via :func:`warnings.warn`. Returns nothing.
    """
    for slo in manifest.slos:
        message = _check_one(slo, manifest.name)
        if message is not None:
            warnings.warn(message, TargetConventionWarning, stacklevel=3)


def _check_one(slo: SLODefinition, service_name: str) -> str | None:
    """Return a warning message for ``slo`` if its target looks like a ratio.

    Return ``None`` if the target is in the canonical percentage range,
    ambiguous at exactly 1.0, or out of valid range entirely.
    """
    target = slo.target
    if target == 1.0 or target <= 0 or target > 100:
        # Ambiguous (100% in either convention) or out of valid range.
        # Out-of-range is a different concern than convention.
        return None

    if target < 1.0:
        kind = "judgment" if slo.is_judgment_slo() else slo.slo_type
        return (
            f"Service '{service_name}' SLO '{slo.name}' has target={target} "
            f"which looks like a ratio (0.0-1.0). This codebase uses 0-100 "
            f"percentage canonical for SLO targets — write '{target * 100}' "
            f"for a {kind} SLO. The OpenSLO surface converts to ratio at "
            f"the boundary; manifest values stay in percentage."
        )

    return None
