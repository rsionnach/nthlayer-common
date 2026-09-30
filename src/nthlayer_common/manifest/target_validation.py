"""SLO target convention validation for parsed manifests.

The ``manifest.SLODefinition.target`` field uses 0-100 percentage convention
canonically (opensrm-5fff: decision recorded in
docs/superpowers/decisions/ and implemented in opensrm-5fff.1):

- ``observe.collector``: ``error_budget = (100 - target) / 100``
- ``measure.worker``: severity classifiers operate on percentage targets
- OpenSLO surface (``slo_models.SLO``): 0.0-1.0 ratio with explicit
  boundary conversion in ``nthlayer-generate.slos.pipeline``

This module owns TWO things. It emits the load-time warning described below,
and it owns the JUDGMENT TARGET POLARITY convention [opensrm-ocvu] — the maps
and the inbound/outbound converters both manifest parsers use, which live here
rather than in either parser so the two cannot disagree. Those names are
package-internal but cross-module, so they are deliberately not re-exported
from manifest/__init__.py (hard rule 3); no consumer outside this package
needs them.

The warning: emitted when an SLO target value looks
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

from nthlayer_common.manifest.models import (
    JudgmentPromise,
    ReliabilityManifest,
    SLODefinition,
)

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

# TWO SPACES, and the whole convention turns on telling them apart:
#
#   DECLARED space   what a manifest writes. A ratio, and for the maxima it
#                    names a BAD rate — "at most 5% reversed" is 0.05.
#   SLI-FLOOR space  what SLODefinition.target holds. 0-100, and always
#                    "how good must this be" — that same SLO is 95.0.
#
# The vocabularies differ per space, which is why one sentence can look
# self-contradictory: a field that is a CEILING when declared becomes a value
# compared with "above" once converted. ceiling/floor describe the declared
# value; above/below describe the converted one; the spec's maximum/minimum
# describe the declared one too.
#
# VALUE means complement-vs-scale. MEMBERSHIP means "this is a rate field we
# convert at all" — see _converts_to_sli_floor(), which names that second
# question so callers do not have to read a dict lookup for it.
#
#   True   declared value is a CEILING (a maximum bad rate) -> complement
#   False  declared value is already a FLOOR                -> scale only
#   ABSENT an error MAGNITUDE, left in its own space         -> untouched
#
# Per decision section 2, only the three maxima complement. `outcomes` and
# `audit_sampling` are floors: complementing them would invert the constraint
# while producing a perfectly plausible number.
#
# `escalation_human_agreement_rate` is PRE-REGISTERED and currently unreachable:
# it is a real optional v2 field on `escalation`, but JUDGMENT_TARGET_FIELDS maps
# that type to `maximum_escalation_rate`, so nothing looks it up yet. It is here
# because it is a FLOOR while its type sits in the maxima set — the exact case
# that made keying polarity by TYPE rather than by FIELD wrong. Registering it
# now means whichever change starts reading it cannot silently invert it.
TARGET_FIELD_IS_CEILING: dict[str, bool] = {
    "maximum_reversal_rate": True,
    "maximum_failure_rate": True,
    "maximum_escalation_rate": True,
    "escalation_human_agreement_rate": False,
    "desired_outcome_rate": False,
    "audit_completion_rate": False,
}


def _converts_to_sli_floor(field_name: str) -> bool:
    """Whether this module converts *field_name* into SLI-floor space at all.

    Membership, not value: the three error magnitudes are absent from
    TARGET_FIELD_IS_CEILING and stay in declared space.
    """
    return field_name in TARGET_FIELD_IS_CEILING


def judgment_target_percent(field_name: str, value: float) -> float:
    """A judgment target as the canonical 0-100 SLI floor.

    Fields absent from TARGET_FIELD_IS_CEILING pass through UNCHANGED rather
    than being guessed at: a wrong conversion is indistinguishable from a right
    one downstream, whereas an unconverted value still trips
    TargetConventionWarning.
    """
    is_ceiling = TARGET_FIELD_IS_CEILING.get(field_name)
    if is_ceiling is None:  # error magnitude — stays in declared space
        return value
    if is_ceiling:
        return (1.0 - value) * 100.0
    return value * 100.0


def judgment_target_ratio(field_name: str, percent: float) -> float:
    """The INVERSE of judgment_target_percent: a 0-100 SLI floor back to what a
    v2 ``target`` block declares.

    Needed because v1 -> v2 migration writes a v2 document that the v2 parser
    then reads back through judgment_target_percent(). Without this, a v1
    percentage was copied verbatim into a ``maximum_*`` field and re-read as a
    ratio: measured on nthlayer/demo/specs/fraud-detect.yaml, a real shipped
    spec, ``reversal_rate.target: 98.5`` loaded directly as 98.5 and came out of
    the migration as **-9750.0**. The emitted document was also schema-illegal —
    v2 types these fields as ``Ratio`` with ``maximum: 1``.

    The classical path has had its counterpart all along (_v1_slo_to_openslo
    divides by 100); only the judgment path lacked one.

    Composes with judgment_target_percent() to the identity, for converted
    fields and for untouched magnitudes alike. Not bit-exact — float round trips
    differ by up to ~7e-15 for some values — but exact for every realistic SLI
    floor, and nothing compares a target with ``==``.
    """
    is_ceiling = TARGET_FIELD_IS_CEILING.get(field_name)
    if is_ceiling is None:  # error magnitude — already in declared space
        return percent
    if is_ceiling:
        return 1.0 - percent / 100.0
    return percent / 100.0


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
    return "above" if _converts_to_sli_floor(field_name) else "below"


def judgment_promise(judgment_type: str, declared: float) -> JudgmentPromise:
    """Build a contract promise for *judgment_type* from its DECLARED value.

    The one place threshold and direction are decided together. Both parsers
    called this as three separate steps — look up the field, convert, derive
    direction — with an unnamed ``""`` default standing in for "type I do not
    know", relying on ``"" not in TARGET_FIELD_IS_CEILING`` to land on the
    passthrough branch. Duplicating that in two modules is how they diverged in
    the first place.

    An unknown judgment_type still yields a self-consistent pair: declared value
    untouched, compared "below". Inert in practice, because validate_contracts()
    finds no SLO for a type the parser could not read.
    """
    field = JUDGMENT_TARGET_FIELDS.get(judgment_type, "")
    return JudgmentPromise(
        judgment_type=judgment_type,
        threshold=judgment_target_percent(field, float(declared)),
        direction=judgment_promise_direction(field),
    )


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
