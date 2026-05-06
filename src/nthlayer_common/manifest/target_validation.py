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
