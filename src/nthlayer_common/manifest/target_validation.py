"""SLO target convention validation for parsed manifests.

The ``manifest.SLODefinition.target`` field is consumed by three different
subsystems with different conventions for the same value:

- ``observe.collector.py``: 0-100 percentage convention
  (e.g., availability ``target=99.9``; ``error_budget = (100 - target) / 100``)
- ``measure.worker.py`` (judgment SLOs): 0.0-1.0 ratio convention
  (e.g., reversal_rate ``target=0.985``; ``budget = 1.0 - target``)
- ``slo_models.SLO`` (OpenSLO model): 0.0-1.0 ratio convention

The cross-subsystem divergence is a known correctness gap tracked in
``opensrm-pa2w.followup`` (unify SLO target convention across subsystems).
Pending that decision, this module provides a load-time warning that flags
likely unit errors — values that look like the wrong convention for the
expected consumer.

The validator never rejects a manifest. Many real manifests are ambiguous
(e.g. ``target=1.0`` could be 100% in either convention, or a reasonable
edge-case ratio). Rejecting would break working configurations. Warnings
are loud enough to catch contributor mistakes; manifest authors who know
what they're doing can ignore them or filter via Python's standard
``warnings`` machinery.

Heuristic:
- ``target <= 1.0`` → looks like ratio (0.0-1.0)
- ``target > 1.0`` → looks like percentage (0-100)
- Expected convention by SLO category:
  - judgment SLO (``judgment_type`` set, consumed by measure) → ratio
  - classical SLO (availability/latency/error_rate/throughput,
    consumed by observe) → percentage
- Mismatch between actual shape and expected convention → warning.

Edge cases skipped (different concern, not a convention warning):
- ``target == 1.0`` exactly: ambiguous (100% either way), pass through silently
- ``target <= 0`` or ``target > 100``: invalid range, caller's responsibility
"""

from __future__ import annotations

import warnings

from nthlayer_common.manifest.models import ReliabilityManifest, SLODefinition


class TargetConventionWarning(UserWarning):
    """Warn when an SLO's target value mismatches its expected consumer convention.

    Raised by :func:`warn_target_convention_mismatches` during manifest load.
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
    """Return a warning message for ``slo`` if it likely mismatches its consumer's
    convention. Return ``None`` if the SLO's target shape looks correct, is in
    the ambiguous ``target == 1.0`` zone, or is out of range entirely.
    """
    target = slo.target
    if target == 1.0 or target <= 0 or target > 100:
        # Ambiguous (100% in either convention) or out of valid range.
        # Out-of-range is a different concern than convention mismatch.
        return None

    is_judgment = slo.is_judgment_slo()
    looks_ratio = target < 1.0
    looks_pct = target > 1.0

    if is_judgment and looks_pct:
        return (
            f"Service '{service_name}' SLO '{slo.name}' has target={target} "
            f"(percentage convention) but is a judgment SLO consumed by measure, "
            f"which uses ratio convention (0.0-1.0; e.g. target=0.985 for a "
            f"≤1.5% reversal rate). See opensrm-pa2w for the cross-subsystem "
            f"divergence; pick the convention that matches your consumer until "
            f"unified."
        )

    if not is_judgment and looks_ratio:
        return (
            f"Service '{service_name}' SLO '{slo.name}' has target={target} "
            f"(ratio convention) but is a {slo.slo_type} SLO consumed by "
            f"observe, which uses percentage convention (0-100; e.g. "
            f"target=99.9 for 99.9% availability). See opensrm-pa2w for the "
            f"cross-subsystem divergence; pick the convention that matches "
            f"your consumer until unified."
        )

    return None
