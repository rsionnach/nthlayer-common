"""
v1 compatibility defaults for producing v2-compatible models from srm/v1 manifests.

When a v1 manifest is parsed, judgment SLOs get sensible default values for
v2-only fields so consumers never see None where a reasonable default exists.

All assumptions are documented here. If a default is wrong for a specific
service, the operator should migrate to a v2 manifest and declare the correct
values explicitly.

Defaults:
  - Statistical requirements: 95% confidence intervals for all judgment types.
    Method inferred from judgment_type.
  - JudgmentMeasurement: source and method inferred from judgment_type.
  - Contract conversion: v1's flat Contract(availability, latency, judgment)
    becomes a single-element contracts list with name derived from service name.
"""

from __future__ import annotations

from nthlayer_common.manifest.models import (
    ContractPromise,
    JudgmentMeasurement,
    JudgmentPromise,
    ReliabilityContract,
    StatisticalRequirements,
)

# =============================================================================
# Statistical Requirements Defaults
# =============================================================================

# Method per judgment type — maps judgment_type to the statistical method
# that v2 spec §5.3 prescribes.
_JUDGMENT_METHOD: dict[str, str] = {
    "reversal_rate": "hypothesis_test",
    "high_confidence_failure": "hypothesis_test",
    "audit_sampling": "hypothesis_test",
    "outcomes": "hypothesis_test",
    "escalation": "hypothesis_test",
    "segments": "hypothesis_test",
    "stability": "hypothesis_test",
    "calibration": "brier_score",
}


def default_statistical_requirements(judgment_type: str) -> StatisticalRequirements:
    """Return default StatisticalRequirements for a judgment_type.

    Assumptions:
      - 95% confidence intervals (standard for SRE metrics)
      - Method inferred from judgment_type: calibration → brier_score,
        all others → hypothesis_test (appropriate for rate comparisons)
      - No minimum_sample_size default — sample size requirements vary
        too much by use case to pick a safe default
    """
    return StatisticalRequirements(
        confidence_interval_pct=95.0,
        method=_JUDGMENT_METHOD.get(judgment_type),
        minimum_sample_size=None,
    )


# =============================================================================
# Measurement Defaults
# =============================================================================

# Default measurement source per judgment type.
_JUDGMENT_SOURCE: dict[str, str] = {
    "reversal_rate": "lineage",
    "high_confidence_failure": "lineage",
    "outcomes": "downstream_signal",
    "calibration": "calibration_sample",
}


def default_measurement(judgment_type: str, window: str) -> JudgmentMeasurement:
    """Return default JudgmentMeasurement for a judgment_type.

    Assumptions:
      - reversal_rate/high_confidence_failure: source is lineage
        (verdict chain tracks reversals naturally)
      - outcomes: source is downstream_signal (outcomes are resolved
        by downstream feedback)
      - calibration: source is calibration_sample, method is brier_score,
        10 bins (standard calibration curve resolution)
      - audit_sampling, escalation, segments, stability: no source default
        (these require service-specific configuration)
      - window: carried from the SLO definition's window field
    """
    source = _JUDGMENT_SOURCE.get(judgment_type)

    measurement = JudgmentMeasurement(
        source=source,
        window=window,
    )

    # Calibration gets additional defaults
    if judgment_type == "calibration":
        measurement.method = "brier_score"
        measurement.bins = 10

    # High-confidence failure gets default threshold
    if judgment_type == "high_confidence_failure":
        measurement.confidence_threshold = 0.9

    return measurement


# =============================================================================
# Contract Conversion
# =============================================================================


def convert_v1_contract(
    service_name: str,
    availability: float | None,
    latency: dict[str, str] | None,
    judgment: dict[str, float] | None,
) -> ReliabilityContract:
    """Convert a v1 flat contract to a v2 ReliabilityContract.

    Assumptions:
      - Contract name derived from service: "{service_name}-api"
      - Judgment dict values are "below" thresholds (error rates, reversal
        rates — lower is better). This matches v1 semantics where judgment
        contract values are maximum acceptable rates.
      - No api_ref, conditions, or breach_semantics (v1 didn't express these)
    """
    promise = ContractPromise(
        availability=availability,
        latency_p99=latency.get("p99") if latency else None,
    )

    if judgment:
        for jtype, threshold in judgment.items():
            promise.judgment.append(
                JudgmentPromise(
                    judgment_type=jtype,
                    threshold=threshold,
                    direction="below",
                )
            )

    return ReliabilityContract(
        name=f"{service_name}-api",
        promise=promise,
    )
