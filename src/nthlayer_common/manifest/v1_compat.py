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

Migration (opensrm-b22.2):
  - ``convert_v1_to_v2`` produces a v2 manifest dict from a v1 manifest dict;
    output round-trips through ``parse_opensrm_v2``.
"""

from __future__ import annotations

from typing import Any

from nthlayer_common.manifest.models import (
    JUDGMENT_SLO_TYPES,
    ContractPromise,
    JudgmentMeasurement,
    JudgmentPromise,
    ReliabilityContract,
    StatisticalRequirements,
    resolve_service_type,
    valid_service_types_phrase,
)
from nthlayer_common.manifest.target_validation import (
    JUDGMENT_TARGET_FIELDS,
    judgment_promise_direction,
    judgment_target_percent,
    judgment_target_ratio,
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
      - No api_ref, conditions, or breach_semantics (v1 didn't express these)

    Judgment thresholds go through judgment_target_percent() and
    judgment_promise_direction(), the SAME helpers parser/v2.py uses
    [opensrm-ocvu]. v1 declares them exactly as v2 does — a raw ratio naming a
    maximum acceptable rate — while v1 judgment SLO targets are already 0-100.
    Emitting the raw ratio with a hardcoded "below" therefore compared a 0-100
    floor against a ratio, and every v1 manifest carrying a judgment contract
    reported a strictly STRICTER SLO as looser. Measured before the fix: an SLO
    at 98.5 against a contract promising 0.05 (a 95.0 floor) produced
    "Judgment SLO 'reversal_rate' (98.5) is looser than contract 'svc-api'
    threshold (0.05)".

    That is the same defect this bead fixed on the v2 side, and leaving it here
    would have kept one shared model with two contradictory producers — which
    is the v1/v2 divergence the bead exists to close, not a separate concern.
    """
    promise = ContractPromise(
        availability=availability,
        latency_p99=latency.get("p99") if latency else None,
    )

    if judgment:
        for jtype, threshold in judgment.items():
            field = JUDGMENT_TARGET_FIELDS.get(jtype, "")
            promise.judgment.append(
                JudgmentPromise(
                    judgment_type=jtype,
                    threshold=judgment_target_percent(field, float(threshold)),
                    direction=judgment_promise_direction(field),
                )
            )

    return ReliabilityContract(
        name=f"{service_name}-api",
        promise=promise,
    )


# =============================================================================
# v1 → v2 Manifest Migration (opensrm-b22.2)
# =============================================================================


def convert_v1_to_v2(v1_data: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 ``srm/v1`` manifest dict to an ``opensrm.nthlayer.io/v2`` dict.

    The output is shape-compatible with ``parse_opensrm_v2`` and round-trips
    through that parser without further edits.

    Mapping summary:

    - ``apiVersion`` ``srm/v1`` → ``opensrm.nthlayer.io/v2``
    - ``kind`` ``ServiceReliabilityManifest`` → ``ServiceManifest``
    - ``metadata.team`` → ``spec.owner.group: "group:default/{team}"``
    - ``metadata.tier`` → ``metadata.labels.tier``
    - ``spec.type`` → ``spec.service.type`` (required in v2; normalised
      through :data:`SERVICE_TYPE_ALIASES`, and an absent value raises)
    - ``spec.slos.<name>`` (dict-of-dicts):

      - judgment SLO names (one of :data:`JUDGMENT_SLO_TYPES`) →
        ``spec.judgment_slo`` entries with the type-specific target field.
      - all other SLOs → ``spec.slo`` OpenSLO list using a thresholdMetric
        carrying the v1 ``indicator.query``. v1 percentage targets
        (e.g. ``99.9``) are normalised to OpenSLO ratio (``0.999``).

    - ``spec.dependencies[*].name`` → ``service: "component:default/{name}"``
      with the v1 ``critical: true`` flag carried as ``criticality: "critical"``.

    Raises:
        ValueError: if the input is not v1 (no ``apiVersion: srm/v1``).
    """
    api_version = v1_data.get("apiVersion")
    if api_version != "srm/v1":
        raise ValueError(
            f"convert_v1_to_v2 expects apiVersion='srm/v1', got {api_version!r}"
        )

    metadata = v1_data.get("metadata", {}) or {}
    spec = v1_data.get("spec", {}) or {}

    name = metadata.get("name")
    team = metadata.get("team")
    tier = metadata.get("tier")
    service_type = spec.get("type")

    if not name:
        raise ValueError("v1 manifest missing metadata.name")

    # spec.type was optional in v1; spec.service.type is REQUIRED in v2
    # (opensrm-6w9d), so a v1 manifest without one cannot produce a valid v2
    # document. Fail loudly rather than defaulting: a guessed 'api' would be
    # indistinguishable downstream from one the author actually declared.
    # Invariant: everything below this point may assume service_type is a
    # non-empty str. The alias lookup hashes it and the validator matches it,
    # so anything else must be rejected HERE, as a ValueError — that is the
    # only exception type nthlayer-generate's migrate_manifest catches around
    # this call, and a TypeError would crash the CLI instead of reporting a
    # bad manifest.
    if not service_type or not isinstance(service_type, str):
        raise ValueError(
            f"v1 manifest '{name}' has no usable spec.type (got "
            f"{service_type!r}), which is required as spec.service.type "
            f"in v2. Declare it as a string before upconverting."
        )

    # Resolve and validate together. This output is checked by schema.json,
    # which knows no aliases, so a raw v1 'background-job' would emit a
    # document the spec rejects — and any value that is neither an alias nor
    # already valid ('Frontend', 'ml') was previously written verbatim.
    # Raising here rather than leaving it to the caller surfaces the problem
    # at its cause: nthlayer-generate's migrate_manifest catches ValueError
    # around this call, whereas ReliabilityManifest's own validation would
    # not fire until much later.
    resolved_type = resolve_service_type(service_type)
    if resolved_type is None:
        raise ValueError(
            f"v1 manifest '{name}' declares spec.type '{service_type}', which "
            f"is not a valid v2 service type. Must be one of: "
            f"{valid_service_types_phrase()}."
        )
    service_type = resolved_type

    # Build v2 metadata + labels
    labels: dict[str, str] = {}
    if tier is not None:
        labels["tier"] = tier

    v2_metadata: dict[str, Any] = {"name": name}
    if labels:
        v2_metadata["labels"] = labels

    # Owner: team → group ref
    v2_spec: dict[str, Any] = {
        "service": {"name": name, "type": service_type},
    }
    if team:
        v2_spec["owner"] = {"group": f"group:default/{team}"}

    # SLOs: split into classical (OpenSLO) + judgment lists
    classical_slos, judgment_slos = _convert_v1_slos(name, spec.get("slos") or {})
    if classical_slos:
        v2_spec["slo"] = classical_slos
    if judgment_slos:
        # v2 permits judgment_slo only on an ai-gate (ServiceManifest.allOf).
        # v1's shipped schema never enforced its own §11 equivalent, so v1
        # manifests pairing a non-ai-gate type with a judgment SLO really do
        # exist — they are precisely what migration runs into. Emitting the
        # document anyway produced output that failed its own round-trip
        # inside migrate_manifest_command, and handed direct callers an
        # invalid manifest with no error at all.
        if service_type != "ai-gate":
            # Names come from the entries actually routed, not from
            # intersecting JUDGMENT_SLO_TYPES with the raw slos keys:
            # _convert_v1_slos skips entries whose value is not a dict, so
            # the intersection would name SLOs that were never routed.
            slo_names = ", ".join(
                sorted(js["spec"]["judgment_type"] for js in judgment_slos)
            )
            raise ValueError(
                f"v1 manifest '{name}' declares spec.type '{service_type}' but "
                f"defines judgment SLOs ({slo_names}). v2 permits judgment SLOs "
                f"only on an 'ai-gate' service. Either declare the service as "
                f"an ai-gate, or remove the judgment SLOs before upconverting."
            )
        v2_spec["judgment_slo"] = judgment_slos

    # Dependencies: name → component ref
    v1_deps = spec.get("dependencies") or []
    if v1_deps:
        v2_spec["dependencies"] = [_convert_v1_dependency(d) for d in v1_deps]

    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": v2_metadata,
        "spec": v2_spec,
    }


def _convert_v1_slos(
    service_name: str, v1_slos: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split v1's flat slos dict into v2's (classical, judgment) lists."""
    classical: list[dict[str, Any]] = []
    judgment: list[dict[str, Any]] = []

    for slo_name, slo_def in v1_slos.items():
        if not isinstance(slo_def, dict):
            continue

        if slo_name in JUDGMENT_SLO_TYPES:
            judgment.append(_v1_slo_to_judgment(service_name, slo_name, slo_def))
        else:
            classical.append(_v1_slo_to_openslo(service_name, slo_name, slo_def))

    return classical, judgment


def _v1_slo_to_openslo(
    service_name: str, slo_name: str, v1_slo: dict[str, Any]
) -> dict[str, Any]:
    """Convert a classical v1 SLO to an OpenSLO v1 SLO document.

    v1 stores the SLI as a single PromQL query in ``indicator.query``;
    the v2 OpenSLO ``thresholdMetric`` shape carries that single query.
    Targets in v1 are 0-100 percentage; OpenSLO uses 0.0-1.0 ratio so
    targets >1.0 are divided by 100.0.
    """
    target = v1_slo.get("target")
    target_ratio: float | None = None
    if target is not None:
        target_ratio = target / 100.0 if target > 1.0 else target

    indicator = v1_slo.get("indicator") or {}
    query = indicator.get("query")

    indicator_spec: dict[str, Any] = {"metadata": {"name": slo_name}}
    if query is not None:
        indicator_spec["spec"] = {
            "thresholdMetric": {
                "metricSource": {
                    "type": "Prometheus",
                    "spec": {"query": query},
                }
            }
        }

    objectives: list[dict[str, Any]] = []
    if target_ratio is not None:
        objectives.append({"target": target_ratio})

    slo_spec: dict[str, Any] = {"indicator": indicator_spec}
    if objectives:
        slo_spec["objectives"] = objectives

    window = v1_slo.get("window")
    if window:
        slo_spec["timeWindow"] = [{"duration": window, "isRolling": True}]

    return {
        "apiVersion": "openslo/v1",
        "kind": "SLO",
        # metadata.name is the SLO's identifier; downstream parsers use it
        # as SLODefinition.name. Keep the bare slo_name (no service prefix)
        # so SLO identity is preserved across migration.
        "metadata": {"name": slo_name},
        "spec": slo_spec,
    }


def _v1_slo_to_judgment(
    service_name: str, slo_name: str, v1_slo: dict[str, Any]
) -> dict[str, Any]:
    """Convert a v1 SLO whose name matches a judgment type into a v2 judgment_slo entry."""
    target_field = JUDGMENT_TARGET_FIELDS[slo_name]
    target = v1_slo.get("target")
    # CONVERTED OUTBOUND, because the document this emits is read back by the
    # v2 parser, which now converts inbound [opensrm-ocvu]. v1 declares a 0-100
    # SLI floor (hard rule 1); v2 declares a ratio in a maximum_*/desired_*
    # field. Copying the percentage verbatim — which is what this did, on the
    # since-falsified reasoning that v2 "carries the operator-specified value
    # as-is" — made the round trip produce -9750.0 from 98.5 on
    # nthlayer/demo/specs/fraud-detect.yaml, and emitted a document that
    # violates v2's Ratio maximum of 1.
    #
    # The classical path beside this one has always had its counterpart
    # (_v1_slo_to_openslo divides by 100). Only the judgment path lacked one,
    # which is why this was a silent no-op until the inbound boundary existed.
    target_block: dict[str, Any] = {}
    if target is not None:
        target_block[target_field] = judgment_target_ratio(
            target_field, float(target)
        )

    spec_block: dict[str, Any] = {
        "judgment_type": slo_name,
        "target": target_block,
    }
    window = v1_slo.get("window")
    if window:
        spec_block["measurement"] = {"window": window}

    return {
        # Bare slo_name preserves identity through the v2 parser
        # (judgment_slo metadata.name → SLODefinition.name).
        "metadata": {"name": slo_name},
        "spec": spec_block,
    }


def _convert_v1_dependency(v1_dep: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 dependency entry to a v2 component-ref dependency."""
    dep_name = v1_dep.get("name")
    if not dep_name:
        return {}
    out: dict[str, Any] = {
        "service": f"component:default/{dep_name}",
    }
    if v1_dep.get("critical"):
        out["criticality"] = "critical"
    return out
