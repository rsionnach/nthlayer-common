"""
OpenSRM v2 format parser (apiVersion: opensrm.nthlayer.io/v2).

Parses ServiceManifest files per OPENSRM-CORE-v2 spec. Translates OpenSLO
v1 SLO documents and JudgmentSLO documents into a unified SLODefinition list.
Template inheritance resolved at parse time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import yaml

from nthlayer_common.manifest.models import (
    JUDGMENT_SLO_TYPES,
    APIRef,
    BreachAction,
    BreachSemantics,
    ContractPromise,
    DecisionValue,
    Dependency,
    DependencyCriticality,
    DependencySLO,
    FailureCost,
    FallbackDeclaration,
    Instrumentation,
    JudgmentMeasurement,
    JudgmentPromise,
    Outcomes,
    Ownership,
    ProbeConfig,
    ReliabilityContract,
    ReliabilityManifest,
    RequiredEvent,
    RequiredLog,
    RequiredMetric,
    RequiredTrace,
    RevenueAttribution,
    SamplingConfig,
    SLODefinition,
    SourceFormat,
    StatisticalRequirements,
    StratifiedSample,
    VolumeEstimate,
    is_valid_service_type,
    valid_service_types_phrase,
)
from nthlayer_common.manifest.openslo.parser import (
    OpenSLOParseError,
    parse_openslo_slos,
)
from nthlayer_common.manifest.parser._shared import parse_observability

logger = structlog.get_logger()


class OpenSRMV2ParseError(Exception):
    """Error parsing OpenSRM v2 manifest."""


def is_opensrm_v2_format(data: dict[str, Any]) -> bool:
    """Check if data is in opensrm.nthlayer.io/v2 format."""
    return (
        data.get("apiVersion") == "opensrm.nthlayer.io/v2"
        and data.get("kind") == "ServiceManifest"
    )


def parse_opensrm_v2(
    data: dict[str, Any],
    source_file: str | None = None,
    base_dir: Path | None = None,
) -> ReliabilityManifest:
    """Parse opensrm.nthlayer.io/v2 format data into a ReliabilityManifest.

    Args:
        data: Parsed YAML data
        source_file: Original file path for provenance
        base_dir: Directory for resolving $ref paths in OpenSLO documents
    """
    if not is_opensrm_v2_format(data):
        raise OpenSRMV2ParseError(
            "Invalid format. Expected apiVersion: opensrm.nthlayer.io/v2 "
            "and kind: ServiceManifest"
        )

    metadata = data.get("metadata", {})
    spec = data.get("spec", {})

    # Required metadata
    name = metadata.get("name")
    if not name:
        raise OpenSRMV2ParseError("metadata.name is required")

    namespace = metadata.get("namespace", "default")

    # Owner → team extraction
    owner = spec.get("owner", {})
    ownership = _parse_v2_owner(owner)
    if not ownership:
        raise OpenSRMV2ParseError("spec.owner is required in v2 manifests")

    team = ownership.team

    # Tier from labels
    labels = metadata.get("labels", {})
    tier = labels.get("tier")
    if not tier:
        raise OpenSRMV2ParseError(
            "metadata.labels.tier is required. Set to: critical, high, standard, or low"
        )

    # Service identity
    service = spec.get("service", {})
    # spec.service is type-bearing now, so a scalar here is reachable — via a
    # hand-written `service: api` or an overrides replace. `.get` on a str
    # raised AttributeError, which escapes even the loader's
    # (OpenSRMV2ParseError, ValueError).
    if not isinstance(service, dict):
        raise OpenSRMV2ParseError(
            f"spec.service must be a mapping, got {type(service).__name__}."
        )
    service_name = service.get("name") or name
    description = service.get("description")

    # spec.service.type is a required first-class field (opensrm-6w9d).
    # It is READ, never inferred: the previous heuristic derived ai-gate
    # from the presence of judgment_slo, which inverts the spec's rule that
    # only an ai-gate may declare judgment SLOs — silently promoting a
    # misconfigured worker into ai-gate-only codepaths instead of letting
    # it be rejected. metadata.labels.type is no longer consulted.
    service_type = service.get("type")
    if not service_type:
        raise OpenSRMV2ParseError(
            "spec.service.type is required. Set it to one of: "
            f"{valid_service_types_phrase()}."
        )
    # Validate here as well as in ReliabilityManifest: an ABSENT type raised
    # the domain error above, while an INVALID one fell through to the
    # model's ValueError — so two spellings of the same authoring mistake
    # arrived as different exception types, and the ValueError escaped every
    # caller that catches only OpenSRMV2ParseError.
    if not is_valid_service_type(service_type):
        raise OpenSRMV2ParseError(
            f"Invalid type '{service_type}' in spec.service.type. "
            f"Must be one of: {valid_service_types_phrase()}."
        )

    # schema.json's ServiceManifest.allOf forbids judgment_slo whenever
    # spec.service.type is present and is not 'ai-gate' — restoring v1 §11's
    # type-specific MUST, which v1's shipped schema never enforced.
    #
    # Enforced here as well as in the schema because deleting the old
    # ai-gate-from-judgment_slo inference only stops the RECLASSIFICATION;
    # without this the same manifest is merely mis-accepted as a worker
    # instead, and get_judgment_slos() still hands judgment SLOs to callers
    # for a service the spec says cannot have them.
    #
    # Presence, not truthiness: the schema's `judgment_slo: false` rejects
    # any value, so `judgment_slo: []` is invalid too.
    if "judgment_slo" in spec and service_type != "ai-gate":
        raise OpenSRMV2ParseError(
            f"spec.judgment_slo is only permitted on an 'ai-gate' service, "
            f"but spec.service.type is '{service_type}'. Judgment SLOs measure "
            f"decision quality and are only meaningful on a decision-making "
            f"service. Remove the judgment_slo block, or declare the service "
            f"as an ai-gate if it does make decisions."
        )

    # Parse classical SLOs (OpenSLO v1 documents)
    slos: list[SLODefinition] = []
    slo_block = spec.get("slo", [])
    if slo_block:
        try:
            slos.extend(parse_openslo_slos(slo_block, base_dir=base_dir))
        except OpenSLOParseError as e:
            raise OpenSRMV2ParseError(f"Error parsing OpenSLO documents: {e}") from e

    # Parse judgment SLOs (§5)
    judgment_slos = spec.get("judgment_slo", [])
    # `.get(key, default)` returns the VALUE when the key is present, so a
    # `judgment_slo:` line with a null or scalar value bypasses the default
    # and reaches the loop below. The schema types this as an array; without
    # the check, iterating None raised a bare TypeError that escaped every
    # caller's `except OpenSRMV2ParseError`.
    if not isinstance(judgment_slos, list):
        raise OpenSRMV2ParseError(
            f"spec.judgment_slo must be a list, got "
            f"{type(judgment_slos).__name__}."
        )
    for js_data in judgment_slos:
        slos.append(_parse_judgment_slo(js_data))

    # Parse contracts (§6)
    contracts = _parse_contracts(spec.get("contracts", []))

    # Parse dependencies (§7)
    dependencies = _parse_dependencies(spec.get("dependencies", []))

    # Parse instrumentation (§8)
    instrumentation = _parse_instrumentation(spec.get("instrumentation"))

    # Parse outcomes (Missing Capabilities § 1) — business value mapping
    outcomes = _parse_outcomes(spec.get("outcomes"))

    # Parse observability (non-standard but commonly included)
    observability = parse_observability(spec.get("observability"))

    # Template ref (§9) — resolved before this function is called.
    # Store the original ref for provenance.
    template_ref = None
    template_block = spec.get("template", {})
    if template_block:
        template_ref = template_block.get("extends")

    return ReliabilityManifest(
        name=service_name,
        team=team,
        tier=tier,
        type=service_type,
        namespace=namespace,
        description=description,
        labels=labels,
        annotations=metadata.get("annotations", {}),
        slos=slos,
        dependencies=dependencies,
        ownership=ownership,
        observability=observability,
        contracts=contracts,
        instrumentation=instrumentation,
        outcomes=outcomes,
        template_ref=template_ref,
        source_format=SourceFormat.OPENSRM_V2,
        source_file=source_file,
        raw_data=data,
    )


def parse_opensrm_v2_file(
    file_path: str | Path,
) -> ReliabilityManifest:
    """Parse an opensrm.nthlayer.io/v2 manifest file.

    Template $ref and OpenSLO $ref are resolved relative to the file's
    directory.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {file_path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise OpenSRMV2ParseError(f"Invalid YAML in {file_path}: {e}") from e

    if not isinstance(data, dict):
        raise OpenSRMV2ParseError(f"Expected YAML object in {file_path}")

    # Resolve template before parsing
    data = resolve_v2_template(data, path.parent)

    return parse_opensrm_v2(data, source_file=str(path), base_dir=path.parent)


# =============================================================================
# Owner Parsing
# =============================================================================


def _parse_v2_owner(owner_data: dict[str, Any]) -> Ownership | None:
    """Parse v2 spec.owner into Ownership.

    Backstage entity refs (group:namespace/name) are parsed to extract
    the team name. Ref resolution failure is a parse error.
    """
    if not owner_data:
        return None

    group_ref = owner_data.get("group")
    if not group_ref:
        raise OpenSRMV2ParseError("spec.owner.group is required")

    # Extract team name from Backstage entity ref
    team = _resolve_backstage_ref(group_ref, "group")

    escalation_ref = owner_data.get("escalation")
    technical_contact_ref = owner_data.get("technical_contact")

    # Derive escalation name from ref if present
    escalation = None
    if escalation_ref:
        escalation = _resolve_backstage_ref(escalation_ref, "group")

    return Ownership(
        team=team,
        group_ref=group_ref,
        escalation=escalation,
        escalation_ref=escalation_ref,
        technical_contact_ref=technical_contact_ref,
    )


def _resolve_backstage_ref(ref: str, expected_kind: str) -> str:
    """Extract the name from a Backstage entity reference.

    Format: "kind:namespace/name" or "kind:name" (default namespace).
    Raises OpenSRMV2ParseError on invalid format.
    """
    if ":" not in ref:
        raise OpenSRMV2ParseError(
            f"Invalid Backstage entity reference: '{ref}'. "
            f"Expected format: '{expected_kind}:namespace/name'"
        )

    kind, rest = ref.split(":", 1)
    if kind != expected_kind:
        raise OpenSRMV2ParseError(
            f"Expected {expected_kind} reference, got '{kind}' in '{ref}'"
        )

    if "/" in rest:
        _namespace, name = rest.rsplit("/", 1)
    else:
        name = rest

    name = name.strip()
    if not name:
        raise OpenSRMV2ParseError(f"Empty name in Backstage reference: '{ref}'")

    return name


# =============================================================================
# Judgment SLO Parsing
# =============================================================================


def _parse_judgment_slo(data: dict[str, Any]) -> SLODefinition:
    """Parse a v2 judgment SLO entry into SLODefinition.

    Handles all 8 standard types from §5.2.
    """
    js_spec = data.get("spec", {})
    js_metadata = data.get("metadata", {})

    name = js_metadata.get("name")
    if not name:
        raise OpenSRMV2ParseError("Judgment SLO missing metadata.name")

    judgment_type = js_spec.get("judgment_type")
    if not judgment_type:
        raise OpenSRMV2ParseError(f"Judgment SLO '{name}' missing spec.judgment_type")
    if judgment_type not in JUDGMENT_SLO_TYPES:
        raise OpenSRMV2ParseError(
            f"Unknown judgment_type '{judgment_type}' in '{name}'. "
            f"Valid types: {', '.join(sorted(JUDGMENT_SLO_TYPES))}"
        )

    # Extract target value from type-specific target block
    target_data = js_spec.get("target", {})
    target = _extract_judgment_target(judgment_type, target_data, name)

    # Measurement config
    measurement = _parse_judgment_measurement(judgment_type, js_spec)

    # Breach actions
    breach_actions = _parse_breach_actions(js_spec.get("breach_actions", []))

    # Statistical requirements
    stats = _parse_statistical_requirements(js_spec)

    # Window from measurement or default
    window = measurement.window if measurement.window else "30d"

    return SLODefinition(
        name=name,
        target=target,
        slo_type="availability",  # judgment SLOs track a rate; slo_type describes indicator shape
        window=window,
        judgment_type=judgment_type,
        measurement=measurement,
        breach_actions=breach_actions,
        statistical_requirements=stats,
        description=js_metadata.get("displayName"),
    )


def _extract_judgment_target(
    judgment_type: str,
    target_data: dict[str, Any],
    name: str,
) -> float:
    """Extract the target value from a judgment SLO's target block.

    Each judgment type has a different target field name in the spec.
    """
    # Type-specific target field names
    target_fields: dict[str, str] = {
        "reversal_rate": "maximum_reversal_rate",
        "high_confidence_failure": "maximum_failure_rate",
        "audit_sampling": "audit_completion_rate",
        "outcomes": "desired_outcome_rate",
        "escalation": "maximum_escalation_rate",
        "segments": "maximum_variance_from_overall",
        "stability": "maximum_drift",
        "calibration": "maximum_brier_score",
    }

    field_name = target_fields.get(judgment_type)
    if not field_name:
        raise OpenSRMV2ParseError(
            f"No target field mapping for judgment_type '{judgment_type}'"
        )

    value = target_data.get(field_name)
    if value is None:
        raise OpenSRMV2ParseError(
            f"Judgment SLO '{name}' missing target.{field_name}"
        )

    return float(value)


def _parse_judgment_measurement(
    judgment_type: str,
    spec: dict[str, Any],
) -> JudgmentMeasurement:
    """Parse measurement config from judgment SLO spec."""
    m_data = spec.get("measurement", {})

    measurement = JudgmentMeasurement(
        source=m_data.get("source"),
        confidence_threshold=m_data.get("confidence_threshold"),
        bins=m_data.get("bins"),
        method=m_data.get("method"),
        window=m_data.get("window"),
        frequency=m_data.get("frequency"),
    )

    # Probe config (stability type)
    probe_data = spec.get("probe")
    if probe_data:
        measurement.probe = ProbeConfig(
            frozen_input_set=probe_data.get("frozen_input_set", ""),
            frequency=probe_data.get("frequency", "daily"),
        )

    # Sampling config (audit_sampling type)
    sampling_data = spec.get("sampling")
    if sampling_data:
        stratified = []
        for s in sampling_data.get("stratified", []):
            stratified.append(StratifiedSample(
                segment=s.get("segment", ""),
                rate=float(s.get("rate", 0)),
            ))
        measurement.sampling = SamplingConfig(
            overall_rate=float(sampling_data.get("overall_rate", 0)),
            stratified=stratified,
        )

    # Segment config (segments type)
    measurement.segment_dimension = spec.get("segment_dimension")
    segment_values = spec.get("segment_values", [])
    if segment_values:
        measurement.segment_values = segment_values

    return measurement


def _parse_breach_actions(actions_data: list[dict[str, Any]]) -> list[BreachAction]:
    """Parse breach actions from v2 spec §5.4."""
    actions = []

    for action in actions_data:
        if not isinstance(action, dict):
            continue

        if "notify" in action:
            actions.append(BreachAction(
                action_type="notify",
                target=action["notify"],
            ))
        elif "create_case" in action:
            case_params = action["create_case"]
            if isinstance(case_params, dict):
                actions.append(BreachAction(
                    action_type="create_case",
                    parameters=case_params,
                ))
        elif "reduce_autonomy" in action:
            ra_params = action["reduce_autonomy"]
            if isinstance(ra_params, dict):
                actions.append(BreachAction(
                    action_type="reduce_autonomy",
                    target=ra_params.get("agent"),
                    parameters=ra_params,
                ))
        elif "action_request" in action:
            ar_params = action["action_request"]
            if isinstance(ar_params, dict):
                actions.append(BreachAction(
                    action_type="action_request",
                    target=ar_params.get("action_id"),
                    parameters=ar_params,
                ))

    return actions


def _parse_statistical_requirements(spec: dict[str, Any]) -> StatisticalRequirements:
    """Parse or default statistical requirements."""
    m_data = spec.get("measurement", {})

    return StatisticalRequirements(
        confidence_interval_pct=95.0,
        method=m_data.get("method"),
        minimum_sample_size=m_data.get("minimum_sample_size"),
    )


# =============================================================================
# Contract Parsing
# =============================================================================


def _parse_contracts(contracts_data: list[dict[str, Any]]) -> list[ReliabilityContract]:
    """Parse v2 reliability contracts from §6."""
    contracts = []

    for c_data in contracts_data:
        if not isinstance(c_data, dict):
            continue

        name = c_data.get("name")
        if not name:
            raise OpenSRMV2ParseError("Contract missing 'name' field")

        # Promise
        promise_data = c_data.get("promise", {})
        judgment_promises = []
        for jtype, threshold in promise_data.get("judgment", {}).items():
            judgment_promises.append(JudgmentPromise(
                judgment_type=jtype,
                threshold=float(threshold),
                direction="below",  # contract thresholds are maximums (lower is better)
            ))

        promise = ContractPromise(
            availability=promise_data.get("availability"),
            latency_p99=promise_data.get("latency_p99"),
            throughput=promise_data.get("throughput"),
            judgment=judgment_promises,
        )

        # API ref
        api_ref = None
        api_ref_data = c_data.get("api_ref")
        if api_ref_data:
            api_ref = APIRef(
                openapi=api_ref_data.get("openapi"),
                asyncapi=api_ref_data.get("asyncapi"),
                operation_id=api_ref_data.get("operation_id"),
            )

        # Breach semantics
        breach_semantics = None
        bs_data = c_data.get("breach_semantics")
        if bs_data:
            comp = bs_data.get("compensation")
            comp_desc = comp.get("description") if isinstance(comp, dict) else comp
            sla = bs_data.get("sla_credits", {})
            breach_semantics = BreachSemantics(
                consumer_notification=bs_data.get("consumer_notification"),
                compensation=comp_desc,
                sla_credits_enabled=sla.get("enabled", False) if isinstance(sla, dict) else False,
                sla_credits_ref=sla.get("calculation_ref") if isinstance(sla, dict) else None,
            )

        contracts.append(ReliabilityContract(
            name=name,
            promise=promise,
            api_ref=api_ref,
            conditions=c_data.get("conditions", []),
            breach_semantics=breach_semantics,
            contract_owner=c_data.get("contract_owner"),
        ))

    return contracts


# =============================================================================
# Dependency Parsing
# =============================================================================


def _parse_dependencies(deps_data: list[dict[str, Any]]) -> list[Dependency]:
    """Parse v2 dependencies from §7."""
    dependencies = []

    for dep_data in deps_data:
        if not isinstance(dep_data, dict):
            continue

        service_ref = dep_data.get("service")
        if not service_ref:
            raise OpenSRMV2ParseError("Dependency missing 'service' field")

        # Extract name from Backstage entity ref
        name = _resolve_backstage_ref(service_ref, "component")

        # Map expected guarantees into DependencySLO
        slo = None
        if any(k in dep_data for k in ("expected_availability", "expected_latency_p99")):
            slo = DependencySLO(
                availability=dep_data.get("expected_availability"),
                latency_p99=dep_data.get("expected_latency_p99"),
            )

        # Infer type from context
        dep_type = dep_data.get("type", "api")

        # Criticality
        criticality = None
        critical = False
        if "criticality" in dep_data:
            try:
                criticality = DependencyCriticality(dep_data["criticality"])
                critical = criticality in (DependencyCriticality.CRITICAL, DependencyCriticality.HIGH)
            except ValueError:
                logger.warning(
                    "invalid_dependency_criticality",
                    criticality=dep_data["criticality"],
                    dependency=name,
                )

        # Fallback declaration
        fallback = None
        fallback_data = dep_data.get("fallback")
        if fallback_data:
            fallback = FallbackDeclaration(
                kind=fallback_data.get("type", "none"),
                description=fallback_data.get("description"),
            )

        dependencies.append(Dependency(
            name=name,
            type=dep_type,
            critical=critical,
            criticality=criticality,
            slo=slo,
            service_ref=service_ref,
            contract_ref=dep_data.get("contract_ref"),
            fallback=fallback,
        ))

    return dependencies


# =============================================================================
# Instrumentation Parsing
# =============================================================================


def _parse_instrumentation(instr_data: dict[str, Any] | None) -> Instrumentation | None:
    """Parse v2 instrumentation requirements from §8."""
    if not instr_data:
        return None

    required_metrics = []
    for m in instr_data.get("required_metrics", []):
        if isinstance(m, dict):
            required_metrics.append(RequiredMetric(
                name=m.get("name", ""),
                metric_type=m.get("type", "counter"),
                labels=m.get("labels", []),
            ))

    required_traces = []
    for t in instr_data.get("required_traces", []):
        if isinstance(t, dict):
            required_traces.append(RequiredTrace(
                span_name=t.get("span_name", ""),
                required_attributes=t.get("required_attributes", []),
            ))

    required_logs = []
    for lg in instr_data.get("required_logs", []):
        if isinstance(lg, dict):
            required_logs.append(RequiredLog(
                format=lg.get("format", "structured"),
                required_fields=lg.get("required_fields", []),
            ))

    required_events = []
    for e in instr_data.get("required_events", []):
        if isinstance(e, dict):
            required_events.append(RequiredEvent(
                event_type=e.get("type", ""),
                schema_ref=e.get("schema_ref"),
            ))

    return Instrumentation(
        required_metrics=required_metrics,
        required_traces=required_traces,
        required_logs=required_logs,
        required_events=required_events,
    )


# =============================================================================
# Outcomes Parsing (Missing Capabilities § 1)
# =============================================================================


def _parse_outcomes(outcomes_data: dict[str, Any] | None) -> Outcomes | None:
    # Distinguish absent ("outcomes:" key not present) from empty
    # ("outcomes: {}", which is malformed since decision_value is
    # required) — the latter must surface as a parse error.
    if outcomes_data is None:
        return None
    if not isinstance(outcomes_data, dict):
        raise OpenSRMV2ParseError(
            "spec.outcomes must be a mapping if declared"
        )

    dv_data = outcomes_data.get("decision_value")
    if not dv_data:
        raise OpenSRMV2ParseError(
            "spec.outcomes.decision_value is required when spec.outcomes is declared"
        )

    try:
        decision_value = DecisionValue(
            correct=dv_data["correct"],
            currency=dv_data["currency"],
            false_positive=_parse_failure_cost(dv_data.get("false_positive")),
            false_negative=_parse_failure_cost(dv_data.get("false_negative")),
        )
    except KeyError as e:
        raise OpenSRMV2ParseError(
            f"spec.outcomes.decision_value missing required field: {e.args[0]}"
        ) from e
    except ValueError as e:
        raise OpenSRMV2ParseError(f"spec.outcomes.decision_value: {e}") from e

    revenue_data = outcomes_data.get("revenue")
    revenue: RevenueAttribution | None = None
    if revenue_data is not None:
        try:
            revenue = RevenueAttribution(
                attribution=revenue_data.get("attribution"),
                signal=revenue_data.get("signal"),
            )
        except ValueError as e:
            raise OpenSRMV2ParseError(f"spec.outcomes.revenue: {e}") from e

    volume_data = outcomes_data.get("volume")
    volume: VolumeEstimate | None = None
    if volume_data is not None:
        # Explicit None handling — `null` in YAML lands as Python None
        # and must not slip past the dataclass numeric guard as a TypeError.
        peak = volume_data.get("peak_multiplier")
        if peak is None:
            peak = 1.0
        try:
            volume = VolumeEstimate(
                estimated_daily_decisions=volume_data.get("estimated_daily_decisions"),
                peak_multiplier=peak,
            )
        except ValueError as e:
            raise OpenSRMV2ParseError(f"spec.outcomes.volume: {e}") from e

    try:
        return Outcomes(
            decision_value=decision_value, revenue=revenue, volume=volume,
        )
    except ValueError as e:
        raise OpenSRMV2ParseError(f"spec.outcomes: {e}") from e


def _parse_failure_cost(data: dict[str, Any] | None) -> FailureCost | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise OpenSRMV2ParseError(
            "failure cost must be a mapping (keys: cost, category)"
        )
    try:
        return FailureCost(
            cost=data["cost"],
            category=data.get("category"),
        )
    except KeyError as e:
        raise OpenSRMV2ParseError(
            f"failure cost missing required field: {e.args[0]}"
        ) from e
    except ValueError as e:
        raise OpenSRMV2ParseError(f"failure cost: {e}") from e


# =============================================================================
# Template Resolution
# =============================================================================


def resolve_v2_template(
    manifest_data: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    """Resolve v2 template inheritance (§9).

    Templates are resolved at parse time. Missing template reference is
    a parse error. Templates may not extend other templates (one level).
    """
    spec = manifest_data.get("spec", {})
    template_block = spec.get("template", {})

    if not template_block:
        return manifest_data

    extends = template_block.get("extends")
    if not extends:
        return manifest_data

    # Resolve Backstage-style template ref: "template:namespace/name"
    if ":" in extends:
        _kind, rest = extends.split(":", 1)
        if "/" in rest:
            _namespace, template_name = rest.rsplit("/", 1)
        else:
            template_name = rest
    else:
        template_name = extends

    # Find template file
    # Search: templates/ directory, then .nthlayer/templates/
    template_file = None
    for search_dir in [base_dir / "templates", base_dir / ".nthlayer" / "templates", base_dir]:
        for ext in (".yaml", ".yml"):
            candidate = search_dir / f"{template_name}{ext}"
            if candidate.exists():
                template_file = candidate
                break
        if template_file:
            break

    if template_file is None:
        raise OpenSRMV2ParseError(
            f"Template '{extends}' not found. Searched for '{template_name}.yaml' "
            f"in {base_dir}/templates/ and {base_dir}/.nthlayer/templates/"
        )

    try:
        with open(template_file) as f:
            template_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise OpenSRMV2ParseError(
            f"Invalid YAML in template '{template_file}': {e}"
        ) from e

    if not isinstance(template_data, dict):
        raise OpenSRMV2ParseError(
            f"Template '{template_file}' is not a valid YAML object"
        )

    # Validate: template kind must be ServiceManifestTemplate
    if template_data.get("kind") != "ServiceManifestTemplate":
        raise OpenSRMV2ParseError(
            f"Template '{template_name}' has kind '{template_data.get('kind')}', "
            f"expected 'ServiceManifestTemplate'"
        )

    # No-chaining: templates cannot extend other templates
    template_spec = template_data.get("spec", {})
    if template_spec.get("template"):
        raise OpenSRMV2ParseError(
            f"Template '{template_name}' itself extends another template. "
            f"Template chaining is not allowed (one level of inheritance only)."
        )

    # `type` is not inheritable. schema.json's TemplateServiceIdentity sets
    # `"type": false` so that "a template's type can never contradict the
    # type of a manifest extending it" (opensrm-6w9d).
    #
    # This must be enforced BEFORE the merge, not after: _merge_template_spec
    # deep-merges the template's service dict under the manifest's, so a
    # template-supplied type becomes indistinguishable from a declared one.
    # Left unchecked, a template typed ai-gate plus an extending manifest
    # with no type parsed cleanly and manufactured an ai-gate identity —
    # unlocking is_ai_gate() and the judgment codepaths from a document pair
    # validate.sh rejects on both sides.
    template_service = template_spec.get("service")
    if isinstance(template_service, dict) and "type" in template_service:
        raise OpenSRMV2ParseError(
            f"Template '{template_name}' declares spec.service.type "
            f"('{template_service['type']}'). A template must not declare a "
            f"service type — it is not inheritable, so that a template can "
            f"never contradict the type of a manifest extending it. Declare "
            f"the type on each extending manifest instead."
        )

    # Apply overrides
    overrides = template_block.get("overrides", {})

    # Same rule via the overrides door: an override may replace `service`
    # wholesale, which would otherwise reintroduce a template-side type.
    override_service = overrides.get("service")
    if isinstance(override_service, dict):
        override_value = override_service.get("replace", override_service)
        if isinstance(override_value, dict) and "type" in override_value:
            raise OpenSRMV2ParseError(
                f"Template override in manifest extending '{template_name}' "
                f"sets spec.service.type ('{override_value['type']}') via "
                f"spec.template.overrides. The service type is not "
                f"inheritable and must be declared on the manifest itself."
            )

    merged_spec = _merge_template_spec(template_spec, spec, overrides)

    result = dict(manifest_data)
    result["spec"] = merged_spec

    return result


def _merge_template_spec(
    template_spec: dict[str, Any],
    manifest_spec: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge template spec with manifest spec using override directives.

    Override directives:
      - append: add items to a list
      - replace: replace a section entirely
      - (default): manifest values override template values
    """
    result = dict(template_spec)

    # Apply manifest's own spec values (override template)
    for key, value in manifest_spec.items():
        if key == "template":
            continue  # Skip template block itself

        if key in result and isinstance(result[key], list) and isinstance(value, list):
            # Lists: manifest replaces template by default
            result[key] = value
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Dicts: deep merge
            merged = dict(result[key])
            merged.update(value)
            result[key] = merged
        else:
            result[key] = value

    # Apply explicit overrides
    for key, directive in overrides.items():
        if not isinstance(directive, dict):
            continue

        if "append" in directive:
            existing = result.get(key, [])
            if isinstance(existing, list):
                append_items = directive["append"]
                if isinstance(append_items, list):
                    result[key] = existing + append_items

        if "replace" in directive:
            result[key] = directive["replace"]

    return result
