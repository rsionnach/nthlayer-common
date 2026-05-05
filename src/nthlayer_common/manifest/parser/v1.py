"""
OpenSRM v1 format parser (apiVersion: srm/v1).

Parses Service Reliability Manifest files and produces a ReliabilityManifest.
Judgment SLOs are detected at parse time and populated with v1_compat defaults
for v2-only fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import yaml

from nthlayer_common.manifest.models import (
    JUDGMENT_SLO_TYPES,
    AuditConfig,
    BudgetPolicy,
    BudgetThresholds,
    Dependency,
    DependencyCriticality,
    DependencySLO,
    DeploymentConfig,
    DeploymentGates,
    ErrorBudgetGate,
    Instrumentation,
    ManifestEscalationStep,

    OnCallConfig,
    Override,
    Ownership,
    PagerDutyConfig,
    RecentIncidentsGate,
    ReliabilityManifest,
    RollbackConfig,
    RosterMember,
    RotationConfig,
    SLOComplianceGate,
    SLODefinition,
    SourceFormat,
    TelemetryEvent,
)
from nthlayer_common.manifest.parser._shared import parse_observability
from nthlayer_common.manifest.v1_compat import (
    convert_v1_contract,
    default_measurement,
    default_statistical_requirements,
)

logger = structlog.get_logger()


class OpenSRMParseError(Exception):
    """Error parsing OpenSRM v1 manifest."""


def is_srm_v1_format(data: dict[str, Any]) -> bool:
    """Check if data is in srm/v1 format."""
    return data.get("apiVersion") == "srm/v1" and data.get("kind") == "ServiceReliabilityManifest"


def parse_srm_v1(
    data: dict[str, Any],
    source_file: str | None = None,
) -> ReliabilityManifest:
    """Parse srm/v1 format data into a ReliabilityManifest."""
    if not is_srm_v1_format(data):
        raise OpenSRMParseError(
            "Invalid format. Expected apiVersion: srm/v1 and "
            "kind: ServiceReliabilityManifest"
        )

    metadata = data.get("metadata", {})
    spec = data.get("spec", {})

    name = metadata.get("name")
    if not name:
        raise OpenSRMParseError("metadata.name is required")

    team = metadata.get("team")
    if not team:
        raise OpenSRMParseError("metadata.team is required")

    tier = metadata.get("tier")
    if not tier:
        raise OpenSRMParseError("metadata.tier is required")

    service_type = spec.get("type")
    if not service_type:
        raise OpenSRMParseError("spec.type is required")

    slos = _parse_slos(spec.get("slos", {}), service_type)
    dependencies = _parse_dependencies(spec.get("dependencies", []))
    ownership = _parse_ownership(spec.get("ownership"))
    observability = parse_observability(spec.get("observability"))
    deployment = _parse_deployment(spec.get("deployment"))
    instrumentation = _parse_instrumentation(spec.get("instrumentation"))

    # Contract: convert v1 flat contract to single-element contracts list
    contracts = []
    contract_data = spec.get("contract")
    if contract_data:
        contracts.append(
            convert_v1_contract(
                service_name=name,
                availability=contract_data.get("availability"),
                latency=contract_data.get("latency"),
                judgment=contract_data.get("judgment"),
            )
        )

    # Alerting: store raw dict, nthlayer-generate post-processes
    alerting = spec.get("alerting")

    return ReliabilityManifest(
        name=name,
        team=team,
        tier=tier,
        type=service_type,
        description=metadata.get("description"),
        labels=metadata.get("labels", {}),
        annotations=metadata.get("annotations", {}),
        slos=slos,
        dependencies=dependencies,
        ownership=ownership,
        observability=observability,
        deployment=deployment,
        contracts=contracts,
        instrumentation=instrumentation,
        alerting=alerting,
        template=metadata.get("template"),
        source_format=SourceFormat.SRM_V1,
        source_file=source_file,
        raw_data=data,
    )


def parse_srm_v1_file(file_path: str | Path) -> ReliabilityManifest:
    """Parse an srm/v1 manifest file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {file_path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise OpenSRMParseError(f"Invalid YAML in {file_path}: {e}") from e

    if not isinstance(data, dict):
        raise OpenSRMParseError(f"Expected YAML object in {file_path}")

    return parse_srm_v1(data, source_file=str(path))


# =============================================================================
# SLO Type Inference
# =============================================================================

# Inference rules for slo_type from SLO name and indicator shape.
# Parser infers; ambiguous cases raise OpenSRMParseError.
_NAME_TO_TYPE: dict[str, str] = {
    "availability": "availability",
    "error_rate": "error_rate",
    "latency": "latency",
    "latency_p50": "latency",
    "latency_p95": "latency",
    "latency_p99": "latency",
    "throughput": "throughput",
}

# Keywords in SLO names that hint at type
_NAME_KEYWORDS: list[tuple[str, str]] = [
    ("latency", "latency"),
    ("duration", "latency"),
    ("avail", "availability"),
    ("error", "error_rate"),
    ("throughput", "throughput"),
    ("rps", "throughput"),
]


def _extract_indicator_query(config: dict[str, Any]) -> str | None:
    """Read the canonical SLI PromQL from ``indicator.query``.

    Per nthlayer-generate's SCHEMA.md, ``indicator`` is a required object
    with ``type`` and ``query`` fields. SLOs that omit ``indicator``
    entirely (target-only validation SLOs) return ``None`` here and surface
    later as NO_DATA when the SLO collector tries to query them — that's
    correct behaviour, and the parser's job is just to round-trip the YAML
    faithfully.

    Top-level ``query`` (without ``indicator``) was previously read by this
    parser but is not in any spec or example; it is no longer accepted.
    """
    indicator = config.get("indicator")
    if not isinstance(indicator, dict):
        return None
    query = indicator.get("query")
    return query if isinstance(query, str) and query else None


def _infer_slo_type(name: str, config: dict[str, Any], service_type: str) -> str:
    """Infer slo_type from name, config, and service_type.

    Raises OpenSRMParseError if inference is ambiguous.
    """
    # Explicit type in config takes precedence
    explicit = config.get("type")
    if explicit:
        return explicit

    # Check if this is a judgment SLO
    if name in JUDGMENT_SLO_TYPES:
        return "availability"  # judgment SLOs track a rate metric; slo_type describes the indicator shape

    # Exact name match
    if name in _NAME_TO_TYPE:
        return _NAME_TO_TYPE[name]

    # Keyword search in name
    name_lower = name.lower()
    for keyword, slo_type in _NAME_KEYWORDS:
        if keyword in name_lower:
            return slo_type

    # Check the canonical indicator.query for type hints (e.g. histogram_quantile → latency).
    query = _extract_indicator_query(config) or ""
    if "histogram_quantile" in query or "duration" in query or "latency" in query:
        return "latency"
    if "error" in query or "5xx" in query or "status" in query:
        return "error_rate"
    if "total" in query and ("success" in query or "good" in query or "200" in query):
        return "availability"

    # Unit hint
    unit = config.get("unit")
    if unit in ("ms", "s"):
        return "latency"
    if unit == "rps":
        return "throughput"

    # Percentile hint
    if config.get("percentile"):
        return "latency"

    raise OpenSRMParseError(
        f"Cannot infer slo_type for SLO '{name}'. Add an explicit 'type' field "
        f"(availability, latency, error_rate, throughput)."
    )


# =============================================================================
# Internal Parsing Functions
# =============================================================================


def _parse_slos(
    slos_data: dict[str, Any],
    service_type: str,
) -> list[SLODefinition]:
    """Parse SLOs from srm/v1 spec.slos section."""
    slos = []

    for name, config in slos_data.items():
        if not isinstance(config, dict):
            config = {"target": config}

        target = config.get("target")
        if target is None:
            target = config.get("minimum")
        if target is None:
            raise OpenSRMParseError(f"SLO '{name}' requires a target or minimum value")

        slo_type = _infer_slo_type(name, config, service_type)

        # Detect judgment SLOs and populate v2 fields
        judgment_type = None
        measurement = None
        statistical_requirements = None

        if name in JUDGMENT_SLO_TYPES:
            judgment_type = name
            window = config.get("window", "30d")
            measurement = default_measurement(judgment_type, window)
            statistical_requirements = default_statistical_requirements(judgment_type)

        slo = SLODefinition(
            name=name,
            target=float(target),
            slo_type=slo_type,
            window=config.get("window", "30d"),
            unit=config.get("unit"),
            percentile=config.get("percentile"),
            indicator_query=_extract_indicator_query(config),
            description=config.get("description"),
            labels=config.get("labels", {}),
            judgment_type=judgment_type,
            measurement=measurement,
            statistical_requirements=statistical_requirements,
        )
        slos.append(slo)

    return slos


def _parse_dependencies(deps_data: list[dict[str, Any]]) -> list[Dependency]:
    """Parse dependencies from srm/v1 spec.dependencies section."""
    dependencies = []

    for dep_data in deps_data:
        if not isinstance(dep_data, dict):
            continue

        name = dep_data.get("name")
        if not name:
            continue

        slo = None
        if "slo" in dep_data:
            slo_data = dep_data["slo"]
            slo = DependencySLO(
                availability=slo_data.get("availability"),
                latency_p99=slo_data.get("latency_p99") or slo_data.get("latency", {}).get("p99"),
            )
        # v2 top-level fields mapped into DependencySLO
        elif "expected_availability" in dep_data or "expected_latency_p99" in dep_data:
            slo = DependencySLO(
                availability=dep_data.get("expected_availability"),
                latency_p99=dep_data.get("expected_latency_p99"),
            )

        criticality = None
        if "criticality" in dep_data:
            try:
                criticality = DependencyCriticality(dep_data["criticality"])
            except ValueError:
                logger.warning(
                    "invalid_dependency_criticality",
                    criticality=dep_data["criticality"],
                    dependency=name,
                )

        dep = Dependency(
            name=name,
            type=dep_data.get("type", "unknown"),
            critical=dep_data.get("critical", False),
            criticality=criticality,
            slo=slo,
            manifest=dep_data.get("manifest"),
            database_type=dep_data.get("database_type"),
        )
        dependencies.append(dep)

    return dependencies


def _parse_ownership(ownership_data: dict[str, Any] | None) -> Ownership | None:
    """Parse ownership from srm/v1 spec.ownership section."""
    if not ownership_data:
        return None

    team = ownership_data.get("team")
    if not team:
        return None

    pagerduty = None
    if "pagerduty" in ownership_data:
        pd_data = ownership_data["pagerduty"]
        pagerduty = PagerDutyConfig(
            service_id=pd_data.get("service_id"),
            escalation_policy_id=pd_data.get("escalation_policy_id"),
        )

    oncall = _parse_oncall(ownership_data.get("oncall"))

    return Ownership(
        team=team,
        slack=ownership_data.get("slack"),
        email=ownership_data.get("email"),
        escalation=ownership_data.get("escalation"),
        pagerduty=pagerduty,
        runbook=ownership_data.get("runbook"),
        documentation=ownership_data.get("documentation"),
        oncall=oncall,
    )


def _parse_oncall(oncall_data: dict[str, Any] | None) -> OnCallConfig | None:
    """Parse on-call configuration from srm/v1 spec.ownership.oncall section."""
    if not oncall_data:
        return None

    rotation_data = oncall_data.get("rotation", {})

    roster = []
    for i, raw_member in enumerate(rotation_data.get("roster", [])):
        try:
            roster.append(
                RosterMember(
                    name=raw_member["name"],
                    slack_id=raw_member["slack_id"],
                    ntfy_topic=raw_member.get("ntfy_topic"),
                    phone=raw_member.get("phone"),
                )
            )
        except KeyError as e:
            raise OpenSRMParseError(
                f"oncall.rotation.roster[{i}] missing required field: {e}"
            ) from e

    overrides = []
    for i, raw_override in enumerate(oncall_data.get("overrides", [])):
        try:
            overrides.append(
                Override(
                    start=raw_override["start"],
                    end=raw_override["end"],
                    user=raw_override["user"],
                    reason=raw_override.get("reason"),
                )
            )
        except KeyError as e:
            raise OpenSRMParseError(
                f"oncall.overrides[{i}] missing required field: {e}"
            ) from e

    escalation = []
    for i, raw_step in enumerate(oncall_data.get("escalation", [])):
        try:
            escalation.append(
                ManifestEscalationStep(
                    after=raw_step["after"],
                    notify=raw_step["notify"],
                    target=raw_step.get("target"),
                    phone=raw_step.get("phone"),
                )
            )
        except KeyError as e:
            raise OpenSRMParseError(
                f"oncall.escalation[{i}] missing required field: {e}"
            ) from e

    tz_name = oncall_data.get("timezone")
    if not tz_name:
        raise OpenSRMParseError("oncall.timezone is required")
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(tz_name)
    except (KeyError, ValueError, OSError):
        raise OpenSRMParseError(
            f"oncall.timezone: invalid IANA timezone: {tz_name!r}"
        ) from None

    return OnCallConfig(
        timezone=tz_name,
        rotation=RotationConfig(
            type=rotation_data.get("type", "weekly"),
            handoff=rotation_data.get("handoff", "monday 09:00"),
            roster=roster,
        ),
        overrides=overrides,
        escalation=escalation,
    )



def _parse_deployment(deploy_data: dict[str, Any] | None) -> DeploymentConfig | None:
    """Parse deployment from srm/v1 spec.deployment section."""
    if not deploy_data:
        return None

    gates = None
    if "gates" in deploy_data:
        gates_data = deploy_data["gates"]
        gates = DeploymentGates(
            error_budget=_parse_error_budget_gate(gates_data.get("error_budget")),
            slo_compliance=_parse_slo_compliance_gate(gates_data.get("slo_compliance")),
            recent_incidents=_parse_recent_incidents_gate(gates_data.get("recent_incidents")),
        )

    rollback = None
    if "rollback" in deploy_data:
        rb_data = deploy_data["rollback"]
        rollback = RollbackConfig(
            automatic=rb_data.get("automatic", False),
            error_rate_increase=rb_data.get("criteria", {}).get("error_rate_increase"),
            latency_increase=rb_data.get("criteria", {}).get("latency_increase"),
        )

    audit = None
    if "audit" in deploy_data:
        audit_data = deploy_data["audit"]
        audit = AuditConfig(
            enabled=audit_data.get("enabled", True),
            retention_days=audit_data.get("retention_days", 90),
        )

    return DeploymentConfig(
        environments=deploy_data.get("environments", []),
        gates=gates,
        rollback=rollback,
        audit=audit,
    )


def _parse_budget_policy(data: dict[str, Any]) -> BudgetPolicy | None:
    """Parse budget policy from error_budget gate config."""
    policy_data = data.get("policy")
    if not policy_data or not isinstance(policy_data, dict):
        return None

    thresholds = BudgetThresholds()
    thresh_data = policy_data.get("thresholds")
    if thresh_data and isinstance(thresh_data, dict):
        thresholds = BudgetThresholds(
            warning=float(thresh_data.get("warning", 0.20)),
            critical=float(thresh_data.get("critical", 0.10)),
        )

    on_exhausted = policy_data.get("on_exhausted", [])
    if not isinstance(on_exhausted, list):
        on_exhausted = []

    policy = BudgetPolicy(
        window=policy_data.get("window", "30d"),
        thresholds=thresholds,
        on_exhausted=on_exhausted,
    )
    policy.validate()
    return policy


def _parse_error_budget_gate(gate_data: dict[str, Any] | None) -> ErrorBudgetGate | None:
    if not gate_data:
        return None
    return ErrorBudgetGate(
        enabled=gate_data.get("enabled", True),
        threshold=gate_data.get("threshold"),
        policy=_parse_budget_policy(gate_data),
    )


def _parse_slo_compliance_gate(gate_data: dict[str, Any] | None) -> SLOComplianceGate | None:
    if not gate_data:
        return None
    return SLOComplianceGate(threshold=gate_data.get("threshold", 0.99))


def _parse_recent_incidents_gate(gate_data: dict[str, Any] | None) -> RecentIncidentsGate | None:
    if not gate_data:
        return None
    return RecentIncidentsGate(
        p1_max=gate_data.get("p1_max", 0),
        p2_max=gate_data.get("p2_max", 2),
        lookback=gate_data.get("lookback", "7d"),
    )


def _parse_instrumentation(instr_data: dict[str, Any] | None) -> Instrumentation | None:
    """Parse instrumentation from srm/v1 spec.instrumentation section."""
    if not instr_data:
        return None

    events = []
    for event_data in instr_data.get("telemetry_events", []):
        if isinstance(event_data, dict):
            events.append(
                TelemetryEvent(
                    name=event_data.get("name", ""),
                    fields=event_data.get("fields", []),
                )
            )

    return Instrumentation(
        telemetry_events=events,
        feedback_loop=instr_data.get("feedback_loop"),
        ground_truth_source=instr_data.get("ground_truth_source"),
    )


# =============================================================================
# Template Resolution
# =============================================================================


def resolve_template(
    manifest_data: dict[str, Any],
    template_dir: str | Path | None,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve srm/v1 template inheritance.

    Returns (resolved_data, warnings). Missing template is a warning for v1
    (backward compat); v2 makes it a parse error.
    """
    warnings: list[str] = []
    metadata = manifest_data.get("metadata", {})
    template_name = metadata.get("template")

    if not template_name:
        return manifest_data, warnings

    if template_dir is None:
        warnings.append(f"Template '{template_name}' specified but no template directory found")
        return manifest_data, warnings

    template_dir = Path(template_dir)

    template_file = template_dir / f"{template_name}.yaml"
    if not template_file.exists():
        template_file = template_dir / f"{template_name}.yml"
    if not template_file.exists():
        warnings.append(f"Template '{template_name}' not found in {template_dir}")
        return manifest_data, warnings

    try:
        with open(template_file) as f:
            template_data = yaml.safe_load(f)
    except Exception as e:
        warnings.append(f"Failed to load template '{template_name}': {e}")
        return manifest_data, warnings

    if not isinstance(template_data, dict):
        warnings.append(f"Template '{template_name}' is not a valid YAML object")
        return manifest_data, warnings

    # No-chaining rule
    template_metadata = template_data.get("metadata", {})
    if template_metadata.get("template"):
        warnings.append(
            f"Template '{template_name}' itself references template "
            f"'{template_metadata['template']}' (chaining not allowed)"
        )
        return manifest_data, warnings

    template_spec = template_data.get("spec", {})
    manifest_spec = manifest_data.get("spec", {})
    merged_spec = _deep_merge_spec(template_spec, manifest_spec)

    result = dict(manifest_data)
    result["spec"] = merged_spec

    return result, warnings


def _deep_merge_spec(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two spec dicts with override-wins semantics."""
    result = dict(base)

    for key, value in override.items():
        if key == "slos":
            base_slos = dict(result.get("slos", {}))
            if isinstance(value, dict):
                for slo_name, slo_def in value.items():
                    base_slos[slo_name] = slo_def
            result["slos"] = base_slos
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_spec(result[key], value)
        else:
            result[key] = value

    return result
