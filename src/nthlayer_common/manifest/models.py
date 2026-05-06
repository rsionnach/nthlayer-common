"""
Unified Reliability Manifest model for OpenSRM v1, v2, and legacy formats.

This module provides the canonical internal representation that all parsers
(srm/v1, opensrm.nthlayer.io/v2, legacy) normalise to. All consumers — core,
workers, bench, nthlayer-generate — import these types.

Design principles:
  - One unified model. Parsers translate format differences; consumers see
    uniform data.
  - v2-only fields are optional with defaults. v1 parser populates sensible
    defaults via v1_compat so consumers never see None for fields that have
    a reasonable default.
  - slo_type is required on SLODefinition. Parsers infer from indicator shape
    and fail loudly if ambiguous.
  - judgment_type discriminates judgment SLOs (not name-based fallback).

Tech debt (v2, not v1.5):
  - JudgmentMeasurement's union-of-fields could become a discriminated union
    per judgment type
  - total_query + good_query + indicator_query could become a discriminated
    SLIIndicator union
  - Classical SLO breach_actions semantic (currently judgment-only; could
    extend for consistent operator notification)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# =============================================================================
# Constants
# =============================================================================

VALID_TIERS = {
    "critical",
    "high",
    "standard",
    "low",
}

VALID_SERVICE_TYPES = {
    "api",
    "worker",
    "stream",
    "ai-gate",
    "batch",
    "database",
    "web",
}

SERVICE_TYPE_ALIASES = {
    "background-job": "worker",
    "pipeline": "batch",
}

# 8 standard judgment SLO types (v2 spec §5.2)
JUDGMENT_SLO_TYPES = {
    "reversal_rate",
    "high_confidence_failure",
    "audit_sampling",
    "outcomes",
    "escalation",
    "segments",
    "stability",
    "calibration",
}

# Standard (classical) SLO types
STANDARD_SLO_TYPES = {
    "availability",
    "latency",
    "error_rate",
    "throughput",
}


# =============================================================================
# Enums
# =============================================================================


class SourceFormat(str, Enum):
    """Source format of the manifest file.

    SRM_V1:     apiVersion: srm/v1, kind: ServiceReliabilityManifest
    OPENSRM_V2: apiVersion: opensrm.nthlayer.io/v2, kind: ServiceManifest
    LEGACY:     service: + resources: (NthLayer legacy format)
    """

    SRM_V1 = "srm_v1"
    OPENSRM = "srm_v1"  # Alias — v1 was historically called "OPENSRM"
    OPENSRM_V2 = "opensrm_v2"
    LEGACY = "legacy"


class DependencyCriticality(str, Enum):
    """Criticality level for dependencies."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Ownership Models
# =============================================================================


@dataclass
class PagerDutyConfig:
    """PagerDuty integration configuration."""

    service_id: str | None = None
    escalation_policy_id: str | None = None


@dataclass
class RosterMember:
    """A person in the on-call rotation roster."""

    name: str
    slack_id: str
    ntfy_topic: str | None = None
    phone: str | None = None


@dataclass
class Override:
    """Manual on-call override (holidays, swaps)."""

    start: str  # ISO 8601
    end: str  # ISO 8601
    user: str
    reason: str | None = None


@dataclass
class ManifestEscalationStep:
    """A single step in the escalation policy (build-time schema).

    Prefixed with 'Manifest' to distinguish from the runtime
    EscalationStep in nthlayer_workers (Phase 3).
    """

    after: str  # e.g. "0m", "5m", "10m"
    notify: str  # "slack_dm" | "ntfy" | "slack_channel" | "phone" | "pagerduty"
    target: str | None = None
    phone: str | None = None


@dataclass
class RotationConfig:
    """On-call rotation schedule configuration."""

    type: str  # "weekly" | "daily"
    handoff: str  # e.g. "monday 09:00" or "09:00"
    roster: list[RosterMember] = field(default_factory=list)


@dataclass
class OnCallConfig:
    """On-call configuration within spec.ownership."""

    timezone: str  # IANA timezone string
    rotation: RotationConfig
    overrides: list[Override] = field(default_factory=list)
    escalation: list[ManifestEscalationStep] = field(default_factory=list)


@dataclass
class Ownership:
    """Service ownership information.

    When group_ref is set (v2), team is populated from the resolved
    Backstage entity reference at parse time. Ref resolution failure
    is a parse error — no silent fallback.
    """

    team: str
    slack: str | None = None
    email: str | None = None
    escalation: str | None = None
    pagerduty: PagerDutyConfig | None = None
    runbook: str | None = None
    documentation: str | None = None
    oncall: OnCallConfig | None = None

    # v2: Backstage entity references
    group_ref: str | None = None
    escalation_ref: str | None = None
    technical_contact_ref: str | None = None


# =============================================================================
# SLO Models
# =============================================================================


@dataclass
class ProbeConfig:
    """Stability judgment SLO: frozen input set for drift measurement."""

    frozen_input_set: str  # URI to probe data
    frequency: str = "daily"


@dataclass
class StratifiedSample:
    """A single segment in audit sampling stratification."""

    segment: str
    rate: float


@dataclass
class SamplingConfig:
    """Audit sampling judgment SLO: sampling rates and stratification."""

    overall_rate: float
    stratified: list[StratifiedSample] = field(default_factory=list)


@dataclass
class JudgmentMeasurement:
    """Measurement configuration for judgment SLOs.

    Tech debt (v2): this union-of-fields shape could become a discriminated
    union per judgment_type. For v1.5, the flat shape is simpler.
    """

    source: str | None = None  # lineage, calibration_sample, downstream_signal
    confidence_threshold: float | None = None
    bins: int | None = None
    method: str | None = None  # brier_score, expected_calibration_error
    window: str | None = None  # overrides SLO-level window for measurement
    frequency: str | None = None  # daily, hourly
    probe: ProbeConfig | None = None  # stability: frozen input set
    sampling: SamplingConfig | None = None  # audit_sampling
    segment_dimension: str | None = None  # segments: dimension to slice on
    segment_values: list[str] = field(default_factory=list)


@dataclass
class BreachAction:
    """Structured breach action declaration (v2 spec §5.4).

    action_type: notify | create_case | reduce_autonomy | action_request
    target: recipient entity ref or agent_id
    parameters: type-specific config (e.g. priority for create_case)
    """

    action_type: str
    target: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalRequirements:
    """Statistical method requirements for judgment SLO evaluation.

    v1 parser populates defaults: 95% confidence intervals, method
    inferred from judgment_type (documented in v1_compat).
    """

    confidence_interval_pct: float = 95.0
    method: str | None = None  # brier_score, expected_calibration_error, hypothesis_test
    minimum_sample_size: int | None = None


@dataclass
class SLODefinition:
    """SLO definition within a reliability manifest.

    Supports both classical SLOs (availability, latency, error_rate,
    throughput) and judgment SLOs for ai-gate services.

    slo_type is required — parser infers from indicator shape and fails
    loudly if ambiguous. "judgment" is NOT a valid slo_type; use
    judgment_type to discriminate judgment SLOs.

    Judgment SLOs use slo_type="availability" because their indicator
    is a rate metric (reversal rate, failure rate, etc.). The slo_type
    describes the indicator shape, not the SLO category. judgment_type
    is what discriminates them from classical SLOs.

    ``target`` convention (KNOWN DIVERGENCE — opensrm-pa2w):

    The ``target`` field is consumed by three subsystems with incompatible
    arithmetic for the same value. Until opensrm-pa2w.followup unifies them:

    - Classical SLOs (slo_type in {availability, latency, error_rate,
      throughput} without ``judgment_type``) are consumed by ``observe``
      which uses 0-100 percentage convention. Example: target=99.9 for
      99.9% availability.
    - Judgment SLOs (``judgment_type`` set, consumed by ``measure``) use
      0.0-1.0 ratio convention. Example: target=0.985 for ≤1.5% reversal
      rate threshold.
    - The unrelated ``slo_models.SLO`` dataclass (OpenSLO-based, used by
      a different code path) is ratio.

    A load-time warning (TargetConventionWarning, see
    nthlayer_common.manifest.target_validation) flags likely mismatches —
    classical SLOs with target<1.0, judgment SLOs with target>1.0.

    Tech debt (v2): total_query + good_query + indicator_query could
    become a discriminated SLIIndicator union.
    """

    name: str
    target: float
    slo_type: str  # availability, latency, error_rate, throughput (required)
    window: str = "30d"

    # Type-specific fields
    unit: str | None = None  # For latency: ms, s; for throughput: rps
    percentile: str | None = None  # For latency: p50, p95, p99

    # Indicator queries
    indicator_query: str | None = None  # Single PromQL or composed ratio
    total_query: str | None = None  # OpenSLO ratio metric: total source
    good_query: str | None = None  # OpenSLO ratio metric: good source

    # Metadata
    description: str | None = None
    labels: dict[str, str] = field(default_factory=dict)

    # Provenance — retained after $ref resolution for traceability
    source_ref: str | None = None  # Original $ref path (OpenSLO)

    # Judgment SLO fields (None for classical SLOs)
    judgment_type: str | None = None  # one of 8 JUDGMENT_SLO_TYPES
    measurement: JudgmentMeasurement | None = None
    breach_actions: list[BreachAction] = field(default_factory=list)
    statistical_requirements: StatisticalRequirements | None = None

    def is_judgment_slo(self) -> bool:
        """Check if this is a judgment SLO. Only checks judgment_type."""
        return self.judgment_type is not None


# =============================================================================
# Contract Models
# =============================================================================


@dataclass
class JudgmentPromise:
    """A single judgment-type promise within a reliability contract."""

    judgment_type: str  # one of JUDGMENT_SLO_TYPES
    threshold: float
    direction: Literal["above", "below"]  # "above" = value must stay above threshold


@dataclass
class ContractPromise:
    """What a reliability contract promises to callers."""

    availability: float | None = None
    latency_p99: str | None = None
    throughput: str | None = None
    judgment: list[JudgmentPromise] = field(default_factory=list)


@dataclass
class APIRef:
    """Reference to an API specification document."""

    openapi: str | None = None
    asyncapi: str | None = None
    operation_id: str | None = None


@dataclass
class BreachSemantics:
    """What happens when a reliability contract breaks."""

    consumer_notification: str | None = None
    compensation: str | None = None
    sla_credits_enabled: bool = False
    sla_credits_ref: str | None = None


@dataclass
class ReliabilityContract:
    """A first-class cross-service reliability promise (v2 spec §6).

    v1 manifests produce single-element contracts lists with name derived
    from service name and promise populated from the flat contract fields.
    """

    name: str
    promise: ContractPromise
    api_ref: APIRef | None = None
    conditions: list[str] = field(default_factory=list)
    breach_semantics: BreachSemantics | None = None
    contract_owner: str | None = None


# =============================================================================
# Dependency Models
# =============================================================================


@dataclass
class DependencySLO:
    """Expected SLO from a dependency.

    Canonical location for expected guarantees. v2 top-level YAML fields
    (expected_availability, expected_latency_p99) are mapped here during
    parsing — internal model has only one representation.
    """

    availability: float | None = None
    latency_p99: str | None = None
    throughput: str | None = None


@dataclass
class FallbackDeclaration:
    """What the service does when a dependency fails (v2 spec §7.3)."""

    kind: str  # graceful_degradation, circuit_breaker, failover, none
    description: str | None = None


@dataclass
class Dependency:
    """Service dependency definition."""

    name: str
    type: str  # database, api, queue, cache, ai-gate, etc.
    critical: bool = False
    criticality: DependencyCriticality | None = None
    slo: DependencySLO | None = None
    manifest: str | None = None  # URL to dependency's manifest

    # Database-specific
    database_type: str | None = None

    # v2 fields
    service_ref: str | None = None  # Backstage entity ref
    contract_ref: str | None = None  # Reference to dependency's contract
    fallback: FallbackDeclaration | None = None


# =============================================================================
# Observability Models
# =============================================================================


@dataclass
class Observability:
    """Observability configuration."""

    metrics_prefix: str | None = None
    logs_label: str | None = None
    traces_service: str | None = None
    prometheus_job: str | None = None
    grafana_url: str | None = None
    labels: dict[str, str] = field(default_factory=dict)


# =============================================================================
# Deployment Models
# =============================================================================

VALID_EXHAUSTION_BEHAVIORS = {"freeze_deploys", "require_approval", "notify"}


@dataclass
class BudgetThresholds:
    """Warning and critical thresholds for error budget policy.

    Values are fractions of remaining budget (e.g., 0.20 = 20% remaining).
    """

    warning: float = 0.20
    critical: float = 0.10


@dataclass
class BudgetPolicy:
    """Error budget policy configuration."""

    window: str = "30d"
    thresholds: BudgetThresholds = field(default_factory=BudgetThresholds)
    on_exhausted: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate policy configuration."""
        for behavior in self.on_exhausted:
            if behavior not in VALID_EXHAUSTION_BEHAVIORS:
                msg = (
                    f"Invalid on_exhausted behavior: {behavior}. "
                    f"Valid values: {', '.join(sorted(VALID_EXHAUSTION_BEHAVIORS))}"
                )
                raise ValueError(msg)
        if self.thresholds.warning < self.thresholds.critical:
            msg = (
                f"warning threshold ({self.thresholds.warning}) must be >= "
                f"critical threshold ({self.thresholds.critical})"
            )
            raise ValueError(msg)


@dataclass
class ErrorBudgetGate:
    """Error budget gate configuration."""

    enabled: bool = True
    threshold: float | None = None
    policy: BudgetPolicy | None = None


@dataclass
class SLOComplianceGate:
    """SLO compliance gate configuration."""

    threshold: float = 0.99


@dataclass
class RecentIncidentsGate:
    """Recent incidents gate configuration."""

    p1_max: int = 0
    p2_max: int = 2
    lookback: str = "7d"


@dataclass
class DeploymentGates:
    """Deployment gate configuration."""

    error_budget: ErrorBudgetGate | None = None
    slo_compliance: SLOComplianceGate | None = None
    recent_incidents: RecentIncidentsGate | None = None


@dataclass
class RollbackConfig:
    """Automatic rollback configuration."""

    automatic: bool = False
    error_rate_increase: str | None = None
    latency_increase: str | None = None


@dataclass
class AuditConfig:
    """Audit logging configuration."""

    enabled: bool = True
    retention_days: int = 90


@dataclass
class DeploymentConfig:
    """Deployment configuration."""

    environments: list[str] = field(default_factory=list)
    gates: DeploymentGates | None = None
    rollback: RollbackConfig | None = None
    audit: AuditConfig | None = None


# =============================================================================
# Instrumentation Models
# =============================================================================


@dataclass
class TelemetryEvent:
    """AI gate telemetry event configuration (v1)."""

    name: str
    fields: list[str] = field(default_factory=list)


@dataclass
class RequiredMetric:
    """Required Prometheus metric (v2 spec §8)."""

    name: str
    metric_type: str  # counter, histogram, gauge, summary
    labels: list[str] = field(default_factory=list)


@dataclass
class RequiredTrace:
    """Required trace span (v2 spec §8)."""

    span_name: str
    required_attributes: list[str] = field(default_factory=list)


@dataclass
class RequiredLog:
    """Required log format (v2 spec §8)."""

    format: str = "structured"
    required_fields: list[str] = field(default_factory=list)


@dataclass
class RequiredEvent:
    """Required telemetry event (v2 spec §8)."""

    event_type: str  # e.g. "io.nthlayer.decision.v1"
    schema_ref: str | None = None


@dataclass
class Instrumentation:
    """Instrumentation requirements.

    v1 fields (telemetry_events, feedback_loop, ground_truth_source) are
    preserved for backward compatibility. v2 adds structured requirements
    for metrics, traces, logs, and events.
    """

    # v1 fields
    telemetry_events: list[TelemetryEvent] = field(default_factory=list)
    feedback_loop: str | None = None
    ground_truth_source: str | None = None

    # v2 fields
    required_metrics: list[RequiredMetric] = field(default_factory=list)
    required_traces: list[RequiredTrace] = field(default_factory=list)
    required_logs: list[RequiredLog] = field(default_factory=list)
    required_events: list[RequiredEvent] = field(default_factory=list)


# =============================================================================
# Main Manifest Model
# =============================================================================


@dataclass
class ReliabilityManifest:
    """Unified internal model for OpenSRM v1, v2, and legacy formats.

    This is the canonical representation that all consumers import.
    All parsers normalise to this model.
    """

    # ==========================================================================
    # Required Fields
    # ==========================================================================
    name: str
    team: str
    tier: str  # critical, high, standard, low
    type: str  # api, worker, stream, ai-gate, batch, database, web

    # ==========================================================================
    # Metadata
    # ==========================================================================
    namespace: str = "default"  # Matches Backstage namespace convention
    description: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)

    # ==========================================================================
    # Spec
    # ==========================================================================
    slos: list[SLODefinition] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    ownership: Ownership | None = None
    observability: Observability | None = None
    deployment: DeploymentConfig | None = None
    contracts: list[ReliabilityContract] = field(default_factory=list)
    instrumentation: Instrumentation | None = None

    # Alerting — type is Any because AlertingConfig lives in nthlayer-generate.
    # Common parser stores raw dict; nthlayer-generate post-processes.
    alerting: Any = None

    # ==========================================================================
    # Template Provenance
    # ==========================================================================
    # Templates are resolved at parse time; consumers see the flattened
    # manifest. template_ref retained for provenance only. Missing template
    # reference is a parse error.
    template_ref: str | None = None

    # ==========================================================================
    # Legacy Fields (backward compat)
    # ==========================================================================
    language: str | None = None
    framework: str | None = None
    template: str | None = None  # v1 legacy template name
    support_model: str = "self"
    environment: str | None = None

    # ==========================================================================
    # Source Tracking
    # ==========================================================================
    source_format: SourceFormat = SourceFormat.SRM_V1
    source_file: str | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalise the manifest."""
        if self.type in SERVICE_TYPE_ALIASES:
            self.type = SERVICE_TYPE_ALIASES[self.type]

        if not self.name:
            raise ValueError("Service name is required")
        if not self.team:
            raise ValueError("Service team is required")
        if not self.tier:
            raise ValueError("Service tier is required")
        if not self.type:
            raise ValueError("Service type is required")

        if self.tier not in VALID_TIERS:
            msg = f"Invalid tier '{self.tier}'. Must be one of: {', '.join(sorted(VALID_TIERS))}"
            raise ValueError(msg)

        if self.type not in VALID_SERVICE_TYPES:
            valid = ", ".join(sorted(VALID_SERVICE_TYPES))
            msg = f"Invalid type '{self.type}'. Must be one of: {valid}"
            raise ValueError(msg)

    def is_ai_gate(self) -> bool:
        """Check if this is an AI gate service."""
        return self.type == "ai-gate"

    def get_judgment_slos(self) -> list[SLODefinition]:
        """Get all judgment SLOs."""
        return [s for s in self.slos if s.is_judgment_slo()]

    def get_standard_slos(self) -> list[SLODefinition]:
        """Get all standard (non-judgment) SLOs."""
        return [s for s in self.slos if not s.is_judgment_slo()]

    def validate_contracts(self) -> list[str]:
        """Validate that internal SLOs are tighter than external contracts."""
        errors: list[str] = []

        for contract in self.contracts:
            promise = contract.promise

            if promise.availability:
                avail_slos = [s for s in self.slos if s.name == "availability"]
                for slo in avail_slos:
                    if slo.target < promise.availability * 100:
                        errors.append(
                            f"Availability SLO ({slo.target}%) is looser than "
                            f"contract '{contract.name}' ({promise.availability * 100}%)"
                        )

            for jp in promise.judgment:
                matching_slos = [
                    s for s in self.slos
                    if s.judgment_type == jp.judgment_type
                ]
                for slo in matching_slos:
                    if jp.direction == "below" and slo.target > jp.threshold:
                        errors.append(
                            f"Judgment SLO '{slo.name}' ({slo.target}) is looser than "
                            f"contract '{contract.name}' threshold ({jp.threshold})"
                        )
                    elif jp.direction == "above" and slo.target < jp.threshold:
                        errors.append(
                            f"Judgment SLO '{slo.name}' ({slo.target}) is looser than "
                            f"contract '{contract.name}' threshold ({jp.threshold})"
                        )

        return errors
