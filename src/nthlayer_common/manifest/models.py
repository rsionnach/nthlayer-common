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

import re
from dataclasses import dataclass, field
from enum import StrEnum
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

# The six values OpenSRM v2 defines for spec.service.type (opensrm-6w9d).
# This set is the spec's, not ours — do not add to it. Anything NthLayer
# needs beyond the six goes through the extension branch below.
VALID_SERVICE_TYPES = {
    "api",
    "worker",
    "stream",
    "ai-gate",
    "batch",
    "database",
}

# The spec's extension escape hatch: `^x-[a-z][a-z0-9-]*$`.
#
# Applied with fullmatch(), never search(). Python's `$` also matches just
# before a trailing newline, so `re.search(r"^x-[a-z][a-z0-9-]*$", "x-web\n")`
# succeeds — while check-jsonschema (ECMA-262) rejects the same string.
# spec/v2/validate.sh runs the strict engine, so a laxer check here would
# accept manifests the spec's own gate refuses. fullmatch() closes that gap.
SERVICE_TYPE_EXTENSION_PATTERN = re.compile(r"x-[a-z][a-z0-9-]*")

# Input conveniences with NO standing in the spec. These resolve in
# ReliabilityManifest.__post_init__ *before* validation, so a manifest never
# stores an alias — which is what stops them leaking back out into documents
# that then fail schema validation.
#
# `web` was an nthlayer-common service type pre-opensrm-ih0v, absent from the
# spec's six. It maps to the extension branch rather than collapsing into
# `api` because nthlayer-generate distinguishes the two in its dashboard
# templates, CLI init choices, and docs generation.
SERVICE_TYPE_ALIASES = {
    "background-job": "worker",
    "pipeline": "batch",
    "web": "x-web",
}


def resolve_service_type(value: object) -> str | None:
    """Resolve an authored service type to its canonical form, or None.

    Returns the canonical type — aliases resolved — or None if the value is
    not one nthlayer-common accepts.

    Prefer this over calling SERVICE_TYPE_ALIASES and is_valid_service_type
    yourself. The two must be applied in that order, because an alias is
    deliberately NOT a valid service type: validating first rejects every
    alias, which is exactly the regression opensrm-ih0v's edge-cases pass
    caught after three call sites had each open-coded the sequence. This
    encodes the order instead of describing it in a comment beside each one.

    Returns None rather than raising so each caller keeps its own exception
    type — OpenSRMV2ParseError in the parser, ValueError in v1_compat.

    One deliberate non-user: ReliabilityManifest.__post_init__ keeps the two
    steps separate so its "Service type is required" guard can sit between
    them, since "" resolves to None here and would otherwise be reported as
    an invalid type rather than a missing one.
    """
    if not isinstance(value, str):
        return None

    resolved = SERVICE_TYPE_ALIASES.get(value, value)
    return resolved if is_valid_service_type(resolved) else None


def valid_service_types_phrase() -> str:
    """Return the accepted service types as a phrase, for use after a
    lead-in such as "Must be one of: " or "Set it to one of: ".

    The only place that wording exists. All three author-facing errors — the
    parser's, v1_compat's and ReliabilityManifest's — describe the accepted
    set identically and stay in step with VALID_SERVICE_TYPES, which tracks
    the spec and therefore changes.
    """
    return (
        f"{', '.join(sorted(VALID_SERVICE_TYPES))}; "
        "or an extension type matching 'x-<lowercase-name>'"
    )


def is_valid_service_type(value: object) -> bool:
    """True if ``value`` is a service type schema.json's ServiceType accepts.

    The single source of truth for the rule. Both ReliabilityManifest's own
    validation and v1_compat's upconversion check consult this, so the two
    cannot drift apart — they previously disagreed about the empty string,
    which one path rejected and the other wrote into a v2 document verbatim.

    Note this takes a *resolved* type: aliases are not valid service types,
    they are inputs that must be normalised through SERVICE_TYPE_ALIASES
    first. Callers that accept author input resolve, then validate.
    """
    # Guard the type: values reach here straight from parsed YAML, where
    # `type: 5` yields an int. Letting that hit fullmatch() would raise
    # TypeError, which sails through the `except ValueError` that callers
    # like nthlayer-generate's migrate_manifest use to catch bad manifests.
    if not isinstance(value, str):
        return False

    return bool(
        value in VALID_SERVICE_TYPES or SERVICE_TYPE_EXTENSION_PATTERN.fullmatch(value)
    )


# Decision 3c (opensrm-l33e) splits the judgment vocabulary in two. Both sets
# are pinned against opensrm/spec/v2/schema.json in BOTH directions by
# tests/test_judgment_thresholds_against_spec.py.
V2_JUDGMENT_SLO_TYPES: frozenset[str] = frozenset({
    "reversal_rate",
    "high_confidence_failure",
    "audit_sampling",
    "outcomes",
    "escalation",
})

# Error magnitudes: bounds, not SLOs. See manifest/thresholds.py.
V2_JUDGMENT_THRESHOLD_TYPES: frozenset[str] = frozenset({
    "segments",
    "stability",
    "calibration",
    "audit_backlog",
    "feedback_latency",
})

# The judgment types srm/v1 can express as thresholds when parsed. srm/v1 writes
# judgment types flat under spec.slos (its SLO names are V2_JUDGMENT_SLO_TYPES).
# segments and stability are absent deliberately: their v2 forms need fields
# (segment_dimension, probe) the flat format cannot express, so srm/v1 rejects
# them. Published for consumers that need the srm/v1 lift set; parser/v1.py
# routes on V2_JUDGMENT_THRESHOLD_TYPES instead, so segments, stability and
# audit_backlog reach their rejection rather than falling through.
SRM_V1_JUDGMENT_THRESHOLD_TYPES: frozenset[str] = frozenset({
    "calibration",
    "feedback_latency",
})

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


class SourceFormat(StrEnum):
    """Source format of the manifest file.

    SRM_V1:     apiVersion: srm/v1, kind: ServiceReliabilityManifest
    OPENSRM_V2: apiVersion: opensrm.nthlayer.io/v2, kind: ServiceManifest
    LEGACY:     service: + resources: (NthLayer legacy format)
    """

    SRM_V1 = "srm_v1"
    OPENSRM = "srm_v1"  # Alias — v1 was historically called "OPENSRM"
    OPENSRM_V2 = "opensrm_v2"
    LEGACY = "legacy"


class DependencyCriticality(StrEnum):
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


@dataclass(frozen=True)
class OutcomeSignal:
    """Where a decision's outcome comes from, and how long to wait for it.

    The ``outcome_signal`` of a segments threshold whose ``metric`` is
    ``outcomes`` (CORE §5.5.1, opensrm-7zpz), with the meaning the outcomes
    judgment SLO gives it. ``window`` is the outcome wait, a Duration string
    kept as declared (``"0s"`` allowed); it is not the population window,
    which is ``ThresholdMeasurement.window``.
    """

    source: str
    window: str


@dataclass
class ThresholdMeasurement:
    """Type-specific measurement fields of a JudgmentThreshold (opensrm-l33e).

    Which fields each threshold type sets (schema.json JudgmentThresholdSpec):

    - calibration: ``bins`` and ``window``
    - stability: ``probe`` (``frozen_input_set``, ``frequency``)
    - segments: ``metric``, ``segment_dimension``, ``segment_values`` and
      ``window`` (the population window; the parser stores the spec default
      ``"30d"`` when ``measurement.window`` is absent), plus the parameter the
      metric takes (CORE §5.5.1, opensrm-7zpz): ``source`` for
      ``reversal_rate``, ``confidence_threshold`` for
      ``high_confidence_failure``, ``outcome_signal`` for ``outcomes``;
      ``escalation`` takes none
    - audit_backlog, feedback_latency: none; every field stays at its default

    Deliberately omits JudgmentMeasurement's SLO-only fields (``method``,
    ``frequency``, ``sampling``). On a threshold they were always None, so an
    evaluator reading ``.frequency`` instead of ``.probe.frequency`` got a
    silent None. Here it is an AttributeError. ``source`` and
    ``confidence_threshold`` are present because a segments metric takes them.

    Nothing here enforces which fields go with which type: the parsers are the
    only invariant enforcer, so a hand-built instance can pair any fields.
    ``window`` keeps the declared Duration string (so it round-trips);
    convert it with ``duration_seconds()``.
    """

    bins: int | None = None
    window: str | None = None  # calibration: ECE window; segments: population window (parser stores 30d)
    probe: ProbeConfig | None = None
    segment_dimension: str | None = None
    segment_values: list[str] = field(default_factory=list)
    # segments only (opensrm-7zpz): the rate compared, and its parameter.
    metric: str | None = None  # reversal_rate | high_confidence_failure | outcomes | escalation
    source: str | None = None  # reversal_rate: lineage | calibration_sample | downstream_signal
    confidence_threshold: float | None = None  # high_confidence_failure, a ratio
    outcome_signal: OutcomeSignal | None = None  # outcomes


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

    ``target`` convention: 0-100 percentage canonical (opensrm-5fff).

    All NthLayer-internal consumers (observe, measure) read ``target`` as
    a 0-100 percentage. Examples: ``target=99.9`` for 99.9% availability,
    ``target=98.5`` for a judgment SLO that triggers when the SLI drops
    below 98.5%.

    The OpenSLO surface (``nthlayer_common.slo_models.SLO``) uses 0.0-1.0
    ratio convention. Conversion happens at the boundary in
    ``nthlayer-generate.slos.pipeline._build_slo_from_manifest`` which
    divides by 100.0 unconditionally.

    A load-time warning (``TargetConventionWarning``, see
    ``nthlayer_common.manifest.target_validation``) flags targets in the
    (0, 1) range as likely ratio author errors.

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
    judgment_type: str | None = None  # one of V2_JUDGMENT_SLO_TYPES
    measurement: JudgmentMeasurement | None = None
    breach_actions: list[BreachAction] = field(default_factory=list)
    statistical_requirements: StatisticalRequirements | None = None

    def is_judgment_slo(self) -> bool:
        """Check if this is a judgment SLO. Only checks judgment_type."""
        return self.judgment_type is not None


@dataclass
class JudgmentThreshold:
    """A bound on a judgment ERROR MAGNITUDE (decision 3c, opensrm-l33e).

    Not an SLO, and deliberately not an SLODefinition: there is no target in
    SLI-floor space, no error budget and no complement. A breach is the
    measured value exceeding ANY declared bound. ``bounds`` holds the values
    exactly as declared: floats for Ratio bounds, duration strings such as
    ``"24h"`` for Duration bounds. Validated by manifest/thresholds.py.

    ``measurement`` is a ThresholdMeasurement holding the type-specific
    fields (calibration bins/window, stability probe, segments
    metric/dimension/values/window and the metric's parameter), not the
    judgment-SLO JudgmentMeasurement.
    """

    # A plain str by design, like every judgment_type in this module; the
    # contract is membership in V2_JUDGMENT_THRESHOLD_TYPES.
    type: str
    bounds: dict[str, float | str]
    measurement: ThresholdMeasurement = field(default_factory=ThresholdMeasurement)


# =============================================================================
# Contract Models
# =============================================================================


@dataclass
class JudgmentPromise:
    """A single judgment-type promise within a reliability contract."""

    judgment_type: str  # one of V2_JUDGMENT_SLO_TYPES
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
# Outcomes — business value mapping for AI decisions (spec § 1)
# =============================================================================


# ISO 4217 alpha-3 currency code shape: three uppercase letters. The
# parser only validates the shape; comprehensive code-validity is the
# responsibility of finance-side consumers.
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")

# Spec § 1 FailureCost.category enum.
VALID_FAILURE_CATEGORIES = frozenset({
    "financial_loss", "friction", "compliance", "reputational", "operational",
})

# Spec § 1 revenue.attribution enum.
VALID_REVENUE_ATTRIBUTIONS = frozenset({"direct", "indirect", "supporting"})


@dataclass
class FailureCost:
    """Per-failure cost for false-positive / false-negative decisions."""

    cost: float
    category: str | None = None

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError(
                f"FailureCost.cost must be >= 0; got {self.cost}"
            )
        if self.category is not None and self.category not in VALID_FAILURE_CATEGORIES:
            raise ValueError(
                f"FailureCost.category must be one of "
                f"{sorted(VALID_FAILURE_CATEGORIES)}; got {self.category!r}"
            )


@dataclass
class DecisionValue:
    """Average value of a correct decision plus per-failure costs."""

    correct: float
    currency: str
    false_positive: FailureCost | None = None
    false_negative: FailureCost | None = None

    def __post_init__(self) -> None:
        if not _CURRENCY_PATTERN.match(self.currency):
            raise ValueError(
                f"DecisionValue.currency must be an ISO 4217 alpha-3 code; "
                f"got {self.currency!r}"
            )


@dataclass
class RevenueAttribution:
    """Logical revenue signal — backend-agnostic."""

    attribution: str | None = None
    signal: str | None = None

    def __post_init__(self) -> None:
        if self.attribution is not None and self.attribution not in VALID_REVENUE_ATTRIBUTIONS:
            raise ValueError(
                f"revenue.attribution must be one of "
                f"{sorted(VALID_REVENUE_ATTRIBUTIONS)}; got {self.attribution!r}"
            )
        # Signal is mapped to a Prometheus metric (or other backend) by
        # nthlayer-generate; identifier shape only here.
        if self.signal is not None and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", self.signal):
            raise ValueError(
                f"revenue.signal must be a valid identifier (alphanumeric + "
                f"underscores); got {self.signal!r}"
            )


@dataclass
class VolumeEstimate:
    """Fallback volume estimate used only when real-time metrics are absent."""

    estimated_daily_decisions: int | None = None
    peak_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.estimated_daily_decisions is not None and self.estimated_daily_decisions < 0:
            raise ValueError(
                f"volume.estimated_daily_decisions must be >= 0; "
                f"got {self.estimated_daily_decisions}"
            )
        if self.peak_multiplier < 0:
            raise ValueError(
                f"volume.peak_multiplier must be >= 0; got {self.peak_multiplier}"
            )


@dataclass
class Outcomes:
    """Business outcome binding — financial impact configuration (spec § 1).

    Optional sibling to ``slos`` and ``contracts``. Services without it
    continue to report SLO percentages only; services with it get
    financial impact in retrospectives and blast-radius assessments.
    """

    decision_value: DecisionValue
    revenue: RevenueAttribution | None = None
    volume: VolumeEstimate | None = None


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
    type: str  # one of VALID_SERVICE_TYPES, or an x- extension type

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
    judgment_thresholds: list[JudgmentThreshold] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    ownership: Ownership | None = None
    observability: Observability | None = None
    deployment: DeploymentConfig | None = None
    contracts: list[ReliabilityContract] = field(default_factory=list)
    instrumentation: Instrumentation | None = None
    outcomes: Outcomes | None = None

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
        # DELIBERATELY NOT resolve_service_type(), despite that helper's
        # docstring saying to prefer it. Resolution and validation are split
        # here so the "Service type is required" guard below can sit between
        # them: resolve_service_type returns None for "", so collapsing the
        # two into one call would report `Invalid type ''` instead of the
        # actionable "you did not set it". Pinned by
        # test_empty_type_reports_required_not_invalid — collapse this and
        # that test fails rather than the message quietly changing.
        #
        # isinstance guard before the lookup, not just inside
        # is_valid_service_type: `self.type in SERVICE_TYPE_ALIASES` hashes
        # the value, so a list or dict from raw YAML raised TypeError here —
        # before validation could turn it into a ValueError.
        if isinstance(self.type, str) and self.type in SERVICE_TYPE_ALIASES:
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
            msg = (
                f"Invalid tier '{self.tier}'. "
                f"Must be one of: {', '.join(sorted(VALID_TIERS))}."
            )
            raise ValueError(msg)

        if not is_valid_service_type(self.type):
            msg = (
                f"Invalid type '{self.type}'. "
                f"Must be one of: {valid_service_types_phrase()}."
            )
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

    def get_judgment_thresholds(self) -> list[JudgmentThreshold]:
        """Get all judgment thresholds (bounds on error magnitudes)."""
        return list(self.judgment_thresholds)

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
                    looser = (
                        (jp.direction == "below" and slo.target > jp.threshold)
                        or (jp.direction == "above" and slo.target < jp.threshold)
                    )
                    if looser:
                        errors.append(
                            f"Judgment SLO '{slo.name}' ({slo.target}) is looser than "
                            f"contract '{contract.name}' threshold ({jp.threshold})"
                        )

        return errors
