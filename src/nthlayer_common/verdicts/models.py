"""Verdict data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

VALID_SUBJECT_TYPES = frozenset({
    "agent_output", "correlation", "triage", "investigation",
    "remediation", "review", "classification", "recommendation",
    "moderation", "communication", "evaluation", "retrospective",
    "custom",
})

VALID_ACTIONS = frozenset({
    "approve", "reject", "flag", "escalate", "defer", "custom",
})

VALID_OUTCOME_STATUSES = frozenset({
    "confirmed", "overridden", "partial", "superseded", "expired",
})

# Verdict types from OPENSRM-RBAC-EXTENSION-v2 §10 + v1.5 additions.
# These are judgment-bearing decisions, not observations. Observations
# (correlation_snapshot, topology_drift, contract_divergence) are
# assessments — see nthlayer_common.cloudevents.ASSESSMENT_KINDS.
VALID_VERDICT_TYPES = frozenset({
    # Auth flow (RBAC §10)
    "action_request",
    "approval",
    "capability",
    "denial",
    "execution",
    "operator_note",
    # Measure outputs (NTHLAYER-MEASURE-v1 §9)
    "autonomy_change",
    "quality_breach",
    # Core (outcome resolution — immutable, creates new verdict)
    "outcome_resolution",
    # Assessment-like (continuous, not decisions)
    "assessment",
})

TTL_DEFAULT = 90 * 24 * 60 * 60  # 90 days in seconds


@dataclass
class Producer:
    system: str
    instance: str | None = None
    model: str | None = None
    prompt_version: str | None = None


@dataclass
class Subject:
    type: str
    ref: str
    summary: str
    agent: str | None = None
    service: str | None = None
    environment: str | None = None
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if self.type not in VALID_SUBJECT_TYPES:
            raise ValueError(
                f"Invalid subject type '{self.type}'. "
                f"Must be one of: {sorted(VALID_SUBJECT_TYPES)}"
            )


@dataclass
class Judgment:
    action: str
    confidence: float
    score: float | None = None
    dimensions: dict[str, float] | None = None
    reasoning: str | None = None
    tags: list[str] | None = None

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{self.action}'. "
                f"Must be one of: {sorted(VALID_ACTIONS)}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if self.score is not None and not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"Score must be between 0.0 and 1.0, got {self.score}"
            )
        if self.dimensions:
            for dim_name, dim_value in self.dimensions.items():
                if not (0.0 <= dim_value <= 1.0):
                    raise ValueError(
                        f"Dimension '{dim_name}' must be between 0.0 and 1.0, "
                        f"got {dim_value}"
                    )


@dataclass
class Override:
    by: str | None = None
    at: datetime | None = None
    action: str | None = None
    reasoning: str | None = None


@dataclass
class GroundTruth:
    signal: str | None = None
    value: str | None = None
    detected_at: datetime | None = None


@dataclass
class Outcome:
    status: str = "pending"
    resolution: str | None = None
    override: Override | None = None
    ground_truth: GroundTruth | None = None
    closed_at: datetime | None = None


@dataclass
class Lineage:
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)


@dataclass
class Metadata:
    cost_tokens: int | None = None
    cost_currency: float | None = None
    latency_ms: int | None = None
    ttl: int = TTL_DEFAULT
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    id: str
    version: int
    timestamp: datetime
    producer: Producer
    subject: Subject
    judgment: Judgment
    outcome: Outcome = field(default_factory=Outcome)
    lineage: Lineage = field(default_factory=Lineage)
    metadata: Metadata = field(default_factory=Metadata)
    # v1.5 transitional fields (NTHLAYER-SERVE-MODE-v2.1 §6)
    verdict_type: str | None = None  # from VALID_VERDICT_TYPES; None for backward compat
    pipeline_latency_ms: int | None = None  # cumulative latency from first verdict in chain
    chain_depth: int = 0  # number of ancestors
    parent_ids: list[str] = field(default_factory=list)  # string IDs in v1.5, CIDs in v2
    service: str | None = None  # affected service (denormalized for query convenience)


@dataclass
class AccuracyReport:
    producer: str
    total: int
    total_resolved: int
    confirmation_rate: float
    override_rate: float
    partial_rate: float
    pending_rate: float
    mean_confidence_on_confirmed: float
    mean_confidence_on_overridden: float
    dimension: str | None = None
