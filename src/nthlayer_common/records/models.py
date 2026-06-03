"""Content-addressed decision record types.

Spec: nthlayer-spec-decision-records.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from enum import StrEnum
from typing import Any


def _require_utc(ts: datetime, cls_name: str) -> None:
    if ts.tzinfo is None:
        raise ValueError(f"{cls_name}.timestamp must be timezone-aware (got naive datetime)")
    if ts.utcoffset() != UTC.utcoffset(None):
        raise ValueError(f"{cls_name}.timestamp must be UTC (got {ts.tzinfo})")


__all__ = [
    "AssessmentType",
    "Severity",
    "VerdictOutcome",
    "EvaluationMethod",
    "EvaluationOutcome",
    "IncidentStatus",
    "Summaries",
    "Assessment",
    "Verdict",
    "Evaluation",
    "Incident",
]

ZERO_HASH = "0" * 64


# --- Enums ---


class AssessmentType(StrEnum):
    THRESHOLD_BREACH = "threshold_breach"
    CORRELATION = "correlation"
    DRIFT = "drift"
    CHANGE_EVENT = "change_event"


class Severity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class VerdictOutcome(StrEnum):
    RECOMMENDED = "recommended"
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class EvaluationMethod(StrEnum):
    METRIC_RECOVERY = "metric_recovery"
    HUMAN_REVIEW = "human_review"
    TIMEOUT = "timeout"
    SLO_RESTORATION = "slo_restoration"


class EvaluationOutcome(StrEnum):
    EFFECTIVE = "effective"
    INEFFECTIVE = "ineffective"
    INCONCLUSIVE = "inconclusive"
    PARTIAL = "partial"


class IncidentStatus(StrEnum):
    OPEN = "open"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    LEARNING = "learning"
    CLOSED = "closed"


# --- Summaries ---


@dataclass(slots=True)
class Summaries:
    """Multi-register summaries for context compression."""

    technical: str  # max 280 chars
    plain: str  # max 280 chars
    executive: str | None = None  # max 140 chars

    def truncated(self) -> Summaries:
        """Return a copy truncated to spec limits."""
        return Summaries(
            technical=self.technical[:280],
            plain=self.plain[:280],
            executive=self.executive[:140] if self.executive is not None else None,
        )


# --- Record Types ---


@dataclass(slots=True)
class Assessment:
    """Deterministic measured fact about system state. Produced by nthlayer-observe."""

    hash: str
    previous_hash: str
    schema_version: str
    timestamp: datetime
    stream: str  # colon-delimited SLI identifier, e.g. "sli:checkout-service:latency-p99"
    incident_id: str | None
    type: AssessmentType
    severity: Severity
    payload: dict[str, Any]
    summaries: Summaries

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "Assessment")


@dataclass(slots=True)
class Verdict:
    """LLM-derived judgment and action. Produced by agentic components."""

    hash: str
    previous_hash: str
    schema_version: str
    timestamp: datetime
    agent: str
    incident_id: str
    input_hashes: list[str]
    prompt_hash: str
    response_hash: str
    model: str
    reasoning: str
    action: dict[str, Any]
    outcome: VerdictOutcome
    summaries: Summaries

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "Verdict")


@dataclass(slots=True)
class Evaluation:
    """Retrospective judgment closing the feedback loop. Produced by nthlayer-learn."""

    hash: str
    previous_hash: str
    schema_version: str
    timestamp: datetime
    incident_id: str
    verdict_hash: str
    method: EvaluationMethod
    outcome: EvaluationOutcome
    evidence_hashes: list[str]
    payload: dict[str, Any]
    summaries: Summaries

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "Evaluation")


@dataclass(slots=True)
class Incident:
    """Lightweight mutable index grouping related records."""

    id: str
    created_at: datetime
    trigger_hash: str
    stream: str  # colon-delimited SLI identifier, e.g. "sli:checkout-service:latency-p99"
    status: IncidentStatus = IncidentStatus.OPEN

    def __post_init__(self) -> None:
        _require_utc(self.created_at, "Incident")
