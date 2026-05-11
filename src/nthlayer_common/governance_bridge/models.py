"""Governance bridge signal schemas (spec § 5).

All signals carry ``version = "v1"`` so receivers can multiplex when
the protocol evolves. The dataclass field is ``signal_type`` (Python)
but ``to_payload()`` emits ``type`` on the wire (matches the spec
shape and avoids shadowing the ``type`` builtin).

Idempotency is receiver-side: per spec principle 3, receivers
deduplicate on ``(type, service, timestamp)``. Senders only need to
ensure those fields are stable across retries — which they are,
since the same signal instance is re-emitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Bump when payload shape breaks compatibility. Receivers route on (type, version).
SIGNAL_VERSION = "v1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _require_tz_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(
            f"{field_name} must be timezone-aware "
            "(use datetime.now(timezone.utc))"
        )


def _require_nonempty(value: object, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} is required (spec § 5)")


# ---------------------------------------------------------------------------
# Outbound: NthLayer → governance platform
# ---------------------------------------------------------------------------


@dataclass
class AutonomyChangeTrigger:
    """Why the autonomy ratchet fired — the SLO that breached.

    ``current_value`` and ``threshold`` use whatever unit the metric
    is recorded in (ratio for reversal_rate, count for HCF, etc.) —
    the receiver consumes whatever nthlayer-measure emits without
    re-normalising.
    """

    metric: str
    current_value: float
    threshold: float
    window: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "window": self.window,
        }


@dataclass
class AutonomyChangeSignal:
    """Spec § 5 signal 1 — emitted when measure changes an agent's autonomy.

    Outbound only. The governance platform is informed; it cannot veto
    (the autonomy ratchet is local and authoritative per principle 6).
    """

    service: str
    previous_level: str
    new_level: str
    trigger: AutonomyChangeTrigger
    recommended_governance_action: str
    incident: str | None = None
    timestamp: datetime = field(default_factory=_utcnow)
    version: str = SIGNAL_VERSION
    signal_type: str = field(default="autonomy_change", init=False)

    def __post_init__(self) -> None:
        _require_tz_aware(self.timestamp, "AutonomyChangeSignal.timestamp")
        for name in (
            "service",
            "previous_level",
            "new_level",
            "recommended_governance_action",
        ):
            _require_nonempty(getattr(self, name), f"AutonomyChangeSignal.{name}")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.signal_type,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "service": self.service,
            "previous_level": self.previous_level,
            "new_level": self.new_level,
            "trigger": self.trigger.to_payload(),
            "recommended_governance_action": self.recommended_governance_action,
        }
        if self.incident is not None:
            payload["incident"] = self.incident
        return payload


@dataclass
class PolicyRecommendation:
    """Inner payload of PolicyRecommendationSignal.

    ``recommendation_type`` matches the SpecRecommendation taxonomy from
    spec § 2 (e.g. ``add_deploy_gate``, ``tighten_slo``, ``add_dependency``).
    ``financial_impact`` is the human-readable string from jmy.1 outcomes
    (e.g. "Would have prevented ~€47,200 in false_negative losses").
    """

    recommendation_type: str
    description: str
    confidence: float
    financial_impact: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.recommendation_type, "PolicyRecommendation.recommendation_type")
        _require_nonempty(self.description, "PolicyRecommendation.description")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "PolicyRecommendation.confidence must be in [0.0, 1.0]; "
                f"got {self.confidence}"
            )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.recommendation_type,
            "description": self.description,
            "confidence": self.confidence,
        }
        if self.financial_impact is not None:
            payload["financial_impact"] = self.financial_impact
        return payload


@dataclass
class PolicyRecommendationSignal:
    """Spec § 5 signal 2 — emitted when learn produces a policy-relevant rec."""

    service: str
    incident: str
    recommendation: PolicyRecommendation
    requires_human_review: bool = True
    timestamp: datetime = field(default_factory=_utcnow)
    version: str = SIGNAL_VERSION
    signal_type: str = field(default="policy_recommendation", init=False)

    def __post_init__(self) -> None:
        _require_tz_aware(self.timestamp, "PolicyRecommendationSignal.timestamp")
        _require_nonempty(self.service, "PolicyRecommendationSignal.service")
        _require_nonempty(self.incident, "PolicyRecommendationSignal.incident")

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.signal_type,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "service": self.service,
            "incident": self.incident,
            "recommendation": self.recommendation.to_payload(),
            "requires_human_review": self.requires_human_review,
        }


@dataclass
class IncidentOpenedSignal:
    """Spec § 5 signal 3 — emitted when respond opens an incident.

    ``service`` here is the root-cause service; the broader blast
    radius lives in ``affected_services``. This matches the
    nthlayer-correlate output shape so the bridge can reuse it
    directly without re-shaping.
    """

    incident: str
    severity: int
    affected_services: list[str]
    root_cause_service: str
    root_cause_type: str
    timestamp: datetime = field(default_factory=_utcnow)
    version: str = SIGNAL_VERSION
    signal_type: str = field(default="incident_opened", init=False)

    @property
    def service(self) -> str:
        """Receivers dedup on (type, service, timestamp); the root-cause
        service is the natural anchor for this signal."""
        return self.root_cause_service

    def __post_init__(self) -> None:
        _require_tz_aware(self.timestamp, "IncidentOpenedSignal.timestamp")
        for name in ("incident", "root_cause_service", "root_cause_type"):
            _require_nonempty(getattr(self, name), f"IncidentOpenedSignal.{name}")
        if not isinstance(self.severity, int) or self.severity < 1:
            raise ValueError(
                "IncidentOpenedSignal.severity must be a positive int; "
                f"got {self.severity!r}"
            )
        if not self.affected_services:
            raise ValueError(
                "IncidentOpenedSignal.affected_services must contain at "
                "least the root-cause service"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.signal_type,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "incident": self.incident,
            "severity": self.severity,
            "affected_services": list(self.affected_services),
            "root_cause_service": self.root_cause_service,
            "root_cause_type": self.root_cause_type,
        }


# ---------------------------------------------------------------------------
# Inbound: governance platform → NthLayer (delivered via the bridge adapter)
# ---------------------------------------------------------------------------


@dataclass
class PolicyUpdatedSignal:
    """Spec § 5 inbound signal 1 — governance platform changed a policy.

    The bridge adapter translates this into an OTel event that
    nthlayer-correlate ingests as a signal source (policy changes
    can cause behavioural changes, analogous to deploys / config
    changes).
    """

    service: str
    policy_id: str
    change_type: str
    description: str
    source_system: str
    timestamp: datetime = field(default_factory=_utcnow)
    version: str = SIGNAL_VERSION
    signal_type: str = field(default="policy_updated", init=False)

    def __post_init__(self) -> None:
        _require_tz_aware(self.timestamp, "PolicyUpdatedSignal.timestamp")
        for name in (
            "service",
            "policy_id",
            "change_type",
            "description",
            "source_system",
        ):
            _require_nonempty(getattr(self, name), f"PolicyUpdatedSignal.{name}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.signal_type,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "service": self.service,
            "policy_id": self.policy_id,
            "change_type": self.change_type,
            "description": self.description,
            "source_system": self.source_system,
        }


@dataclass
class AutonomyRestoreRequest:
    """Spec § 5 inbound signal 2 — human requests autonomy restoration.

    The bridge adapter forwards this to nthlayer-measure, which
    validates against current metrics. Principle 4: NthLayer can
    tighten autonomy automatically; only a human can loosen it.
    """

    service: str
    requested_by: str
    requested_level: str
    justification: str
    timestamp: datetime = field(default_factory=_utcnow)
    version: str = SIGNAL_VERSION
    signal_type: str = field(default="autonomy_restore_request", init=False)

    def __post_init__(self) -> None:
        _require_tz_aware(self.timestamp, "AutonomyRestoreRequest.timestamp")
        for name in ("service", "requested_by", "requested_level", "justification"):
            _require_nonempty(getattr(self, name), f"AutonomyRestoreRequest.{name}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.signal_type,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "service": self.service,
            "requested_by": self.requested_by,
            "requested_level": self.requested_level,
            "justification": self.justification,
        }
