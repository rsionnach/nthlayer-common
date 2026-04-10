"""
Deployment gate data models.

Pure data structures for gate results and policies — no runtime logic.
The runtime DeploymentGate class lives in nthlayer (moving to observe in P4).

These models are extracted early (P0) so that generators/backstage.py can
import GateResult from common, avoiding a cross-boundary dependency when
the runtime gate code moves to observe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

__all__ = [
    "GateResult",
    "GatePolicy",
    "DeploymentGateCheck",
]


class GateResult(IntEnum):
    """
    Deployment gate exit codes.

    Following common CI/CD conventions:
    - 0 = Success/Approved
    - 1 = Warning (advisory, doesn't block)
    - 2 = Error (blocked)
    """

    APPROVED = 0
    WARNING = 1
    BLOCKED = 2


@dataclass
class GatePolicy:
    """
    Custom gate policy from DeploymentGate resource.

    Allows overriding default tier-based thresholds and adding conditions.
    """

    # Custom thresholds (override defaults)
    warning: float | None = None  # Warn when budget remaining < this %
    blocking: float | None = None  # Block when budget remaining < this %

    # Conditional policies
    conditions: list[dict[str, Any]] = field(default_factory=list)

    # Exceptions (teams that can bypass)
    exceptions: list[dict[str, Any]] = field(default_factory=list)

    # Behaviors when error budget is exhausted (0% remaining)
    on_exhausted: list[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> GatePolicy:
        """Create GatePolicy from DeploymentGate resource spec."""
        thresholds = spec.get("thresholds", {})
        return cls(
            warning=thresholds.get("warning"),
            blocking=thresholds.get("blocking"),
            conditions=spec.get("conditions", []),
            exceptions=spec.get("exceptions", []),
            on_exhausted=spec.get("on_exhausted", []),
        )


@dataclass
class DeploymentGateCheck:
    """Result of deployment gate check."""

    service: str
    tier: str
    result: GateResult

    # Error budget status
    budget_total_minutes: int
    budget_consumed_minutes: int
    budget_remaining_minutes: int
    budget_remaining_percentage: float

    # Thresholds
    warning_threshold: float
    blocking_threshold: float | None

    # Blast radius
    downstream_services: list[str]
    high_criticality_downstream: list[str]

    # Messages
    message: str
    recommendations: list[str]

    @property
    def is_approved(self) -> bool:
        """Check if deployment is approved."""
        return self.result == GateResult.APPROVED

    @property
    def is_warning(self) -> bool:
        """Check if deployment has warnings."""
        return self.result == GateResult.WARNING

    @property
    def is_blocked(self) -> bool:
        """Check if deployment is blocked."""
        return self.result == GateResult.BLOCKED
