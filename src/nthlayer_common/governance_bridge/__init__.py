"""Governance bridge — webhook protocol for external policy platforms.

Spec: ``NTHLAYER_MISSING_CAPABILITIES_SPEC.md`` § 5 (opensrm-jmy.5).

The bridge is the open, platform-agnostic handshake between NthLayer
(post-execution reliability) and governance platforms like CortexHub
(pre-execution policy enforcement). It defines signal shapes, not
implementations: anything that speaks HTTP can participate.

This module ships only the foundation — signal schemas + the outbound
emitter. Integration into the workers (measure → autonomy_change,
learn → policy_recommendation, respond → incident_opened) and the
inbound adapter sidecar are tracked in follow-up beads.

Canonical import: ``from nthlayer_common.governance_bridge import
GovernanceBridgeEmitter, AutonomyChangeSignal``.
"""

from nthlayer_common.governance_bridge.emitter import (
    EmitResult,
    GovernanceBridgeEmitter,
)
from nthlayer_common.governance_bridge.models import (
    SIGNAL_VERSION,
    AutonomyChangeSignal,
    AutonomyChangeTrigger,
    AutonomyRestoreRequest,
    IncidentOpenedSignal,
    PolicyRecommendation,
    PolicyRecommendationSignal,
    PolicyUpdatedSignal,
)

__all__ = [
    # Constants
    "SIGNAL_VERSION",
    # Outbound signals (NthLayer → governance platform)
    "AutonomyChangeSignal",
    "AutonomyChangeTrigger",
    "PolicyRecommendationSignal",
    "PolicyRecommendation",
    "IncidentOpenedSignal",
    # Inbound signals (governance platform → NthLayer)
    "PolicyUpdatedSignal",
    "AutonomyRestoreRequest",
    # Emitter
    "EmitResult",
    "GovernanceBridgeEmitter",
]
