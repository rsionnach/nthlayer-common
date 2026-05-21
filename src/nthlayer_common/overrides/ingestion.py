"""Bind a canonical OverrideEvent to a verdict in the verdict store.

The integration spec point (§ 4 nthlayer-measure integration steps 1-2):
look up the verdict by ``decision_id``, set ``outcome.status =
"overridden"``, populate ``outcome.override`` with the gen_ai.override
attributes (reviewer hashed by default).

Steps 3-5 (Prometheus reversal/HCF counter increments and OTel metrics)
are emitted by the agent-side instrumentation already — this function
is the verdict-store side of the same event.
"""
from __future__ import annotations

import structlog

from nthlayer_common.overrides.models import (
    OverrideEvent,
    OverridePrivacyConfig,
    hash_reviewer,
)
from nthlayer_common.verdicts.models import Outcome, Override, Verdict
from nthlayer_common.verdicts.store import OutcomeStatusMismatch, VerdictStore

logger = structlog.get_logger(__name__)


# Allowlist gate: only "pending" verdicts accept a fresh override.
# "overridden" is handled separately below for idempotency / conflict.
_OPEN_FOR_OVERRIDE = frozenset({"pending"})


def _build_override(
    event: OverrideEvent, privacy: OverridePrivacyConfig,
) -> Override:
    trust_wire = privacy.plaintext_reviewer or privacy.pre_redacted
    reviewer = event.reviewer if trust_wire else hash_reviewer(event.reviewer)
    return Override(
        by=reviewer,
        at=event.timestamp,
        action=event.corrected_action,
        reasoning=None if privacy.exclude_reason else event.reason,
        original_action=event.original_action,
        confidence_at_decision=event.confidence_at_decision,
        source_system=event.source_system,
    )


def apply_override_to_verdict(
    store: VerdictStore,
    event: OverrideEvent,
    *,
    privacy: OverridePrivacyConfig | None = None,
) -> Verdict | None:
    """Bind ``event`` to the verdict ``event.decision_id`` references.

    Returns the updated Verdict on success, or None when:
    - no verdict matches the decision_id (unmatched; logs warning),
    - the verdict is already overridden with conflicting content,
    - the verdict is in a non-pending status that the override
      cannot legitimately override (confirmed / superseded / expired
      / partial / any unknown future status).

    Semantic is first-writer-wins: ``event.timestamp`` is recorded on
    the override but never used to order replays. Idempotent only when
    the replay carries identical payload including timestamp.

    Reviewer is hashed by default (SHA-256, GDPR-pseudonymisation
    baseline — not anonymisation; reversible by enumeration on
    low-entropy inputs). Pass ``privacy=OverridePrivacyConfig(
    plaintext_reviewer=True)`` to opt in.
    """
    privacy = privacy or OverridePrivacyConfig()

    verdict = store.get(event.decision_id)
    if verdict is None:
        logger.warning(
            "override_unmatched_decision_id",
            decision_id=event.decision_id,
            service=event.service,
            source_system=event.source_system,
        )
        return None

    new_override = _build_override(event, privacy)

    current_status = verdict.outcome.status
    existing = verdict.outcome.override
    if current_status == "overridden":
        if existing == new_override:
            return verdict
        logger.warning(
            "override_conflicts_with_existing",
            decision_id=event.decision_id,
            existing_reviewer=existing.by if existing else None,
            incoming_reviewer=new_override.by,
            source_system=event.source_system,
        )
        return None

    if current_status not in _OPEN_FOR_OVERRIDE:
        logger.warning(
            "override_blocked_by_status",
            decision_id=event.decision_id,
            status=current_status,
            source_system=event.source_system,
        )
        return None

    new_outcome = Outcome(
        status="overridden",
        resolution=verdict.outcome.resolution,
        override=new_override,
        ground_truth=verdict.outcome.ground_truth,
        closed_at=event.timestamp,
    )

    try:
        return store.update_outcome(
            event.decision_id, new_outcome, expected_status=current_status,
        )
    except KeyError:
        # TOCTOU race with a concurrent delete.
        logger.warning(
            "override_lost_race_to_concurrent_delete",
            decision_id=event.decision_id,
            source_system=event.source_system,
        )
        return None
    except OutcomeStatusMismatch:
        # A concurrent writer transitioned the verdict between our
        # status read and our update — last-writer-wins would silently
        # drop one of two simultaneous Slack/Jira overrides. The CAS
        # ensures only one wins; the other backs off without effect.
        logger.warning(
            "override_lost_race_to_concurrent_writer",
            decision_id=event.decision_id,
            source_system=event.source_system,
        )
        return None
