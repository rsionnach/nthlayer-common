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
from nthlayer_common.verdicts.store import VerdictStore

logger = structlog.get_logger(__name__)


# Outcome statuses that already represent a settled judgment. Re-applying
# an override on top would silently rewrite the audit trail, so the
# integration boundary refuses and logs instead. "overridden" is treated
# specially below to preserve idempotency on retries.
_TERMINAL_STATUSES = frozenset({"confirmed", "superseded", "expired"})


def _build_override(
    event: OverrideEvent, privacy: OverridePrivacyConfig,
) -> Override:
    reviewer = (
        event.reviewer
        if privacy.plaintext_reviewer
        else hash_reviewer(event.reviewer)
    )
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
    """Apply ``event`` to the verdict ``event.decision_id`` references.

    Returns the updated Verdict on success, or None when no verdict
    matches (spec § 4 test scenario 5) or the verdict is already in a
    terminal state that the override cannot override.

    Idempotent: re-applying the same event to an already-overridden
    verdict with the same Override content is a no-op and returns the
    existing verdict (test scenario 11). A second event with *different*
    content (different reviewer, different corrected_action, etc.) on an
    already-overridden verdict is rejected with a warning — silently
    rewriting an audit trail is worse than refusing and surfacing the
    conflict for operator attention. Verdicts in confirmed / superseded
    / expired status are similarly protected.

    The reviewer field is hashed by default; pass ``privacy`` with
    ``plaintext_reviewer=True`` to opt in. ``exclude_reason=True`` drops
    the reason. Note: SHA-256 of low-entropy reviewer identifiers
    (emails, usernames) is reversible by enumeration — the hash is a
    GDPR-pseudonymisation baseline, not anonymisation. Operators with
    higher requirements should pre-hash with a per-deployment HMAC.

    Unmatched decision_ids log a warning. The future sidecar
    (opensrm-jmy.7) will buffer for retry; today the source replays.
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
    if current_status == "overridden":
        if verdict.outcome.override == new_override:
            return verdict
        logger.warning(
            "override_conflicts_with_existing",
            decision_id=event.decision_id,
            existing_reviewer=verdict.outcome.override.by
                if verdict.outcome.override else None,
            incoming_reviewer=new_override.by,
            source_system=event.source_system,
        )
        return None

    if current_status in _TERMINAL_STATUSES:
        logger.warning(
            "override_blocked_by_terminal_status",
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

    return store.update_outcome(event.decision_id, new_outcome)
