"""Tests for the override ingestion module (opensrm-jmy.4).

Covers spec § 4 test scenarios 1, 5, 6, 7, 8, 10, 11. Scenarios 2, 3, 4,
9 (sidecar HTTP, batch endpoint, Slack reaction adapter) live behind
opensrm-jmy.7 along with the standalone adapter process.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from nthlayer_common.overrides import (
    OverrideEvent,
    OverridePrivacyConfig,
    apply_override_to_verdict,
    hash_reviewer,
    map_webhook_to_override,
)
from nthlayer_common.verdicts import (
    Judgment,
    MemoryStore,
    Outcome,
    Producer,
    Subject,
    Verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verdict(
    *,
    vid: str = "vrd-test-1",
    service: str = "fraud-detection",
    action: str = "approve",
    confidence: float = 0.71,
) -> Verdict:
    return Verdict(
        id=vid,
        version=1,
        timestamp=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        producer=Producer(system=service),
        subject=Subject(
            type="agent_output", ref=service, summary="t", service=service,
        ),
        judgment=Judgment(action=action, confidence=confidence, reasoning=""),
        service=service,
    )


def _event(**overrides: object) -> OverrideEvent:
    base: dict[str, object] = {
        "decision_id": "vrd-test-1",
        "service": "fraud-detection",
        "corrected_action": "escalate_to_review",
        "reviewer": "analyst-047",
        "original_action": "approve",
        "reason": "Model regression - underwriting-v3 miscalibrated",
        "confidence_at_decision": 0.71,
        "source_system": "internal-review-ui",
        "timestamp": datetime(2026, 5, 6, 12, 5, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return OverrideEvent(**base)


# ---------------------------------------------------------------------------
# OverrideEvent validation
# ---------------------------------------------------------------------------


class TestOverrideEvent:
    def test_required_fields_enforced(self):
        for missing in ("decision_id", "service", "corrected_action", "reviewer"):
            with pytest.raises(ValueError, match=missing):
                _event(**{missing: ""})

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="confidence_at_decision"):
            _event(confidence_at_decision=1.5)
        with pytest.raises(ValueError, match="confidence_at_decision"):
            _event(confidence_at_decision=-0.1)

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            _event(timestamp=datetime(2026, 5, 6, 12, 0))

    def test_high_confidence_failure_threshold(self):
        # Spec §4 step 4: increment HCF when confidence > 0.85.
        assert _event(confidence_at_decision=0.86).is_high_confidence_failure
        assert not _event(confidence_at_decision=0.85).is_high_confidence_failure
        assert not _event(confidence_at_decision=0.71).is_high_confidence_failure
        assert not _event(confidence_at_decision=None).is_high_confidence_failure

    def test_to_otel_attributes_drops_none(self):
        event = _event(reason=None, confidence_at_decision=None)
        attrs = event.to_otel_attributes()
        assert attrs["gen_ai.override.decision_id"] == "vrd-test-1"
        assert attrs["gen_ai.override.corrected_action"] == "escalate_to_review"
        assert "gen_ai.override.reason" not in attrs
        assert "gen_ai.override.confidence_at_decision" not in attrs


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


class TestPrivacy:
    def test_hash_reviewer_is_stable_sha256(self):
        expected = hashlib.sha256(b"analyst-047").hexdigest()
        assert hash_reviewer("analyst-047") == expected

    def test_default_hashes_reviewer(self):
        # Spec §4 test scenario 7.
        store = MemoryStore()
        store.put(_verdict())
        result = apply_override_to_verdict(store, _event())
        assert result is not None
        assert result.outcome.override is not None
        assert result.outcome.override.by == hash_reviewer("analyst-047")
        assert result.outcome.override.by != "analyst-047"

    def test_plaintext_opt_in(self):
        # Spec §4 test scenario 8.
        store = MemoryStore()
        store.put(_verdict())
        result = apply_override_to_verdict(
            store, _event(),
            privacy=OverridePrivacyConfig(plaintext_reviewer=True),
        )
        assert result is not None
        assert result.outcome.override is not None
        assert result.outcome.override.by == "analyst-047"

    def test_exclude_reason(self):
        store = MemoryStore()
        store.put(_verdict())
        result = apply_override_to_verdict(
            store, _event(),
            privacy=OverridePrivacyConfig(exclude_reason=True),
        )
        assert result is not None
        assert result.outcome.override is not None
        assert result.outcome.override.reasoning is None


# ---------------------------------------------------------------------------
# Verdict binding
# ---------------------------------------------------------------------------


class TestApplyOverride:
    def test_native_otel_event_updates_verdict(self):
        # Spec §4 test scenario 1.
        store = MemoryStore()
        store.put(_verdict())
        result = apply_override_to_verdict(store, _event())
        assert result is not None
        assert result.outcome.status == "overridden"
        assert result.outcome.override is not None
        assert result.outcome.override.action == "escalate_to_review"
        assert result.outcome.override.original_action == "approve"
        assert result.outcome.override.confidence_at_decision == 0.71
        assert result.outcome.override.source_system == "internal-review-ui"
        assert result.outcome.closed_at is not None

    def test_unknown_decision_id_returns_none(self):
        # Spec §4 test scenario 5.
        store = MemoryStore()
        result = apply_override_to_verdict(
            store, _event(decision_id="vrd-does-not-exist"),
        )
        assert result is None

    def test_high_confidence_event_preserves_signal(self):
        # Spec §4 test scenario 6: HCF flag must remain readable from
        # the persisted Override so measure can recompute counters.
        store = MemoryStore()
        store.put(_verdict(confidence=0.92))
        result = apply_override_to_verdict(
            store, _event(confidence_at_decision=0.92),
        )
        assert result is not None
        assert result.outcome.override is not None
        assert result.outcome.override.confidence_at_decision == 0.92

    def test_idempotent_reapplication(self):
        # Spec §4 test scenario 11.
        store = MemoryStore()
        store.put(_verdict())
        first = apply_override_to_verdict(store, _event())
        second = apply_override_to_verdict(store, _event())
        assert first is not None and second is not None
        assert first.outcome.status == second.outcome.status == "overridden"
        assert first.outcome.override == second.outcome.override

    def test_conflicting_override_rejected(self):
        # Audit-trail safety: replacing an existing override with
        # different content silently rewrites history; refuse instead.
        store = MemoryStore()
        store.put(_verdict())
        first = apply_override_to_verdict(store, _event())
        assert first is not None
        conflict = apply_override_to_verdict(
            store,
            _event(reviewer="other-analyst", corrected_action="approve_anyway"),
        )
        assert conflict is None
        # Verdict still carries the original override.
        v = store.get("vrd-test-1")
        assert v is not None
        assert v.outcome.override is not None
        assert v.outcome.override.action == "escalate_to_review"

    def test_terminal_status_blocks_override(self):
        # A verdict already resolved with ground truth must not be
        # silently flipped to "overridden" by a late webhook.
        store = MemoryStore()
        verdict = _verdict()
        verdict.outcome = Outcome(status="confirmed", resolution="confirmed")
        store.put(verdict)
        result = apply_override_to_verdict(store, _event())
        assert result is None
        v = store.get("vrd-test-1")
        assert v is not None
        assert v.outcome.status == "confirmed"

    def test_partial_status_blocks_override(self):
        # 'partial' is non-terminal but settled — overriding it would
        # contradict the partial ground-truth signal already attached.
        store = MemoryStore()
        verdict = _verdict()
        verdict.outcome = Outcome(status="partial")
        store.put(verdict)
        result = apply_override_to_verdict(store, _event())
        assert result is None
        v = store.get("vrd-test-1")
        assert v is not None
        assert v.outcome.status == "partial"


# ---------------------------------------------------------------------------
# Webhook field mapping
# ---------------------------------------------------------------------------


class TestWebhookMapping:
    def test_jira_field_mapping(self):
        # Spec §4 test scenario 10 — Jira event shape from the spec.
        payload = {
            "issue": {
                "customfield_10042": "vrd-jira-42",
                "resolution": {
                    "name": "escalate_to_review",
                    "description": "Operator override after triage",
                },
                "assignee": {"emailAddress": "alice@example.com"},
                "updated": "2026-05-06T14:23:00Z",
            }
        }
        mapping = {
            "decision_id": "issue.customfield_10042",
            "corrected_action": "issue.resolution.name",
            "reviewer": "issue.assignee.emailAddress",
            "reason": "issue.resolution.description",
            "timestamp": "issue.updated",
        }
        event = map_webhook_to_override(
            payload, mapping, defaults={"service": "fraud-detection"},
        )
        assert event.decision_id == "vrd-jira-42"
        assert event.corrected_action == "escalate_to_review"
        assert event.reviewer == "alice@example.com"
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.year == 2026

    def test_missing_required_path_raises(self):
        with pytest.raises(ValueError, match="reviewer"):
            map_webhook_to_override(
                {"decision_id": "x", "service": "y", "corrected_action": "z"},
                mapping={
                    "decision_id": "decision_id",
                    "service": "service",
                    "corrected_action": "corrected_action",
                    "reviewer": "missing.path",
                },
            )

    def test_naive_iso_timestamp_rejected(self):
        # Relabelling naive timestamps as UTC silently corrupts
        # reversal-rate windowing when the source's actual offset != 0.
        with pytest.raises(ValueError, match="tz-aware"):
            map_webhook_to_override(
                {"id": "vrd-1", "ts": "2026-05-06T14:23:00"},
                mapping={
                    "decision_id": "id",
                    "service": "service",
                    "corrected_action": "action",
                    "reviewer": "reviewer",
                    "timestamp": "ts",
                },
                defaults={
                    "service": "fraud-detection",
                    "corrected_action": "escalate",
                    "reviewer": "alice@example.com",
                },
            )

    def test_stringified_confidence_coerced(self):
        # Many webhook sources stringify floats; coercion at the
        # mapping boundary keeps the error surface ValueError-only.
        event = map_webhook_to_override(
            {"id": "vrd-1", "conf": "0.71"},
            mapping={
                "decision_id": "id",
                "confidence_at_decision": "conf",
            },
            defaults={
                "service": "fraud-detection",
                "corrected_action": "escalate",
                "reviewer": "alice@example.com",
            },
        )
        assert event.confidence_at_decision == 0.71

    def test_defaults_fill_absent_fields(self):
        payload = {"id": "vrd-99", "action": "deny"}
        mapping = {
            "decision_id": "id",
            "corrected_action": "action",
        }
        event = map_webhook_to_override(
            payload, mapping,
            defaults={
                "service": "fraud-detection",
                "reviewer": "system-reviewer",
            },
        )
        assert event.service == "fraud-detection"
        assert event.reviewer == "system-reviewer"
