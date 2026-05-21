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
    OutcomeStatusMismatch,
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


def _make_pending_verdict(decision_id: str) -> "Verdict":
    return _verdict(vid=decision_id, service="fraud-detect")


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

    def test_to_dict_canonical_wire_shape(self) -> None:
        """opensrm-jmy.18: to_dict produces a JSON-serializable canonical dict."""
        event = OverrideEvent(
            decision_id="dec-1",
            service="fraud-detect",
            corrected_action="approve",
            reviewer="reviewer-hash",
            original_action="reject",
            reason="false positive",
            confidence_at_decision=0.92,
            source_system="slack-adapter",
            timestamp=datetime(2026, 5, 20, 10, 33, 0, tzinfo=timezone.utc),
        )
        body = event.to_dict()
        assert body == {
            "decision_id": "dec-1",
            "service": "fraud-detect",
            "corrected_action": "approve",
            "reviewer": "reviewer-hash",
            "original_action": "reject",
            "reason": "false positive",
            "confidence_at_decision": 0.92,
            "source_system": "slack-adapter",
            "timestamp": "2026-05-20T10:33:00+00:00",
        }
        import json
        assert json.dumps(body)  # raises on non-serialisable

    def test_to_dict_drops_none_optional_fields(self) -> None:
        event = OverrideEvent(
            decision_id="dec-2",
            service="fraud-detect",
            corrected_action="approve",
            reviewer="reviewer-hash",
        )
        body = event.to_dict()
        assert "original_action" not in body
        assert "reason" not in body
        assert "confidence_at_decision" not in body
        assert "source_system" not in body
        assert "timestamp" in body  # timestamp is always present (default utcnow)


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


# ---------------------------------------------------------------------------
# opensrm-jmy.11 correctness fixes (surfaced by /r5-supervise dry-run on jmy.4)
# ---------------------------------------------------------------------------


class TestJmy11ConcurrencyRace:
    """Verdict-store CAS prevents last-writer-wins on simultaneous overrides."""

    def test_update_outcome_cas_raises_on_mismatch(self):
        store = MemoryStore()
        store.put(_verdict())
        new_outcome = Outcome(status="overridden", resolution=None)
        # Actual is "pending"; claim "overridden" → mismatch.
        with pytest.raises(OutcomeStatusMismatch):
            store.update_outcome(
                "vrd-test-1", new_outcome, expected_status="overridden",
            )
        # Matching expected_status succeeds.
        result = store.update_outcome(
            "vrd-test-1", new_outcome, expected_status="pending",
        )
        assert result.outcome.status == "overridden"

    def test_update_outcome_without_expected_status_is_unconditional(self):
        # Backward compat: default expected_status=None preserves
        # the prior last-writer-wins contract for non-override callers.
        store = MemoryStore()
        store.put(_verdict())
        result = store.update_outcome(
            "vrd-test-1", Outcome(status="overridden"),
        )
        assert result.outcome.status == "overridden"

    def test_sqlite_cas_rejects_stale_pending_write(self, tmp_path):
        # SQLite is the production store: deserialise-per-call avoids the
        # in-process aliasing MemoryStore exhibits, so this is the actual
        # race we're protecting against. Simulate the interleave: an
        # unconditional first write (status pending → overridden), then a
        # second CAS write that still believes status is pending.
        from nthlayer_common.verdicts import SQLiteVerdictStore

        db_path = str(tmp_path / "verdicts.db")
        store = SQLiteVerdictStore(db_path)
        try:
            store.put(_verdict())
            store.update_outcome(
                "vrd-test-1", Outcome(status="overridden"),
            )
            with pytest.raises(OutcomeStatusMismatch):
                store.update_outcome(
                    "vrd-test-1",
                    Outcome(status="overridden"),
                    expected_status="pending",
                )
        finally:
            store.close()

    def test_sqlite_cas_against_deleted_row_reports_deleted(self, tmp_path):
        # Edge case: a concurrent delete between our SELECT and our
        # CAS UPDATE leaves rowcount==0 and the followup diagnostic
        # SELECT returns None. The error message should distinguish
        # this from a status transition so operators don't chase a
        # phantom race when the verdict was actually removed.
        from nthlayer_common.verdicts import SQLiteVerdictStore

        db_path = str(tmp_path / "verdicts.db")
        store = SQLiteVerdictStore(db_path)
        try:
            store.put(_verdict())
            # Simulate a concurrent delete by dropping the row out of
            # band, then attempting a CAS on the now-missing row.
            conn = store._conn()
            conn.execute("DELETE FROM verdicts WHERE id = ?", ("vrd-test-1",))
            conn.commit()
            with pytest.raises(KeyError, match="vrd-test-1"):
                store.update_outcome(
                    "vrd-test-1",
                    Outcome(status="overridden"),
                    expected_status="pending",
                )
        finally:
            store.close()

    def test_resolve_path_non_dict_intermediate_returns_absent(self):
        # Defensive branch: a dotted path that walks through a
        # non-dict value (e.g. a string) must return _ABSENT, not
        # crash with TypeError. Verified via the public boundary —
        # missing required field reports cleanly.
        with pytest.raises(ValueError, match="decision_id"):
            map_webhook_to_override(
                {"a": "not-a-dict"},
                mapping={"decision_id": "a.b.c"},
                defaults={
                    "service": "fraud-detection",
                    "corrected_action": "escalate",
                    "reviewer": "alice@example.com",
                },
            )

    def test_apply_override_returns_none_when_cas_fails(self):
        # apply_override_to_verdict's contract: a CAS miss (concurrent
        # writer slipped in between our read and our update) surfaces as
        # None + a lost_race_to_concurrent_writer log, not a silent
        # last-writer-wins overwrite.
        class RacyStore(MemoryStore):
            def update_outcome(self, verdict_id, outcome, *, expected_status=None):
                raise OutcomeStatusMismatch(
                    f"simulated concurrent writer for {verdict_id}"
                )

        store = RacyStore()
        store.put(_verdict())
        result = apply_override_to_verdict(store, _event())
        assert result is None


class TestJmy11ResolvePathSentinel:
    """_resolve_path distinguishes explicit null from absent."""

    def test_explicit_null_required_field_raises_clearly(self):
        with pytest.raises(ValueError, match="decision_id"):
            map_webhook_to_override(
                {"id": None},
                mapping={"decision_id": "id"},
                defaults={
                    "service": "fraud-detection",
                    "corrected_action": "escalate",
                    "reviewer": "alice@example.com",
                },
            )

    def test_explicit_null_optional_field_preserved_not_collapsed(self):
        # Before the fix, a payload setting confidence_at_decision to
        # JSON null was indistinguishable from absence and silently
        # disabled HCF accounting. The sentinel preserves the explicit
        # null, which OverrideEvent then keeps as None (HCF off, but
        # the operator's intent is no longer hidden behind a default).
        event = map_webhook_to_override(
            {"id": "vrd-1", "conf": None},
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
        assert event.confidence_at_decision is None
        assert not event.is_high_confidence_failure


class TestJmy11EmptyStringRequiredField:
    """Empty-string required field gets the canonical 'is required' message."""

    def test_empty_string_required_field_message_names_field(self):
        # Pre-fix: "could not be resolved" (misleading — value WAS
        # resolved, just to ""). Post-fix: the empty string passes the
        # mapping layer and OverrideEvent.__post_init__ surfaces the
        # canonical "is required (spec § 4)" message.
        with pytest.raises(ValueError, match=r"decision_id is required"):
            map_webhook_to_override(
                {"id": ""},
                mapping={"decision_id": "id"},
                defaults={
                    "service": "fraud-detection",
                    "corrected_action": "escalate",
                    "reviewer": "alice@example.com",
                },
            )


class TestPreRedactedFlag:
    """opensrm-jmy.18: pre_redacted flag with plaintext_reviewer as deprecated alias."""

    def test_pre_redacted_true_skips_reviewer_hashing(self) -> None:
        event = OverrideEvent(
            decision_id="dec-1",
            service="fraud-detect",
            corrected_action="approve",
            reviewer="already-hashed-hex",
        )
        privacy = OverridePrivacyConfig(pre_redacted=True)
        store = MemoryStore()
        store.put(_make_pending_verdict("dec-1"))

        result = apply_override_to_verdict(store, event, privacy=privacy)

        assert result is not None
        assert result.outcome.override.by == "already-hashed-hex"

    def test_plaintext_reviewer_remains_an_alias(self) -> None:
        event = OverrideEvent(
            decision_id="dec-2",
            service="fraud-detect",
            corrected_action="approve",
            reviewer="already-hashed-hex",
        )
        store_a = MemoryStore()
        store_a.put(_make_pending_verdict("dec-2"))
        store_b = MemoryStore()
        store_b.put(_make_pending_verdict("dec-2"))

        r_pre = apply_override_to_verdict(
            store_a, event, privacy=OverridePrivacyConfig(pre_redacted=True),
        )
        r_plain = apply_override_to_verdict(
            store_b, event, privacy=OverridePrivacyConfig(plaintext_reviewer=True),
        )

        assert r_pre is not None and r_plain is not None
        assert r_pre.outcome.override.by == r_plain.outcome.override.by == "already-hashed-hex"

    def test_both_flags_set_together_behaves_identically(self) -> None:
        event = OverrideEvent(
            decision_id="dec-3",
            service="fraud-detect",
            corrected_action="approve",
            reviewer="already-hashed-hex",
        )
        store = MemoryStore()
        store.put(_make_pending_verdict("dec-3"))

        result = apply_override_to_verdict(
            store, event,
            privacy=OverridePrivacyConfig(pre_redacted=True, plaintext_reviewer=True),
        )

        assert result is not None
        assert result.outcome.override.by == "already-hashed-hex"


class TestJmy11EmptyStringOptionalNormalisation:
    """Empty-string optional fields normalise to None on OverrideEvent."""

    def test_empty_string_optional_fields_become_none(self):
        event = _event(reason="", original_action="", source_system="")
        assert event.reason is None
        assert event.original_action is None
        assert event.source_system is None

    def test_empty_string_replay_is_idempotent_not_conflict(self):
        # An upstream webhook that started sending reason="" instead of
        # omitting the field would, pre-fix, flip a legitimate replay
        # from idempotent no-op into override_conflicts_with_existing.
        store = MemoryStore()
        store.put(_verdict())

        first = apply_override_to_verdict(store, _event(reason=None))
        assert first is not None
        assert first.outcome.override is not None
        assert first.outcome.override.reasoning is None

        second = apply_override_to_verdict(store, _event(reason=""))
        assert second is not None, "empty-string replay must be idempotent"
        assert second.outcome.status == "overridden"
        assert second.outcome.override == first.outcome.override
