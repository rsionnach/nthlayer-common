"""Tests for the verdict bridge — building content-addressed Verdict records."""

import hashlib
from datetime import datetime, timezone

from nthlayer_common.records.hashing import verify_hash
from nthlayer_common.records.models import (
    ZERO_HASH,
    Verdict,
    VerdictOutcome,
)
from nthlayer_common.records.verdict_bridge import (
    build_decision_verdict,
    hash_content,
)

NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)


# --- hash_content ---


class TestHashContent:
    def test_returns_64_char_hex(self):
        h = hash_content("hello world")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_matches_sha256(self):
        h = hash_content("hello world")
        expected = hashlib.sha256("hello world".encode("utf-8")).hexdigest()
        assert h == expected

    def test_deterministic(self):
        assert hash_content("test") == hash_content("test")

    def test_different_input_different_hash(self):
        assert hash_content("a") != hash_content("b")


# --- build_decision_verdict ---


class TestBuildDecisionVerdict:
    def test_builds_valid_verdict(self):
        v = build_decision_verdict(
            agent="triage",
            incident_id="inc-001",
            timestamp=NOW,
            model="claude-sonnet-4-20250514",
            reasoning="High severity due to customer impact",
            action={"type": "escalate", "target": "payments-team"},
            outcome=VerdictOutcome.RECOMMENDED,
            prompt_text="You are a triage agent...",
            response_text='{"severity": 3, "reasoning": "..."}',
            input_hashes=[],
            previous_hash=ZERO_HASH,
            summaries_technical="Triage: P1 severity, payments-team assigned",
            summaries_plain="Incident triaged as high severity",
            summaries_executive="P1 incident triaged",
        )
        assert isinstance(v, Verdict)
        assert v.schema_version == "verdict/v1"
        assert v.agent == "triage"
        assert v.incident_id == "inc-001"
        assert v.model == "claude-sonnet-4-20250514"
        assert v.outcome == VerdictOutcome.RECOMMENDED

    def test_hash_is_valid(self):
        v = build_decision_verdict(
            agent="investigation",
            incident_id="inc-001",
            timestamp=NOW,
            model="claude-sonnet-4-20250514",
            reasoning="Root cause identified",
            action={"type": "identify_root_cause"},
            outcome=VerdictOutcome.RECOMMENDED,
            prompt_text="system prompt",
            response_text="response text",
            input_hashes=[],
            previous_hash=ZERO_HASH,
            summaries_technical="Investigation complete",
            summaries_plain="Root cause found",
            summaries_executive="Investigated",
        )
        assert verify_hash(v) is True

    def test_prompt_and_response_hashed(self):
        prompt = "You are a triage agent. Assess severity."
        response = '{"severity": 3}'
        v = build_decision_verdict(
            agent="triage",
            incident_id="inc-001",
            timestamp=NOW,
            model="test-model",
            reasoning="test",
            action={},
            outcome=VerdictOutcome.RECOMMENDED,
            prompt_text=prompt,
            response_text=response,
            input_hashes=[],
            previous_hash=ZERO_HASH,
            summaries_technical="t",
            summaries_plain="p",
            summaries_executive="e",
        )
        assert v.prompt_hash == hash_content(prompt)
        assert v.response_hash == hash_content(response)

    def test_input_hashes_preserved(self):
        v = build_decision_verdict(
            agent="remediation",
            incident_id="inc-001",
            timestamp=NOW,
            model="test-model",
            reasoning="rollback recommended",
            action={"type": "rollback"},
            outcome=VerdictOutcome.RECOMMENDED,
            prompt_text="p",
            response_text="r",
            input_hashes=["a" * 64, "b" * 64],
            previous_hash=ZERO_HASH,
            summaries_technical="t",
            summaries_plain="p",
            summaries_executive="e",
        )
        assert v.input_hashes == ["a" * 64, "b" * 64]

    def test_different_reasoning_different_hash(self):
        kwargs = dict(
            agent="triage",
            incident_id="inc-001",
            timestamp=NOW,
            model="test-model",
            action={},
            outcome=VerdictOutcome.RECOMMENDED,
            prompt_text="p",
            response_text="r",
            input_hashes=[],
            previous_hash=ZERO_HASH,
            summaries_technical="t",
            summaries_plain="p",
            summaries_executive="e",
        )
        v1 = build_decision_verdict(reasoning="reason A", **kwargs)
        v2 = build_decision_verdict(reasoning="reason B", **kwargs)
        assert v1.hash != v2.hash

    def test_summaries_truncated(self):
        v = build_decision_verdict(
            agent="triage",
            incident_id="inc-001",
            timestamp=NOW,
            model="test-model",
            reasoning="test",
            action={},
            outcome=VerdictOutcome.RECOMMENDED,
            prompt_text="p",
            response_text="r",
            input_hashes=[],
            previous_hash=ZERO_HASH,
            summaries_technical="x" * 300,
            summaries_plain="y" * 300,
            summaries_executive="z" * 200,
        )
        assert len(v.summaries.technical) == 280
        assert len(v.summaries.plain) == 280
        assert len(v.summaries.executive) == 140

    def test_chains_with_previous_hash(self):
        v1 = build_decision_verdict(
            agent="triage",
            incident_id="inc-001",
            timestamp=NOW,
            model="test-model",
            reasoning="first",
            action={},
            outcome=VerdictOutcome.RECOMMENDED,
            prompt_text="p1",
            response_text="r1",
            input_hashes=[],
            previous_hash=ZERO_HASH,
            summaries_technical="t",
            summaries_plain="p",
            summaries_executive="e",
        )
        v2 = build_decision_verdict(
            agent="triage",
            incident_id="inc-001",
            timestamp=datetime(2026, 4, 11, 12, 5, 0, tzinfo=timezone.utc),
            model="test-model",
            reasoning="second",
            action={},
            outcome=VerdictOutcome.RECOMMENDED,
            prompt_text="p2",
            response_text="r2",
            input_hashes=[v1.hash],
            previous_hash=v1.hash,
            summaries_technical="t2",
            summaries_plain="p2",
            summaries_executive="e2",
        )
        assert v2.previous_hash == v1.hash
        assert v1.hash in v2.input_hashes
