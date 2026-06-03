"""Tests for core verdict operations."""

import pytest

from nthlayer_common.verdicts.core import create, link, resolve, supersede


def _make_verdict(**overrides):
    """Helper to create a verdict with sensible defaults."""
    kwargs = dict(
        subject={"type": "review", "ref": "git:abc123", "summary": "Test change"},
        judgment={"action": "approve", "confidence": 0.8},
        producer={"system": "test-producer"},
    )
    kwargs.update(overrides)
    return create(**kwargs)


class TestCreate:
    def test_creates_verdict_with_pending_outcome(self):
        v = _make_verdict()
        assert v.id.startswith("vrd-")
        assert v.version == 1
        assert v.outcome.status == "pending"
        assert v.outcome.closed_at is None

    def test_creates_verdict_from_dicts(self):
        v = _make_verdict()
        assert v.producer.system == "test-producer"
        assert v.subject.type == "review"
        assert v.judgment.action == "approve"
        assert v.judgment.confidence == 0.8

    def test_creates_verdict_from_dataclasses(self):
        from nthlayer_common.verdicts.models import Judgment, Producer, Subject
        v = create(
            subject=Subject(type="review", ref="ref", summary="s"),
            judgment=Judgment(action="approve", confidence=0.5),
            producer=Producer(system="test"),
        )
        assert v.producer.system == "test"

    def test_creates_unique_ids(self):
        ids = {_make_verdict().id for _ in range(100)}
        assert len(ids) == 100

    def test_default_metadata(self):
        v = _make_verdict()
        assert v.metadata.ttl == 90 * 24 * 60 * 60
        assert v.metadata.custom == {}

    def test_custom_metadata(self):
        v = _make_verdict(metadata={"cost_tokens": 500, "latency_ms": 100})
        assert v.metadata.cost_tokens == 500
        assert v.metadata.latency_ms == 100

    def test_validates_subject_type(self):
        with pytest.raises(ValueError, match="Invalid subject type"):
            _make_verdict(subject={"type": "invalid", "ref": "r", "summary": "s"})

    def test_validates_action(self):
        with pytest.raises(ValueError, match="Invalid action"):
            _make_verdict(judgment={"action": "nope", "confidence": 0.5})

    def test_validates_confidence_range(self):
        with pytest.raises(ValueError, match="Confidence must be"):
            _make_verdict(judgment={"action": "approve", "confidence": 5.0})


class TestLink:
    def test_sets_parent(self):
        v = _make_verdict()
        v = link(v, parent="vrd-parent-001")
        assert v.lineage.parent == "vrd-parent-001"

    def test_sets_context(self):
        v = _make_verdict()
        v = link(v, context=["vrd-ctx-001", "vrd-ctx-002"])
        assert v.lineage.context == ["vrd-ctx-001", "vrd-ctx-002"]

    def test_sets_both(self):
        v = _make_verdict()
        v = link(v, parent="vrd-parent", context=["vrd-ctx"])
        assert v.lineage.parent == "vrd-parent"
        assert v.lineage.context == ["vrd-ctx"]

    def test_none_values_do_not_overwrite(self):
        v = _make_verdict()
        v = link(v, parent="vrd-parent")
        v = link(v, context=["vrd-ctx"])
        assert v.lineage.parent == "vrd-parent"


class TestResolve:
    def test_resolve_confirmed(self):
        v = _make_verdict()
        v = resolve(v, status="confirmed")
        assert v.outcome.status == "confirmed"
        assert v.outcome.closed_at is not None

    def test_resolve_overridden_with_override(self):
        v = _make_verdict()
        v = resolve(
            v, status="overridden",
            override={"by": "human:rob", "action": "reject", "reasoning": "missed bug"},
        )
        assert v.outcome.status == "overridden"
        assert v.outcome.override.by == "human:rob"
        assert v.outcome.override.action == "reject"
        assert v.outcome.override.at is not None

    def test_resolve_with_ground_truth(self):
        v = _make_verdict()
        v = resolve(
            v, status="overridden",
            ground_truth={"signal": "test_failure", "value": "reject"},
        )
        assert v.outcome.ground_truth.signal == "test_failure"

    def test_resolve_with_resolution_text(self):
        v = _make_verdict()
        v = resolve(v, status="confirmed", resolution="Deployed without issues")
        assert v.outcome.resolution == "Deployed without issues"

    def test_invalid_status_raises(self):
        v = _make_verdict()
        with pytest.raises(ValueError, match="Invalid status 'nope'"):
            resolve(v, status="nope")

    def test_double_resolve_raises(self):
        v = _make_verdict()
        v = resolve(v, status="confirmed")
        with pytest.raises(ValueError, match="expected 'pending'"):
            resolve(v, status="overridden")

    def test_all_valid_statuses(self):
        for status in ("confirmed", "overridden", "partial", "superseded", "expired"):
            v = _make_verdict()
            v = resolve(v, status=status)
            assert v.outcome.status == status


class TestSupersede:
    def test_supersede_links_verdicts(self):
        old = _make_verdict()
        new = _make_verdict()
        old, new = supersede(old, new)
        assert old.outcome.status == "superseded"
        assert new.lineage.parent == old.id
        assert new.id in old.lineage.children

    def test_self_supersede_raises(self):
        v = _make_verdict()
        with pytest.raises(ValueError, match="Cannot supersede a verdict with itself"):
            supersede(v, v)

    def test_supersede_already_resolved_raises(self):
        old = _make_verdict()
        old = resolve(old, status="confirmed")
        new = _make_verdict()
        with pytest.raises(ValueError, match="expected 'pending'"):
            supersede(old, new)
