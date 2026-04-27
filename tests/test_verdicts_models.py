"""Tests for verdict data models and validation."""

import math

import pytest

from nthlayer_common.verdicts.models import Judgment, Subject, VALID_ACTIONS, VALID_SUBJECT_TYPES


class TestSubjectValidation:
    def test_valid_subject_types(self):
        for t in VALID_SUBJECT_TYPES:
            s = Subject(type=t, ref="ref", summary="summary")
            assert s.type == t

    def test_invalid_subject_type_raises(self):
        with pytest.raises(ValueError, match="Invalid subject type 'banana'"):
            Subject(type="banana", ref="ref", summary="summary")


class TestJudgmentValidation:
    def test_valid_actions(self):
        for action in VALID_ACTIONS:
            j = Judgment(action=action, confidence=0.5)
            assert j.action == action

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="Invalid action 'yolo'"):
            Judgment(action="yolo", confidence=0.5)

    def test_confidence_must_be_between_0_and_1(self):
        Judgment(action="approve", confidence=0.0)
        Judgment(action="approve", confidence=1.0)
        Judgment(action="approve", confidence=0.5)

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValueError, match="Confidence must be between"):
            Judgment(action="approve", confidence=-0.1)

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValueError, match="Confidence must be between"):
            Judgment(action="approve", confidence=1.1)

    def test_confidence_nan_raises(self):
        with pytest.raises(ValueError, match="Confidence must be between"):
            Judgment(action="approve", confidence=math.nan)

    def test_score_must_be_between_0_and_1(self):
        Judgment(action="approve", confidence=0.5, score=0.0)
        Judgment(action="approve", confidence=0.5, score=1.0)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Score must be between"):
            Judgment(action="approve", confidence=0.5, score=1.5)

    def test_score_none_is_valid(self):
        j = Judgment(action="approve", confidence=0.5, score=None)
        assert j.score is None

    def test_dimension_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Dimension 'safety'"):
            Judgment(action="approve", confidence=0.5, dimensions={"safety": 2.0})

    def test_valid_dimensions(self):
        j = Judgment(
            action="approve", confidence=0.5,
            dimensions={"correctness": 0.9, "safety": 0.8},
        )
        assert j.dimensions["correctness"] == 0.9
