"""Tests for NTHLAYER_LLM_STUB=canned canned-response mode.

Covers:
- llm_call() short-circuits to canned text indexed by detected role
- structured_call() returns canned validated Pydantic model
- structured_call_with_usage() returns canned StructuredCallResult
- env var unset preserves normal HTTP behaviour
- unknown structured response_model raises NotImplementedError
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from nthlayer_common.llm import llm_call
from nthlayer_common.llm_structured import (
    StructuredCallResult,
    structured_call,
    structured_call_with_usage,
)
from nthlayer_common.llm_stub import (
    _detect_role,
    is_stub_enabled,
    stub_structured_response,
    stub_text_response,
)

# --- Pydantic models matching the two known structured callers --------------

class DimensionScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class EvaluationResult(BaseModel):
    # Mirrors the real model in
    # nthlayer_workers.measure.pipeline.evaluator.EvaluationResult —
    # dimensions is a dict keyed by dimension name, not a list.
    dimensions: dict[str, DimensionScore]
    confidence: float = 0.0


class SnapshotSummary(BaseModel):
    summary: str = Field(max_length=500)
    notable_omissions: list[str] = Field(default_factory=list)


class UnregisteredModel(BaseModel):
    foo: str


# --- is_stub_enabled --------------------------------------------------------

class TestIsStubEnabled:
    def test_returns_true_when_set_to_canned(self, monkeypatch):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", "canned")
        assert is_stub_enabled() is True

    def test_returns_false_when_unset(self, monkeypatch):
        monkeypatch.delenv("NTHLAYER_LLM_STUB", raising=False)
        assert is_stub_enabled() is False

    def test_returns_false_when_other_value(self, monkeypatch):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", "other")
        assert is_stub_enabled() is False

    @pytest.mark.parametrize("value", ["CANNED", "Canned", " canned ", "canned\n"])
    def test_accepts_case_and_whitespace_variants(self, monkeypatch, value):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", value)
        assert is_stub_enabled() is True


# --- Role detection ---------------------------------------------------------

class TestDetectRole:
    @pytest.mark.parametrize(
        "system,expected",
        [
            ("You are a triage agent for incident response.", "triage"),
            ("You are a communication agent.", "communication"),
            ("You are an investigation agent.", "investigation"),
            ("You are a remediation agent.", "remediation"),
            ("Random text with no marker.", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_detect_role(self, system, expected):
        assert _detect_role(system) == expected

    def test_case_insensitive(self):
        assert _detect_role("YOU ARE A TRIAGE AGENT") == "triage"

    def test_none_system_returns_unknown(self):
        # Type says str, but a misbehaving caller should surface as "unknown"
        # rather than crashing with AttributeError.
        assert _detect_role(None) == "unknown"


# --- llm_call short-circuit ------------------------------------------------

class TestLLMCallStub:
    def test_triage_returns_canned_json(self, monkeypatch):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", "canned")
        with patch("nthlayer_common.llm.httpx.post") as mock_post:
            resp = llm_call("You are a triage agent.", "user prompt")
        mock_post.assert_not_called()
        assert resp.provider == "stub"
        body = json.loads(resp.text)
        assert body["severity"] == 1
        assert body["confidence"] == 0.85
        assert "fraud-detect" in body["blast_radius"]

    def test_remediation_returns_registry_action(self, monkeypatch):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", "canned")
        resp = llm_call("You are a remediation agent.", "user")
        body = json.loads(resp.text)
        assert body["proposed_action"] == "rollback"
        assert body["target"] == "fraud-detect"
        assert body["requires_human_approval"] is True

    def test_unknown_role_returns_generic(self, monkeypatch):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", "canned")
        resp = llm_call("opaque system prompt", "user")
        body = json.loads(resp.text)
        assert "reasoning" in body
        assert "confidence" in body

    def test_stub_disabled_makes_http_call(self, monkeypatch):
        monkeypatch.delenv("NTHLAYER_LLM_STUB", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        # If stub were enabled, httpx.post would not be called and this test
        # would fail at the stub-not-an-LLMResponse boundary. Instead we
        # assert httpx.post IS reached.
        import httpx
        mock_resp = httpx.Response(
            status_code=200,
            json={"content": [{"text": "real"}], "usage": {}},
            request=httpx.Request("POST", "https://mock"),
        )
        with patch("nthlayer_common.llm.httpx.post", return_value=mock_resp) as mock_post:
            llm_call("sys", "user", model="anthropic/claude-sonnet-4-20250514")
        mock_post.assert_called_once()

    def test_returned_model_falls_back_when_none(self, monkeypatch):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", "canned")
        # Direct call to stub_text_response with model=None — exercises the
        # "stub/canned" default since llm_call always passes a resolved model.
        resp = stub_text_response("You are a triage agent.", None)
        assert resp.model == "stub/canned"


# --- structured_call short-circuit -----------------------------------------

class TestStructuredCallStub:
    def test_evaluation_result_returns_canned(self, monkeypatch):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", "canned")
        result = structured_call("sys", "user", EvaluationResult)
        assert isinstance(result, EvaluationResult)
        assert len(result.dimensions) == 1
        assert "stub_dimension" in result.dimensions
        assert 0.0 <= result.dimensions["stub_dimension"].score <= 1.0
        assert result.confidence == 0.8

    def test_snapshot_summary_returns_canned(self, monkeypatch):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", "canned")
        result = structured_call("sys", "user", SnapshotSummary)
        assert isinstance(result, SnapshotSummary)
        assert len(result.summary) <= 500
        assert result.notable_omissions == []

    def test_unregistered_model_raises(self, monkeypatch):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", "canned")
        with pytest.raises(NotImplementedError, match="UnregisteredModel"):
            structured_call("sys", "user", UnregisteredModel)

    def test_direct_helper_unregistered(self):
        with pytest.raises(NotImplementedError, match="UnregisteredModel"):
            stub_structured_response(UnregisteredModel)

    def test_name_collision_with_incompatible_shape_raises_clearly(self):
        # A foreign class named "EvaluationResult" with a different required
        # shape would cause Pydantic to reject the canned dict. The stub
        # converts that into a NotImplementedError naming the qualified path.
        class EvaluationResult(BaseModel):  # noqa: F811 — intentional name reuse
            required_other_field: str

        with pytest.raises(NotImplementedError, match="does not match the shape"):
            stub_structured_response(EvaluationResult)


# --- structured_call_with_usage short-circuit ------------------------------

class TestStructuredCallWithUsageStub:
    def test_returns_structured_call_result(self, monkeypatch):
        monkeypatch.setenv("NTHLAYER_LLM_STUB", "canned")
        result = structured_call_with_usage("sys", "user", EvaluationResult)
        assert isinstance(result, StructuredCallResult)
        assert isinstance(result.data, EvaluationResult)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
