"""Tests for structured LLM call via Instructor."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from nthlayer_common.llm import LLMError
from nthlayer_common.llm_structured import _parse_model, structured_call


class EvalResult(BaseModel):
    outcome: str
    confidence: float
    reasoning: str


class TestParseModel:
    def test_explicit_provider(self):
        assert _parse_model("anthropic/claude-sonnet-4-20250514") == (
            "anthropic",
            "claude-sonnet-4-20250514",
        )

    def test_openai_provider(self):
        assert _parse_model("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_bare_claude_guesses_anthropic(self):
        assert _parse_model("claude-sonnet-4-20250514") == (
            "anthropic",
            "claude-sonnet-4-20250514",
        )

    def test_bare_gpt_guesses_openai(self):
        assert _parse_model("gpt-4o") == ("openai", "gpt-4o")

    def test_unknown_defaults_to_openai(self):
        assert _parse_model("llama-3.1") == ("openai", "llama-3.1")


class TestAPIKeyGuard:
    def test_rejects_anthropic_key(self):
        with pytest.raises(LLMError, match="looks like an API key"):
            structured_call("sys", "user", EvalResult, model="sk-ant-abc123")

    def test_rejects_openai_key(self):
        with pytest.raises(LLMError, match="looks like an API key"):
            structured_call("sys", "user", EvalResult, model="sk-proj-abc123")

    def test_rejects_bearer_token(self):
        with pytest.raises(LLMError, match="looks like an API key"):
            structured_call("sys", "user", EvalResult, model="Bearer abc123")


class TestStructuredCallAnthropic:
    @patch("nthlayer_common.llm_structured._get_anthropic_client")
    def test_returns_validated_model(self, mock_get_client):
        expected = EvalResult(outcome="success", confidence=0.95, reasoning="Test")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = expected
        mock_get_client.return_value = mock_client

        result = structured_call(
            "You are an evaluator",
            "Evaluate this",
            EvalResult,
            model="anthropic/claude-sonnet-4-20250514",
        )

        assert isinstance(result, EvalResult)
        assert result.outcome == "success"
        assert result.confidence == 0.95
        mock_client.messages.create.assert_called_once()

    @patch("nthlayer_common.llm_structured._get_anthropic_client")
    def test_passes_max_retries(self, mock_get_client):
        expected = EvalResult(outcome="ok", confidence=0.8, reasoning="r")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = expected
        mock_get_client.return_value = mock_client

        structured_call("s", "u", EvalResult, model="anthropic/claude-sonnet-4-20250514", max_retries=5)

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("max_retries") == 5 or call_kwargs[1].get("max_retries") == 5


class TestStructuredCallOpenAI:
    @patch("nthlayer_common.llm_structured._get_openai_client")
    def test_returns_validated_model(self, mock_get_client):
        expected = EvalResult(outcome="partial", confidence=0.6, reasoning="Partial match")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = expected
        mock_get_client.return_value = mock_client

        result = structured_call(
            "You are an evaluator",
            "Evaluate this",
            EvalResult,
            model="openai/gpt-4o",
        )

        assert isinstance(result, EvalResult)
        assert result.outcome == "partial"
        mock_client.chat.completions.create.assert_called_once()


class TestErrorWrapping:
    @patch("nthlayer_common.llm_structured._get_anthropic_client")
    def test_wraps_provider_error_in_llm_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMError, match="Structured call failed"):
            structured_call("s", "u", EvalResult, model="anthropic/claude-sonnet-4-20250514")
