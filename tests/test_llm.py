# tests/test_llm.py
"""Unit tests for the unified LLM wrapper."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from nthlayer_common.llm import (
    LLMError,
    LLMResponse,
    _guess_provider,
    llm_call,
)


def _mock_response(body: dict, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("POST", "https://mock"),
    )
    return resp


class TestAnthropicPath:
    def test_successful_call(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        mock_resp = _mock_response({"content": [{"text": "hello from claude"}]})

        with patch("nthlayer_common.llm.httpx.post", return_value=mock_resp) as mock_post:
            result = llm_call("system", "user", model="anthropic/claude-sonnet-4-20250514")

        assert result.text == "hello from claude"
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-20250514"

        call_args = mock_post.call_args
        assert "api.anthropic.com/v1/messages" in call_args.args[0]
        assert call_args.kwargs["headers"]["x-api-key"] == "sk-ant-test"
        assert call_args.kwargs["headers"]["anthropic-version"] == "2023-06-01"


class TestOpenAIPath:
    def test_successful_call(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_resp = _mock_response({"choices": [{"message": {"content": "hello from gpt"}}]})

        with patch("nthlayer_common.llm.httpx.post", return_value=mock_resp) as mock_post:
            result = llm_call("system", "user", model="openai/gpt-4o")

        assert result.text == "hello from gpt"
        assert result.provider == "openai"
        assert result.model == "gpt-4o"

        call_args = mock_post.call_args
        assert "api.openai.com/v1/chat/completions" in call_args.args[0]
        assert "Bearer sk-test" in call_args.kwargs["headers"]["Authorization"]


class TestOllamaPath:
    def test_correct_url(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        mock_resp = _mock_response({"choices": [{"message": {"content": "local response"}}]})

        with patch("nthlayer_common.llm.httpx.post", return_value=mock_resp) as mock_post:
            result = llm_call("system", "user", model="ollama/llama3.1")

        call_args = mock_post.call_args
        assert "localhost:11434/v1/chat/completions" in call_args.args[0]
        assert result.provider == "ollama"


class TestCustomBaseURL:
    def test_openai_api_base_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_BASE", "http://custom:1234/v1")
        mock_resp = _mock_response({"choices": [{"message": {"content": "custom"}}]})

        with patch("nthlayer_common.llm.httpx.post", return_value=mock_resp) as mock_post:
            result = llm_call("system", "user", model="custom/my-model")

        call_args = mock_post.call_args
        assert "http://custom:1234/v1/chat/completions" == call_args.args[0]


class TestMissingAPIKey:
    def test_anthropic_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY not set"):
            llm_call("system", "user", model="anthropic/claude-sonnet-4-20250514")


class TestTimeout:
    def test_timeout_raises_llm_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        with patch("nthlayer_common.llm.httpx.post", side_effect=httpx.TimeoutException("timed out")):
            with pytest.raises(LLMError, match="Timeout"):
                llm_call("system", "user", model="anthropic/claude-sonnet-4-20250514", timeout=5)


class TestHTTPError:
    def test_429_raises_llm_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        resp = httpx.Response(
            status_code=429,
            text="Rate limit exceeded",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        with patch("nthlayer_common.llm.httpx.post", return_value=resp):
            with pytest.raises(LLMError, match="HTTP 429"):
                llm_call("system", "user", model="openai/gpt-4o")


class TestGuessProvider:
    def test_claude_is_anthropic(self):
        assert _guess_provider("claude-sonnet-4-20250514") == "anthropic"

    def test_gpt_is_openai(self):
        assert _guess_provider("gpt-4o") == "openai"

    def test_o_series_is_openai(self):
        assert _guess_provider("o1-preview") == "openai"
        assert _guess_provider("o3-mini") == "openai"

    def test_llama_is_ollama(self):
        assert _guess_provider("llama3.1") == "ollama"

    def test_unknown_defaults_to_openai(self):
        assert _guess_provider("some-unknown-model") == "openai"


class TestLLMResponseFields:
    def test_all_fields_populated(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_resp = _mock_response({"choices": [{"message": {"content": "test output"}}]})

        with patch("nthlayer_common.llm.httpx.post", return_value=mock_resp):
            result = llm_call("system", "user", model="openai/gpt-4o")

        assert isinstance(result, LLMResponse)
        assert result.text == "test output"
        assert result.model == "gpt-4o"
        assert result.provider == "openai"
