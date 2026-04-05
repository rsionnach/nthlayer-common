# tests/test_llm.py
"""Unit tests for the unified LLM wrapper."""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from nthlayer_common.llm import (
    LLMError,
    LLMResponse,
    _guess_provider,
    _is_transient,
    _parse_retry_after,
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
        mock_resp = _mock_response({
            "content": [{"text": "hello from claude"}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        })

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
        mock_resp = _mock_response({
            "choices": [{"message": {"content": "hello from gpt"}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 40},
        })

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
            llm_call("system", "user", model="custom/my-model")

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
                llm_call("system", "user", model="anthropic/claude-sonnet-4-20250514", timeout=5, retry=0)


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
                llm_call("system", "user", model="openai/gpt-4o", retry=0)


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
        mock_resp = _mock_response({
            "choices": [{"message": {"content": "test output"}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 40},
        })

        with patch("nthlayer_common.llm.httpx.post", return_value=mock_resp):
            result = llm_call("system", "user", model="openai/gpt-4o")

        assert isinstance(result, LLMResponse)
        assert result.text == "test output"
        assert result.model == "gpt-4o"
        assert result.provider == "openai"
        assert result.input_tokens == 80
        assert result.output_tokens == 40

    def test_token_counts_from_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        mock_resp = _mock_response({
            "content": [{"text": "hello"}],
            "usage": {"input_tokens": 150, "output_tokens": 75},
        })

        with patch("nthlayer_common.llm.httpx.post", return_value=mock_resp):
            result = llm_call("system", "user", model="anthropic/claude-sonnet-4-20250514")

        assert result.input_tokens == 150
        assert result.output_tokens == 75

    def test_missing_usage_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_resp = _mock_response({"choices": [{"message": {"content": "no usage"}}]})

        with patch("nthlayer_common.llm.httpx.post", return_value=mock_resp):
            result = llm_call("system", "user", model="openai/gpt-4o")

        assert result.input_tokens is None
        assert result.output_tokens is None


class TestStatusCodeClassification:
    def test_429_is_transient(self):
        assert _is_transient(429) is True

    def test_502_is_transient(self):
        assert _is_transient(502) is True

    def test_503_is_transient(self):
        assert _is_transient(503) is True

    def test_408_is_transient(self):
        assert _is_transient(408) is True

    def test_401_is_permanent(self):
        assert _is_transient(401) is False

    def test_400_is_permanent(self):
        assert _is_transient(400) is False

    def test_403_is_permanent(self):
        assert _is_transient(403) is False

    def test_404_is_permanent(self):
        assert _is_transient(404) is False

    def test_422_is_permanent(self):
        assert _is_transient(422) is False

    def test_500_is_transient(self):
        """500 assumed transient — some providers return 500 for internal errors."""
        assert _is_transient(500) is True

    def test_504_is_transient(self):
        assert _is_transient(504) is True

    def test_200_is_not_transient(self):
        assert _is_transient(200) is False


class TestParseRetryAfter:
    def test_integer_seconds(self):
        resp = httpx.Response(
            status_code=429, text="",
            headers={"Retry-After": "3"},
            request=httpx.Request("POST", "https://mock"),
        )
        assert _parse_retry_after(resp) == 3.0

    def test_missing_header_returns_zero(self):
        resp = httpx.Response(
            status_code=429, text="",
            request=httpx.Request("POST", "https://mock"),
        )
        assert _parse_retry_after(resp) == 0.0

    def test_invalid_header_returns_zero(self):
        resp = httpx.Response(
            status_code=429, text="",
            headers={"Retry-After": "not-a-number"},
            request=httpx.Request("POST", "https://mock"),
        )
        assert _parse_retry_after(resp) == 0.0

    def test_float_seconds(self):
        resp = httpx.Response(
            status_code=429, text="",
            headers={"Retry-After": "1.5"},
            request=httpx.Request("POST", "https://mock"),
        )
        assert _parse_retry_after(resp) == 1.5


class TestRetryTransient:
    def test_429_retries_then_succeeds(self, monkeypatch):
        """429 twice then 200 — llm_call returns successfully."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fail_resp = httpx.Response(
            status_code=429, text="Rate limited",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        ok_resp = _mock_response({
            "choices": [{"message": {"content": "finally"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
        call_count = {"n": 0}

        def mock_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise httpx.HTTPStatusError("429", request=fail_resp.request, response=fail_resp)
            return ok_resp

        with patch("nthlayer_common.llm.httpx.post", side_effect=mock_post):
            with patch("nthlayer_common.llm.time.sleep"):
                result = llm_call("system", "user", model="openai/gpt-4o", retry=3)

        assert result.text == "finally"
        assert call_count["n"] == 3

    def test_permanent_401_fails_immediately(self, monkeypatch):
        """401 raises LLMError immediately — no retry."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        resp = httpx.Response(
            status_code=401, text="Unauthorized",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        call_count = {"n": 0}

        def mock_post(*args, **kwargs):
            call_count["n"] += 1
            raise httpx.HTTPStatusError("401", request=resp.request, response=resp)

        with patch("nthlayer_common.llm.httpx.post", side_effect=mock_post):
            with pytest.raises(LLMError, match="HTTP 401"):
                llm_call("system", "user", model="openai/gpt-4o", retry=3)

        assert call_count["n"] == 1

    def test_retry_exhaustion_raises(self, monkeypatch):
        """503 on every attempt — raises LLMError after all retries."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        resp = httpx.Response(
            status_code=503, text="Service Unavailable",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        def mock_post(*args, **kwargs):
            raise httpx.HTTPStatusError("503", request=resp.request, response=resp)

        with patch("nthlayer_common.llm.httpx.post", side_effect=mock_post):
            with patch("nthlayer_common.llm.time.sleep"):
                with pytest.raises(LLMError, match="HTTP 503"):
                    llm_call("system", "user", model="openai/gpt-4o", retry=2)

    def test_retry_zero_disables_retry(self, monkeypatch):
        """retry=0 raises on first transient failure."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        resp = httpx.Response(
            status_code=429, text="Rate limited",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        call_count = {"n": 0}

        def mock_post(*args, **kwargs):
            call_count["n"] += 1
            raise httpx.HTTPStatusError("429", request=resp.request, response=resp)

        with patch("nthlayer_common.llm.httpx.post", side_effect=mock_post):
            with pytest.raises(LLMError, match="HTTP 429"):
                llm_call("system", "user", model="openai/gpt-4o", retry=0)

        assert call_count["n"] == 1

    def test_connect_error_retries(self, monkeypatch):
        """ConnectError is transient — retries then succeeds."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        ok_resp = _mock_response({
            "choices": [{"message": {"content": "recovered"}}],
        })
        call_count = {"n": 0}

        def mock_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.ConnectError("Connection refused")
            return ok_resp

        with patch("nthlayer_common.llm.httpx.post", side_effect=mock_post):
            with patch("nthlayer_common.llm.time.sleep"):
                result = llm_call("system", "user", model="openai/gpt-4o", retry=2)

        assert result.text == "recovered"
        assert call_count["n"] == 2

    def test_timeout_retries(self, monkeypatch):
        """TimeoutException is transient — retries then succeeds."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        ok_resp = _mock_response({
            "content": [{"text": "recovered"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        call_count = {"n": 0}

        def mock_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.TimeoutException("timed out")
            return ok_resp

        with patch("nthlayer_common.llm.httpx.post", side_effect=mock_post):
            with patch("nthlayer_common.llm.time.sleep"):
                result = llm_call("system", "user", model="anthropic/claude-sonnet-4-20250514", retry=2)

        assert result.text == "recovered"


class TestRetryAfterRespected:
    def test_retry_after_header_used_as_floor(self, monkeypatch):
        """Retry-After header sets minimum delay."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fail_resp = httpx.Response(
            status_code=429, text="Rate limited",
            headers={"Retry-After": "5"},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        ok_resp = _mock_response({"choices": [{"message": {"content": "ok"}}]})
        call_count = {"n": 0}

        def mock_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.HTTPStatusError("429", request=fail_resp.request, response=fail_resp)
            return ok_resp

        sleep_times = []

        def mock_sleep(seconds):
            sleep_times.append(seconds)

        with patch("nthlayer_common.llm.httpx.post", side_effect=mock_post):
            with patch("nthlayer_common.llm.time.sleep", side_effect=mock_sleep):
                result = llm_call("system", "user", model="openai/gpt-4o", retry=2, timeout=30)

        assert result.text == "ok"
        assert sleep_times[0] >= 5.0


class TestTimeoutBudget:
    def test_skips_retry_when_budget_exhausted(self, monkeypatch):
        """Don't retry if remaining timeout is less than backoff delay."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        resp = httpx.Response(
            status_code=429, text="Rate limited",
            headers={"Retry-After": "60"},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        def mock_post(*args, **kwargs):
            raise httpx.HTTPStatusError("429", request=resp.request, response=resp)

        with patch("nthlayer_common.llm.httpx.post", side_effect=mock_post):
            with pytest.raises(LLMError, match="429"):
                llm_call("system", "user", model="openai/gpt-4o", retry=3, timeout=5)


class TestLLMErrorStatusCode:
    def test_status_code_on_permanent_error(self, monkeypatch):
        """LLMError carries status_code for HTTP errors."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        resp = httpx.Response(
            status_code=401, text="Unauthorized",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        with patch("nthlayer_common.llm.httpx.post", return_value=resp):
            with pytest.raises(LLMError) as exc_info:
                llm_call("system", "user", model="openai/gpt-4o", retry=0)

        assert exc_info.value.status_code == 401

    def test_status_code_on_transient_exhaustion(self, monkeypatch):
        """LLMError carries status_code after retry exhaustion."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        resp = httpx.Response(
            status_code=503, text="Unavailable",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        def mock_post(*args, **kwargs):
            raise httpx.HTTPStatusError("503", request=resp.request, response=resp)

        with patch("nthlayer_common.llm.httpx.post", side_effect=mock_post):
            with patch("nthlayer_common.llm.time.sleep"):
                with pytest.raises(LLMError) as exc_info:
                    llm_call("system", "user", model="openai/gpt-4o", retry=1)

        assert exc_info.value.status_code == 503

    def test_status_code_none_on_timeout(self, monkeypatch):
        """LLMError has no status_code for timeout errors."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        with patch("nthlayer_common.llm.httpx.post", side_effect=httpx.TimeoutException("timed out")):
            with patch("nthlayer_common.llm.time.sleep"):
                with pytest.raises(LLMError) as exc_info:
                    llm_call("system", "user", model="anthropic/claude-sonnet-4-20250514", retry=1)

        assert exc_info.value.status_code is None

    def test_backward_compat_no_status_code_attr(self):
        """LLMError without status_code defaults to None."""
        err = LLMError("test", "provider", "model")
        assert err.status_code is None
