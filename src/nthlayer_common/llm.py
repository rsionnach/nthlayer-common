"""
Unified LLM interface for NthLayer agentic components.

Two API formats cover the entire market:
- Anthropic Messages API (Anthropic only)
- OpenAI Chat Completions API (everyone else)

No third-party LLM libraries. No LiteLLM.

Usage:
    from nthlayer_common.llm import llm_call

    response = llm_call(
        system="You are a triage agent...",
        user="Evaluate this incident...",
    )

Configuration via environment:
    NTHLAYER_MODEL          - provider/model (default: anthropic/claude-sonnet-4-20250514)
    NTHLAYER_LLM_TIMEOUT    - seconds (default: 60)
    ANTHROPIC_API_KEY       - for anthropic/* models
    OPENAI_API_KEY          - for openai/*, together/*, groq/*, mistral/*, azure/* models
    OPENAI_API_BASE         - override endpoint URL for any provider
    AZURE_OPENAI_ENDPOINT   - Azure OpenAI resource URL
"""

import logging
import os
import random
import time
from dataclasses import dataclass

import httpx

DEFAULT_MODEL = os.environ.get("NTHLAYER_MODEL", "anthropic/claude-sonnet-4-20250514")
try:
    TIMEOUT = int(os.environ.get("NTHLAYER_LLM_TIMEOUT", "60"))
except (ValueError, TypeError):
    TIMEOUT = 60

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = frozenset({429, 408, 502, 503})
_PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


def _is_transient(status_code: int) -> bool:
    """Classify an HTTP status code as transient (retryable) or permanent.

    Transient: 429, 408, 502, 503 — retry with backoff.
    Permanent: 400, 401, 403, 404, 422 — fail immediately.
    Unknown 5xx: assumed transient. Note: some providers return 500 for
    genuinely permanent problems (malformed request their validation missed),
    but we can't distinguish that from a transient internal error at the HTTP
    layer, so retrying is the safer default.
    """
    if status_code in _TRANSIENT_STATUS_CODES:
        return True
    if status_code in _PERMANENT_STATUS_CODES:
        return False
    return status_code >= 500


def _parse_retry_after(response: httpx.Response) -> float:
    """Parse Retry-After header from an HTTP response.

    Supports integer/float seconds. Returns 0.0 if header is missing or unparseable.
    HTTP-date format is not supported (returns 0.0).
    """
    header = response.headers.get("Retry-After")
    if header is None:
        return 0.0
    try:
        return float(header)
    except (ValueError, TypeError):
        return 0.0


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str           # The response content
    model: str          # Model that was used
    provider: str       # Provider that was used
    input_tokens: int | None = None   # Token count for input (if available)
    output_tokens: int | None = None  # Token count for output (if available)


class LLMError(Exception):
    """Raised when an LLM call fails."""
    def __init__(self, message: str, provider: str, model: str, cause: Exception | None = None, status_code: int | None = None):
        self.provider = provider
        self.model = model
        self.cause = cause
        self.status_code = status_code
        super().__init__(f"[{provider}/{model}] {message}")


def llm_call(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 2000,
    timeout: int | None = None,
    retry: int = 3,
) -> LLMResponse:
    """
    Unified LLM call for all NthLayer agentic components.

    Model format: "provider/model-name"
    Returns LLMResponse with the text content, model, and provider.
    Raises LLMError on failure with provider/model context.

    Transient errors (429, 502, 503, 408, connection errors, timeouts)
    are retried with exponential backoff and full jitter. Permanent errors
    (400, 401, 403, 404, 422) raise immediately. Default retry=3;
    pass retry=0 to disable.

    Note: callers that wrap llm_call() in asyncio.wait_for(timeout=T) should
    use the same timeout value. httpx fires the network timeout first; the
    asyncio.wait_for is a safety net for thread scheduling delays.
    """
    model = model or DEFAULT_MODEL
    _timeout = timeout if timeout is not None else TIMEOUT

    # CI integration test fast-path: NTHLAYER_LLM_STUB=canned bypasses HTTP.
    # See nthlayer_common.llm_stub for the canned-response policy. Function-
    # local import is cycle-safe (llm_stub imports LLMResponse from this
    # module) and cheap on the hot path (sys.modules lookup after first call).
    from nthlayer_common.llm_stub import is_stub_enabled, stub_text_response
    if is_stub_enabled():
        return stub_text_response(system, model)

    # Guard: detect API keys accidentally used as model names
    if model.startswith(("sk-ant-", "sk-", "key-", "Bearer ")):
        raise LLMError(
            f"'{model[:20]}...' looks like an API key, not a model name. "
            "Set NTHLAYER_MODEL to a model (e.g. 'anthropic/claude-sonnet-4-20250514') "
            "and ANTHROPIC_API_KEY or OPENAI_API_KEY to your key.",
            "unknown", model,
        )

    # Parse provider from model string
    if "/" in model:
        provider, _, model_name = model.partition("/")
    else:
        provider = _guess_provider(model)
        model_name = model

    start_time = time.monotonic()
    last_error: Exception | None = None
    last_status_code: int | None = None

    for attempt in range(retry + 1):
        try:
            if provider == "anthropic":
                text, in_tok, out_tok = _call_anthropic(system, user, model_name, max_tokens, _timeout)
            else:
                text, in_tok, out_tok = _call_openai_compat(system, user, model_name, provider, max_tokens, _timeout)

            return LLMResponse(
                text=text, model=model_name, provider=provider,
                input_tokens=in_tok, output_tokens=out_tok,
            )

        except httpx.HTTPStatusError as e:
            last_status_code = e.response.status_code
            if not _is_transient(last_status_code):
                logger.warning(
                    "LLM call failed (permanent, failing): HTTP %d",
                    last_status_code,
                )
                raise LLMError(
                    f"HTTP {last_status_code}: {e.response.text[:200]}",
                    provider, model_name, e, status_code=last_status_code,
                ) from e
            last_error = e
            retry_after = _parse_retry_after(e.response)

        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_error = e
            last_status_code = None
            retry_after = 0.0

        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(str(e), provider, model_name, e) from e

        # Check if we have retries left
        if attempt >= retry:
            break

        # Calculate backoff with full jitter
        delay_cap = min(1.0 * (2 ** attempt), 30.0)
        delay = random.uniform(0, delay_cap)
        delay = max(delay, retry_after)

        # Timeout budget check — don't sleep if we'll exceed the timeout
        elapsed = time.monotonic() - start_time
        remaining = _timeout - elapsed
        if delay > remaining or remaining <= 0:
            break

        error_desc = f"HTTP {last_status_code}" if last_status_code else str(last_error)
        logger.warning(
            "LLM call failed (attempt %d/%d, transient, retrying in %.1fs): %s",
            attempt + 1, retry + 1, delay, error_desc,
        )
        time.sleep(delay)

    # All retries exhausted
    if isinstance(last_error, httpx.HTTPStatusError):
        raise LLMError(
            f"HTTP {last_status_code}: {last_error.response.text[:200]}",
            provider, model_name, last_error, status_code=last_status_code,
        ) from last_error
    elif isinstance(last_error, httpx.TimeoutException):
        raise LLMError(
            f"Timeout after {_timeout}s",
            provider, model_name, last_error,
        ) from last_error
    elif last_error is not None:
        raise LLMError(
            str(last_error),
            provider, model_name, last_error,
        ) from last_error
    else:
        raise LLMError("Unknown error", provider, model_name)


def _call_anthropic(system: str, user: str, model: str, max_tokens: int, timeout: int) -> tuple[str, int | None, int | None]:
    """Call Anthropic Messages API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY not set", "anthropic", model)

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    content = data.get("content", [])
    if not content:
        raise LLMError("Model returned empty content", "anthropic", model)
    text = content[0].get("text", "")
    usage = data.get("usage", {})
    return text, usage.get("input_tokens"), usage.get("output_tokens")


def _call_openai_compat(
    system: str, user: str, model: str, provider: str, max_tokens: int, timeout: int
) -> tuple[str, int | None, int | None]:
    """
    Call OpenAI-compatible Chat Completions API.

    Works with: OpenAI, Azure OpenAI, Ollama, vLLM, Together AI,
    Groq, Mistral, LM Studio, any OpenAI-compatible server.
    """
    base_url = os.environ.get("OPENAI_API_BASE") or _default_base_url(provider)
    if not base_url and provider == "azure":
        raise LLMError("AZURE_OPENAI_ENDPOINT not set", "azure", model)
    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")  # Ollama/vLLM don't require keys

    # Azure uses api-key header; everything else uses Bearer token
    if provider == "azure":
        headers = {
            "api-key": api_key,
            "content-type": "application/json",
        }
        url = f"{base_url}/{model}/chat/completions?api-version=2024-02-01"
    else:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        url = f"{base_url}/chat/completions"

    response = httpx.post(
        url,
        headers=headers,
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise LLMError("Model returned empty choices", provider, model)
    text = (choices[0].get("message") or {}).get("content", "")
    usage = data.get("usage", {})
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens")


def _default_base_url(provider: str) -> str:
    """Default API base URLs by provider."""
    defaults = {
        "openai": "https://api.openai.com/v1",
        "ollama": "http://localhost:11434/v1",
        "vllm": "http://localhost:8000/v1",
        "lmstudio": "http://localhost:1234/v1",
        "together": "https://api.together.xyz/v1",
        "groq": "https://api.groq.com/openai/v1",
        "mistral": "https://api.mistral.ai/v1",
        "azure": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    }
    return defaults.get(provider, "https://api.openai.com/v1")


def _guess_provider(model: str) -> str:
    """Guess provider from bare model name."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    if model.startswith("llama") or model.startswith("mistral") or model.startswith("gemma"):
        return "ollama"
    return "openai"  # Default: assume OpenAI-compatible
