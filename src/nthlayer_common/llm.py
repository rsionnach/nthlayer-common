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

import os
import json
import httpx
from dataclasses import dataclass

DEFAULT_MODEL = os.environ.get("NTHLAYER_MODEL", "anthropic/claude-sonnet-4-20250514")
try:
    TIMEOUT = int(os.environ.get("NTHLAYER_LLM_TIMEOUT", "60"))
except (ValueError, TypeError):
    TIMEOUT = 60


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
    def __init__(self, message: str, provider: str, model: str, cause: Exception | None = None):
        self.provider = provider
        self.model = model
        self.cause = cause
        super().__init__(f"[{provider}/{model}] {message}")


def llm_call(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 2000,
    timeout: int | None = None,
) -> LLMResponse:
    """
    Unified LLM call for all NthLayer agentic components.

    Model format: "provider/model-name"
      - anthropic/claude-sonnet-4-20250514
      - openai/gpt-4o
      - ollama/llama3.1
      - azure/my-deployment
      - together/meta-llama/Llama-3-70b
      - groq/llama-3.1-70b-versatile
      - mistral/mistral-large-latest
      - vllm/my-model
      - lmstudio/my-model
      - custom/my-model (with OPENAI_API_BASE set)

    Provider determines the API format and endpoint:
      - "anthropic/*"  -> Anthropic Messages API
      - Everything else -> OpenAI-compatible Chat Completions API

    Returns LLMResponse with the text content, model, and provider.
    Raises LLMError on failure with provider/model context.

    Note: callers that wrap llm_call() in asyncio.wait_for(timeout=T) should
    use the same timeout value. httpx fires the network timeout first; the
    asyncio.wait_for is a safety net for thread scheduling delays.
    """
    model = model or DEFAULT_MODEL
    _timeout = timeout if timeout is not None else TIMEOUT

    # Parse provider from model string
    if "/" in model:
        provider, _, model_name = model.partition("/")
    else:
        # Bare model name - guess provider from known prefixes
        provider = _guess_provider(model)
        model_name = model

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
        raise LLMError(
            f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            provider, model_name, e,
        ) from e
    except httpx.TimeoutException as e:
        raise LLMError(
            f"Timeout after {_timeout}s",
            provider, model_name, e,
        ) from e
    except Exception as e:
        if isinstance(e, LLMError):
            raise
        raise LLMError(str(e), provider, model_name, e) from e


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
