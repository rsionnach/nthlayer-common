"""Structured LLM outputs via Instructor.

Provides structured_call() as a complement to llm_call().
llm_call() returns raw text; structured_call() returns validated Pydantic models.
structured_call_with_usage() additionally returns token usage for cost accounting.

Canonical import: ``from nthlayer_common.llm_structured import structured_call, structured_call_with_usage``
"""

from __future__ import annotations

import os
from typing import TypeVar

from pydantic import BaseModel

from nthlayer_common.llm import DEFAULT_MODEL, TIMEOUT, LLMError

import threading

T = TypeVar("T", bound=BaseModel)

# Lazy-initialised clients (one per provider), thread-safe for asyncio.to_thread
_anthropic_client = None
_openai_client = None
_client_lock = threading.Lock()


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        with _client_lock:
            if _anthropic_client is None:
                import instructor
                from anthropic import Anthropic

                _anthropic_client = instructor.from_anthropic(Anthropic())
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        with _client_lock:
            if _openai_client is None:
                import instructor
                from openai import OpenAI

                base_url = os.environ.get("OPENAI_API_BASE")
                _openai_client = instructor.from_openai(
                    OpenAI(base_url=base_url) if base_url else OpenAI()
                )
    return _openai_client


def _parse_model(model: str) -> tuple[str, str]:
    """Parse 'provider/model-name' into (provider, model_name)."""
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    # Guess provider from model name prefix
    if model.startswith("claude"):
        return "anthropic", model
    if model.startswith(("gpt", "o1", "o3")):
        return "openai", model
    return "openai", model


def structured_call(
    system: str,
    user: str,
    response_model: type[T],
    model: str | None = None,
    max_tokens: int = 2000,
    timeout: int | None = None,
    max_retries: int = 3,
) -> T:
    """Call an LLM and return a validated Pydantic model instance.

    Uses Instructor for automatic JSON schema enforcement, validation,
    and retry on malformed responses.

    Args:
        system: System prompt.
        user: User message.
        response_model: Pydantic model class to validate against.
        model: Model string in "provider/model-name" format.
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on validation failure.

    Returns:
        Validated instance of response_model.

    Raises:
        LLMError: On provider errors or exhausted retries.
    """
    model = model or DEFAULT_MODEL
    _timeout = timeout if timeout is not None else TIMEOUT

    if model.startswith(("sk-ant-", "sk-", "key-", "Bearer ")):
        raise LLMError(
            f"'{model[:20]}...' looks like an API key, not a model name. "
            f"Set NTHLAYER_MODEL or pass model='provider/model-name'.",
            provider="unknown",
            model=model,
        )

    provider, model_name = _parse_model(model)

    try:
        if provider == "anthropic":
            client = _get_anthropic_client()
            result = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                max_retries=max_retries,
                messages=[{"role": "user", "content": user}],
                system=system,
                response_model=response_model,
                timeout=_timeout,
            )
        else:
            client = _get_openai_client()
            result = client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                max_retries=max_retries,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_model=response_model,
                timeout=_timeout,
            )
        return result
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(
            f"Structured call failed: {e}",
            provider=provider,
            model=model_name,
            cause=e,
        ) from e


# ---------------------------------------------------------------------------
# structured_call_with_usage — returns model + token usage for cost accounting
# ---------------------------------------------------------------------------


from dataclasses import dataclass
from typing import Generic


@dataclass
class StructuredCallUsage:
    """Token usage from an LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class StructuredCallResult(Generic[T]):
    """Validated model + token usage from a structured LLM call."""
    data: T
    usage: StructuredCallUsage


def structured_call_with_usage(
    system: str,
    user: str,
    response_model: type[T],
    model: str | None = None,
    max_tokens: int = 2000,
    timeout: int | None = None,
    max_retries: int = 3,
) -> StructuredCallResult[T]:
    """Like structured_call but also returns token usage for cost accounting.

    Uses Instructor's create_with_completion() — public API that returns
    both the validated model and the raw completion with usage data.
    Stable across Instructor upgrades.

    Returns:
        StructuredCallResult with validated model and usage info.

    Raises:
        LLMError: On provider errors or exhausted retries.
    """
    model = model or DEFAULT_MODEL
    _timeout = timeout if timeout is not None else TIMEOUT

    if model.startswith(("sk-ant-", "sk-", "key-", "Bearer ")):
        raise LLMError(
            f"'{model[:20]}...' looks like an API key, not a model name.",
            provider="unknown",
            model=model,
        )

    provider, model_name = _parse_model(model)

    try:
        if provider == "anthropic":
            client = _get_anthropic_client()
            validated, completion = client.messages.create_with_completion(
                model=model_name,
                max_tokens=max_tokens,
                max_retries=max_retries,
                messages=[{"role": "user", "content": user}],
                system=system,
                response_model=response_model,
                timeout=_timeout,
            )
            usage = StructuredCallUsage(
                input_tokens=getattr(completion.usage, "input_tokens", 0) or 0,
                output_tokens=getattr(completion.usage, "output_tokens", 0) or 0,
            )
        else:
            client = _get_openai_client()
            validated, completion = client.chat.completions.create_with_completion(
                model=model_name,
                max_tokens=max_tokens,
                max_retries=max_retries,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_model=response_model,
                timeout=_timeout,
            )
            usage = StructuredCallUsage(
                input_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
            )

        return StructuredCallResult(data=validated, usage=usage)

    except LLMError:
        raise
    except Exception as e:
        raise LLMError(
            f"Structured call with usage failed: {e}",
            provider=provider,
            model=model_name,
            cause=e,
        ) from e
