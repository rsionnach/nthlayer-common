"""OTel telemetry emission for NthLayer components.

Emits cost-accounting events for LLM calls and provides helpers
for component-level telemetry.

Graceful no-op when OTel SDK is not configured — import succeeds,
events are silently discarded, no errors raised.

Canonical import: ``from nthlayer_common.telemetry import emit_llm_event``

Spec: NTHLAYER-COMMON-v1 §3.4, §7
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import OTel API. If not available or not configured,
# all emit functions become no-ops.
try:
    from opentelemetry import trace

    _tracer = trace.get_tracer("nthlayer")
    _otel_available = True
except ImportError:
    _otel_available = False
    _tracer = None


def emit_llm_event(
    *,
    model: str,
    provider: str,
    caller: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    verdict_id: str | None = None,
    duration_ms: int | None = None,
    success: bool = True,
    error: str | None = None,
) -> None:
    """Emit an OTel event for an LLM call with gen_ai.* attributes.

    Cost attribution: which component/caller used how many tokens
    of which model. Every LLM call should emit this.

    Args:
        model: The model name (e.g., "claude-sonnet-4-20250514").
        provider: The provider (e.g., "anthropic", "openai").
        caller: The calling component/module (e.g., "measure.evaluator").
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cached_tokens: Cached input tokens (Anthropic prompt caching).
        reasoning_tokens: Reasoning/thinking tokens.
        verdict_id: The verdict being produced, if applicable.
        duration_ms: Wall-clock time of the call in milliseconds.
        success: Whether the call succeeded.
        error: Error message if the call failed.
    """
    if not _otel_available or _tracer is None:
        return

    span = trace.get_current_span()
    if not span.is_recording():
        # No active span — create a minimal event on a new span
        # This handles the case where emit is called outside a traced context
        return

    attributes: dict[str, Any] = {
        "gen_ai.request.model": model,
        "gen_ai.system": provider,
        "nthlayer.llm.caller": caller,
    }

    if input_tokens is not None:
        attributes["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attributes["gen_ai.usage.output_tokens"] = output_tokens
    if cached_tokens is not None:
        attributes["gen_ai.usage.cached_tokens"] = cached_tokens
    if reasoning_tokens is not None:
        attributes["gen_ai.usage.reasoning_tokens"] = reasoning_tokens
    if verdict_id is not None:
        attributes["nthlayer.llm.caller_verdict_cid"] = verdict_id
    if duration_ms is not None:
        attributes["nthlayer.llm.duration_ms"] = duration_ms

    attributes["nthlayer.llm.success"] = success
    if error:
        attributes["nthlayer.llm.error"] = error

    span.add_event("nthlayer.llm.call", attributes=attributes)


def is_otel_available() -> bool:
    """Check if OTel API is importable and a tracer is available."""
    return _otel_available
