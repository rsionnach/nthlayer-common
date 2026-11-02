"""OTel telemetry emission for NthLayer components, as log records.

Every NthLayer event (the OpenSRM Addendum A1 §4 events and NthLayer's own,
such as ``nthlayer.llm.call``) is an OpenTelemetry log record (A1 §4.0;
opensrm-quqo). Events used to be span events on the current span, but no
component ever made a span recording, so all of them were dropped.

Carriage (ruling Q2, Rob 2026-11-02; A1 is silent on it, opensrm-pdx7):
``LogRecord.event_name`` is the event name, there is no body, every field is
an attribute, severity is INFO, and ``Timestamp`` is the event's own time
when the caller supplies it (A1 T1), else the emit time; it is never left to
ObservedTimestamp.

A process calls ``init_telemetry(service_name)`` once at startup. Without an
OTLP endpoint nothing is installed and every emit is a no-op; with an
endpoint but without the SDK (``pip install nthlayer-common[otel]``), the
same, after a warning.

Canonical imports::

    from nthlayer_common.telemetry import emit_event, emit_llm_event, init_telemetry

Spec: NTHLAYER-COMMON-v1 §3.4, §7; OpenSRM Addendum A1 §4.0.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from opentelemetry import _logs
from opentelemetry._logs import SeverityNumber

logger = logging.getLogger(__name__)

_OTEL_LOGGER_NAME = "nthlayer"

# The provider init_telemetry installed, so a second call does not try to
# replace it (the API allows the global provider to be set only once).
_installed_provider: Any = None
_install_lock = threading.Lock()

# OTLP time_unix_nano is a fixed64.
_MAX_UNIX_NANO = 2**64 - 1


def _epoch_ns(ts: datetime) -> int:
    # A naive datetime is read as UTC, the ecosystem's convention, rather than
    # as local time, which is what datetime.timestamp() would assume.
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    delta = ts - datetime(1970, 1, 1, tzinfo=UTC)
    # Integer arithmetic: a float timestamp loses microseconds at ns scale.
    ns = (delta.days * 86_400 + delta.seconds) * 10**9 + delta.microseconds * 1_000
    # OTLP time_unix_nano is unsigned 64-bit: a value outside it (pre-epoch,
    # or past ~2554, e.g. a datetime.max sentinel) fails to encode in the
    # export thread and takes the whole batch of unrelated events with it.
    return min(max(ns, 0), _MAX_UNIX_NANO)


def emit_event(
    name: str,
    attributes: Mapping[str, Any],
    *,
    timestamp: datetime | None = None,
) -> None:
    """Emit one NthLayer event as an OTel log record.

    ``attributes`` with a value of ``None`` are dropped. ``timestamp`` is when
    the event happened; omitted, it is now. Either way the record carries a
    Timestamp: the SDK would otherwise leave it unset and keep only
    ObservedTimestamp, which A1 T1 forbids measuring latency from. Never
    raises: a telemetry failure must not break the caller.
    """
    try:
        _logs.get_logger(_OTEL_LOGGER_NAME).emit(
            event_name=name,
            attributes={k: v for k, v in attributes.items() if v is not None},
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
            timestamp=_epoch_ns(timestamp) if timestamp is not None else time.time_ns(),
        )
    except Exception:
        logger.debug("telemetry: failed to emit %s", name, exc_info=True)


def init_telemetry(
    service_name: str,
    *,
    endpoint: str | None = None,
    exporter: Any = None,
) -> bool:
    """Install a LoggerProvider exporting OTLP log records; True if installed.

    The endpoint is ``endpoint``, else ``OTEL_EXPORTER_OTLP_LOGS_ENDPOINT``,
    else ``OTEL_EXPORTER_OTLP_ENDPOINT``; with none, nothing is installed.
    ``service.name`` is ``OTEL_SERVICE_NAME`` if set, else ``service_name``.
    ``exporter`` replaces the OTLP gRPC exporter (tests, debugging). The
    exporter is gRPC only: ``OTEL_EXPORTER_OTLP_PROTOCOL`` is not read, so
    point the endpoint at a collector's gRPC port (4317), not 4318.

    Returns False, installing nothing, if another LoggerProvider already owns
    the global slot (e.g. one set by ``opentelemetry-instrument``); events
    then go to that provider.

    Idempotent and thread-safe: a second call keeps the first provider.
    Events emitted before this call are dropped, so call it first thing. In a
    forking server, call it in each worker after the fork (the gRPC channel is
    not fork-safe). The provider shuts down at exit, so short-lived CLI
    commands flush what they emitted; with an unreachable collector that
    flush waits out ``OTEL_EXPORTER_OTLP_TIMEOUT`` (default 10s).
    """
    with _install_lock:
        return _install(service_name, endpoint, exporter)


def _install(service_name: str, endpoint: str | None, exporter: Any) -> bool:
    global _installed_provider
    if _installed_provider is not None:
        return True
    # Blank counts as unset, so a whitespace LOGS var cannot hide the general one.
    endpoint = next(
        (
            e.strip()
            for e in (
                endpoint,
                os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"),
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            )
            if e and e.strip()
        ),
        None,
    )
    if endpoint is None:
        return False
    try:
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource

        if exporter is None:
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

            exporter = OTLPLogExporter(endpoint=endpoint)
    except ImportError as exc:
        logger.warning(
            "telemetry: an OTLP endpoint is set but %s is not installed; "
            "install nthlayer-common[otel] to export events",
            exc.name,
        )
        return False

    resource = Resource.create(
        {"service.name": os.environ.get("OTEL_SERVICE_NAME") or service_name}
    )
    # shutdown_on_exit flushes the batch at interpreter exit (the SDK
    # registers the atexit hook itself).
    provider = LoggerProvider(resource=resource, shutdown_on_exit=True)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    _logs.set_logger_provider(provider)
    if _logs.get_logger_provider() is not provider:
        # The API keeps an already-set provider and only warns; ours would
        # never receive a record, so stop its batch thread and say so.
        provider.shutdown()
        return False
    _installed_provider = provider
    return True


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
    """Emit ``nthlayer.llm.call`` with gen_ai.* attributes.

    Cost attribution: which component/caller used how many tokens
    of which model. Every LLM call should emit this.

    Args:
        model: The model name (e.g., "claude-sonnet-4-20250514").
        provider: The provider (e.g., "anthropic", "openai").
        caller: The calling component/module (e.g., "measure.evaluator").
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cached_tokens: Cache-read input tokens (e.g. Anthropic's
            ``cache_read_input_tokens``), not cache writes; emitted as
            gen_ai.usage.cache_read.input_tokens.
        reasoning_tokens: Reasoning/thinking tokens; emitted as
            gen_ai.usage.reasoning.output_tokens.
        verdict_id: The verdict being produced, if applicable.
        duration_ms: Wall-clock time of the call in milliseconds.
        success: Whether the call succeeded.
        error: Error message if the call failed.
    """
    emit_event(
        "nthlayer.llm.call",
        {
            "gen_ai.request.model": model,
            # Addendum A1 §5 P4: gen_ai.system MUST NOT be emitted (dropped
            # from the upstream registry).
            "gen_ai.provider.name": provider,
            "nthlayer.llm.caller": caller,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "gen_ai.usage.cache_read.input_tokens": cached_tokens,
            "gen_ai.usage.reasoning.output_tokens": reasoning_tokens,
            "nthlayer.llm.caller_verdict_cid": verdict_id,
            "nthlayer.llm.duration_ms": duration_ms,
            "nthlayer.llm.success": success,
            # `or None`: an empty error string is no error, as before.
            "nthlayer.llm.error": error or None,
        },
    )


def is_otel_available() -> bool:
    """Always True: the OTel API is a hard dependency. Kept for API
    compatibility; it does not mean events export (see init_telemetry)."""
    return True
