"""Tests for OTel telemetry emission as log records (opensrm-quqo).

Events used to be span events on the current span, and nothing ever made a
span recording, so every one was dropped in production while tests that
mocked the span passed. These tests run a real SDK LoggerProvider with an
in-memory exporter, so they see what an OTLP collector would receive.

Carriage (ruling Q2, Rob 2026-11-02; A1 §4.0 says the events are OTel
log-based events and is silent on the rest): LogRecord.event_name is the
event name, there is no body, every field is an attribute, severity is INFO,
and Timestamp is the event's own time when supplied (A1 T1), else the emit
time; it is never left to ObservedTimestamp.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta, timezone

import pytest
from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

try:
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
except ImportError:  # SDK 1.38-1.40 (inside the declared range): the pre-1.41 name
    from opentelemetry.sdk._logs.export import (
        InMemoryLogExporter as InMemoryLogRecordExporter,
    )

from nthlayer_common import telemetry
from nthlayer_common.telemetry import emit_event, emit_llm_event, init_telemetry


def _record(data):
    # SDK 1.38 exports LogData(log_record=...); later releases export the
    # readable record itself.
    return getattr(data, "log_record", data)


@pytest.fixture
def get_exported(monkeypatch):
    """Route emit_event to a private provider; yield its exported records.

    Patches the lookup rather than setting the global provider, which the API
    allows only once per process.
    """
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    monkeypatch.setattr(telemetry._logs, "get_logger", provider.get_logger)
    yield lambda: [_record(d) for d in exporter.get_finished_logs()]
    provider.shutdown()


class TestEmitEvent:
    def test_carriage(self, get_exported):
        emit_event("nthlayer.reversal", {"nthlayer.decision.id": "d-1", "count": 2})

        (rec,) = get_exported()
        assert rec.event_name == "nthlayer.reversal"
        assert rec.body is None
        assert dict(rec.attributes) == {"nthlayer.decision.id": "d-1", "count": 2}
        assert rec.severity_number == SeverityNumber.INFO
        assert rec.severity_text == "INFO"

    def test_timestamp_is_the_event_time(self, get_exported):
        # A1 T1: feedback latency is measured from Timestamp, so it must be
        # when the event happened, not when it was emitted.
        when = datetime(2026, 11, 1, 12, 30, 15, 250000, tzinfo=UTC)
        emit_event("nthlayer.reversal", {}, timestamp=when)

        (rec,) = get_exported()
        assert rec.timestamp == 1_793_536_215_250_000_000

    def test_timestamp_honours_the_offset(self, get_exported):
        when = datetime(2026, 11, 1, 13, 30, 15, tzinfo=timezone(timedelta(hours=1)))
        emit_event("nthlayer.reversal", {}, timestamp=when)

        (rec,) = get_exported()
        assert rec.timestamp == 1_793_536_215_000_000_000

    def test_a_naive_timestamp_is_read_as_utc(self, get_exported):
        emit_event("nthlayer.reversal", {}, timestamp=datetime(2026, 11, 1, 12, 30, 15))

        (rec,) = get_exported()
        assert rec.timestamp == 1_793_536_215_000_000_000

    def test_no_timestamp_means_now(self, get_exported):
        # Left to the SDK, Timestamp would be unset (only ObservedTimestamp).
        before = time.time_ns()
        emit_event("nthlayer.llm.call", {})
        after = time.time_ns()

        (rec,) = get_exported()
        assert before <= rec.timestamp <= after

    def test_none_attribute_values_are_dropped(self, get_exported):
        emit_event("nthlayer.reversal", {"kept": "x", "absent": None})

        (rec,) = get_exported()
        assert dict(rec.attributes) == {"kept": "x"}

    def test_a_failing_logger_never_breaks_the_caller(self, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("exporter down")

        monkeypatch.setattr(telemetry._logs, "get_logger", boom)
        emit_event("nthlayer.reversal", {"k": "v"})  # must not raise

    def test_without_an_sdk_provider_it_is_a_no_op(self):
        # The API's default provider discards: no SDK, no record, no error.
        assert emit_event("nthlayer.reversal", {"k": "v"}) is None

    def test_a_far_future_timestamp_is_clamped(self, get_exported):
        # datetime.max is past OTLP's fixed64 time_unix_nano; unclamped it
        # failed the exporter's whole batch.
        emit_event("nthlayer.reversal", {}, timestamp=datetime.max)

        (rec,) = get_exported()
        assert rec.timestamp == 2**64 - 1

    def test_a_pre_epoch_timestamp_is_clamped(self, get_exported):
        # OTLP time_unix_nano is unsigned; a negative value would fail the
        # exporter's whole batch.
        emit_event("nthlayer.reversal", {}, timestamp=datetime(1969, 12, 31, tzinfo=UTC))

        (rec,) = get_exported()
        assert rec.timestamp == 0


class TestEmitLLMEvent:
    def test_emits_a_log_record(self, get_exported):
        emit_llm_event(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            caller="measure.evaluator",
            input_tokens=500,
            output_tokens=200,
        )

        (rec,) = get_exported()
        assert rec.event_name == "nthlayer.llm.call"
        attrs = dict(rec.attributes)
        assert attrs["gen_ai.request.model"] == "claude-sonnet-4-20250514"
        # Addendum A1 §5 P4: gen_ai.provider.name, never the retired gen_ai.system.
        assert attrs["gen_ai.provider.name"] == "anthropic"
        assert "gen_ai.system" not in attrs
        assert attrs["nthlayer.llm.caller"] == "measure.evaluator"
        assert attrs["gen_ai.usage.input_tokens"] == 500
        assert attrs["gen_ai.usage.output_tokens"] == 200
        assert attrs["nthlayer.llm.success"] is True

    def test_includes_optional_fields(self, get_exported):
        emit_llm_event(
            model="claude-opus-4-7",
            provider="anthropic",
            caller="respond.triage",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=300,
            reasoning_tokens=100,
            verdict_id="vrd-test-001",
            duration_ms=2500,
        )

        attrs = dict(get_exported()[0].attributes)
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 300
        assert attrs["gen_ai.usage.reasoning.output_tokens"] == 100
        assert attrs["nthlayer.llm.caller_verdict_cid"] == "vrd-test-001"
        assert attrs["nthlayer.llm.duration_ms"] == 2500

    def test_error_attributes(self, get_exported):
        emit_llm_event(model="m", provider="anthropic", caller="c", success=False, error="timeout")

        attrs = dict(get_exported()[0].attributes)
        assert attrs["nthlayer.llm.success"] is False
        assert attrs["nthlayer.llm.error"] == "timeout"

    def test_omits_none_fields(self, get_exported):
        emit_llm_event(model="m", provider="anthropic", caller="c")

        attrs = dict(get_exported()[0].attributes)
        for key in (
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.cache_read.input_tokens",
            "gen_ai.usage.reasoning.output_tokens",
            "nthlayer.llm.caller_verdict_cid",
            "nthlayer.llm.duration_ms",
            "nthlayer.llm.error",
        ):
            assert key not in attrs


class TestInitTelemetry:
    @pytest.fixture
    def installed(self, monkeypatch):
        """Capture the provider init_telemetry installs, without installing it
        globally."""
        captured: dict = {}
        monkeypatch.setattr(telemetry, "_installed_provider", None)
        monkeypatch.setattr(
            telemetry._logs, "set_logger_provider", lambda p: captured.setdefault("provider", p)
        )
        monkeypatch.setattr(
            telemetry._logs, "get_logger_provider", lambda: captured.get("provider")
        )
        for var in (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
            "OTEL_SERVICE_NAME",
        ):
            monkeypatch.delenv(var, raising=False)
        yield captured
        # Stop the batch thread of any provider the test built.
        if "provider" in captured:
            captured["provider"].shutdown()

    def test_without_an_endpoint_it_installs_nothing(self, installed):
        assert init_telemetry("nthlayer-workers") is False
        assert installed == {}

    def test_an_endpoint_installs_a_provider_that_exports(self, installed, monkeypatch):
        exporter = InMemoryLogRecordExporter()
        assert init_telemetry("nthlayer-workers", endpoint="http://c:4317", exporter=exporter)

        provider = installed["provider"]
        assert provider.resource.attributes["service.name"] == "nthlayer-workers"
        # Flushed at exit, so short-lived CLI commands export what they emit.
        assert provider._at_exit_handler is not None

        monkeypatch.setattr(telemetry._logs, "get_logger", provider.get_logger)
        emit_event("nthlayer.reversal", {"k": "v"})
        provider.force_flush()
        (rec,) = [_record(d) for d in exporter.get_finished_logs()]
        assert rec.event_name == "nthlayer.reversal"

    def test_the_standard_env_var_is_an_endpoint(self, installed, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4317")
        assert init_telemetry("nthlayer-workers", exporter=InMemoryLogRecordExporter())

    def test_the_logs_specific_env_var_is_an_endpoint(self, installed, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://c:4317")
        assert init_telemetry("nthlayer-workers", exporter=InMemoryLogRecordExporter())

    def test_a_blank_logs_var_does_not_hide_the_general_one(self, installed, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "   ")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4317")
        assert init_telemetry("nthlayer-workers", exporter=InMemoryLogRecordExporter())

    def test_only_blank_endpoints_install_nothing(self, installed, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", " ")
        assert init_telemetry("nthlayer-workers") is False
        assert "provider" not in installed

    def test_only_blank_endpoints_install_nothing_even_with_an_exporter(
        self, installed, monkeypatch
    ):
        # With an exporter injected, the missing gRPC exporter cannot be what
        # makes this return False: only the blank handling can.
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "  ")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", " ")
        assert init_telemetry("w", exporter=InMemoryLogRecordExporter()) is False
        assert "provider" not in installed

    def test_concurrent_calls_install_one_provider(self, installed, monkeypatch):
        calls = []
        monkeypatch.setattr(
            telemetry._logs,
            "set_logger_provider",
            lambda p: (calls.append(p), installed.setdefault("provider", p)),
        )
        results = []
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    init_telemetry(
                        "w", endpoint="http://c:4317", exporter=InMemoryLogRecordExporter()
                    )
                )
            )
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == [True] * 8
        assert len(calls) == 1

    def test_otel_service_name_wins(self, installed, monkeypatch):
        monkeypatch.setenv("OTEL_SERVICE_NAME", "fraud-workers")
        init_telemetry(
            "nthlayer-workers", endpoint="http://c:4317", exporter=InMemoryLogRecordExporter()
        )
        assert installed["provider"].resource.attributes["service.name"] == "fraud-workers"

    def test_a_second_call_is_a_no_op(self, installed):
        init_telemetry(
            "nthlayer-workers", endpoint="http://c:4317", exporter=InMemoryLogRecordExporter()
        )
        first = installed["provider"]
        installed.clear()

        assert init_telemetry("nthlayer-workers", endpoint="http://c:4317") is True
        assert installed == {}
        assert telemetry._installed_provider is first

    def test_a_provider_already_set_elsewhere_is_left_alone(self, installed, monkeypatch):
        # e.g. opentelemetry-instrument set one first: the API keeps it, so
        # ours would receive nothing. init_telemetry must say so.
        monkeypatch.setattr(telemetry._logs, "get_logger_provider", lambda: object())

        assert (
            init_telemetry("w", endpoint="http://c:4317", exporter=InMemoryLogRecordExporter())
            is False
        )
        assert telemetry._installed_provider is None


class TestResolveEndpoint:
    """Precedence per the OTel SDK environment spec: the signal-specific var
    overrides the general one; an explicit argument overrides both."""

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    def test_logs_var_beats_general(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://logs:4317")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://general:4317")
        assert telemetry._resolve_endpoint(None) == "http://logs:4317"

    def test_explicit_beats_both(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://logs:4317")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://general:4317")
        assert telemetry._resolve_endpoint("http://arg:4317") == "http://arg:4317"

    def test_general_alone(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://general:4317")
        assert telemetry._resolve_endpoint(None) == "http://general:4317"

    def test_a_blank_logs_var_falls_through(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "   ")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://general:4317")
        assert telemetry._resolve_endpoint(None) == "http://general:4317"

    def test_surrounding_whitespace_is_stripped(self):
        assert telemetry._resolve_endpoint("  http://arg:4317 ") == "http://arg:4317"

    def test_nothing_set(self):
        assert telemetry._resolve_endpoint(None) is None
        assert telemetry._resolve_endpoint("  ") is None


def test_is_otel_available():
    assert telemetry.is_otel_available() is True
