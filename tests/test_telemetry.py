"""Tests for OTel telemetry emission."""

from unittest.mock import MagicMock, patch

import pytest

from nthlayer_common.telemetry import emit_llm_event, is_otel_available


class TestOTelAvailability:
    def test_otel_is_available(self):
        """OTel API should be importable (it's in our deps)."""
        assert is_otel_available() is True


class TestEmitLLMEvent:
    @patch("nthlayer_common.telemetry.trace")
    def test_emits_event_on_active_span(self, mock_trace):
        """When there's an active recording span, event is added."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace.get_current_span.return_value = mock_span

        emit_llm_event(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            caller="measure.evaluator",
            input_tokens=500,
            output_tokens=200,
        )

        mock_span.add_event.assert_called_once()
        call_args = mock_span.add_event.call_args
        assert call_args[0][0] == "nthlayer.llm.call"
        attrs = call_args[1]["attributes"]
        assert attrs["gen_ai.request.model"] == "claude-sonnet-4-20250514"
        assert attrs["gen_ai.system"] == "anthropic"
        assert attrs["nthlayer.llm.caller"] == "measure.evaluator"
        assert attrs["gen_ai.usage.input_tokens"] == 500
        assert attrs["gen_ai.usage.output_tokens"] == 200
        assert attrs["nthlayer.llm.success"] is True

    @patch("nthlayer_common.telemetry.trace")
    def test_includes_optional_fields(self, mock_trace):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace.get_current_span.return_value = mock_span

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

        attrs = mock_span.add_event.call_args[1]["attributes"]
        assert attrs["gen_ai.usage.cached_tokens"] == 300
        assert attrs["gen_ai.usage.reasoning_tokens"] == 100
        assert attrs["nthlayer.llm.caller_verdict_cid"] == "vrd-test-001"
        assert attrs["nthlayer.llm.duration_ms"] == 2500

    @patch("nthlayer_common.telemetry.trace")
    def test_error_attributes(self, mock_trace):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace.get_current_span.return_value = mock_span

        emit_llm_event(
            model="gpt-4o",
            provider="openai",
            caller="correlate.summary",
            success=False,
            error="Rate limited",
        )

        attrs = mock_span.add_event.call_args[1]["attributes"]
        assert attrs["nthlayer.llm.success"] is False
        assert attrs["nthlayer.llm.error"] == "Rate limited"

    @patch("nthlayer_common.telemetry.trace")
    def test_no_event_when_span_not_recording(self, mock_trace):
        """When no active span is recording, emit is a no-op."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        mock_trace.get_current_span.return_value = mock_span

        emit_llm_event(
            model="test", provider="test", caller="test",
        )

        mock_span.add_event.assert_not_called()

    @patch("nthlayer_common.telemetry._otel_available", False)
    def test_no_event_when_otel_not_available(self):
        """When OTel is not available, emit is a silent no-op."""
        # Should not raise
        emit_llm_event(
            model="test", provider="test", caller="test",
            input_tokens=100, output_tokens=50,
        )

    @patch("nthlayer_common.telemetry.trace")
    def test_omits_none_fields(self, mock_trace):
        """None-valued optional fields are not included in attributes."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace.get_current_span.return_value = mock_span

        emit_llm_event(
            model="test", provider="test", caller="test",
            # All optional fields left as None
        )

        attrs = mock_span.add_event.call_args[1]["attributes"]
        assert "gen_ai.usage.input_tokens" not in attrs
        assert "gen_ai.usage.output_tokens" not in attrs
        assert "gen_ai.usage.cached_tokens" not in attrs
        assert "nthlayer.llm.caller_verdict_cid" not in attrs
        assert "nthlayer.llm.duration_ms" not in attrs
