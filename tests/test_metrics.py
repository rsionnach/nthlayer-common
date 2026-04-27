"""Tests for self-observability Prometheus metrics."""

import pytest

from nthlayer_common.metrics import (
    api_requests_total,
    assessments_written_total,
    cycle_duration_seconds,
    errors_total,
    heartbeats_emitted_total,
    llm_calls_total,
    metrics_content_type,
    render_metrics,
    store_size_bytes,
    stuck_action_requests,
    verdicts_written_total,
    wal_size_bytes,
)


class TestMetricDefinitions:
    def test_cycle_duration_is_histogram(self):
        assert cycle_duration_seconds._type == "histogram"

    def test_verdicts_written_is_counter(self):
        assert verdicts_written_total._type == "counter"

    def test_assessments_written_is_counter(self):
        assert assessments_written_total._type == "counter"

    def test_heartbeats_is_counter(self):
        assert heartbeats_emitted_total._type == "counter"

    def test_llm_calls_is_counter(self):
        assert llm_calls_total._type == "counter"

    def test_errors_is_counter(self):
        assert errors_total._type == "counter"

    def test_api_requests_is_counter(self):
        assert api_requests_total._type == "counter"

    def test_store_size_is_gauge(self):
        assert store_size_bytes._type == "gauge"

    def test_wal_size_is_gauge(self):
        assert wal_size_bytes._type == "gauge"

    def test_stuck_action_requests_is_gauge(self):
        assert stuck_action_requests._type == "gauge"


class TestMetricUsage:
    def test_increment_counter(self):
        verdicts_written_total.labels(component="measure", type="quality_breach").inc()
        # No assertion on value — just verify it doesn't crash

    def test_observe_histogram(self):
        cycle_duration_seconds.labels(component="observe").observe(1.5)

    def test_set_gauge(self):
        store_size_bytes.set(1024 * 1024)

    def test_labeled_gauge(self):
        stuck_action_requests.labels(service="payment-api").set(3)


class TestRenderMetrics:
    def test_render_returns_bytes(self):
        result = render_metrics()
        assert isinstance(result, bytes)
        assert b"nthlayer_" in result

    def test_content_type(self):
        ct = metrics_content_type()
        assert "text/plain" in ct or "openmetrics" in ct
