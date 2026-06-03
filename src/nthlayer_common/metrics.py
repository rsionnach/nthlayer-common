"""Self-observability Prometheus metrics for NthLayer components.

Each tier exposes /metrics via prometheus-client. Metrics are pre-defined
here so all components use consistent names and labels.

Canonical import: ``from nthlayer_common.metrics import render_metrics, cycle_duration_seconds``

Spec: NTHLAYER-COMMON-v1 §7.3
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# -- Worker metrics --

cycle_duration_seconds = Histogram(
    "nthlayer_cycle_duration_seconds",
    "Duration of a worker module processing cycle",
    ["component"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

verdicts_written_total = Counter(
    "nthlayer_verdicts_written_total",
    "Total verdicts submitted to core",
    ["component", "type"],
)

assessments_written_total = Counter(
    "nthlayer_assessments_written_total",
    "Total assessments submitted to core",
    ["component", "kind"],
)

heartbeats_emitted_total = Counter(
    "nthlayer_heartbeats_emitted_total",
    "Total heartbeats emitted to core",
    ["component"],
)

llm_calls_total = Counter(
    "nthlayer_llm_calls_total",
    "Total LLM calls made",
    ["component", "model", "outcome"],
)

errors_total = Counter(
    "nthlayer_errors_total",
    "Total errors by type",
    ["component", "error_type"],
)

# -- Core metrics --

api_requests_total = Counter(
    "nthlayer_api_requests_total",
    "Total HTTP API requests handled by core",
    ["method", "route", "status"],  # route = URL template (e.g. /verdicts/{id}), never raw path
)

store_size_bytes = Gauge(
    "nthlayer_store_size_bytes",
    "Size of the SQLite store file in bytes",
)

wal_size_bytes = Gauge(
    "nthlayer_wal_size_bytes",
    "Size of the SQLite WAL file in bytes",
)

stuck_action_requests = Gauge(
    "nthlayer_stuck_action_requests",
    "Number of action_request verdicts without corresponding cases",
    ["service"],
)


def render_metrics() -> bytes:
    """Render all metrics in Prometheus text format."""
    return generate_latest()


def metrics_content_type() -> str:
    """Content-Type header for Prometheus metrics response."""
    return CONTENT_TYPE_LATEST
