"""Tests for the governance bridge foundation (opensrm-jmy.5).

Covers spec § 5: outbound signal schemas (autonomy_change,
policy_recommendation, incident_opened), inbound signal schemas
(policy_updated, autonomy_restore_request), and the fail-open
webhook emitter with retry. Worker integration (wiring measure /
learn / respond to emit) and the inbound adapter sidecar are
tracked in follow-up beads.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from nthlayer_common.governance_bridge import (
    SIGNAL_VERSION,
    AutonomyChangeSignal,
    AutonomyChangeTrigger,
    AutonomyRestoreRequest,
    EmitResult,
    GovernanceBridgeEmitter,
    IncidentOpenedSignal,
    PolicyRecommendation,
    PolicyRecommendationSignal,
    PolicyUpdatedSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FIXED_TS = datetime(2026, 3, 24, 14, 23, 0, tzinfo=timezone.utc)


def _autonomy_signal(**overrides: object) -> AutonomyChangeSignal:
    base: dict[str, object] = {
        "service": "fraud-detection",
        "previous_level": "autonomous",
        "new_level": "human_review",
        "trigger": AutonomyChangeTrigger(
            metric="reversal_rate",
            current_value=0.031,
            threshold=0.015,
            window="7d",
        ),
        "recommended_governance_action": (
            "Tighten approval gate for fraud-detection decisions until "
            "autonomy is restored."
        ),
        "incident": "INC-4821",
        "timestamp": _FIXED_TS,
    }
    base.update(overrides)
    return AutonomyChangeSignal(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Outbound signal schemas
# ---------------------------------------------------------------------------


class TestAutonomyChangeSignal:
    def test_payload_matches_spec_shape(self):
        sig = _autonomy_signal()
        payload = sig.to_payload()
        assert payload == {
            "type": "autonomy_change",
            "version": "v1",
            "timestamp": "2026-03-24T14:23:00+00:00",
            "service": "fraud-detection",
            "previous_level": "autonomous",
            "new_level": "human_review",
            "trigger": {
                "metric": "reversal_rate",
                "current_value": 0.031,
                "threshold": 0.015,
                "window": "7d",
            },
            "incident": "INC-4821",
            "recommended_governance_action": (
                "Tighten approval gate for fraud-detection decisions until "
                "autonomy is restored."
            ),
        }

    def test_incident_omitted_when_none(self):
        sig = _autonomy_signal(incident=None)
        assert "incident" not in sig.to_payload()

    def test_required_fields_rejected_when_empty(self):
        for missing in ("service", "previous_level", "new_level", "recommended_governance_action"):
            with pytest.raises(ValueError, match=missing):
                _autonomy_signal(**{missing: ""})

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            _autonomy_signal(timestamp=datetime(2026, 3, 24, 14, 23, 0))

    def test_signal_type_is_immutable(self):
        # signal_type is init=False to prevent callers from forging
        # one signal class as another on the wire.
        with pytest.raises(TypeError):
            AutonomyChangeSignal(  # type: ignore[call-arg]
                service="x",
                previous_level="a",
                new_level="b",
                trigger=AutonomyChangeTrigger("m", 0.0, 0.0, "1m"),
                recommended_governance_action="r",
                signal_type="forged",
            )


class TestPolicyRecommendationSignal:
    def test_payload_matches_spec_shape(self):
        sig = PolicyRecommendationSignal(
            service="fraud-detection",
            incident="INC-4821",
            recommendation=PolicyRecommendation(
                recommendation_type="add_deploy_gate",
                description=(
                    "Gate deploys on reversal rate exceeding 2% during "
                    "15-minute evaluation window"
                ),
                confidence=0.82,
                financial_impact="Would have prevented ~€47,200 in false_negative losses",
            ),
            timestamp=datetime(2026, 3, 24, 15, 0, 0, tzinfo=timezone.utc),
        )
        payload = sig.to_payload()
        assert payload["type"] == "policy_recommendation"
        assert payload["version"] == "v1"
        assert payload["service"] == "fraud-detection"
        assert payload["incident"] == "INC-4821"
        assert payload["requires_human_review"] is True
        assert payload["recommendation"]["type"] == "add_deploy_gate"
        assert payload["recommendation"]["confidence"] == 0.82
        assert payload["recommendation"]["financial_impact"].startswith(
            "Would have prevented"
        )

    def test_financial_impact_omitted_when_none(self):
        rec = PolicyRecommendation(
            recommendation_type="tighten_slo", description="x", confidence=0.5,
        )
        assert "financial_impact" not in rec.to_payload()

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            PolicyRecommendation(recommendation_type="x", description="y", confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            PolicyRecommendation(recommendation_type="x", description="y", confidence=-0.1)


class TestIncidentOpenedSignal:
    def test_payload_matches_spec_shape(self):
        sig = IncidentOpenedSignal(
            incident="INC-4821",
            severity=2,
            affected_services=["fraud-detection", "payment-api", "checkout-service"],
            root_cause_service="fraud-detection",
            root_cause_type="model_regression",
            timestamp=datetime(2026, 3, 24, 14, 15, 0, tzinfo=timezone.utc),
        )
        payload = sig.to_payload()
        assert payload["type"] == "incident_opened"
        assert payload["severity"] == 2
        assert payload["affected_services"] == [
            "fraud-detection", "payment-api", "checkout-service",
        ]
        assert payload["root_cause_service"] == "fraud-detection"
        # Receivers dedup on (type, service, ts); service property is the anchor.
        assert sig.service == "fraud-detection"

    def test_empty_affected_services_rejected(self):
        with pytest.raises(ValueError, match="affected_services"):
            IncidentOpenedSignal(
                incident="INC-1",
                severity=2,
                affected_services=[],
                root_cause_service="x",
                root_cause_type="y",
            )

    def test_non_positive_severity_rejected(self):
        with pytest.raises(ValueError, match="severity"):
            IncidentOpenedSignal(
                incident="INC-1",
                severity=0,
                affected_services=["x"],
                root_cause_service="x",
                root_cause_type="y",
            )


# ---------------------------------------------------------------------------
# Inbound signal schemas (delivered via the bridge adapter sidecar)
# ---------------------------------------------------------------------------


class TestInboundSignals:
    def test_policy_updated_payload(self):
        sig = PolicyUpdatedSignal(
            service="fraud-detection",
            policy_id="pol-2847",
            change_type="tightened",
            description="Added human approval gate for transactions > €5000",
            source_system="cortexhub",
            timestamp=datetime(2026, 3, 24, 16, 0, 0, tzinfo=timezone.utc),
        )
        payload = sig.to_payload()
        assert payload["type"] == "policy_updated"
        assert payload["policy_id"] == "pol-2847"
        assert payload["source_system"] == "cortexhub"

    def test_autonomy_restore_request_payload(self):
        sig = AutonomyRestoreRequest(
            service="fraud-detection",
            requested_by="oncall-lead-047",
            requested_level="autonomous",
            justification="Model v2.2 restored, reversal rate back to baseline for 2 hours",
            timestamp=datetime(2026, 3, 24, 17, 0, 0, tzinfo=timezone.utc),
        )
        payload = sig.to_payload()
        assert payload["type"] == "autonomy_restore_request"
        assert payload["requested_by"] == "oncall-lead-047"
        assert payload["requested_level"] == "autonomous"

    def test_inbound_signals_reject_naive_timestamp(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            PolicyUpdatedSignal(
                service="x",
                policy_id="p1",
                change_type="t",
                description="d",
                source_system="s",
                timestamp=datetime(2026, 3, 24, 16, 0, 0),
            )


# ---------------------------------------------------------------------------
# Idempotency — receivers dedup on (type, service, timestamp)
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_signal_emits_stable_payload_on_repeat(self):
        # Receivers dedup on (type, service, timestamp). Same signal
        # instance emitted twice must produce byte-identical payloads
        # so dedup is reliable across SENDER-side retries.
        sig = _autonomy_signal()
        assert sig.to_payload() == sig.to_payload()

    def test_dedup_keys_match_across_signal_kinds(self):
        # The three outbound signals all expose `signal_type` and
        # `service` so a receiver can apply a uniform dedup rule.
        outbound = [
            _autonomy_signal(),
            PolicyRecommendationSignal(
                service="fraud-detection",
                incident="INC-1",
                recommendation=PolicyRecommendation(
                    recommendation_type="x", description="y", confidence=0.5,
                ),
            ),
            IncidentOpenedSignal(
                incident="INC-1",
                severity=2,
                affected_services=["fraud-detection"],
                root_cause_service="fraud-detection",
                root_cause_type="model_regression",
            ),
        ]
        for sig in outbound:
            assert sig.signal_type
            assert sig.service == "fraud-detection"


# ---------------------------------------------------------------------------
# Emitter — happy path, retry, fail-open, auth, lifecycle
# ---------------------------------------------------------------------------


def _make_emitter(handler, **kwargs: object) -> GovernanceBridgeEmitter:
    """Wire an httpx.MockTransport-backed client into the emitter."""
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    defaults: dict[str, object] = {"initial_backoff": 0.001, "max_attempts": 3}
    defaults.update(kwargs)
    return GovernanceBridgeEmitter(
        "http://governance.example.com/api/v1/nthlayer-signals",
        client=client,
        **defaults,  # type: ignore[arg-type]
    )


class TestEmitter:
    def test_webhook_url_required(self):
        with pytest.raises(ValueError, match="webhook_url"):
            GovernanceBridgeEmitter("")

    def test_max_attempts_lower_bound(self):
        with pytest.raises(ValueError, match="max_attempts"):
            GovernanceBridgeEmitter("http://x", max_attempts=0)

    @pytest.mark.asyncio
    async def test_emit_success_returns_ok(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            captured["payload"] = request.read().decode("utf-8")
            return httpx.Response(200, json={"status": "accepted"})

        emitter = _make_emitter(handler, auth_token="secret-token")
        try:
            result = await emitter.emit(_autonomy_signal())
        finally:
            await emitter.close()

        assert result.ok
        assert result.status_code == 200
        assert result.attempts == 1
        assert result.signal_type == "autonomy_change"
        # Auth header is attached when token is supplied.
        assert captured["headers"]["authorization"] == "Bearer secret-token"  # type: ignore[index]
        # Payload is JSON-shaped per spec § 5 (compact form, no spaces).
        assert '"type":"autonomy_change"' in captured["payload"]  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_emit_retries_transient_then_succeeds(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(503, json={"error": "unavailable"})
            return httpx.Response(200)

        emitter = _make_emitter(handler, max_attempts=5)
        try:
            result = await emitter.emit(_autonomy_signal())
        finally:
            await emitter.close()

        assert result.ok
        assert result.status_code == 200
        assert result.attempts == 3
        assert attempts["n"] == 3

    @pytest.mark.asyncio
    async def test_emit_fails_open_when_exhausted(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(503)

        emitter = _make_emitter(handler, max_attempts=2)
        try:
            result = await emitter.emit(_autonomy_signal())
        finally:
            await emitter.close()

        # Fail-open: returns a result, never raises.
        assert result.ok is False
        assert result.status_code == 503
        assert result.attempts == 2
        assert attempts["n"] == 2

    @pytest.mark.asyncio
    async def test_emit_4xx_not_retried(self):
        # Permanent failure: 4xx (except 408/429) returns immediately.
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(422, json={"error": "invalid_payload"})

        emitter = _make_emitter(handler, max_attempts=5)
        try:
            result = await emitter.emit(_autonomy_signal())
        finally:
            await emitter.close()

        assert result.ok is False
        assert result.status_code == 422
        assert result.attempts == 1
        assert attempts["n"] == 1
        assert "HTTP 422" in (result.error or "")

    @pytest.mark.asyncio
    async def test_emit_connection_error_fails_open(self):
        # Transport-level failure (connection refused, DNS, timeout).
        # The emitter should retry and ultimately fail-open without
        # raising — NthLayer must continue operating.
        emitter = GovernanceBridgeEmitter(
            "http://127.0.0.1:1/nope",
            max_attempts=2,
            initial_backoff=0.001,
            timeout=0.05,
        )
        try:
            result = await emitter.emit(_autonomy_signal())
        finally:
            await emitter.close()

        assert result.ok is False
        assert result.status_code is None
        assert result.attempts == 2
        assert result.error  # populated with the transport error

    @pytest.mark.asyncio
    async def test_emit_omits_auth_header_when_no_token(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            return httpx.Response(200)

        emitter = _make_emitter(handler)
        try:
            await emitter.emit(_autonomy_signal())
        finally:
            await emitter.close()

        assert "authorization" not in captured["headers"]  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_context_manager_closes_owned_client(self):
        # When the emitter creates its own client, the context manager
        # must close it on exit.
        async with GovernanceBridgeEmitter("http://x") as emitter:
            client = emitter._get_client()
            assert client is not None
        # _client cleared after close.
        assert emitter._client is None


# ---------------------------------------------------------------------------
# EmitResult shape
# ---------------------------------------------------------------------------


def test_emit_result_carries_signal_type():
    # Ensure callers can correlate result back to which signal kind
    # they fired without re-inspecting their input.
    result = EmitResult(
        ok=True, status_code=200, signal_type="autonomy_change", attempts=1,
    )
    assert result.signal_type == "autonomy_change"


def test_signal_version_constant():
    # Single source of truth for the protocol version. Bumping this
    # value cascades to every signal class.
    assert SIGNAL_VERSION == "v1"
