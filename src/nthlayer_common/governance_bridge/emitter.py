"""Outbound webhook emitter for governance bridge signals (spec § 5).

Fail-open per spec principle 6: NthLayer does not depend on the
governance bridge for safety. If the webhook is unreachable, the
autonomy ratchet remains authoritative locally; the bridge is a
coordination enhancement, not a safety dependency. Every emit call
returns an :class:`EmitResult` and never raises.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import structlog

logger = structlog.get_logger(__name__)


# HTTP statuses that warrant a retry (transient class). 4xx (except 408/429)
# is permanent — re-sending won't help.
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class _SignalLike(Protocol):
    """Duck-type contract for anything the emitter can ship."""

    signal_type: str
    service: str

    def to_payload(self) -> dict[str, Any]: ...


@dataclass
class EmitResult:
    """Outcome of an emit attempt. Always returned, never raised.

    Callers may record ``ok`` / ``status_code`` for observability but
    must not gate any reliability decision on a successful emit — the
    bridge is fail-open.
    """

    ok: bool
    status_code: int | None
    signal_type: str
    attempts: int
    error: str | None = None


class GovernanceBridgeEmitter:
    """Async webhook emitter for governance bridge outbound signals.

    Reuses ``httpx.AsyncClient`` patterns from the existing
    ``BaseHTTPClient`` / ``CoreAPIClient``: caller-injectable transport
    for tests, owned-client lifecycle by default, structured-logging
    on every failure path.
    """

    def __init__(
        self,
        webhook_url: str,
        *,
        auth_token: str | None = None,
        max_attempts: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_factor: float = 2.0,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not webhook_url:
            raise ValueError(
                "GovernanceBridgeEmitter.webhook_url is required (spec § 5)"
            )
        if max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1; got {max_attempts}"
            )
        self.webhook_url = webhook_url
        self.auth_token = auth_token
        self.max_attempts = max_attempts
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self._client = client
        self._owned_client = client is None

    async def __aenter__(self) -> GovernanceBridgeEmitter:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Release the owned httpx client, if any.

        Safe to call repeatedly. A subsequent ``emit()`` lazily
        re-opens a fresh owned client — the emitter remains usable
        after close, just at the cost of one new TCP connection.
        """
        if self._owned_client and self._client is not None:
            await self._client.aclose()
            self._client = None
            self._owned_client = False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Lazy-create on first emit() OR re-open after close().
            # The ``_owned_client = True`` here matters only for the
            # re-open case — ``__init__`` already set it when no
            # external client was injected.
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owned_client = True
        return self._client

    async def emit(self, signal: _SignalLike) -> EmitResult:
        """POST the signal payload. Retries transient failures, fails open.

        Concurrency note: ``client`` is captured once before the retry
        loop, so a concurrent ``close()`` does not orphan the in-flight
        call — the closed transport instead raises a transport error
        that the existing handler catches, and the call fails-open
        normally.
        """
        payload = signal.to_payload()
        # Pre-encode JSON outside the retry loop. A payload containing
        # non-JSON-serialisable values (caller bug) would otherwise
        # raise TypeError from ``client.post(json=...)`` mid-loop and
        # bypass the fail-open contract.
        try:
            # Compact form (no inter-token whitespace) — matches what
            # httpx's ``json=`` kwarg would have emitted internally, so
            # this change of mechanism is transparent on the wire.
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            logger.warning(
                "governance_bridge_emit_payload_unserialisable",
                signal_type=signal.signal_type,
                service=signal.service,
                error=str(exc),
            )
            return EmitResult(
                ok=False,
                status_code=None,
                signal_type=signal.signal_type,
                attempts=0,
                error=f"payload_serialisation: {exc}",
            )

        client = self._get_client()
        backoff = self.initial_backoff
        last_status: int | None = None
        last_error: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await client.post(
                    self.webhook_url,
                    content=body,
                    headers=self._headers(),
                )
            except (httpx.HTTPError, OSError) as exc:
                # Connection refused, timeout, DNS — all retryable.
                last_status = None
                last_error = str(exc) or exc.__class__.__name__
                logger.warning(
                    "governance_bridge_emit_transport_error",
                    signal_type=signal.signal_type,
                    service=signal.service,
                    attempt=attempt,
                    error=last_error,
                )
            else:
                last_status = response.status_code
                if 200 <= response.status_code < 300:
                    return EmitResult(
                        ok=True,
                        status_code=response.status_code,
                        signal_type=signal.signal_type,
                        attempts=attempt,
                    )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    # Permanent failure — no retry, but still fail-open.
                    logger.warning(
                        "governance_bridge_emit_permanent_failure",
                        signal_type=signal.signal_type,
                        service=signal.service,
                        status_code=response.status_code,
                    )
                    return EmitResult(
                        ok=False,
                        status_code=response.status_code,
                        signal_type=signal.signal_type,
                        attempts=attempt,
                        error=f"HTTP {response.status_code}",
                    )
                last_error = f"HTTP {response.status_code}"
                logger.warning(
                    "governance_bridge_emit_transient_failure",
                    signal_type=signal.signal_type,
                    service=signal.service,
                    attempt=attempt,
                    status_code=response.status_code,
                )

            if attempt < self.max_attempts:
                await asyncio.sleep(min(backoff, self.max_backoff))
                # Keep ``backoff`` bounded so very long retry tails on
                # pathological configs don't grow the variable beyond
                # its useful clamped range.
                backoff = min(backoff * self.backoff_factor, self.max_backoff)

        logger.warning(
            "governance_bridge_emit_exhausted",
            signal_type=signal.signal_type,
            service=signal.service,
            attempts=self.max_attempts,
            last_status=last_status,
            last_error=last_error,
        )
        return EmitResult(
            ok=False,
            status_code=last_status,
            signal_type=signal.signal_type,
            attempts=self.max_attempts,
            error=last_error,
        )
