"""HTTP client for the nthlayer-core API.

Workers and bench communicate with core exclusively through this client.
Provides typed methods for all core endpoints with retry on transient failures.

Canonical import: ``from nthlayer_common.api_client import CoreAPIClient``
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Transient HTTP status codes that trigger retry
_TRANSIENT_CODES = frozenset({502, 503, 504, 429})


@dataclass
class APIResult:
    """Result from a core API call. Check ok before using data."""

    ok: bool
    status_code: int
    data: Any = None
    error: str | None = None
    detail: dict | None = None


@dataclass
class CoreAPIClient:
    """Async HTTP client for nthlayer-core API.

    Retries on transient failures (connection refused, 502, 503, 504, 429)
    with exponential backoff. Never raises on API errors — returns APIResult.
    """

    base_url: str = "http://localhost:8000"
    max_retries: int = 3
    initial_backoff: float = 1.0
    max_backoff: float = 30.0
    timeout: float = 30.0
    _client: httpx.AsyncClient = field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")

    async def __aenter__(self) -> CoreAPIClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            return self._client

    async def _reset_client(self) -> None:
        """Close current client so next call gets a fresh one."""
        async with self._lock:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
            self._client = None

    async def close(self) -> None:
        await self._reset_client()

    async def _request(
        self,
        method: str,
        path: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> APIResult:
        """Make an HTTP request with retry on transient failures."""
        backoff = self.initial_backoff
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.request(method, path, json=json, params=params)

                if response.status_code in _TRANSIENT_CODES and attempt < self.max_retries:
                    logger.warning(
                        "core_api_transient status=%d path=%s attempt=%d",
                        response.status_code, path, attempt + 1,
                    )
                    await asyncio.sleep(min(backoff, self.max_backoff))
                    backoff *= 2
                    continue

                # Parse response
                try:
                    body = response.json()
                except Exception:
                    body = None

                if 200 <= response.status_code < 300:
                    return APIResult(ok=True, status_code=response.status_code, data=body)

                # Error path: extract meaningful error text
                if isinstance(body, dict):
                    error_msg = body.get("error", f"http_{response.status_code}")
                    detail = body.get("detail")
                else:
                    error_msg = (response.text[:500] if response.text else f"http_{response.status_code}")
                    detail = None

                return APIResult(
                    ok=False,
                    status_code=response.status_code,
                    error=error_msg,
                    detail=detail,
                )

            except (httpx.ConnectError, httpx.ConnectTimeout, OSError) as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "core_api_connection_error error=%s path=%s attempt=%d",
                        e, path, attempt + 1,
                    )
                    await asyncio.sleep(min(backoff, self.max_backoff))
                    backoff *= 2
                    await self._reset_client()
                    continue

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "core_api_timeout error=%s path=%s attempt=%d",
                        e, path, attempt + 1,
                    )
                    await asyncio.sleep(min(backoff, self.max_backoff))
                    backoff *= 2
                    continue

        # All retries exhausted
        return APIResult(
            ok=False,
            status_code=0,
            error="connection_failed",
            detail={"message": f"All {self.max_retries + 1} attempts failed: {last_error}"},
        )

    # -- Health --

    async def health(self) -> APIResult:
        return await self._request("GET", "/health")

    # -- Verdicts --

    async def submit_verdict(self, verdict: dict) -> APIResult:
        return await self._request("POST", "/verdicts", json=verdict)

    async def get_verdict(self, verdict_id: str) -> APIResult:
        return await self._request("GET", f"/verdicts/{verdict_id}")

    async def get_verdicts(
        self,
        *,
        service: str | None = None,
        verdict_type: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
    ) -> APIResult:
        params = {"limit": str(limit)}
        if service:
            params["service"] = service
        if verdict_type:
            params["type"] = verdict_type
        if created_after:
            params["created_after"] = created_after
        if created_before:
            params["created_before"] = created_before
        return await self._request("GET", "/verdicts", params=params)

    async def get_ancestors(self, verdict_id: str, max_hops: int | None = None) -> APIResult:
        params = {}
        if max_hops is not None:
            params["max_hops"] = str(max_hops)
        return await self._request("GET", f"/verdicts/{verdict_id}/ancestors", params=params)

    async def get_descendants(self, verdict_id: str) -> APIResult:
        return await self._request("GET", f"/verdicts/{verdict_id}/descendants")

    async def resolve_outcome(self, verdict_id: str, outcome: dict) -> APIResult:
        return await self._request("POST", f"/verdicts/{verdict_id}/outcome", json=outcome)

    async def apply_override(
        self,
        verdict_id: str,
        payload: dict,
    ) -> APIResult:
        """Apply an operator override to a verdict (opensrm-jmy.18).

        Calls POST /verdicts/{verdict_id}/override on the core API.

        Status code mapping (interpret via result.status_code, not result.ok):
            200 - applied (including idempotent re-apply)
            404 - verdict_not_found (no record or concurrent delete)
            409 - conflict (existing override differs OR CAS miss)
            422 - validation_error (terminal status block or schema failure)
            0   - connection failed (transport layer; result.error populated)

        Does not raise; check result.ok and result.status_code.
        """
        return await self._request("POST", f"/verdicts/{verdict_id}/override", json=payload)

    # -- Assessments --

    async def submit_assessment(self, assessment: dict) -> APIResult:
        return await self._request("POST", "/assessments", json=assessment)

    async def get_assessments(
        self,
        *,
        service: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> APIResult:
        params = {"limit": str(limit)}
        if service:
            params["service"] = service
        if kind:
            params["kind"] = kind
        return await self._request("GET", "/assessments", params=params)

    # -- Cases --

    async def create_case(self, case: dict) -> APIResult:
        return await self._request("POST", "/cases", json=case)

    async def get_case(self, case_id: str) -> APIResult:
        return await self._request("GET", f"/cases/{case_id}")

    async def get_cases(
        self,
        *,
        state: str | None = None,
        priority: str | None = None,
        service: str | None = None,
        limit: int = 100,
    ) -> APIResult:
        params = {"limit": str(limit)}
        if state:
            params["state"] = state
        if priority:
            params["priority"] = priority
        if service:
            params["service"] = service
        return await self._request("GET", "/cases", params=params)

    async def acquire_lease(self, case_id: str, holder: str, expires_at: str) -> APIResult:
        return await self._request(
            "PUT", f"/cases/{case_id}/lease",
            json={"holder": holder, "expires_at": expires_at},
        )

    async def release_lease(self, case_id: str) -> APIResult:
        return await self._request("DELETE", f"/cases/{case_id}/lease")

    async def resolve_case(self, case_id: str, resolution_id: str) -> APIResult:
        return await self._request(
            "PUT", f"/cases/{case_id}/resolve",
            json={"resolution_id": resolution_id},
        )

    # -- Change Freezes --

    async def create_change_freeze(self, freeze: dict) -> APIResult:
        return await self._request("POST", "/change-freezes", json=freeze)

    async def get_active_freezes(self) -> APIResult:
        return await self._request("GET", "/change-freezes")

    async def lift_change_freeze(self, name: str, lifted_by: str) -> APIResult:
        return await self._request(
            "PUT", f"/change-freezes/{name}/lift",
            json={"lifted_by": lifted_by},
        )

    # -- Heartbeats --

    async def heartbeat(self, component: str, instance_id: str, state: dict | None = None) -> APIResult:
        return await self._request(
            "POST", "/heartbeats",
            json={"component": component, "instance_id": instance_id, "state": state},
        )

    async def get_heartbeats(self, threshold: int = 30) -> APIResult:
        return await self._request("GET", "/heartbeats", params={"threshold": str(threshold)})

    # -- Manifests --

    async def get_manifests(self) -> APIResult:
        return await self._request("GET", "/manifests")

    async def get_manifest(self, service: str) -> APIResult:
        return await self._request("GET", f"/manifests/{service}")

    # -- Component State --

    async def put_component_state(self, component: str, state: dict) -> APIResult:
        return await self._request("PUT", f"/component-state/{component}", json=state)

    async def get_component_state(self, component: str) -> APIResult:
        return await self._request("GET", f"/component-state/{component}")

    # -- Monitoring --

    async def get_stuck_action_requests(self, threshold: int = 60) -> APIResult:
        return await self._request(
            "GET", "/monitoring/stuck-action-requests",
            params={"threshold": str(threshold)},
        )
