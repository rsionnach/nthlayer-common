"""Mimir Ruler API client — push alert/recording rules to Mimir/Cortex.

This is a BaseHTTPClient subclass (standalone HTTP client with retry +
circuit breaker), NOT a Provider protocol implementer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nthlayer_common.clients.base import BaseHTTPClient, PermanentHTTPError, RetryableHTTPError
from nthlayer_common.errors import ProviderError

DEFAULT_USER_AGENT = "nthlayer-provider-mimir/0.1.0"


class MimirRulerError(ProviderError):
    """Error communicating with Mimir Ruler API."""


@dataclass
class RulerPushResult:
    """Result of pushing rules to Mimir."""

    success: bool
    namespace: str
    status_code: int = 0
    message: str = ""
    groups_pushed: int = 0


class MimirRulerProvider(BaseHTTPClient):
    """Push alert rules to Mimir/Cortex Ruler API.

    API endpoints:
        POST /api/v1/rules/{namespace} - Create/update rule groups
        DELETE /api/v1/rules/{namespace}/{groupName} - Delete rule group
        GET /api/v1/rules - List all rules
    """

    def __init__(
        self,
        ruler_url: str,
        *,
        tenant_id: str | None = None,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 30.0,
        user_agent: str = DEFAULT_USER_AGENT,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        super().__init__(
            base_url=ruler_url,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
        self._tenant_id = tenant_id
        self._api_key = api_key
        self._user_agent = user_agent
        self._auth = (username, password) if username and password else None

    def _auth_tuple(self) -> tuple[str, str] | None:
        """Provide basic auth credentials if configured."""
        return self._auth

    def _headers(self) -> dict[str, str]:
        """Build request headers with auth via _headers() override."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": self._user_agent,
        }
        if self._tenant_id:
            headers["X-Scope-OrgID"] = self._tenant_id
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def push_rules(self, namespace: str, rules_yaml: str) -> RulerPushResult:
        """Push rule groups to a namespace."""
        try:
            await self._request(
                "POST",
                f"/api/v1/rules/{namespace}",
                content=rules_yaml,
                headers={"Content-Type": "application/yaml"},
            )
            groups_count = rules_yaml.count("- name:")
            return RulerPushResult(
                success=True,
                namespace=namespace,
                status_code=200,
                message="Rules pushed successfully",
                groups_pushed=groups_count,
            )
        except (RetryableHTTPError, PermanentHTTPError) as e:
            raise MimirRulerError(
                f"Mimir push failed for {self._base_url}: {e}"
            ) from e

    async def delete_rules(
        self, namespace: str, group_name: str | None = None
    ) -> bool:
        """Delete rules from a namespace."""
        path = f"/api/v1/rules/{namespace}"
        if group_name:
            path = f"{path}/{group_name}"
        try:
            await self._request("DELETE", path)
            return True
        except (RetryableHTTPError, PermanentHTTPError) as e:
            raise MimirRulerError(f"Failed to delete rules: {e}") from e

    async def list_rules(self) -> dict[str, Any]:
        """List all rules across all namespaces."""
        try:
            return await self._request("GET", "/api/v1/rules")
        except (RetryableHTTPError, PermanentHTTPError) as e:
            raise MimirRulerError(f"Failed to list rules: {e}") from e

    async def health_check(self) -> bool:
        """Check if Mimir Ruler is reachable."""
        try:
            await self.list_rules()
            return True
        except MimirRulerError:
            return False
