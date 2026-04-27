"""Tests for CoreAPIClient.

Tests use a mock HTTP server (httpx MockTransport) to verify client behavior
without requiring a running core process.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nthlayer_common.api_client import APIResult, CoreAPIClient


class TestAPIResult:
    def test_ok_result(self):
        r = APIResult(ok=True, status_code=200, data={"id": "v1"})
        assert r.ok
        assert r.data["id"] == "v1"

    def test_error_result(self):
        r = APIResult(ok=False, status_code=404, error="not_found")
        assert not r.ok
        assert r.error == "not_found"


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_success(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=200, data={"status": "ok"})
            result = await client.health()
            assert result.ok
            mock.assert_called_once_with("GET", "/health")


class TestVerdictMethods:
    @pytest.mark.asyncio
    async def test_submit_verdict(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=201, data={"id": "vrd-001"})
            verdict = {"id": "vrd-001", "type": "action_request", "created_at": "2026-04-23T00:00:00Z"}
            result = await client.submit_verdict(verdict)
            assert result.ok
            assert result.data["id"] == "vrd-001"
            mock.assert_called_once_with("POST", "/verdicts", json=verdict)

    @pytest.mark.asyncio
    async def test_get_verdict(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=200, data={"id": "vrd-001"})
            result = await client.get_verdict("vrd-001")
            assert result.ok
            mock.assert_called_once_with("GET", "/verdicts/vrd-001")

    @pytest.mark.asyncio
    async def test_get_verdicts_with_filters(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=200, data=[])
            await client.get_verdicts(service="fraud-detect", verdict_type="quality_breach", limit=50)
            call_args = mock.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params["service"] == "fraud-detect"
            assert params["type"] == "quality_breach"
            assert params["limit"] == "50"

    @pytest.mark.asyncio
    async def test_get_ancestors(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=200, data=[])
            await client.get_ancestors("vrd-001", max_hops=2)
            call_args = mock.call_args
            assert call_args[0] == ("GET", "/verdicts/vrd-001/ancestors")
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params["max_hops"] == "2"

    @pytest.mark.asyncio
    async def test_resolve_outcome(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=201, data={"id": "out-vrd-001"})
            result = await client.resolve_outcome("vrd-001", {"outcome_status": "confirmed"})
            assert result.ok
            mock.assert_called_once_with("POST", "/verdicts/vrd-001/outcome", json={"outcome_status": "confirmed"})


class TestCaseMethods:
    @pytest.mark.asyncio
    async def test_create_case(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=201, data={"id": "c1", "priority": "P0"})
            result = await client.create_case({"id": "c1", "kind": "approval_required"})
            assert result.data["priority"] == "P0"

    @pytest.mark.asyncio
    async def test_acquire_lease(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=200, data={"leased": True})
            result = await client.acquire_lease("c1", "op-1", "2026-04-23T01:00:00Z")
            assert result.data["leased"] is True
            call_args = mock.call_args
            assert call_args[1]["json"]["holder"] == "op-1"

    @pytest.mark.asyncio
    async def test_resolve_case(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=200, data={"resolved": True})
            result = await client.resolve_case("c1", "vrd-resolution")
            assert result.data["resolved"] is True


class TestHeartbeatMethods:
    @pytest.mark.asyncio
    async def test_heartbeat(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=200, data={"ok": True})
            await client.heartbeat("workers", "i-001", {"cycles": 42})
            call_args = mock.call_args
            body = call_args[1]["json"]
            assert body["component"] == "workers"
            assert body["instance_id"] == "i-001"
            assert body["state"]["cycles"] == 42


class TestRetryBehavior:
    @pytest.mark.asyncio
    async def test_retries_on_503_then_succeeds(self):
        """Client retries on 503 and succeeds on third attempt."""
        attempt = 0

        def handler(request):
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                return httpx.Response(503, json={"error": "unavailable"})
            return httpx.Response(200, json={"status": "ok"})

        transport = httpx.MockTransport(handler)
        client = CoreAPIClient(base_url="http://test", max_retries=2, initial_backoff=0.01)
        client._client = httpx.AsyncClient(transport=transport, base_url="http://test")
        result = await client._request("GET", "/health")
        assert result.ok
        assert attempt == 3
        await client.close()

    @pytest.mark.asyncio
    async def test_retries_exhausted_returns_last_error(self):
        """When all retries fail with 503, returns the 503 result."""
        def handler(request):
            return httpx.Response(503, json={"error": "unavailable"})

        transport = httpx.MockTransport(handler)
        client = CoreAPIClient(base_url="http://test", max_retries=1, initial_backoff=0.01)
        client._client = httpx.AsyncClient(transport=transport, base_url="http://test")
        result = await client._request("GET", "/health")
        assert not result.ok
        assert result.status_code == 503
        await client.close()

    @pytest.mark.asyncio
    async def test_4xx_not_retried(self):
        """4xx errors are returned immediately, not retried."""
        attempt = 0

        def handler(request):
            nonlocal attempt
            attempt += 1
            return httpx.Response(422, json={"error": "missing_fields"})

        transport = httpx.MockTransport(handler)
        client = CoreAPIClient(base_url="http://test", max_retries=3, initial_backoff=0.01)
        client._client = httpx.AsyncClient(transport=transport, base_url="http://test")
        result = await client._request("POST", "/verdicts")
        assert not result.ok
        assert result.status_code == 422
        assert attempt == 1  # No retry
        await client.close()

    @pytest.mark.asyncio
    async def test_connection_refused_returns_error(self):
        """Connection refused after retries returns APIResult, never raises."""
        client = CoreAPIClient(base_url="http://localhost:1", max_retries=1, initial_backoff=0.01)
        result = await client._request("GET", "/health")
        assert not result.ok
        assert result.status_code == 0
        assert result.error == "connection_failed"
        assert "attempts failed" in result.detail["message"]
        await client.close()

    @pytest.mark.asyncio
    async def test_non_json_error_preserves_content(self):
        """Non-JSON error body (e.g., HTML from nginx) captures text, not 'None'."""
        def handler(request):
            return httpx.Response(
                502,
                content=b"<html>Bad Gateway</html>",
                headers={"content-type": "text/html"},
            )

        transport = httpx.MockTransport(handler)
        client = CoreAPIClient(base_url="http://test", max_retries=0, initial_backoff=0.01)
        client._client = httpx.AsyncClient(transport=transport, base_url="http://test")
        result = await client._request("GET", "/health")
        assert not result.ok
        assert result.status_code == 502
        assert "None" not in result.error
        assert "Bad Gateway" in result.error
        await client.close()


class TestManifestMethods:
    @pytest.mark.asyncio
    async def test_get_manifests(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=200, data=[{"name": "fraud-detect"}])
            result = await client.get_manifests()
            assert result.ok
            assert len(result.data) == 1
            mock.assert_called_once_with("GET", "/manifests")

    @pytest.mark.asyncio
    async def test_get_manifest(self):
        client = CoreAPIClient(base_url="http://test", max_retries=0)
        with patch.object(client, "_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResult(ok=True, status_code=200, data={"name": "fraud-detect"})
            result = await client.get_manifest("fraud-detect")
            assert result.ok
            assert result.data["name"] == "fraud-detect"
            mock.assert_called_once_with("GET", "/manifests/fraud-detect")


class TestContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Client can be used as async context manager."""
        async with CoreAPIClient(base_url="http://test", max_retries=0) as client:
            assert client.base_url == "http://test"

    @pytest.mark.asyncio
    async def test_trailing_slash_stripped(self):
        """Trailing slash in base_url is normalized."""
        client = CoreAPIClient(base_url="http://test:8000/")
        assert client.base_url == "http://test:8000"
        await client.close()
