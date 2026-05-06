"""Tests for nthlayer_common.providers module."""

from unittest.mock import AsyncMock, patch

import pytest

from nthlayer_common.providers.base import (
    PlanChange,
    PlanResult,
    ProviderHealth,
    ProviderResourceSchema,
)
from nthlayer_common.providers.lock import (
    ProviderLock,
    load_lock,
    save_lock,
)
from nthlayer_common.providers.prometheus import (
    DEFAULT_USER_AGENT,
    PrometheusProvider,
    PrometheusProviderError,
)
from nthlayer_common.providers.registry import (
    ProviderRegistry,
)
from nthlayer_common.errors import ProviderError


class TestProviderResourceSchema:
    def test_frozen(self):
        s = ProviderResourceSchema(name="test", description="desc", attributes={"a": "b"})
        assert s.name == "test"
        with pytest.raises(AttributeError):
            s.name = "other"


class TestPlanResult:
    def test_no_changes(self):
        r = PlanResult(changes=[])
        assert not r.has_changes

    def test_has_changes(self):
        c = PlanChange(action="create", details={"name": "test"})
        r = PlanResult(changes=[c])
        assert r.has_changes


class TestProviderHealth:
    def test_healthy(self):
        h = ProviderHealth(status="healthy")
        assert h.status == "healthy"
        assert h.details is None

    def test_unreachable(self):
        h = ProviderHealth(status="unreachable", details="timeout")
        assert h.details == "timeout"


class TestProviderRegistry:
    def test_register_and_create(self):
        reg = ProviderRegistry()
        reg.register("test", lambda: "instance", version="1.0")
        result = reg.create("test")
        assert result == "instance"

    def test_register_empty_name_raises(self):
        reg = ProviderRegistry()
        with pytest.raises(ValueError, match="required"):
            reg.register("", lambda: None)

    def test_create_unknown_raises(self):
        reg = ProviderRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.create("nonexistent")

    def test_list(self):
        reg = ProviderRegistry()
        reg.register("a", lambda: None)
        reg.register("b", lambda: None)
        specs = reg.list()
        assert len(specs) == 2
        names = {s.name for s in specs}
        assert names == {"a", "b"}


class TestProviderLock:
    def test_set_and_get(self):
        lock = ProviderLock()
        lock.set("grafana", "1.0.0")
        assert lock.get("grafana") == "1.0.0"

    def test_get_missing(self):
        lock = ProviderLock()
        assert lock.get("nonexistent") is None

    def test_load_missing_file(self, tmp_path):
        lock = load_lock(tmp_path / "nonexistent.lock")
        assert lock.providers == {}

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "test.lock"
        lock = ProviderLock()
        lock.set("grafana", "1.0.0")
        save_lock(lock, path)

        loaded = load_lock(path)
        assert loaded.get("grafana") == "1.0.0"


class TestPrometheusProvider:
    def test_name(self):
        p = PrometheusProvider("http://localhost:9090")
        assert p.name == "prometheus"

    def test_strips_trailing_slash(self):
        p = PrometheusProvider("http://localhost:9090/")
        assert p._base_url == "http://localhost:9090"

    def test_default_user_agent(self):
        assert "nthlayer" in DEFAULT_USER_AGENT

    def test_error_is_provider_error(self):
        assert issubclass(PrometheusProviderError, ProviderError)

    def test_parse_step_to_seconds(self):
        p = PrometheusProvider("http://localhost:9090")
        assert p._parse_step_to_seconds("5m") == 300.0
        assert p._parse_step_to_seconds("1h") == 3600.0
        assert p._parse_step_to_seconds("30s") == 30.0
        assert p._parse_step_to_seconds("1d") == 86400.0
        assert p._parse_step_to_seconds("unknown") == 300.0  # default

    def test_parse_step_numeric_string(self):
        p = PrometheusProvider("http://localhost:9090")
        assert p._parse_step_to_seconds("300") == 300.0
        assert p._parse_step_to_seconds("60") == 60.0
        assert p._parse_step_to_seconds("3600.5") == 3600.5


class TestGetSliValueNoneVsZero:
    """Pin the no-data-vs-total-outage distinction (opensrm-e1gk).

    Empty result, malformed value tuples, and non-numeric strings all
    return None so callers can distinguish "Prometheus returned no
    data" from "Prometheus returned 0.0" (a real total-outage SLI).
    """

    @pytest.mark.asyncio
    async def test_empty_result_returns_none(self):
        p = PrometheusProvider("http://localhost:9090")
        with patch.object(p, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = {"data": {"result": []}}
            assert await p.get_sli_value("up") is None

    @pytest.mark.asyncio
    async def test_zero_value_returns_zero(self):
        p = PrometheusProvider("http://localhost:9090")
        with patch.object(p, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = {
                "data": {"result": [{"value": [1609459200, "0"]}]}
            }
            assert await p.get_sli_value("up") == 0.0

    @pytest.mark.asyncio
    async def test_short_value_tuple_returns_none(self):
        p = PrometheusProvider("http://localhost:9090")
        with patch.object(p, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = {
                "data": {"result": [{"value": [1609459200]}]}
            }
            assert await p.get_sli_value("up") is None

    @pytest.mark.asyncio
    async def test_non_numeric_value_returns_none(self):
        p = PrometheusProvider("http://localhost:9090")
        with patch.object(p, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = {
                "data": {"result": [{"value": [1609459200, "not-a-number"]}]}
            }
            assert await p.get_sli_value("up") is None

    @pytest.mark.asyncio
    async def test_real_value_returns_float(self):
        p = PrometheusProvider("http://localhost:9090")
        with patch.object(p, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = {
                "data": {"result": [{"value": [1609459200, "0.9995"]}]}
            }
            assert await p.get_sli_value("up") == pytest.approx(0.9995)
