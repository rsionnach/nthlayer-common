"""Tests for tier-boundary error types and retry decorator."""


import pytest

from nthlayer_common.errors import (
    DegradedError,
    NthLayerError,
    PermanentError,
    TransientError,
    classify_http_error,
    retry,
)


class TestErrorHierarchy:
    def test_transient_is_nthlayer_error(self):
        assert issubclass(TransientError, NthLayerError)

    def test_permanent_is_nthlayer_error(self):
        assert issubclass(PermanentError, NthLayerError)

    def test_degraded_is_nthlayer_error(self):
        assert issubclass(DegradedError, NthLayerError)

    def test_transient_has_details(self):
        e = TransientError("core unavailable", {"status": 503})
        assert e.message == "core unavailable"
        assert e.details["status"] == 503

    def test_permanent_has_details(self):
        e = PermanentError("bad input", {"field": "id"})
        assert e.message == "bad input"
        assert e.details["field"] == "id"


class TestClassifyHTTPError:
    def test_503_is_transient(self):
        e = classify_http_error(503, "unavailable")
        assert isinstance(e, TransientError)

    def test_502_is_transient(self):
        assert isinstance(classify_http_error(502), TransientError)

    def test_504_is_transient(self):
        assert isinstance(classify_http_error(504), TransientError)

    def test_429_is_transient(self):
        assert isinstance(classify_http_error(429), TransientError)

    def test_400_is_permanent(self):
        assert isinstance(classify_http_error(400), PermanentError)

    def test_404_is_permanent(self):
        assert isinstance(classify_http_error(404), PermanentError)

    def test_422_is_permanent(self):
        assert isinstance(classify_http_error(422), PermanentError)

    def test_409_is_permanent(self):
        assert isinstance(classify_http_error(409), PermanentError)

    def test_unknown_5xx_is_transient(self):
        assert isinstance(classify_http_error(500), TransientError)
        assert isinstance(classify_http_error(599), TransientError)

    def test_unknown_4xx_is_permanent(self):
        assert isinstance(classify_http_error(418), PermanentError)

    def test_detail_forwarded(self):
        e = classify_http_error(503, "down", {"retry_after": 30})
        assert e.details["retry_after"] == 30


class TestRetryDecorator:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self):
        call_count = 0

        @retry(on=TransientError, max_attempts=3, initial_backoff=0.01)
        async def my_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await my_func()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_then_succeeds(self):
        call_count = 0

        @retry(on=TransientError, max_attempts=3, initial_backoff=0.01)
        async def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("temporary")
            return "recovered"

        result = await my_func()
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        @retry(on=TransientError, max_attempts=2, initial_backoff=0.01)
        async def my_func():
            raise TransientError("always fails")

        with pytest.raises(TransientError, match="always fails"):
            await my_func()

    @pytest.mark.asyncio
    async def test_permanent_error_not_retried(self):
        call_count = 0

        @retry(on=TransientError, max_attempts=3, initial_backoff=0.01)
        async def my_func():
            nonlocal call_count
            call_count += 1
            raise PermanentError("bad input")

        with pytest.raises(PermanentError):
            await my_func()
        assert call_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_multiple_exception_types(self):
        call_count = 0

        @retry(on=(TransientError, DegradedError), max_attempts=3, initial_backoff=0.01)
        async def my_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TransientError("temp")
            if call_count == 2:
                raise DegradedError("degraded")
            return "ok"

        result = await my_func()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_backoff_increases(self):
        """Verify backoff factor is applied between retries."""
        from unittest.mock import patch as _patch

        sleeps = []

        async def fake_sleep(t):
            sleeps.append(t)

        @retry(on=TransientError, max_attempts=3, initial_backoff=0.01, backoff_factor=2.0, max_backoff=1.0)
        async def my_func():
            raise TransientError("fail")

        with _patch("asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(TransientError):
                await my_func()

        assert len(sleeps) == 2  # 3 attempts = 2 sleeps
        assert sleeps[1] > sleeps[0]  # backoff increased

    def test_retry_rejects_sync_function(self):
        """@retry must reject synchronous functions at decoration time."""
        with pytest.raises(TypeError, match="async"):
            @retry(on=TransientError)
            def sync_func():
                pass
