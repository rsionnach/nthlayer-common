"""Tests for SlackNotifier transport."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nthlayer_common.slack import SlackNotifier


class TestSlackNotifier:
    @pytest.mark.asyncio
    async def test_send_posts_to_webhook(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.text = "ok"
        mock_resp.raise_for_status = MagicMock()

        with patch("nthlayer_common.slack.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            notifier = SlackNotifier("https://hooks.slack.com/test")
            await notifier.send(
                [{"type": "section", "text": {"type": "mrkdwn", "text": "test"}}],
                "fallback text",
            )

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "hooks.slack.com" in call_args[0][0]
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["text"] == "fallback text"
        assert len(body["blocks"]) == 1

    @pytest.mark.asyncio
    async def test_send_with_thread_ts(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.raise_for_status = MagicMock()

        with patch("nthlayer_common.slack.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            notifier = SlackNotifier("https://hooks.slack.com/test")
            await notifier.send([], "test", thread_ts="parent_ts")

        body = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
        assert body["thread_ts"] == "parent_ts"

    @pytest.mark.asyncio
    async def test_send_without_thread_ts_no_field(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.raise_for_status = MagicMock()

        with patch("nthlayer_common.slack.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            notifier = SlackNotifier("https://hooks.slack.com/test")
            await notifier.send([], "test")

        body = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
        assert "thread_ts" not in body

    @pytest.mark.asyncio
    async def test_send_returns_ts_from_json_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"ok": True, "ts": "1234567890.123456"}
        mock_resp.raise_for_status = MagicMock()

        with patch("nthlayer_common.slack.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            notifier = SlackNotifier("https://hooks.slack.com/test")
            ts = await notifier.send([], "test")

        assert ts == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_send_failure_returns_none(self):
        with patch("nthlayer_common.slack.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            notifier = SlackNotifier("https://hooks.slack.com/test")
            ts = await notifier.send([], "test")

        assert ts is None

    @pytest.mark.asyncio
    async def test_empty_webhook_url_returns_none(self):
        notifier = SlackNotifier("")
        ts = await notifier.send([], "test")
        assert ts is None
