"""Tests for SlackWebClient — Slack Web API for interactive messages."""
from __future__ import annotations

import hashlib
import hmac
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nthlayer_common.slack_web import SlackWebClient


@pytest.fixture
def client():
    return SlackWebClient("xoxb-test-token")


async def test_post_message_returns_ts(client):
    """post_message sends to Slack Web API and returns ts."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "ts": "1234567890.123456"}
    mock_response.raise_for_status = MagicMock()

    with patch("nthlayer_common.slack_web.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        ts = await client.post_message(
            channel="C12345",
            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "test"}}],
            text="test",
        )
        assert ts == "1234567890.123456"
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "chat.postMessage" in call_kwargs[0][0]


async def test_post_message_with_thread_ts(client):
    """post_message includes thread_ts when provided."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "ts": "111.222"}
    mock_response.raise_for_status = MagicMock()

    with patch("nthlayer_common.slack_web.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        await client.post_message(
            channel="C12345",
            blocks=[],
            text="reply",
            thread_ts="999.888",
        )
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["thread_ts"] == "999.888"


async def test_post_message_fail_open(client):
    """post_message returns None on error, never raises."""
    with patch("nthlayer_common.slack_web.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        mock_client_cls.return_value = mock_client

        ts = await client.post_message("C12345", [], "test")
        assert ts is None


async def test_update_message(client):
    """update_message calls chat.update."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()

    with patch("nthlayer_common.slack_web.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        await client.update_message(
            channel="C12345",
            ts="1234567890.123456",
            blocks=[],
            text="updated",
        )
        call_kwargs = mock_client.post.call_args
        assert "chat.update" in call_kwargs[0][0]
        payload = call_kwargs[1]["json"]
        assert payload["ts"] == "1234567890.123456"


def test_verify_signature_valid():
    """verify_signature returns True for valid HMAC."""
    secret = "test-signing-secret"
    timestamp = str(int(time.time()))
    body = b'{"type":"block_actions"}'
    sig_basestring = f"v0:{timestamp}:{body.decode()}"
    expected = "v0=" + hmac.new(
        secret.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()

    assert SlackWebClient.verify_signature(secret, timestamp, body, expected) is True


def test_verify_signature_invalid():
    """verify_signature returns False for tampered signature."""
    assert SlackWebClient.verify_signature(
        "secret", "12345", b"body", "v0=bad"
    ) is False


def test_verify_signature_stale_timestamp():
    """verify_signature returns False for timestamp older than 5 minutes."""
    secret = "secret"
    stale_ts = str(int(time.time()) - 600)
    body = b"body"
    sig_basestring = f"v0:{stale_ts}:{body.decode()}"
    sig = "v0=" + hmac.new(
        secret.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()

    assert SlackWebClient.verify_signature(secret, stale_ts, body, sig) is False


async def test_empty_token_returns_none():
    """SlackWebClient with empty token returns None immediately."""
    client = SlackWebClient("")
    ts = await client.post_message("C12345", [], "test")
    assert ts is None
