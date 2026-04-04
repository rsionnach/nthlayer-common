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


def _patch_client(client, mock_response=None, side_effect=None):
    """Inject a mock httpx client into the SlackWebClient."""
    mock_http = MagicMock()
    mock_http.is_closed = False
    if side_effect:
        mock_http.post = AsyncMock(side_effect=side_effect)
    else:
        mock_http.post = AsyncMock(return_value=mock_response)
    client._client = mock_http
    return mock_http


async def test_post_message_returns_ts(client):
    """post_message sends to Slack Web API and returns ts."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "ts": "1234567890.123456"}
    mock_response.raise_for_status = MagicMock()

    mock_http = _patch_client(client, mock_response=mock_response)

    ts = await client.post_message(
        channel="C12345",
        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "test"}}],
        text="test",
    )
    assert ts == "1234567890.123456"
    mock_http.post.assert_called_once()
    call_kwargs = mock_http.post.call_args
    assert "chat.postMessage" in call_kwargs[0][0]


async def test_post_message_with_thread_ts(client):
    """post_message includes thread_ts when provided."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "ts": "111.222"}
    mock_response.raise_for_status = MagicMock()

    mock_http = _patch_client(client, mock_response=mock_response)

    await client.post_message(
        channel="C12345",
        blocks=[],
        text="reply",
        thread_ts="999.888",
    )
    call_kwargs = mock_http.post.call_args
    payload = call_kwargs[1]["json"]
    assert payload["thread_ts"] == "999.888"


async def test_post_message_fail_open(client):
    """post_message returns None on error, never raises."""
    _patch_client(client, side_effect=Exception("network error"))

    ts = await client.post_message("C12345", [], "test")
    assert ts is None


async def test_update_message(client):
    """update_message calls chat.update."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()

    mock_http = _patch_client(client, mock_response=mock_response)

    await client.update_message(
        channel="C12345",
        ts="1234567890.123456",
        blocks=[],
        text="updated",
    )
    call_kwargs = mock_http.post.call_args
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


async def test_close_client(client):
    """close() closes the underlying httpx client."""
    mock_http = MagicMock()
    mock_http.is_closed = False
    mock_http.aclose = AsyncMock()
    client._client = mock_http

    await client.close()
    mock_http.aclose.assert_called_once()
    assert client._client is None
