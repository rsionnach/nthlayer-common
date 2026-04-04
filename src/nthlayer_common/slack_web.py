"""Slack Web API client for interactive messages (buttons, message updates).

Complements SlackNotifier (incoming webhooks) with Web API features:
- chat.postMessage: send messages with interactive buttons
- chat.update: replace buttons after action taken
- Signature verification: validate Slack interaction callbacks

Fail-open: errors log warnings and return None, never raise.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SLACK_API_BASE = "https://slack.com/api"


class SlackWebClient:
    """Slack Web API client for interactive messages.

    Reuses a single httpx.AsyncClient for connection pooling across calls.
    The client is created lazily on first use.
    """

    def __init__(self, bot_token: str) -> None:
        self.bot_token = bot_token
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared httpx client, creating it on first call."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def post_message(
        self,
        channel: str,
        blocks: list[dict[str, Any]],
        text: str,
        thread_ts: str | None = None,
    ) -> str | None:
        """Post via chat.postMessage. Returns message ts, or None on failure."""
        if not self.bot_token:
            return None

        payload: dict[str, Any] = {
            "channel": channel,
            "blocks": blocks,
            "text": text,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts

        try:
            client = self._get_client()
            resp = await client.post(
                f"{SLACK_API_BASE}/chat.postMessage",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                logger.warning("Slack API error: %s", data.get("error"))
                return None
            return data.get("ts")
        except Exception as exc:
            logger.warning("Slack post_message failed: %s", exc)
            return None

    async def update_message(
        self,
        channel: str,
        ts: str,
        blocks: list[dict[str, Any]],
        text: str,
    ) -> None:
        """Update a message via chat.update (e.g. remove buttons after action)."""
        if not self.bot_token:
            return

        payload: dict[str, Any] = {
            "channel": channel,
            "ts": ts,
            "blocks": blocks,
            "text": text,
        }

        try:
            client = self._get_client()
            resp = await client.post(
                f"{SLACK_API_BASE}/chat.update",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                logger.warning("Slack chat.update error: %s", data.get("error"))
        except Exception as exc:
            logger.warning("Slack update_message failed: %s", exc)

    @staticmethod
    def verify_signature(
        signing_secret: str, timestamp: str, body: bytes, signature: str
    ) -> bool:
        """Verify Slack request signature (HMAC-SHA256).

        Returns False if signature is invalid or timestamp is stale (>5 min).
        """
        try:
            ts = int(timestamp)
        except (ValueError, TypeError):
            return False
        if abs(time.time() - ts) > 300:
            return False

        try:
            body_str = body.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            return False

        sig_basestring = f"v0:{timestamp}:{body_str}"
        expected = "v0=" + hmac.new(
            signing_secret.encode(), sig_basestring.encode(), hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected, signature)
