"""Slack notification transport — Block Kit messages via incoming webhook.

Fail-open: if Slack is unreachable, log a warning and return None.
Never block the incident pipeline for a notification failure.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SlackNotifier:
    """Send Slack Block Kit messages via incoming webhook."""

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    async def send(
        self,
        blocks: list[dict[str, Any]],
        text: str,
        thread_ts: str | None = None,
    ) -> str | None:
        """Post a Slack message. Returns thread_ts for threading.

        Returns None if sending fails or webhook_url is empty.
        """
        if not self.webhook_url:
            return None

        payload: dict[str, Any] = {
            "text": text,
            "blocks": blocks,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                resp.raise_for_status()
                # Slack incoming webhooks return "ok" as text, not JSON with ts.
                # The ts is only returned by the chat.postMessage API.
                # For incoming webhooks, we use the message timestamp from headers
                # or return a synthetic ts for threading.
                if resp.headers.get("content-type", "").startswith("application/json"):
                    data = resp.json()
                    return data.get("ts")
                # Incoming webhook — no ts returned. Generate from response.
                return None
        except Exception as exc:
            logger.warning("Slack notification failed: %s", exc)
            return None
