"""Shared verdict bridge for building content-addressed Verdict records.

Used by nthlayer-respond, nthlayer-correlate, and nthlayer-measure to convert
their verdict outputs into content-addressed decision records.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from nthlayer_common.records.hashing import canonical_json, compute_hash
from nthlayer_common.records.models import ZERO_HASH, Summaries, Verdict, VerdictOutcome

__all__ = ["build_decision_verdict", "hash_content", "write_decision_verdict"]

_log = logging.getLogger(__name__)

_MAX_RESPONSE_TEXT = 10_000  # truncate large response texts before hashing/storing


def hash_content(text: str) -> str:
    """SHA-256 hex digest of a text string (for prompt/response hashing)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_utc(ts: datetime) -> datetime:
    """Normalize a datetime to UTC. Naive datetimes are assumed UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def build_decision_verdict(
    *,
    agent: str,
    incident_id: str,
    timestamp: datetime,
    model: str,
    reasoning: str,
    action: dict[str, Any],
    outcome: VerdictOutcome = VerdictOutcome.RECOMMENDED,
    prompt_text: str,
    response_text: str,
    input_hashes: list[str],
    previous_hash: str,
    summaries_technical: str,
    summaries_plain: str,
    summaries_executive: str | None = None,
) -> Verdict:
    """Build a content-addressed Verdict record.

    Computes prompt_hash and response_hash from the raw text, then computes
    the record's SHA-256 hash from its canonical form.

    Timestamps are normalized to UTC (naive datetimes assumed UTC).
    Response text is truncated to 10KB before hashing.
    """
    timestamp = _ensure_utc(timestamp)
    response_text = response_text[:_MAX_RESPONSE_TEXT]
    prompt_hash = hash_content(prompt_text)
    response_hash = hash_content(response_text)
    summaries = Summaries(
        technical=summaries_technical[:280],
        plain=summaries_plain[:280],
        executive=summaries_executive[:140] if summaries_executive else None,
    )

    # Build placeholder to compute canonical form (hash/previous_hash excluded)
    placeholder = Verdict(
        hash="placeholder",
        previous_hash=previous_hash,
        schema_version="verdict/v1",
        timestamp=timestamp,
        agent=agent,
        incident_id=incident_id,
        input_hashes=input_hashes,
        prompt_hash=prompt_hash,
        response_hash=response_hash,
        model=model,
        reasoning=reasoning,
        action=action,
        outcome=outcome,
        summaries=summaries,
    )
    canonical = canonical_json(placeholder)
    record_hash = compute_hash(canonical)

    return Verdict(
        hash=record_hash,
        previous_hash=previous_hash,
        schema_version="verdict/v1",
        timestamp=timestamp,
        agent=agent,
        incident_id=incident_id,
        input_hashes=input_hashes,
        prompt_hash=prompt_hash,
        response_hash=response_hash,
        model=model,
        reasoning=reasoning,
        action=action,
        outcome=outcome,
        summaries=summaries,
    )


def write_decision_verdict(
    store: Any,
    *,
    agent: str,
    incident_id: str,
    timestamp: datetime,
    model: str,
    reasoning: str,
    action: dict[str, Any],
    outcome: VerdictOutcome = VerdictOutcome.RECOMMENDED,
    prompt_text: str,
    response_text: str,
    input_hashes: list[str] | None = None,
    summaries_technical: str,
    summaries_plain: str,
    summaries_executive: str | None = None,
    max_retries: int = 1,
) -> None:
    """Build and write a content-addressed Verdict record with retry on chain fork.

    Reads the chain tail, builds the record, writes it. On chain fork (concurrent
    writer), retries once with a refreshed chain tail.

    Stores prompt and response text alongside the verdict.
    Fail-open: all errors logged, never raised.
    """
    from nthlayer_common.records.errors import ChainForkError

    for attempt in range(max_retries + 1):
        try:
            tail = store.get_chain_tail("verdict", agent)
            previous_hash = tail.hash if tail else ZERO_HASH

            record = build_decision_verdict(
                agent=agent,
                incident_id=incident_id,
                timestamp=timestamp,
                model=model,
                reasoning=reasoning,
                action=action,
                outcome=outcome,
                prompt_text=prompt_text,
                response_text=response_text,
                input_hashes=input_hashes or [],
                previous_hash=previous_hash,
                summaries_technical=summaries_technical,
                summaries_plain=summaries_plain,
                summaries_executive=summaries_executive,
            )
            store.put_verdict(record)
            store.put_prompt(record.prompt_hash, prompt_text)
            store.put_response(record.response_hash, response_text[:_MAX_RESPONSE_TEXT])
            return
        except ChainForkError:
            if attempt < max_retries:
                _log.warning("Chain fork on agent=%s, retrying with refreshed tail", agent)
                continue
            _log.error("Chain fork on agent=%s after %d retries, record dropped", agent, max_retries)
            return
        except Exception as exc:
            _log.warning("Decision verdict write failed for agent=%s: %s", agent, exc)
            return
