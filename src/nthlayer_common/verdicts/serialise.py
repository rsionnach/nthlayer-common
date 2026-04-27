"""Verdict serialisation: JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from nthlayer_common.verdicts.models import (
    GroundTruth,
    Judgment,
    Lineage,
    Metadata,
    Outcome,
    Override,
    Producer,
    Subject,
    Verdict,
    TTL_DEFAULT,
)


def _prepare_value(v: Any) -> Any:
    """Recursively convert datetime objects to ISO strings for JSON output."""
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _prepare_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_prepare_value(item) for item in v]
    return v


def to_dict(verdict: Verdict) -> dict:
    """Convert a verdict to a plain dict suitable for JSON serialisation."""
    d = asdict(verdict)
    return _prepare_value(d)


def to_json(verdict: Verdict, indent: int = 2) -> str:
    """Serialise a verdict to JSON."""
    return json.dumps(to_dict(verdict), indent=indent)


def _parse_datetime(value: str | None, field_name: str) -> datetime | None:
    """Parse an ISO 8601 datetime string, with a clear error on failure."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid datetime in '{field_name}': {value!r} ({e})") from e


def from_dict(data: dict) -> Verdict:
    """Reconstruct a verdict from a plain dict.

    Validates required fields and schema version.
    """
    for required in ("id", "version", "timestamp", "producer", "subject", "judgment"):
        if required not in data:
            raise ValueError(f"Missing required field: '{required}'")

    if data["version"] != 1:
        raise ValueError(
            f"Unsupported schema version {data['version']}. This library supports version 1."
        )

    timestamp = _parse_datetime(data["timestamp"], "timestamp")
    if timestamp is None:
        raise ValueError("Field 'timestamp' must not be null")

    producer_data = data["producer"]
    producer = Producer(
        system=producer_data["system"],
        instance=producer_data.get("instance"),
        model=producer_data.get("model"),
        prompt_version=producer_data.get("prompt_version"),
    )

    subject_data = data["subject"]
    subject = Subject(
        type=subject_data["type"],
        ref=subject_data["ref"],
        summary=subject_data["summary"],
        agent=subject_data.get("agent"),
        service=subject_data.get("service"),
        environment=subject_data.get("environment"),
        content_hash=subject_data.get("content_hash"),
    )

    judgment_data = data["judgment"]
    judgment = Judgment(
        action=judgment_data["action"],
        confidence=judgment_data["confidence"],
        score=judgment_data.get("score"),
        dimensions=judgment_data.get("dimensions"),
        reasoning=judgment_data.get("reasoning"),
        tags=judgment_data.get("tags"),
    )

    outcome_data = data.get("outcome", {})
    override_data = outcome_data.get("override") or {}
    ground_truth_data = outcome_data.get("ground_truth") or {}
    outcome = Outcome(
        status=outcome_data.get("status", "pending"),
        resolution=outcome_data.get("resolution"),
        override=Override(
            by=override_data.get("by"),
            at=_parse_datetime(override_data.get("at"), "outcome.override.at"),
            action=override_data.get("action"),
            reasoning=override_data.get("reasoning"),
        ) if override_data else None,
        ground_truth=GroundTruth(
            signal=ground_truth_data.get("signal"),
            value=ground_truth_data.get("value"),
            detected_at=_parse_datetime(
                ground_truth_data.get("detected_at"),
                "outcome.ground_truth.detected_at",
            ),
        ) if ground_truth_data else None,
        closed_at=_parse_datetime(outcome_data.get("closed_at"), "outcome.closed_at"),
    )

    lineage_data = data.get("lineage", {})
    lineage = Lineage(
        parent=lineage_data.get("parent"),
        children=lineage_data.get("children", []),
        context=lineage_data.get("context", []),
    )

    metadata_data = data.get("metadata", {})
    metadata = Metadata(
        cost_tokens=metadata_data.get("cost_tokens"),
        cost_currency=metadata_data.get("cost_currency"),
        latency_ms=metadata_data.get("latency_ms"),
        ttl=metadata_data.get("ttl", TTL_DEFAULT),
        custom=metadata_data.get("custom", {}),
    )

    return Verdict(
        id=data["id"],
        version=data["version"],
        timestamp=timestamp,
        producer=producer,
        subject=subject,
        judgment=judgment,
        outcome=outcome,
        lineage=lineage,
        metadata=metadata,
        # v1.5 transitional fields (optional, backward compatible)
        verdict_type=data.get("verdict_type"),
        pipeline_latency_ms=data.get("pipeline_latency_ms"),
        chain_depth=data.get("chain_depth", 0),
        parent_ids=data.get("parent_ids", []),
        service=data.get("service"),
    )


def from_json(s: str) -> Verdict:
    """Deserialise a verdict from JSON."""
    return from_dict(json.loads(s))
