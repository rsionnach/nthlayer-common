"""Core verdict operations: create, link, resolve, supersede."""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from nthlayer_common.verdicts.models import (
    VALID_OUTCOME_STATUSES,
    GroundTruth,
    Judgment,
    Lineage,
    Metadata,
    Outcome,
    Override,
    Producer,
    Subject,
    Verdict,
)

_id_lock = threading.Lock()
_id_sequence = 0


def _generate_id() -> str:
    """Generate a unique verdict ID. Thread-safe."""
    global _id_sequence
    with _id_lock:
        _id_sequence += 1
        seq = _id_sequence
    short_uuid = uuid.uuid4().hex[:8]
    now = datetime.now(UTC)
    date_part = now.strftime("%Y-%m-%d")
    return f"vrd-{date_part}-{short_uuid}-{seq:05d}"


def _coerce(value: dict | object, cls: type) -> Any:
    """Convert a dict to a dataclass instance if needed."""
    if isinstance(value, dict):
        return cls(**value)
    return value


def create(
    subject: dict | Subject,
    judgment: dict | Judgment,
    producer: dict | Producer,
    metadata: dict | Metadata | None = None,
) -> Verdict:
    """Create a new verdict with a generated ID, current timestamp, and pending outcome."""
    subject = _coerce(subject, Subject)
    judgment = _coerce(judgment, Judgment)
    producer = _coerce(producer, Producer)
    metadata = Metadata() if metadata is None else _coerce(metadata, Metadata)

    return Verdict(
        id=_generate_id(),
        version=1,
        timestamp=datetime.now(UTC),
        producer=producer,
        subject=subject,
        judgment=judgment,
        outcome=Outcome(),
        lineage=Lineage(),
        metadata=metadata,
    )


def link(
    verdict: Verdict,
    parent: str | None = None,
    context: list[str] | None = None,
) -> Verdict:
    """Set lineage fields on a verdict. Mutates and returns the verdict."""
    if parent is not None:
        verdict.lineage.parent = parent
    if context is not None:
        verdict.lineage.context = context
    return verdict


def resolve(
    verdict: Verdict,
    status: str,
    override: dict | Override | None = None,
    ground_truth: dict | GroundTruth | None = None,
    resolution: str | None = None,
) -> Verdict:
    """Update the outcome phase. Mutates and returns the verdict.

    Transitions status from pending to the resolved state.
    """
    if status not in VALID_OUTCOME_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {sorted(VALID_OUTCOME_STATUSES)}")

    if verdict.outcome.status != "pending":
        raise ValueError(
            f"Cannot resolve verdict {verdict.id}: "
            f"status is '{verdict.outcome.status}', expected 'pending'"
        )

    verdict.outcome.status = status
    verdict.outcome.closed_at = datetime.now(UTC)

    if resolution is not None:
        verdict.outcome.resolution = resolution

    if override is not None:
        if isinstance(override, dict):
            override = Override(**override)
        override.at = override.at or datetime.now(UTC)
        verdict.outcome.override = override

    if ground_truth is not None:
        if isinstance(ground_truth, dict):
            ground_truth = GroundTruth(**ground_truth)
        verdict.outcome.ground_truth = ground_truth

    return verdict


def supersede(old_verdict: Verdict, new_verdict: Verdict) -> tuple[Verdict, Verdict]:
    """Mark old_verdict as superseded, set new_verdict as the replacement.

    Mutates both verdicts. Updates lineage in both directions.
    """
    if old_verdict.id == new_verdict.id:
        raise ValueError(
            f"Cannot supersede a verdict with itself (id: {old_verdict.id})"
        )
    old_verdict = resolve(old_verdict, status="superseded")
    new_verdict.lineage.parent = old_verdict.id
    old_verdict.lineage.children.append(new_verdict.id)
    return old_verdict, new_verdict
