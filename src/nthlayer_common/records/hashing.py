"""Canonical serialization and content-addressed hashing for decision records."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any

from nthlayer_common.records.models import Assessment, Evaluation, Verdict

__all__ = ["canonical_json", "compute_hash", "verify_hash"]

_EXCLUDED_FIELDS = frozenset({"hash", "previous_hash"})


def canonical_json(record: Assessment | Verdict | Evaluation) -> bytes:
    """Produce deterministic canonical JSON for hashing.

    Sorted keys, no whitespace, UTF-8 encoded.
    The ``hash`` and ``previous_hash`` fields are excluded (they are derived, not input).
    """
    d = dataclasses.asdict(record)
    for key in _EXCLUDED_FIELDS:
        d.pop(key, None)
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=_json_default).encode("utf-8")


def compute_hash(canonical: bytes) -> str:
    """SHA-256 hex digest of canonical bytes."""
    return hashlib.sha256(canonical).hexdigest()


def verify_hash(record: Assessment | Verdict | Evaluation) -> bool:
    """Verify that a record's hash matches its canonical form."""
    canonical = canonical_json(record)
    expected = compute_hash(canonical)
    return record.hash == expected


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
