"""Domain error hierarchy for the decision record store."""

from __future__ import annotations

__all__ = [
    "RecordStoreError",
    "RecordStoreCorrupt",
    "RecordStoreLocked",
    "ChainForkError",
    "InvalidTransitionError",
]


class RecordStoreError(Exception):
    """Base error for decision record store operations."""


class RecordStoreCorrupt(RecordStoreError):
    """Database is corrupt or malformed."""


class RecordStoreLocked(RecordStoreError):
    """Database is locked beyond the busy timeout."""


class ChainForkError(RecordStoreError):
    """Two records claim the same predecessor in the same chain."""


class InvalidTransitionError(RecordStoreError):
    """Invalid incident status transition."""
