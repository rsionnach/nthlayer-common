"""Verdict data model — the atomic unit of AI judgment.

Originally defined in nthlayer-learn. Moved to nthlayer-common so that
nthlayer-measure, nthlayer-correlate, nthlayer-respond, and nthlayer-observe
can all import the verdict model without circular dependencies.

Canonical import path: ``from nthlayer_common.verdicts import Verdict``

The old import path (``from nthlayer_learn import Verdict``) continues to
work via re-export shims and resolves to the same classes.
"""

from nthlayer_common.verdicts.core import create, link, resolve, supersede
from nthlayer_common.verdicts.models import (
    TTL_DEFAULT,
    VALID_ACTIONS,
    VALID_OUTCOME_STATUSES,
    VALID_SUBJECT_TYPES,
    VALID_VERDICT_TYPES,
    AccuracyReport,
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
from nthlayer_common.verdicts.serialise import from_dict, from_json, to_dict, to_json
from nthlayer_common.verdicts.sqlite_store import SQLiteVerdictStore
from nthlayer_common.verdicts.store import (
    AccuracyFilter,
    MemoryStore,
    OutcomeStatusMismatch,
    VerdictFilter,
    VerdictStore,
)

__all__ = [
    # Core operations
    "create",
    "link",
    "resolve",
    "supersede",
    # Models
    "AccuracyReport",
    "GroundTruth",
    "Judgment",
    "Lineage",
    "Metadata",
    "Outcome",
    "Override",
    "Producer",
    "Subject",
    "Verdict",
    # Constants
    "TTL_DEFAULT",
    "VALID_ACTIONS",
    "VALID_OUTCOME_STATUSES",
    "VALID_SUBJECT_TYPES",
    "VALID_VERDICT_TYPES",
    # Serialisation
    "from_dict",
    "from_json",
    "to_dict",
    "to_json",
    # Store
    "AccuracyFilter",
    "MemoryStore",
    "OutcomeStatusMismatch",
    "SQLiteVerdictStore",
    "VerdictFilter",
    "VerdictStore",
]
