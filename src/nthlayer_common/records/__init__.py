"""Content-addressed decision records for the NthLayer ecosystem."""

from nthlayer_common.records.errors import (
    ChainForkError,
    InvalidTransitionError,
    RecordStoreCorrupt,
    RecordStoreError,
    RecordStoreLocked,
)
from nthlayer_common.records.hashing import canonical_json, compute_hash, verify_hash
from nthlayer_common.records.models import (
    ZERO_HASH,
    Assessment,
    AssessmentType,
    Evaluation,
    EvaluationMethod,
    EvaluationOutcome,
    Incident,
    IncidentStatus,
    Severity,
    Summaries,
    Verdict,
    VerdictOutcome,
)
from nthlayer_common.records.sqlite_store import SQLiteDecisionRecordStore
from nthlayer_common.records.store import DecisionRecordStore
from nthlayer_common.records.verdict_bridge import build_decision_verdict, hash_content, write_decision_verdict
from nthlayer_common.records.verification import (
    ChainVerificationResult,
    IncidentVerificationResult,
    verify_chain,
    verify_incident,
)

__all__ = [
    # Models
    "Assessment",
    "AssessmentType",
    "Evaluation",
    "EvaluationMethod",
    "EvaluationOutcome",
    "Incident",
    "IncidentStatus",
    "Severity",
    "Summaries",
    "Verdict",
    "VerdictOutcome",
    "ZERO_HASH",
    # Hashing
    "canonical_json",
    "compute_hash",
    "hash_content",
    "verify_hash",
    # Verdict bridge
    "build_decision_verdict",
    "write_decision_verdict",
    # Errors
    "ChainForkError",
    "InvalidTransitionError",
    "RecordStoreCorrupt",
    "RecordStoreError",
    "RecordStoreLocked",
    # Store
    "DecisionRecordStore",
    "SQLiteDecisionRecordStore",
    # Verification
    "ChainVerificationResult",
    "IncidentVerificationResult",
    "verify_chain",
    "verify_incident",
]
