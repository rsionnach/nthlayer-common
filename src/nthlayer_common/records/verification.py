"""Hash chain and incident verification for decision records."""

from __future__ import annotations

from dataclasses import dataclass, field

from nthlayer_common.records.hashing import verify_hash
from nthlayer_common.records.models import ZERO_HASH
from nthlayer_common.records.store import DecisionRecordStore

__all__ = [
    "ChainVerificationResult",
    "IncidentVerificationResult",
    "verify_chain",
    "verify_incident",
]


@dataclass(slots=True)
class ChainVerificationResult:
    verified: bool
    record_count: int
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IncidentVerificationResult:
    verified: bool
    assessment_count: int = 0
    verdict_count: int = 0
    evaluation_count: int = 0
    errors: list[str] = field(default_factory=list)


def verify_chain(store: DecisionRecordStore, record_type: str, chain_key: str) -> ChainVerificationResult:
    """Walk a chain from genesis and verify hash integrity and contiguity.

    Checks:
    1. Each record's hash matches the SHA-256 of its canonical form.
    2. Each record's previous_hash matches the hash of the preceding record.
    3. The first record's previous_hash is the zero hash.
    4. Timestamps are non-decreasing (contiguous within the stream).
    """
    chain = store.get_chain(record_type, chain_key)
    if not chain:
        return ChainVerificationResult(verified=True, record_count=0)

    errors: list[str] = []

    # Check genesis
    first = chain[0]
    if first.previous_hash != ZERO_HASH:
        errors.append(f"Record {first.hash[:12]}...: genesis record previous_hash is not zero hash")

    for i, record in enumerate(chain):
        # Verify content hash
        if not verify_hash(record):
            errors.append(f"Record {record.hash[:12]}...: hash mismatch — content does not match hash")

        # Verify chain linkage (skip first, already checked genesis)
        if i > 0:
            expected_prev = chain[i - 1].hash
            if record.previous_hash != expected_prev:
                errors.append(
                    f"Record {record.hash[:12]}...: chain link broken — "
                    f"previous_hash {record.previous_hash[:12]}... does not match "
                    f"prior record {expected_prev[:12]}..."
                )

            # Verify timestamp contiguity
            if record.timestamp < chain[i - 1].timestamp:
                errors.append(
                    f"Record {record.hash[:12]}...: timestamp {record.timestamp.isoformat()} "
                    f"is before prior record {chain[i - 1].timestamp.isoformat()}"
                )

    return ChainVerificationResult(
        verified=len(errors) == 0,
        record_count=len(chain),
        errors=errors,
    )


def verify_incident(store: DecisionRecordStore, incident_id: str) -> IncidentVerificationResult:
    """Verify all records within an incident.

    Checks:
    1. Incident exists.
    2. All record hashes are valid.
    3. All verdict input_hashes resolve to existing records.
    4. All verdict prompt_hash and response_hash resolve.
    5. All evaluation verdict_hash and evidence_hashes resolve to existing records.
    """
    incident = store.get_incident(incident_id)
    if incident is None:
        return IncidentVerificationResult(
            verified=False,
            errors=[f"Incident {incident_id} not found"],
        )

    records = store.get_incident_records(incident_id)
    errors: list[str] = []

    # Verify assessment hashes
    for a in records["assessments"]:
        if not verify_hash(a):
            errors.append(f"Assessment {a.hash[:12]}...: hash mismatch")

    # Verify verdict hashes and cross-references
    for v in records["verdicts"]:
        if not verify_hash(v):
            errors.append(f"Verdict {v.hash[:12]}...: hash mismatch")

        for ih in v.input_hashes:
            if store.get_by_hash(ih) is None:
                errors.append(f"Verdict {v.hash[:12]}...: unresolved input_hash {ih[:12]}...")

        if store.get_prompt(v.prompt_hash) is None:
            errors.append(f"Verdict {v.hash[:12]}...: missing prompt {v.prompt_hash[:12]}...")

        if store.get_response(v.response_hash) is None:
            errors.append(f"Verdict {v.hash[:12]}...: missing response {v.response_hash[:12]}...")

    # Verify evaluation hashes and cross-references
    for e in records["evaluations"]:
        if not verify_hash(e):
            errors.append(f"Evaluation {e.hash[:12]}...: hash mismatch")

        if store.get_by_hash(e.verdict_hash) is None:
            errors.append(f"Evaluation {e.hash[:12]}...: unresolved verdict_hash {e.verdict_hash[:12]}...")

        for eh in e.evidence_hashes:
            if store.get_by_hash(eh) is None:
                errors.append(f"Evaluation {e.hash[:12]}...: unresolved evidence_hash {eh[:12]}...")

    return IncidentVerificationResult(
        verified=len(errors) == 0,
        assessment_count=len(records["assessments"]),
        verdict_count=len(records["verdicts"]),
        evaluation_count=len(records["evaluations"]),
        errors=errors,
    )
