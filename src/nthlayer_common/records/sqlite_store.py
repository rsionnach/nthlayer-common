"""SQLite WAL implementation of DecisionRecordStore."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from datetime import datetime
from typing import Any

from nthlayer_common.records.models import (
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

__all__ = ["SQLiteDecisionRecordStore"]

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS assessments (
    hash            TEXT PRIMARY KEY,
    previous_hash   TEXT NOT NULL,
    schema_version  TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    stream          TEXT NOT NULL,
    incident_id     TEXT,
    type            TEXT NOT NULL,
    severity        TEXT NOT NULL,
    payload         TEXT NOT NULL,
    summaries       TEXT NOT NULL,
    canonical       TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(stream, previous_hash)
);

CREATE TABLE IF NOT EXISTS verdicts (
    hash            TEXT PRIMARY KEY,
    previous_hash   TEXT NOT NULL,
    schema_version  TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    agent           TEXT NOT NULL,
    incident_id     TEXT NOT NULL,
    input_hashes    TEXT NOT NULL,
    prompt_hash     TEXT NOT NULL,
    response_hash   TEXT NOT NULL,
    model           TEXT NOT NULL,
    reasoning       TEXT NOT NULL,
    action          TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    summaries       TEXT NOT NULL,
    canonical       TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(agent, previous_hash)
);

CREATE TABLE IF NOT EXISTS evaluations (
    hash            TEXT PRIMARY KEY,
    previous_hash   TEXT NOT NULL,
    schema_version  TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    incident_id     TEXT NOT NULL,
    verdict_hash    TEXT NOT NULL,
    method          TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    evidence_hashes TEXT NOT NULL,
    payload         TEXT NOT NULL,
    summaries       TEXT NOT NULL,
    canonical       TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(incident_id, previous_hash)
);

CREATE TABLE IF NOT EXISTS incidents (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    trigger_hash    TEXT NOT NULL,
    stream          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS prompts (
    hash       TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS responses (
    hash       TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_assessments_stream ON assessments(stream, timestamp);
CREATE INDEX IF NOT EXISTS idx_assessments_incident ON assessments(incident_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_agent ON verdicts(agent, timestamp);
CREATE INDEX IF NOT EXISTS idx_verdicts_incident ON verdicts(incident_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_incident ON evaluations(incident_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_verdict ON evaluations(verdict_hash);
"""


def _raise_operational(exc: sqlite3.OperationalError) -> None:
    """Translate SQLite operational errors into domain exceptions."""
    from nthlayer_common.records.errors import RecordStoreCorrupt, RecordStoreLocked

    msg = str(exc).lower()
    if "locked" in msg or "busy" in msg:
        raise RecordStoreLocked(str(exc)) from exc
    if "corrupt" in msg or "malformed" in msg:
        raise RecordStoreCorrupt(str(exc)) from exc
    from nthlayer_common.records.errors import RecordStoreError
    raise RecordStoreError(str(exc)) from exc


class SQLiteDecisionRecordStore:
    """SQLite WAL-mode store for content-addressed decision records.

    Thread-safe via thread-local connections with WAL mode and busy timeout.
    Append-only: no UPDATE or DELETE on record tables (assessments, verdicts, evaluations).
    Chain fork protection: UNIQUE constraints on (stream, previous_hash) etc. prevent
    two records from claiming the same predecessor in the same chain.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            with self._conn_lock:
                self._connections.append(conn)
        return conn

    def close(self) -> None:
        """Close all thread-local connections."""
        with self._conn_lock:
            for conn in self._connections:
                with contextlib.suppress(Exception):
                    conn.close()
            self._connections.clear()
        self._local = threading.local()

    def __enter__(self) -> SQLiteDecisionRecordStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    # --- Assessments ---

    def put_assessment(self, a: Assessment) -> None:
        from nthlayer_common.records.hashing import canonical_json

        canonical = canonical_json(a).decode("utf-8")
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO assessments "
                "(hash, previous_hash, schema_version, timestamp, stream, incident_id, "
                "type, severity, payload, summaries, canonical) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    a.hash,
                    a.previous_hash,
                    a.schema_version,
                    a.timestamp.isoformat(),
                    a.stream,
                    a.incident_id,
                    a.type.value,
                    a.severity.value,
                    json.dumps(a.payload, sort_keys=True),
                    _summaries_to_json(a.summaries),
                    canonical,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            if conn.execute("SELECT 1 FROM assessments WHERE hash = ?", (a.hash,)).fetchone():
                return  # Idempotent: same record already stored
            from nthlayer_common.records.errors import ChainForkError
            raise ChainForkError(
                "Chain fork detected in assessments: another record with the same "
                "previous_hash already exists in this stream"
            ) from exc
        except sqlite3.OperationalError as exc:
            conn.rollback()
            _raise_operational(exc)

    # --- Verdicts ---

    def put_verdict(self, v: Verdict) -> None:
        from nthlayer_common.records.hashing import canonical_json

        canonical = canonical_json(v).decode("utf-8")
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO verdicts "
                "(hash, previous_hash, schema_version, timestamp, agent, incident_id, "
                "input_hashes, prompt_hash, response_hash, model, reasoning, action, "
                "outcome, summaries, canonical) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    v.hash,
                    v.previous_hash,
                    v.schema_version,
                    v.timestamp.isoformat(),
                    v.agent,
                    v.incident_id,
                    json.dumps(v.input_hashes),
                    v.prompt_hash,
                    v.response_hash,
                    v.model,
                    v.reasoning,
                    json.dumps(v.action, sort_keys=True),
                    v.outcome.value,
                    _summaries_to_json(v.summaries),
                    canonical,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            if conn.execute("SELECT 1 FROM verdicts WHERE hash = ?", (v.hash,)).fetchone():
                return  # Idempotent: same record already stored
            from nthlayer_common.records.errors import ChainForkError
            raise ChainForkError(
                "Chain fork detected in verdicts: another record with the same "
                "previous_hash already exists for this agent"
            ) from exc
        except sqlite3.OperationalError as exc:
            conn.rollback()
            _raise_operational(exc)

    # --- Evaluations ---

    def put_evaluation(self, e: Evaluation) -> None:
        from nthlayer_common.records.hashing import canonical_json

        canonical = canonical_json(e).decode("utf-8")
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO evaluations "
                "(hash, previous_hash, schema_version, timestamp, incident_id, "
                "verdict_hash, method, outcome, evidence_hashes, payload, summaries, canonical) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    e.hash,
                    e.previous_hash,
                    e.schema_version,
                    e.timestamp.isoformat(),
                    e.incident_id,
                    e.verdict_hash,
                    e.method.value,
                    e.outcome.value,
                    json.dumps(e.evidence_hashes),
                    json.dumps(e.payload, sort_keys=True),
                    _summaries_to_json(e.summaries),
                    canonical,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            if conn.execute("SELECT 1 FROM evaluations WHERE hash = ?", (e.hash,)).fetchone():
                return  # Idempotent: same record already stored
            from nthlayer_common.records.errors import ChainForkError
            raise ChainForkError(
                "Chain fork detected in evaluations: another record with the same "
                "previous_hash already exists for this incident"
            ) from exc
        except sqlite3.OperationalError as exc:
            conn.rollback()
            _raise_operational(exc)

    # --- Incidents ---

    def create_incident(self, incident: Incident) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO incidents (id, created_at, trigger_hash, stream, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                incident.id,
                incident.created_at.isoformat(),
                incident.trigger_hash,
                incident.stream,
                incident.status.value,
            ),
        )
        conn.commit()

    # Valid incident status transitions (from → set of allowed targets)
    _VALID_TRANSITIONS: dict[IncidentStatus, frozenset[IncidentStatus]] = {
        IncidentStatus.OPEN: frozenset({IncidentStatus.MITIGATED, IncidentStatus.RESOLVED, IncidentStatus.CLOSED}),
        IncidentStatus.MITIGATED: frozenset({IncidentStatus.RESOLVED, IncidentStatus.CLOSED}),
        IncidentStatus.RESOLVED: frozenset({IncidentStatus.LEARNING, IncidentStatus.CLOSED}),
        IncidentStatus.LEARNING: frozenset({IncidentStatus.CLOSED}),
        IncidentStatus.CLOSED: frozenset(),
    }

    def update_incident_status(self, incident_id: str, status: IncidentStatus) -> None:
        from nthlayer_common.records.errors import InvalidTransitionError

        conn = self._get_conn()
        row = conn.execute("SELECT status FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        if row is None:
            raise ValueError(f"Incident {incident_id} not found")

        current = IncidentStatus(row["status"])
        if status not in self._VALID_TRANSITIONS.get(current, frozenset()):
            raise InvalidTransitionError(
                f"Cannot transition incident {incident_id} from {current.value} to {status.value}"
            )

        conn.execute("UPDATE incidents SET status = ? WHERE id = ?", (status.value, incident_id))
        conn.commit()

    # --- Generic retrieval ---

    def get_by_hash(self, hash_val: str) -> Assessment | Verdict | Evaluation | None:
        conn = self._get_conn()
        for table, builder in [
            ("assessments", _row_to_assessment),
            ("verdicts", _row_to_verdict),
            ("evaluations", _row_to_evaluation),
        ]:
            row = conn.execute(f"SELECT * FROM {table} WHERE hash = ?", (hash_val,)).fetchone()
            if row is not None:
                return builder(row)
        return None

    def get_chain(self, record_type: str, chain_key: str, *, limit: int = 0) -> list[Assessment | Verdict | Evaluation]:
        """Retrieve a chain ordered by timestamp.

        ``chain_key`` meaning depends on ``record_type``:
        - ``"assessment"``: the ``stream`` value
        - ``"verdict"``: the ``agent`` value
        - ``"evaluation"``: the ``incident_id``

        ``limit``: max records to return (0 = unlimited).
        """
        conn = self._get_conn()
        table_map = {
            "assessment": ("assessments", "stream", _row_to_assessment),
            "verdict": ("verdicts", "agent", _row_to_verdict),
            "evaluation": ("evaluations", "incident_id", _row_to_evaluation),
        }
        if record_type not in table_map:
            raise ValueError(f"Unknown record type: {record_type}")

        table, column, builder = table_map[record_type]
        sql = f"SELECT * FROM {table} WHERE {column} = ? ORDER BY timestamp ASC"
        if limit > 0:
            sql += f" LIMIT {limit}"
        rows = conn.execute(sql, (chain_key,)).fetchall()
        return [builder(r) for r in rows]

    def get_chain_tail(self, record_type: str, chain_key: str) -> Assessment | Verdict | Evaluation | None:
        """Get the most recent record in a chain. Returns None for empty chains."""
        conn = self._get_conn()
        table_map = {
            "assessment": ("assessments", "stream", _row_to_assessment),
            "verdict": ("verdicts", "agent", _row_to_verdict),
            "evaluation": ("evaluations", "incident_id", _row_to_evaluation),
        }
        if record_type not in table_map:
            raise ValueError(f"Unknown record type: {record_type}")

        table, column, builder = table_map[record_type]
        row = conn.execute(
            f"SELECT * FROM {table} WHERE {column} = ? ORDER BY timestamp DESC LIMIT 1",
            (chain_key,),
        ).fetchone()
        return builder(row) if row is not None else None

    def get_incident(self, incident_id: str) -> Incident | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        if row is None:
            return None
        return _row_to_incident(row)

    def get_incident_records(self, incident_id: str) -> dict[str, list[Any]]:
        conn = self._get_conn()
        assessments = conn.execute(
            "SELECT * FROM assessments WHERE incident_id = ? ORDER BY timestamp ASC",
            (incident_id,),
        ).fetchall()
        verdicts = conn.execute(
            "SELECT * FROM verdicts WHERE incident_id = ? ORDER BY timestamp ASC",
            (incident_id,),
        ).fetchall()
        evaluations = conn.execute(
            "SELECT * FROM evaluations WHERE incident_id = ? ORDER BY timestamp ASC",
            (incident_id,),
        ).fetchall()
        return {
            "assessments": [_row_to_assessment(r) for r in assessments],
            "verdicts": [_row_to_verdict(r) for r in verdicts],
            "evaluations": [_row_to_evaluation(r) for r in evaluations],
        }

    # --- Prompts/Responses ---

    def put_prompt(self, hash_val: str, content: str) -> None:
        conn = self._get_conn()
        conn.execute("INSERT OR IGNORE INTO prompts (hash, content) VALUES (?, ?)", (hash_val, content))
        conn.commit()

    def put_response(self, hash_val: str, content: str) -> None:
        conn = self._get_conn()
        conn.execute("INSERT OR IGNORE INTO responses (hash, content) VALUES (?, ?)", (hash_val, content))
        conn.commit()

    def get_prompt(self, hash_val: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute("SELECT content FROM prompts WHERE hash = ?", (hash_val,)).fetchone()
        return row["content"] if row else None

    def get_response(self, hash_val: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute("SELECT content FROM responses WHERE hash = ?", (hash_val,)).fetchone()
        return row["content"] if row else None


# --- Row deserializers ---


def _summaries_to_json(s: Summaries) -> str:
    d: dict[str, str | None] = {"technical": s.technical, "plain": s.plain, "executive": s.executive}
    return json.dumps(d, sort_keys=True)


def _json_to_summaries(raw: str) -> Summaries:
    d = json.loads(raw)
    return Summaries(technical=d["technical"], plain=d["plain"], executive=d.get("executive"))


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def _row_to_assessment(row: sqlite3.Row) -> Assessment:
    return Assessment(
        hash=row["hash"],
        previous_hash=row["previous_hash"],
        schema_version=row["schema_version"],
        timestamp=_parse_ts(row["timestamp"]),
        stream=row["stream"],
        incident_id=row["incident_id"],
        type=AssessmentType(row["type"]),
        severity=Severity(row["severity"]),
        payload=json.loads(row["payload"]),
        summaries=_json_to_summaries(row["summaries"]),
    )


def _row_to_verdict(row: sqlite3.Row) -> Verdict:
    return Verdict(
        hash=row["hash"],
        previous_hash=row["previous_hash"],
        schema_version=row["schema_version"],
        timestamp=_parse_ts(row["timestamp"]),
        agent=row["agent"],
        incident_id=row["incident_id"],
        input_hashes=json.loads(row["input_hashes"]),
        prompt_hash=row["prompt_hash"],
        response_hash=row["response_hash"],
        model=row["model"],
        reasoning=row["reasoning"],
        action=json.loads(row["action"]),
        outcome=VerdictOutcome(row["outcome"]),
        summaries=_json_to_summaries(row["summaries"]),
    )


def _row_to_evaluation(row: sqlite3.Row) -> Evaluation:
    return Evaluation(
        hash=row["hash"],
        previous_hash=row["previous_hash"],
        schema_version=row["schema_version"],
        timestamp=_parse_ts(row["timestamp"]),
        incident_id=row["incident_id"],
        verdict_hash=row["verdict_hash"],
        method=EvaluationMethod(row["method"]),
        outcome=EvaluationOutcome(row["outcome"]),
        evidence_hashes=json.loads(row["evidence_hashes"]),
        payload=json.loads(row["payload"]),
        summaries=_json_to_summaries(row["summaries"]),
    )


def _row_to_incident(row: sqlite3.Row) -> Incident:
    return Incident(
        id=row["id"],
        created_at=_parse_ts(row["created_at"]),
        trigger_hash=row["trigger_hash"],
        stream=row["stream"],
        status=IncidentStatus(row["status"]),
    )
