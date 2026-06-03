"""SQLite-backed verdict store with WAL mode."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path

from nthlayer_common.verdicts.core import resolve as _core_resolve
from nthlayer_common.verdicts.models import AccuracyReport, Outcome, Verdict
from nthlayer_common.verdicts.serialise import from_dict, to_dict
from nthlayer_common.verdicts.store import (
    AccuracyFilter,
    OutcomeStatusMismatch,
    VerdictFilter,
    VerdictStore,
)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS verdicts (
    id               TEXT PRIMARY KEY,
    version          INTEGER NOT NULL,
    timestamp        TEXT NOT NULL,
    data             TEXT NOT NULL,
    producer_system  TEXT NOT NULL,
    subject_type     TEXT NOT NULL,
    subject_agent    TEXT,
    subject_service  TEXT,
    outcome_status   TEXT NOT NULL DEFAULT 'pending',
    ttl              INTEGER NOT NULL,
    closed_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON verdicts(timestamp);
CREATE INDEX IF NOT EXISTS idx_producer_ts ON verdicts(producer_system, timestamp);
CREATE INDEX IF NOT EXISTS idx_subject_type ON verdicts(subject_type);
CREATE INDEX IF NOT EXISTS idx_outcome_status ON verdicts(outcome_status);
"""


class SQLiteVerdictStore(VerdictStore):
    """SQLite-backed verdict store using WAL mode for concurrent read access."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._local = threading.local()
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local connection with WAL mode enabled."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def put(self, verdict: Verdict) -> None:
        data = json.dumps(to_dict(verdict))
        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO verdicts
               (id, version, timestamp, data, producer_system, subject_type,
                subject_agent, subject_service, outcome_status, ttl, closed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                verdict.id,
                verdict.version,
                verdict.timestamp.isoformat(),
                data,
                verdict.producer.system,
                verdict.subject.type,
                verdict.subject.agent,
                verdict.subject.service,
                verdict.outcome.status,
                verdict.metadata.ttl,
                verdict.outcome.closed_at.isoformat()
                if verdict.outcome.closed_at
                else None,
            ),
        )
        except sqlite3.IntegrityError:
            raise ValueError(f"Verdict {verdict.id} already exists")
        conn.commit()

    def get(self, verdict_id: str) -> Verdict | None:
        row = self._conn().execute(
            "SELECT data FROM verdicts WHERE id = ?",
            (verdict_id,),
        ).fetchone()
        if row is None:
            return None
        return from_dict(json.loads(row["data"]))

    def query(self, criteria: VerdictFilter) -> list[Verdict]:
        clauses: list[str] = []
        params: list = []

        if criteria.producer_system:
            clauses.append("producer_system = ?")
            params.append(criteria.producer_system)
        if criteria.subject_type:
            clauses.append("subject_type = ?")
            params.append(criteria.subject_type)
        if criteria.subject_agent:
            clauses.append("subject_agent = ?")
            params.append(criteria.subject_agent)
        if criteria.subject_service:
            clauses.append("subject_service = ?")
            params.append(criteria.subject_service)
        if criteria.status:
            clauses.append("outcome_status = ?")
            params.append(criteria.status)
        if criteria.from_time:
            clauses.append("timestamp >= ?")
            params.append(criteria.from_time.isoformat())
        if criteria.to_time:
            clauses.append("timestamp <= ?")
            params.append(criteria.to_time.isoformat())

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT data FROM verdicts {where} ORDER BY timestamp DESC"

        # When tag filtering is needed, skip SQL LIMIT so we filter the full
        # result set in Python first (tags live inside the JSON blob).
        if criteria.limit > 0 and not criteria.tags:
            sql += " LIMIT ?"
            params.append(criteria.limit)

        rows = self._conn().execute(sql, params).fetchall()
        results = [from_dict(json.loads(row["data"])) for row in rows]

        if criteria.tags:
            results = [
                v
                for v in results
                if v.judgment.tags and set(criteria.tags) & set(v.judgment.tags)
            ]
            if criteria.limit > 0:
                results = results[: criteria.limit]

        return results

    def resolve(
        self,
        verdict_id: str,
        status: str,
        override=None,
        ground_truth=None,
        resolution=None,
    ) -> Verdict:
        """Atomic resolve — uses conditional UPDATE to prevent double-resolution."""
        conn = self._conn()
        row = conn.execute(
            "SELECT data FROM verdicts WHERE id = ? AND outcome_status = 'pending'",
            (verdict_id,),
        ).fetchone()
        if row is None:
            # Distinguish not-found from already-resolved
            exists = conn.execute(
                "SELECT outcome_status FROM verdicts WHERE id = ?",
                (verdict_id,),
            ).fetchone()
            if exists is None:
                raise KeyError(f"Verdict {verdict_id} not found")
            raise ValueError(
                f"Cannot resolve verdict {verdict_id}: status is "
                f"'{exists['outcome_status']}', expected 'pending'"
            )

        verdict = from_dict(json.loads(row["data"]))
        _core_resolve(
            verdict, status, override=override,
            ground_truth=ground_truth, resolution=resolution,
        )
        data = json.dumps(to_dict(verdict))

        cursor = conn.execute(
            """UPDATE verdicts SET data = ?, outcome_status = ?, closed_at = ?
               WHERE id = ? AND outcome_status = 'pending'""",
            (
                data,
                verdict.outcome.status,
                verdict.outcome.closed_at.isoformat() if verdict.outcome.closed_at else None,
                verdict_id,
            ),
        )
        conn.commit()
        if cursor.rowcount == 0:
            raise ValueError(
                f"Verdict {verdict_id} was resolved concurrently"
            )
        return verdict

    def update_outcome(
        self,
        verdict_id: str,
        outcome: Outcome,
        *,
        expected_status: str | None = None,
    ) -> Verdict:
        conn = self._conn()
        row = conn.execute(
            "SELECT data, outcome_status FROM verdicts WHERE id = ?",
            (verdict_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Verdict {verdict_id} not found")

        verdict = from_dict(json.loads(row["data"]))
        verdict.outcome = outcome
        data = json.dumps(to_dict(verdict))
        closed_at = (
            outcome.closed_at.isoformat() if outcome.closed_at else None
        )

        if expected_status is None:
            conn.execute(
                "UPDATE verdicts SET data = ?, outcome_status = ?, "
                "closed_at = ? WHERE id = ?",
                (data, outcome.status, closed_at, verdict_id),
            )
            conn.commit()
            return verdict

        # Cheap pre-check on the value we just read — gives a clear
        # error before we even try the UPDATE in the common stale-read
        # case.
        if row["outcome_status"] != expected_status:
            raise OutcomeStatusMismatch(
                f"Verdict {verdict_id}: outcome.status is "
                f"{row['outcome_status']!r}, expected {expected_status!r}"
            )

        # Conditional UPDATE: only one writer can transition the
        # verdict away from expected_status. rowcount==0 means a
        # concurrent writer beat us between SELECT and UPDATE.
        cursor = conn.execute(
            "UPDATE verdicts SET data = ?, outcome_status = ?, "
            "closed_at = ? WHERE id = ? AND outcome_status = ?",
            (data, outcome.status, closed_at, verdict_id, expected_status),
        )
        conn.commit()
        if cursor.rowcount == 0:
            after = conn.execute(
                "SELECT outcome_status FROM verdicts WHERE id = ?",
                (verdict_id,),
            ).fetchone()
            current = after["outcome_status"] if after else "<deleted>"
            raise OutcomeStatusMismatch(
                f"Verdict {verdict_id}: outcome.status raced to "
                f"{current!r}, expected {expected_status!r}"
            )
        return verdict

    def accuracy(self, criteria: AccuracyFilter) -> AccuracyReport:
        conn = self._conn()
        clauses = ["producer_system = ?"]
        params: list = [criteria.producer_system]

        if criteria.from_time:
            clauses.append("timestamp >= ?")
            params.append(criteria.from_time.isoformat())
        if criteria.to_time:
            clauses.append("timestamp <= ?")
            params.append(criteria.to_time.isoformat())

        where = "WHERE " + " AND ".join(clauses)

        agg = conn.execute(
            f"""SELECT
                COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN outcome_status IN ('confirmed','overridden','partial')
                    THEN 1 ELSE 0 END), 0) AS total_resolved,
                COALESCE(SUM(CASE WHEN outcome_status = 'confirmed'
                    THEN 1 ELSE 0 END), 0) AS confirmed,
                COALESCE(SUM(CASE WHEN outcome_status = 'overridden'
                    THEN 1 ELSE 0 END), 0) AS overridden,
                COALESCE(SUM(CASE WHEN outcome_status = 'partial'
                    THEN 1 ELSE 0 END), 0) AS partial_cnt,
                COALESCE(SUM(CASE WHEN outcome_status = 'pending'
                    THEN 1 ELSE 0 END), 0) AS pending
            FROM verdicts {where}""",
            params,
        ).fetchone()

        total = agg["total"]
        total_resolved = agg["total_resolved"]
        confirmed = agg["confirmed"]
        overridden = agg["overridden"]
        partial_cnt = agg["partial_cnt"]
        pending = agg["pending"]

        def safe_div(a: float, b: float) -> float:
            return a / b if b > 0 else 0.0

        # Mean confidence requires reading the JSON data column.
        conf_rows = conn.execute(
            f"""SELECT data, outcome_status FROM verdicts
                {where} AND outcome_status IN ('confirmed', 'overridden')""",
            params,
        ).fetchall()

        confirmed_confs: list[float] = []
        overridden_confs: list[float] = []
        for r in conf_rows:
            c = json.loads(r["data"])["judgment"]["confidence"]
            if r["outcome_status"] == "confirmed":
                confirmed_confs.append(c)
            else:
                overridden_confs.append(c)

        return AccuracyReport(
            producer=criteria.producer_system,
            total=total,
            total_resolved=total_resolved,
            confirmation_rate=safe_div(confirmed, total_resolved),
            override_rate=safe_div(overridden, total_resolved),
            partial_rate=safe_div(partial_cnt, total_resolved),
            pending_rate=safe_div(pending, total),
            mean_confidence_on_confirmed=(
                sum(confirmed_confs) / len(confirmed_confs)
                if confirmed_confs
                else 0.0
            ),
            mean_confidence_on_overridden=(
                sum(overridden_confs) / len(overridden_confs)
                if overridden_confs
                else 0.0
            ),
            dimension=criteria.dimension,
        )

    def by_lineage(
        self,
        verdict_id: str,
        direction: str = "both",
        max_depth: int = 500,
    ) -> list[Verdict]:
        if direction not in ("up", "down", "both"):
            raise ValueError(
                f"direction must be 'up', 'down', or 'both', got '{direction}'"
            )

        visited: set[str] = set()
        result: list[Verdict] = []
        visited.add(verdict_id)

        start = self.get(verdict_id)
        if start is None:
            return []

        # Iterative BFS to avoid RecursionError on deep chains
        queue: list[tuple[str, int]] = []

        if direction in ("up", "both"):
            if start.lineage.parent:
                queue.append((start.lineage.parent, 1))
            for ctx_id in start.lineage.context:
                queue.append((ctx_id, 1))

        if direction in ("down", "both"):
            for child_id in start.lineage.children:
                queue.append((child_id, 1))

        while queue:
            vid, depth = queue.pop(0)
            if vid in visited or depth > max_depth:
                continue
            v = self.get(vid)
            if v is None:
                continue
            visited.add(vid)
            result.append(v)

            if depth < max_depth:
                if direction in ("up", "both"):
                    if v.lineage.parent:
                        queue.append((v.lineage.parent, depth + 1))
                    for ctx_id in v.lineage.context:
                        queue.append((ctx_id, depth + 1))
                if direction in ("down", "both"):
                    for child_id in v.lineage.children:
                        queue.append((child_id, depth + 1))

        return result

    def expire(self) -> int:
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        conn = self._conn()
        # Use cursor iteration to avoid loading all rows into memory
        cursor = conn.execute(
            "SELECT id, data FROM verdicts WHERE outcome_status = 'pending'",
        )

        count = 0
        batch: list[tuple[str, str, str]] = []
        for row in cursor:
            verdict = from_dict(json.loads(row["data"]))
            expiry_time = verdict.timestamp + timedelta(seconds=verdict.metadata.ttl)
            if expiry_time < now:
                verdict.outcome.status = "expired"
                verdict.outcome.closed_at = now
                data = json.dumps(to_dict(verdict))
                batch.append((data, now_iso, row["id"]))

        for data, closed, vid in batch:
            conn.execute(
                """UPDATE verdicts
                   SET data = ?, outcome_status = 'expired', closed_at = ?
                   WHERE id = ?""",
                (data, closed, vid),
            )
            count += 1

        if count > 0:
            conn.commit()
        return count

    def close(self) -> None:
        """Close the database connection for the calling thread only.

        Other threads' connections are not affected. This is a limitation
        of the thread-local connection design.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> SQLiteVerdictStore:
        return self

    def __exit__(self, *args) -> None:
        self.close()
