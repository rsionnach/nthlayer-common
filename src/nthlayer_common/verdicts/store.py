"""Verdict store interface and query operations."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC

from nthlayer_common.verdicts.core import resolve as _core_resolve
from nthlayer_common.verdicts.models import (
    AccuracyReport,
    GroundTruth,
    Outcome,
    Override,
    Verdict,
)


class OutcomeStatusMismatch(ValueError):
    """Compare-and-swap miss on update_outcome's expected_status guard.

    Raised when a caller supplies ``expected_status`` to ``update_outcome``
    and the verdict's current status differs — either because a concurrent
    writer transitioned it, or the caller's read is stale. Subclass of
    ValueError so existing broad handlers stay backward compatible.
    """


@dataclass
class VerdictFilter:
    producer_system: str | None = None
    subject_type: str | None = None
    subject_agent: str | None = None
    subject_service: str | None = None
    status: str | None = None
    tags: list[str] | None = None
    from_time: datetime | None = None
    to_time: datetime | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        for field_name in ("from_time", "to_time"):
            val = getattr(self, field_name)
            if val is not None and val.tzinfo is None:
                raise ValueError(
                    f"VerdictFilter.{field_name} must be timezone-aware "
                    f"(got naive datetime). Use datetime.now(timezone.utc)."
                )


@dataclass
class AccuracyFilter:
    producer_system: str
    from_time: datetime | None = None
    to_time: datetime | None = None
    dimension: str | None = None  # TODO: dimension-level filtering not yet implemented


class VerdictStore(ABC):
    """Abstract interface for verdict storage."""

    @abstractmethod
    def put(self, verdict: Verdict) -> None:
        """Store a verdict."""

    @abstractmethod
    def get(self, verdict_id: str) -> Verdict | None:
        """Retrieve a verdict by ID."""

    @abstractmethod
    def query(self, criteria: VerdictFilter) -> list[Verdict]:
        """Query verdicts matching a filter."""

    @abstractmethod
    def update_outcome(
        self,
        verdict_id: str,
        outcome: Outcome,
        *,
        expected_status: str | None = None,
    ) -> Verdict:
        """Update a verdict's outcome.

        When ``expected_status`` is provided, the update is conditional:
        the store performs a compare-and-swap on the current
        ``outcome.status`` and raises :class:`OutcomeStatusMismatch` if
        it does not equal ``expected_status``. Callers use this to detect
        a concurrent writer that transitioned the verdict between their
        read and their write.
        """

    @abstractmethod
    def accuracy(self, criteria: AccuracyFilter) -> AccuracyReport:
        """Compute accuracy metrics from resolved verdicts."""

    @abstractmethod
    def by_lineage(
        self, verdict_id: str, direction: str = "both",
    ) -> list[Verdict]:
        """Traverse the verdict chain.

        Args:
            verdict_id: Starting verdict ID.
            direction: "up" (parents/context), "down" (children), or "both".
        """

    @abstractmethod
    def expire(self) -> int:
        """Expire pending verdicts past their TTL. Returns count expired."""

    def resolve(
        self,
        verdict_id: str,
        status: str,
        override: dict | Override | None = None,
        ground_truth: dict | GroundTruth | None = None,
        resolution: str | None = None,
    ) -> Verdict:
        """Resolve a verdict in the store: get, validate, persist in one call.

        Delegates validation to core.resolve(). Returns the persisted verdict.
        Thread-safety depends on the concrete store's get/update_outcome methods.

        Raises:
            KeyError: If verdict_id not found in store.
            ValueError: If verdict is not pending, or status is invalid.
        """
        verdict = self.get(verdict_id)
        if verdict is None:
            raise KeyError(f"Verdict {verdict_id} not found")
        _core_resolve(
            verdict, status, override=override,
            ground_truth=ground_truth, resolution=resolution,
        )
        return self.update_outcome(verdict_id, verdict.outcome)


class MemoryStore(VerdictStore):
    """In-memory verdict store for testing and development.

    Thread-safe for individual operations; compound operations
    (e.g., get-then-update) may see concurrent mutations between steps.
    """

    def __init__(self) -> None:
        self._verdicts: dict[str, Verdict] = {}
        self._lock = threading.Lock()

    def put(self, verdict: Verdict) -> None:
        with self._lock:
            self._verdicts[verdict.id] = verdict

    def get(self, verdict_id: str) -> Verdict | None:
        with self._lock:
            return self._verdicts.get(verdict_id)

    def query(self, criteria: VerdictFilter) -> list[Verdict]:
        with self._lock:
            results = list(self._verdicts.values())

        if criteria.producer_system:
            results = [v for v in results if v.producer.system == criteria.producer_system]
        if criteria.subject_type:
            results = [v for v in results if v.subject.type == criteria.subject_type]
        if criteria.subject_agent:
            results = [v for v in results if v.subject.agent == criteria.subject_agent]
        if criteria.subject_service:
            results = [v for v in results if v.subject.service == criteria.subject_service]
        if criteria.status:
            results = [v for v in results if v.outcome.status == criteria.status]
        if criteria.tags:
            results = [
                v for v in results
                if v.judgment.tags and set(criteria.tags) & set(v.judgment.tags)
            ]
        if criteria.from_time:
            results = [v for v in results if v.timestamp >= criteria.from_time]
        if criteria.to_time:
            results = [v for v in results if v.timestamp <= criteria.to_time]

        results.sort(key=lambda v: v.timestamp, reverse=True)

        if criteria.limit > 0:
            return results[: criteria.limit]
        return results

    def _query_all(
        self,
        producer_system: str | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Verdict]:
        """Internal query without limit, for accuracy computation."""
        return self.query(VerdictFilter(
            producer_system=producer_system,
            from_time=from_time,
            to_time=to_time,
            limit=0,
        ))

    def update_outcome(
        self,
        verdict_id: str,
        outcome: Outcome,
        *,
        expected_status: str | None = None,
    ) -> Verdict:
        with self._lock:
            verdict = self._verdicts.get(verdict_id)
            if verdict is None:
                raise KeyError(f"Verdict {verdict_id} not found")
            if (
                expected_status is not None
                and verdict.outcome.status != expected_status
            ):
                raise OutcomeStatusMismatch(
                    f"Verdict {verdict_id}: outcome.status is "
                    f"{verdict.outcome.status!r}, expected "
                    f"{expected_status!r}"
                )
            verdict.outcome = outcome
            return verdict

    def accuracy(self, criteria: AccuracyFilter) -> AccuracyReport:
        verdicts = self._query_all(
            producer_system=criteria.producer_system,
            from_time=criteria.from_time,
            to_time=criteria.to_time,
        )

        total = len(verdicts)
        confirmed = [v for v in verdicts if v.outcome.status == "confirmed"]
        overridden = [v for v in verdicts if v.outcome.status == "overridden"]
        partial = [v for v in verdicts if v.outcome.status == "partial"]
        pending = [v for v in verdicts if v.outcome.status == "pending"]

        total_resolved = len(confirmed) + len(overridden) + len(partial)

        def safe_div(a: float, b: float) -> float:
            return a / b if b > 0 else 0.0

        def mean_confidence(vs: list[Verdict]) -> float:
            if not vs:
                return 0.0
            return sum(v.judgment.confidence for v in vs) / len(vs)

        confirmation_rate = safe_div(len(confirmed), total_resolved)
        override_rate = safe_div(len(overridden), total_resolved)

        mean_conf_confirmed = mean_confidence(confirmed)
        mean_conf_overridden = mean_confidence(overridden)

        return AccuracyReport(
            producer=criteria.producer_system,
            total=total,
            total_resolved=total_resolved,
            confirmation_rate=confirmation_rate,
            override_rate=override_rate,
            partial_rate=safe_div(len(partial), total_resolved),
            pending_rate=safe_div(len(pending), total),
            mean_confidence_on_confirmed=mean_conf_confirmed,
            mean_confidence_on_overridden=mean_conf_overridden,
            dimension=criteria.dimension,
        )

    def by_lineage(
        self, verdict_id: str, direction: str = "both", max_depth: int = 500,
    ) -> list[Verdict]:
        if direction not in ("up", "down", "both"):
            raise ValueError(f"direction must be 'up', 'down', or 'both', got '{direction}'")

        visited: set[str] = set()
        result: list[Verdict] = []
        visited.add(verdict_id)

        with self._lock:
            start = self._verdicts.get(verdict_id)
        if start is None:
            return []

        # Iterative BFS to avoid RecursionError on deep chains
        queue: list[tuple[str, int]] = []  # (verdict_id, depth)

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
            with self._lock:
                v = self._verdicts.get(vid)
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
        count = 0
        with self._lock:
            for verdict in self._verdicts.values():
                if verdict.outcome.status != "pending":
                    continue
                expiry_time = verdict.timestamp + timedelta(seconds=verdict.metadata.ttl)
                if expiry_time < now:
                    verdict.outcome.status = "expired"
                    verdict.outcome.closed_at = now
                    count += 1
        return count
