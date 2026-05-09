"""Override event data contract + privacy + webhook field mapping.

Spec: ``NTHLAYER_MISSING_CAPABILITIES_SPEC.md`` § 4. The OTel canonical
form is ``gen_ai.override`` with attributes ``gen_ai.override.*``. Field
names here are the snake_case Python equivalents — ``to_otel_attributes()``
flips them back to the dotted form on emission.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Spec § 4: confidence > 0.85 increments the HCF judgment SLO counter.
HIGH_CONFIDENCE_THRESHOLD = 0.85


def hash_reviewer(reviewer: str) -> str:
    """Stable SHA-256 hex of a reviewer identifier.

    Privacy default per spec § 4: reviewer identities are hashed unless
    the operator opts in to plaintext. Hex digest (not bytes) so the
    result round-trips through JSON / YAML / Prometheus labels without
    further encoding.
    """
    return hashlib.sha256(reviewer.encode("utf-8")).hexdigest()


@dataclass
class OverridePrivacyConfig:
    """Privacy posture for override ingestion.

    Defaults are the spec § 4 GDPR-safe baseline: reviewer hashed,
    reason stored. Operators opt in to plaintext / opt out of reason
    storage explicitly.
    """

    plaintext_reviewer: bool = False
    exclude_reason: bool = False


@dataclass
class OverrideEvent:
    """Canonical override event — the gen_ai.override OTel form.

    Required fields per spec § 4: ``decision_id``, ``service``,
    ``corrected_action``, ``reviewer``. Everything else is optional but
    populated when the source has it.
    """

    decision_id: str
    service: str
    corrected_action: str
    reviewer: str
    original_action: str | None = None
    reason: str | None = None
    confidence_at_decision: float | None = None
    source_system: str | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def __post_init__(self) -> None:
        for name in ("decision_id", "service", "corrected_action", "reviewer"):
            value = getattr(self, name)
            if not value:
                raise ValueError(
                    f"OverrideEvent.{name} is required (spec § 4)"
                )
        if self.confidence_at_decision is not None:
            if not 0.0 <= self.confidence_at_decision <= 1.0:
                raise ValueError(
                    "confidence_at_decision must be in [0.0, 1.0]; got "
                    f"{self.confidence_at_decision}"
                )
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "OverrideEvent.timestamp must be timezone-aware "
                "(use datetime.now(timezone.utc))"
            )

    @property
    def is_high_confidence_failure(self) -> bool:
        """None confidence does not count — HCF cannot be proven without
        a confidence signal at decision time.
        """
        return (
            self.confidence_at_decision is not None
            and self.confidence_at_decision > HIGH_CONFIDENCE_THRESHOLD
        )

    def to_otel_attributes(self) -> dict[str, Any]:
        """None-valued fields are dropped so the emitted span event
        matches the spec § 4 example shape.
        """
        attrs: dict[str, Any] = {
            "gen_ai.override.decision_id": self.decision_id,
            "gen_ai.override.service": self.service,
            "gen_ai.override.corrected_action": self.corrected_action,
            "gen_ai.override.reviewer": self.reviewer,
        }
        for name, otel_key in (
            ("original_action", "gen_ai.override.original_action"),
            ("reason", "gen_ai.override.reason"),
            ("confidence_at_decision", "gen_ai.override.confidence_at_decision"),
            ("source_system", "gen_ai.override.source_system"),
        ):
            value = getattr(self, name)
            if value is not None:
                attrs[otel_key] = value
        return attrs


def _resolve_path(payload: dict[str, Any], dotted: str) -> Any | None:
    """Walk a dotted path through a nested dict; return None on any miss."""
    cursor: Any = payload
    for part in dotted.split("."):
        if not isinstance(cursor, dict):
            return None
        if part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def map_webhook_to_override(
    payload: dict[str, Any],
    mapping: dict[str, str],
    *,
    defaults: dict[str, Any] | None = None,
) -> OverrideEvent:
    """Build an OverrideEvent from an arbitrary webhook payload.

    ``mapping`` maps OverrideEvent field names to dotted paths in the
    payload (e.g. ``{"reviewer": "issue.assignee.emailAddress"}``);
    ``defaults`` supplies fallback values for fields absent from both
    the mapping and the payload. Required-field validation runs in
    ``OverrideEvent.__post_init__`` — a missing required mapping path
    raises there with a clear message.
    """
    defaults = defaults or {}
    kwargs: dict[str, Any] = {}
    for field_name, path in mapping.items():
        value = _resolve_path(payload, path)
        if value is not None:
            kwargs[field_name] = value
    for field_name, fallback in defaults.items():
        if field_name not in kwargs or kwargs[field_name] is None:
            kwargs[field_name] = fallback

    if "timestamp" in kwargs and isinstance(kwargs["timestamp"], str):
        kwargs["timestamp"] = _parse_iso_timestamp(kwargs["timestamp"])
    # Coerce so the range check raises ValueError, not TypeError.
    if isinstance(kwargs.get("confidence_at_decision"), str):
        try:
            kwargs["confidence_at_decision"] = float(
                kwargs["confidence_at_decision"],
            )
        except ValueError as exc:
            raise ValueError(
                "confidence_at_decision could not be parsed as float: "
                f"{kwargs['confidence_at_decision']!r}"
            ) from exc

    # Surface mapping-level gaps as ValueError naming the field, so a
    # webhook integrator sees one error type at the boundary instead of
    # a Python TypeError from the dataclass constructor.
    for required in ("decision_id", "service", "corrected_action", "reviewer"):
        if not kwargs.get(required):
            raise ValueError(
                f"OverrideEvent.{required} could not be resolved from "
                f"webhook payload (mapping or defaults must supply it)"
            )

    return OverrideEvent(**kwargs)


def _parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp from a webhook payload.

    Naive timestamps are rejected: relabelling them UTC would silently
    skew override timestamps by the source's actual offset, corrupting
    reversal-rate windowing and HCF burn calculations. Webhook authors
    must supply a tz-aware timestamp (``Z`` or ``+HH:MM``).
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"timestamp could not be parsed as ISO 8601: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(
            f"timestamp must be tz-aware (got naive: {value!r}); "
            "include a 'Z' or '+HH:MM' offset"
        )
    return parsed
