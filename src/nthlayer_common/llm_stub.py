"""Canned-response LLM stub for CI integration tests.

Activated by ``NTHLAYER_LLM_STUB=canned``. When active, both ``llm_call()``
(raw text) and ``structured_call()``/``structured_call_with_usage()``
(Instructor-validated Pydantic models) short-circuit before any HTTP call
and return deterministic responses indexed by agent role.

Role detection for raw text calls reads a marker phrase from the system
prompt (e.g. "You are a triage agent"). Structured calls dispatch on the
``response_model`` class name.

The stub is not a behavioural fake: every call of a given role returns the
same canned data regardless of input. Its job is to exercise wiring (HTTP
plumbing, verdict shape, lineage propagation), not LLM behaviour.

DO NOT enable ``NTHLAYER_LLM_STUB`` in production. The integration test
harness (``test/integration-three-tier.sh``) sets it on the worker process
only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, TypeVar

from nthlayer_common.llm import LLMResponse

T = TypeVar("T")

_STUB_ENV_VAR = "NTHLAYER_LLM_STUB"
_STUB_VALUE = "canned"


def is_stub_enabled() -> bool:
    """Return True iff NTHLAYER_LLM_STUB=canned is set in the environment.

    Match is case-insensitive and trim-tolerant — operators routinely set env
    values as ``CANNED`` or with a trailing newline in YAML, and silently
    falling through to a real HTTP call (potentially leaking an API key) is
    the wrong default.
    """
    raw = os.environ.get(_STUB_ENV_VAR)
    if raw is None:
        return False
    return raw.strip().lower() == _STUB_VALUE


# ---------------------------------------------------------------------------
# Role detection (raw text path)
# ---------------------------------------------------------------------------

# Lower-case marker phrases looked up in the system prompt. Order matters
# only when markers could collide; current set is unambiguous.
_ROLE_MARKERS: tuple[tuple[str, str], ...] = (
    ("triage", "you are a triage agent"),
    ("communication", "you are a communication agent"),
    ("investigation", "you are an investigation agent"),
    ("remediation", "you are a remediation agent"),
)


def _detect_role(system: str | None) -> str:
    # Guard against None — type says str but a misbehaving caller surfaces
    # here clearly as "unknown" rather than crashing with AttributeError.
    haystack = (system or "").lower()
    for role, marker in _ROLE_MARKERS:
        if marker in haystack:
            return role
    return "unknown"


# ---------------------------------------------------------------------------
# Canned text responses (parsed by respond agents via base._parse_json)
# ---------------------------------------------------------------------------

_TRIAGE_BODY = {
    "severity": 1,
    "blast_radius": ["fraud-detect"],
    "affected_slos": ["reversal_rate"],
    "assigned_team": "payments-ml",
    "reasoning": "Stub triage: reversal_rate breach detected on fraud-detect; "
                 "blast radius limited to caller services.",
    "confidence": 0.85,
}

_INVESTIGATION_BODY = {
    "hypotheses": [
        {
            "description": "Recent model deploy degraded judgment quality.",
            "confidence": 0.9,
            "evidence": [
                "reversal_rate above target",
                "timing aligns with most recent deploy window",
            ],
            "change_candidate": "fraud-detect-model-v2.4",
        }
    ],
    "root_cause": "model_deploy_regression",
    "root_cause_confidence": 0.9,
    "reasoning": "Stub investigation: model deploy is the highest-confidence cause.",
    "confidence": 0.85,
}

_COMMUNICATION_BODY = {
    "updates": [
        {
            "channel": "status_page",
            "update_type": "initial",
            "content": "Stub comms: investigating elevated reversal rate on fraud-detect.",
        }
    ],
    "reasoning": "Stub communication: initial advisory; no resolution claim.",
    "confidence": 0.8,
}

# "rollback" exists in the safe-actions registry with requires_approval=True,
# so respond's RemediationAgent treats this canned response as a real action
# and exercises the approval path.
_REMEDIATION_BODY = {
    "proposed_action": "rollback",
    "target": "fraud-detect",
    "risk_assessment": "Stub remediation: rollback to previous model version is low-risk.",
    "requires_human_approval": True,
    "reasoning": "Stub remediation: rollback recommended pending human approval.",
    "confidence": 0.85,
}

_GENERIC_BODY = {
    "reasoning": "Stub response (no role detected from system prompt).",
    "confidence": 0.5,
}

_TEXT_BY_ROLE: dict[str, str] = {
    "triage": json.dumps(_TRIAGE_BODY),
    "investigation": json.dumps(_INVESTIGATION_BODY),
    "communication": json.dumps(_COMMUNICATION_BODY),
    "remediation": json.dumps(_REMEDIATION_BODY),
    "unknown": json.dumps(_GENERIC_BODY),
}


def stub_text_response(system: str, model: str | None) -> LLMResponse:
    """Return a canned LLMResponse for the role detected in ``system``."""
    role = _detect_role(system)
    return LLMResponse(
        text=_TEXT_BY_ROLE[role],
        model=model or "stub/canned",
        provider="stub",
        input_tokens=0,
        output_tokens=0,
    )


# ---------------------------------------------------------------------------
# Canned structured responses (Instructor path)
# ---------------------------------------------------------------------------

# Each factory builds a minimal valid instance of the caller's
# response_model. Pydantic v2 auto-coerces nested dicts into BaseModel
# fields, so factories pass plain dicts where convenient.

def _build_evaluation_result(response_model: type[T]) -> T:
    # Used by nthlayer_workers.measure.pipeline.evaluator.ModelEvaluator.
    # Real EvaluationResult.dimensions is dict[str, DimensionScore] keyed by
    # dimension name (not a list). Pydantic v2 coerces the nested dict into
    # DimensionScore.
    return response_model(  # type: ignore[call-arg]
        dimensions={"stub_dimension": {"score": 0.85, "reasoning": "Stub: dimension passes."}},
        confidence=0.8,
    )


def _build_snapshot_summary(response_model: type[T]) -> T:
    # Used by nthlayer_workers.correlate.summary.generate_summary
    return response_model(  # type: ignore[call-arg]
        summary="Stub correlation summary: events grouped by domain.",
        notable_omissions=[],
    )


# Registry is keyed by class __name__ — keeping the test fixtures simple
# (they reuse the production names rather than coupling to production module
# paths). The trade-off: an unrelated class with the same name will dispatch
# here. Mismatched shape surfaces as a Pydantic ValidationError wrapped
# below, which points at this module so the cause is greppable.
_STRUCTURED_FACTORIES: dict[str, Callable[[type[Any]], Any]] = {
    "EvaluationResult": _build_evaluation_result,
    "SnapshotSummary": _build_snapshot_summary,
}


def stub_structured_response(response_model: type[T]) -> T:
    """Return a canned validated instance of ``response_model``.

    Raises NotImplementedError if no factory is registered for the model's
    class name. Add an entry to ``_STRUCTURED_FACTORIES`` when introducing a
    new structured-call site that must be exercised under the stub.

    If a registered factory's canned shape does not satisfy the supplied
    ``response_model`` (e.g. an unrelated class shares the same ``__name__``
    as a registered one), the underlying Pydantic ValidationError is
    re-raised as ``NotImplementedError`` with the fully-qualified class path
    so the collision is obvious.
    """
    name = response_model.__name__
    factory = _STRUCTURED_FACTORIES.get(name)
    if factory is None:
        raise NotImplementedError(
            f"NTHLAYER_LLM_STUB=canned: no canned response registered for "
            f"response_model={name!r}. Register a factory in "
            f"nthlayer_common/llm_stub.py:_STRUCTURED_FACTORIES."
        )
    try:
        return factory(response_model)
    except Exception as exc:  # noqa: BLE001 — convert to clear stub error
        qualified = f"{response_model.__module__}.{name}"
        raise NotImplementedError(
            f"NTHLAYER_LLM_STUB=canned: registered factory for {name!r} "
            f"does not match the shape of {qualified!r}. Either rename the "
            f"class or register a dedicated factory in "
            f"nthlayer_common/llm_stub.py:_STRUCTURED_FACTORIES. "
            f"Underlying error: {exc!r}"
        ) from exc
