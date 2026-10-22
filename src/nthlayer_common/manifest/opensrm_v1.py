"""opensrm/v1 → opensrm.nthlayer.io/v2 migration (opensrm-pnmv).

``opensrm/v1`` is the published OpenSRM v1 schema (opensrm/spec/v1/schema.json).
NthLayer does not load it — its own v1 dialect is ``srm/v1`` (parser/v1.py),
which differs from the schema in nearly every section. This module converts a
schema-shaped ``opensrm/v1`` document to v2 following
opensrm/spec/v2/AUTHORING.md §7, and never drops a field silently: anything
with no v2 home is returned in ``MigrationResult.unmapped`` for the caller to
refuse or accept. Targets are 0–1 ratios on both sides; nothing here
multiplies by 100. See docs/superpowers/decisions/opensrm-v1-support.md in the
nthlayer front door.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nthlayer_common.manifest.models import (
    VALID_TIERS,
    JudgmentThreshold,
    ThresholdMeasurement,
    resolve_service_type,
    valid_service_types_phrase,
)
from nthlayer_common.manifest.openslo.parser import OpenSLOParseError, _infer_openslo_type
from nthlayer_common.manifest.thresholds import (
    _SRM_V1_CALIBRATION_BINS,
    is_duration,
    threshold_to_v2_entry,
)

OPENSRM_V1_API_VERSION = "opensrm/v1"
_V1_KIND = "ServiceReliabilityManifest"
_V2_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_ENTITY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NOT_IN_SCHEMA = "not part of the opensrm/v1 schema"

OPENSRM_V1_REJECTION = (
    "{path}: apiVersion 'opensrm/v1' is the published OpenSRM v1 schema, which "
    "NthLayer does not load. NthLayer reads opensrm.nthlayer.io/v2 (and srm/v1, "
    "its own v1 dialect). Convert it with: nthlayer migrate-manifest {path}"
)

# Per opensrm/spec/v1/schema.json definition: (keys converted, {key: why not}).
# tests/test_opensrm_v1_schema_completeness.py checks every definition's
# properties are covered, both ways.
_SECTIONS: dict[str, tuple[frozenset[str], dict[str, str]]] = {
    "Metadata": (
        frozenset({"name", "team", "tier", "description", "labels", "annotations"}),
        {"template": "v2 templates are a different mechanism (AUTHORING §4.9); re-declare by hand"},
    ),
    "Spec": (
        frozenset({"type", "slos", "contract", "dependencies", "ownership",
                   "observability", "instrumentation"}),
        {
            "notifications": "no v2 equivalent (AUTHORING §7.6)",
            "deployment": "no v2 equivalent (AUTHORING §7.6)",
        },
    ),
}


@dataclass(frozen=True)
class UnmappedField:
    """A v1 field with no v2 home. *path* is dotted into the v1 document."""

    path: str
    reason: str


@dataclass(frozen=True)
class MigrationResult:
    manifest: dict[str, Any]
    unmapped: tuple[UnmappedField, ...]


@dataclass
class _Ctx:
    service: str
    unmapped: list[UnmappedField] = field(default_factory=list)
    annotations: dict[str, str] = field(default_factory=dict)

    def skip(self, path: str, reason: str) -> None:
        self.unmapped.append(UnmappedField(path, reason))


def is_opensrm_v1_format(data: object) -> bool:
    return isinstance(data, dict) and data.get("apiVersion") == OPENSRM_V1_API_VERSION


def _mapping(path: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping, got {type(value).__name__}")
    return value


def _entity_name(path: str, value: object) -> str:
    if not isinstance(value, str) or not _ENTITY_NAME.fullmatch(value):
        raise ValueError(
            f"{path} must be a name usable in an entity ref ([A-Za-z0-9._-], "
            f"starting alphanumeric), got {value!r}"
        )
    return value


def _string_map(path: str, value: object) -> dict[str, str]:
    m = _mapping(path, value)
    for k, v in m.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"{path} must map strings to strings, got {k!r}: {v!r}")
    return dict(m)


def _service_type(declared: object, supplied: str | None) -> str:
    if declared is None:
        if supplied is None:
            raise ValueError(
                "spec.type is absent: it is optional in opensrm/v1 but required in "
                "v2 (spec.service.type). Supply it (migrate-manifest --service-type). "
                f"Must be one of: {valid_service_types_phrase()}."
            )
        declared_type = supplied
    else:
        declared_type = declared
        if supplied is not None and resolve_service_type(supplied) != resolve_service_type(declared):
            raise ValueError(
                f"spec.type is {declared!r} but {supplied!r} was supplied; "
                "a supplied service type only fills an absent spec.type"
            )
    resolved = resolve_service_type(declared_type) if isinstance(declared_type, str) else None
    if resolved is None:
        raise ValueError(
            f"spec.type {declared_type!r} is not a valid v2 service type. "
            f"Must be one of: {valid_service_types_phrase()}."
        )
    return resolved


def convert_opensrm_v1_to_v2(
    data: dict[str, Any], *, service_type: str | None = None
) -> MigrationResult:
    """Convert an ``opensrm/v1`` document to an ``opensrm.nthlayer.io/v2`` dict.

    Raises ValueError only for a document that cannot be converted at all
    (wrong apiVersion or kind, unusable name/team/tier/type, a malformed value on
    a field that is converted). Fields with no v2 home are reported in
    ``unmapped``, never raised and never dropped.
    """
    if not is_opensrm_v1_format(data):
        got = data.get("apiVersion") if isinstance(data, dict) else type(data).__name__
        raise ValueError(f"expects apiVersion {OPENSRM_V1_API_VERSION!r}, got {got!r}")
    if data.get("kind") != _V1_KIND:
        raise ValueError(
            f"only kind {_V1_KIND!r} converts, got {data.get('kind')!r} "
            "(a v1 Template has no v2 equivalent)"
        )
    metadata = _mapping("metadata", data.get("metadata"))
    spec = _mapping("spec", data.get("spec", {}))

    name = metadata.get("name")
    if not isinstance(name, str) or not _V2_NAME.fullmatch(name):
        raise ValueError(
            f"metadata.name must match ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ in v2, got {name!r}"
        )
    ctx = _Ctx(service=name)

    tier = metadata.get("tier")
    if not isinstance(tier, str) or tier not in VALID_TIERS:
        raise ValueError(f"metadata.tier must be one of {sorted(VALID_TIERS)}, got {tier!r}")
    labels = _string_map("metadata.labels", metadata.get("labels", {}))
    if labels.get("tier", tier) != tier:
        raise ValueError(f"metadata.labels.tier {labels['tier']!r} disagrees with metadata.tier {tier!r}")
    labels["tier"] = tier
    if "annotations" in metadata:
        ctx.annotations.update(_string_map("metadata.annotations", metadata["annotations"]))

    for key in metadata:
        converted, unmapped = _SECTIONS["Metadata"]
        if key in unmapped:
            ctx.skip(f"metadata.{key}", unmapped[key])
        elif key not in converted:
            ctx.skip(f"metadata.{key}", _NOT_IN_SCHEMA)

    team = _entity_name("metadata.team", metadata.get("team"))
    service: dict[str, Any] = {"name": name, "type": _service_type(spec.get("type"), service_type)}
    if "description" in metadata:
        if not isinstance(metadata["description"], str):
            raise ValueError(
                f"metadata.description must be a string, got {type(metadata['description']).__name__}"
            )
        service["description"] = metadata["description"]

    out_spec: dict[str, Any] = {"owner": {"group": f"group:default/{team}"}, "service": service}
    _convert_spec(spec, out_spec, ctx)

    out_metadata: dict[str, Any] = {"name": name, "labels": labels}
    if ctx.annotations:
        out_metadata["annotations"] = ctx.annotations
    manifest = {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": out_metadata,
        "spec": out_spec,
    }
    return MigrationResult(manifest=manifest, unmapped=tuple(ctx.unmapped))


def _convert_spec(spec: dict[str, Any], out: dict[str, Any], ctx: _Ctx) -> None:
    converted, unmapped = _SECTIONS["Spec"]
    for key, value in spec.items():
        if key == "type":
            continue
        if key in unmapped:
            ctx.skip(f"spec.{key}", unmapped[key])
        elif key not in converted:
            ctx.skip(f"spec.{key}", _NOT_IN_SCHEMA)
        else:
            _SPEC_HANDLERS[key](value, out, ctx)


_SPEC_HANDLERS: dict[str, Callable[[object, dict[str, Any], _Ctx], None]] = {}


# v1 classical SLO key → the slo_type common's OpenSLO parser must infer.
_CLASSICAL = {
    "availability": "availability", "connection_availability": "availability",
    "error_rate": "error_rate",
    "latency": "latency", "duration": "latency", "query_latency": "latency",
}
# spec/v1/schema.json window defaults for the converted types (all 30d).
_WINDOW_DEFAULT = dict.fromkeys(_CLASSICAL, "30d")
# Percentile properties per v1 definition (spec/v1/schema.json): DurationSLO has no p999.
_PERCENTILES_BY_DEFN = {
    "LatencySLO": ("p50", "p90", "p95", "p99", "p999"),
    "DurationSLO": ("p50", "p90", "p95", "p99"),
    "QueryLatencySLO": ("p50", "p90", "p95", "p99", "p999"),
}
_DEFN_FOR_KEY = {"latency": "LatencySLO", "duration": "DurationSLO", "query_latency": "QueryLatencySLO"}
_PERCENTILE_WHY = ("kept only in the OpenSLO displayName: an objective without an "
                   "indicator carries no machine-readable percentile")

_SECTIONS["SLOs"] = (
    frozenset({*_CLASSICAL, "openslo", "judgment"}),
    {
        "throughput": "v1 declares a rate (e.g. 1000 rps); an OpenSLO objective needs a ratio target (AUTHORING §7.3)",
        "success_rate": "common's v2 parser cannot infer this SLO's type (opensrm-47tt)",
        "processing_time": "common's v2 parser cannot infer this SLO's type (opensrm-47tt)",
        "lag": "common's v2 parser cannot infer this SLO's type (opensrm-47tt)",
        "schedule_adherence": "maximum-only SLO (max_delay) with no target; an OpenSLO objective needs one",
        "data_freshness": "maximum-only SLO (max_age) with no target; an OpenSLO objective needs one",
        "replication_lag": "maximum-only SLO (max_lag) with no target; an OpenSLO objective needs one",
        "custom": "custom SLOs have no v2 mapping",
    },
)
for _defn, _pcts in _PERCENTILES_BY_DEFN.items():
    _SECTIONS[_defn] = (frozenset({"target", "window"}), dict.fromkeys(_pcts, _PERCENTILE_WHY))
for _defn in ("AvailabilitySLO", "ErrorRateSLO", "ConnectionAvailabilitySLO"):
    _SECTIONS[_defn] = (frozenset({"target", "window"}), {})


def _ratio(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{path} must be a ratio in [0, 1], got {value!r}")
    return float(value)


def _duration(path: str, value: object) -> str:
    if not is_duration(value):
        raise ValueError(f"{path} must be a duration like '30d' or '200ms', got {value!r}")
    return str(value)


def _pct(ratio: float) -> str:
    return f"{ratio * 100:g}%"


def _openslo(key: str, cfg: object, ctx: _Ctx) -> dict[str, Any]:
    path = f"spec.slos.{key}"
    cfg = _mapping(path, cfg)
    target = _ratio(f"{path}.target", cfg.get("target"))
    window = _duration(f"{path}.window", cfg.get("window", _WINDOW_DEFAULT[key]))
    kind = _CLASSICAL[key]
    if kind == "error_rate":
        # v1 ErrorRateSLO.target is a MAXIMUM error ratio; OpenSLO targets are
        # good-ratios (the spec's reference output turns 0.001 into 0.999).
        good = round(1.0 - target, 12)
        display = f"Error rate under {_pct(target)} over {window}"
    elif kind == "latency":
        pct_keys = _PERCENTILES_BY_DEFN[_DEFN_FOR_KEY[key]]
        pcts = [(p, _duration(f"{path}.{p}", cfg[p])) for p in pct_keys if p in cfg]
        for p, _ in pcts:
            ctx.skip(f"{path}.{p}", _PERCENTILE_WHY)
        good = target
        bits = ", ".join(f"{p} under {v}" for p, v in pcts)
        display = f"{_pct(target)} of requests {bits} over {window}" if bits else f"{_pct(target)} {key.replace('_', ' ')} over {window}"
    else:
        good = target
        display = f"{_pct(target)} {key.replace('_', ' ')} over {window}"
    for extra in cfg:
        if extra not in ("target", "window", *_PERCENTILES_BY_DEFN.get(_DEFN_FOR_KEY.get(key, ""), ())):
            ctx.skip(f"{path}.{extra}", _NOT_IN_SCHEMA)

    slo_name = f"{ctx.service}-{key.replace('_', '-')}"
    objective = {"displayName": display, "target": good}
    try:
        inferred: str | None = _infer_openslo_type(slo_name, good * 100, None, None, None, objective)
    except OpenSLOParseError:
        inferred = None
    if inferred != kind:
        raise ValueError(
            f"SLO '{slo_name}' would be read as {inferred!r}, not {kind!r}: common's v2 "
            "parser infers the type from the SLO name and the service name contains "
            "another type's word (opensrm-47tt). Rename the service or migrate this SLO by hand."
        )
    return {
        "apiVersion": "openslo/v1",
        "kind": "SLO",
        "metadata": {"name": slo_name},
        "spec": {
            "service": ctx.service,
            "timeWindow": [{"duration": window, "isRolling": True}],
            "objectives": [objective],
        },
    }


def _convert_slos(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
    slos = _mapping("spec.slos", value)
    converted, unmapped = _SECTIONS["SLOs"]
    docs: list[dict[str, Any]] = []
    for key, cfg in slos.items():
        if key in unmapped:
            ctx.skip(f"spec.slos.{key}", unmapped[key])
        elif key not in converted:
            ctx.skip(f"spec.slos.{key}", _NOT_IN_SCHEMA)
        elif key == "openslo":
            if not isinstance(cfg, list):
                raise ValueError(f"spec.slos.openslo must be a list, got {type(cfg).__name__}")
            for i, item in enumerate(cfg):
                item = _mapping(f"spec.slos.openslo[{i}]", item)
                for extra in item:
                    if extra != "path":
                        ctx.skip(f"spec.slos.openslo[{i}].{extra}", _NOT_IN_SCHEMA)
                ref = item.get("path")
                if not isinstance(ref, str) or not ref:
                    raise ValueError(f"spec.slos.openslo[{i}].path must be a non-empty string")
                docs.append({"$ref": ref})
        elif key == "judgment":
            _convert_judgment(cfg, out, ctx)
        else:
            docs.append(_openslo(key, cfg, ctx))
    if docs:
        out["slo"] = docs


_JUDGMENT_SLO_WHY = (
    "judgment_slo is new capability, not migration (AUTHORING §7.7 step 8), and no "
    "v2 judgment-SLO shape passes both the schema and common's parser (opensrm-a742)"
)
_SECTIONS["JudgmentSLOs"] = (
    frozenset({"calibration", "feedback_latency"}),
    {"reversal_rate": _JUDGMENT_SLO_WHY, "high_confidence_failure": _JUDGMENT_SLO_WHY},
)
_SECTIONS["FlatCalibrationSLO"] = (frozenset({"target", "window"}), {})
_SECTIONS["FeedbackLatencySLO"] = (frozenset({"p50", "p90"}), {})


def _report_extra_keys(path: str, cfg: dict[str, Any], defn: str, ctx: _Ctx) -> None:
    converted, unmapped = _SECTIONS[defn]
    for extra in cfg:
        if extra in unmapped:
            ctx.skip(f"{path}.{extra}", unmapped[extra])
        elif extra not in converted:
            ctx.skip(f"{path}.{extra}", _NOT_IN_SCHEMA)


def _convert_judgment(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
    judgment = _mapping("spec.slos.judgment", value)
    if out["service"]["type"] != "ai-gate":
        ctx.skip("spec.slos.judgment",
                 "v2 allows judgment blocks only on spec.service.type ai-gate (AUTHORING §3.1)")
        return
    converted, unmapped = _SECTIONS["JudgmentSLOs"]
    entries: list[dict[str, Any]] = []
    for key, cfg in judgment.items():
        path = f"spec.slos.judgment.{key}"
        if key in unmapped:
            ctx.skip(path, unmapped[key])
        elif key not in converted:
            ctx.skip(path, _NOT_IN_SCHEMA)
        elif key == "calibration":
            cfg = _mapping(path, cfg)
            threshold = JudgmentThreshold(
                type="calibration",
                bounds={"maximum_expected_calibration_error": _ratio(f"{path}.target", cfg.get("target"))},
                measurement=ThresholdMeasurement(
                    window=_duration(f"{path}.window", cfg.get("window")),
                    bins=_SRM_V1_CALIBRATION_BINS,
                ),
            )
            entries.append(threshold_to_v2_entry(threshold))
            ctx.skip(path, f"converted with measurement.bins: {_SRM_V1_CALIBRATION_BINS}, "
                           "a value opensrm/v1 does not declare; check it")
            _report_extra_keys(path, cfg, "FlatCalibrationSLO", ctx)
        else:  # feedback_latency
            cfg = _mapping(path, cfg)
            bounds = {f"maximum_{p}": _duration(f"{path}.{p}", cfg[p]) for p in ("p50", "p90") if p in cfg}
            _report_extra_keys(path, cfg, "FeedbackLatencySLO", ctx)
            if not bounds:
                ctx.skip(path, "declares neither p50 nor p90, so there is no bound to convert")
                continue
            entries.append(threshold_to_v2_entry(JudgmentThreshold(type="feedback_latency", bounds=bounds)))
    if entries:
        out["judgment_threshold"] = entries


_SECTIONS["Contract"] = (frozenset({"availability", "latency", "throughput"}),
                         {"judgment": "v2 contracts[].promise has availability, latency_p99 and throughput only (AUTHORING §7.4)"})
_SECTIONS["ContractLatency"] = (frozenset({"p99"}), dict.fromkeys(
    ("p50", "p999"), "v2 contracts[].promise has latency_p99 only (AUTHORING §7.4)"))
_SECTIONS["ContractThroughput"] = (frozenset({"max_rps"}),
                                   {"burst": "v2 contracts[].promise.throughput is a single rate (AUTHORING §7.4)"})
_SECTIONS["Dependency"] = (frozenset({"service", "type", "expects"}), {
    "critical": "v2 expresses this as a fallback, which only the author can choose (AUTHORING §7.5)",
    "manifest": "v2 resolves dependencies through the catalogue (AUTHORING §7.5)"})
# spec/v1/schema.json Dependency.expects and its inline latency object.
_SECTIONS["Dependency.expects"] = (frozenset({"availability", "latency"}), {})
_SECTIONS["Dependency.expects.latency"] = (frozenset({"p99"}), {})
_SECTIONS["Ownership"] = (frozenset({"team", "slack", "escalation", "pagerduty", "runbook", "documentation"}), {
    "email": "no v2 field or AUTHORING §7.2 annotation key",
    "oncall_required": "no v2 field"})
# spec/v1/schema.json Ownership.pagerduty (inline).
_SECTIONS["Ownership.pagerduty"] = (frozenset({"service_id"}), {
    "escalation_policy_id": "no AUTHORING §7.2 annotation key"})
_SECTIONS["Observability"] = (frozenset(), {
    "metrics": "v2 instrumentation needs a type per metric (AUTHORING §7.7 step 6)",
    "dashboards": "no v2 equivalent (AUTHORING §7.6)",
    "alerts": "no v2 equivalent (AUTHORING §7.6)",
    "tracing": "no v2 equivalent (AUTHORING §7.6)"})
_SECTIONS["Instrumentation"] = (frozenset({"events"}), {"attributes": "v2 instrumentation has no attribute list"})
# spec/v1/schema.json Instrumentation.events: each key maps to an event-name string.
_SECTIONS["Instrumentation.events"] = (frozenset({"decision", "reversal", "outcome", "audit_result"}), {})

_RESOURCE_TYPES = ("database", "cache", "queue")
_OWNERSHIP_ANNOTATIONS = {   # AUTHORING §7.2
    "slack": "slack.com/channel",
    "runbook": "opensrm.nthlayer.io/runbook",
    "documentation": "opensrm.nthlayer.io/documentation",
}


def _keys(section: str, path: str, value: dict[str, Any], ctx: _Ctx) -> list[str]:
    """Report the unmapped and unknown keys of *value*; return the converted ones, in order."""
    converted, unmapped = _SECTIONS[section]
    keep = []
    for key in value:
        if key in unmapped:
            ctx.skip(f"{path}.{key}", unmapped[key])
        elif key not in converted:
            ctx.skip(f"{path}.{key}", _NOT_IN_SCHEMA)
        else:
            keep.append(key)
    return keep


def _annotate(ctx: _Ctx, key: str, value: object, path: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string, got {value!r}")
    if ctx.annotations.get(key, value) != value:
        raise ValueError(f"{path} would set annotation {key} to {value!r}, but metadata.annotations has {ctx.annotations[key]!r}")
    ctx.annotations[key] = value


def _convert_contract(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
    contract = _mapping("spec.contract", value)
    promise: dict[str, Any] = {}
    for key in _keys("Contract", "spec.contract", contract, ctx):
        if key == "availability":
            promise["availability"] = _ratio("spec.contract.availability", contract[key])
        elif key == "latency":
            latency = _mapping("spec.contract.latency", contract[key])
            if "p99" in _keys("ContractLatency", "spec.contract.latency", latency, ctx):
                promise["latency_p99"] = _duration("spec.contract.latency.p99", latency["p99"])
        else:  # throughput
            throughput = _mapping("spec.contract.throughput", contract[key])
            if "max_rps" in _keys("ContractThroughput", "spec.contract.throughput", throughput, ctx):
                rps = throughput["max_rps"]
                if isinstance(rps, bool) or not isinstance(rps, int) or rps < 1:
                    raise ValueError(f"spec.contract.throughput.max_rps must be an integer >= 1, got {rps!r}")
                promise["throughput"] = f"{rps}rps"
    if promise:
        out["contracts"] = [{"name": ctx.service, "promise": promise,
                             "contract_owner": out["owner"]["group"]}]


def _convert_dependencies(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
    if not isinstance(value, list):
        raise ValueError(f"spec.dependencies must be a list, got {type(value).__name__}")
    deps = []
    for i, raw in enumerate(value):
        path = f"spec.dependencies[{i}]"
        dep = _mapping(path, raw)
        keys = _keys("Dependency", path, dep, ctx)
        name = _entity_name(f"{path}.service", dep.get("service"))
        dtype = dep.get("type", "service")
        if dtype in _RESOURCE_TYPES:
            ref = f"resource:default/{name}"
        elif dtype == "external":
            ref = f"component:external/{name}"
        elif dtype == "service":
            ref = f"component:default/{name}"
        else:
            raise ValueError(f"{path}.type must be one of service, database, cache, queue, external; got {dtype!r}")
        entry: dict[str, Any] = {"service": ref}
        if "expects" in keys:
            expects = _mapping(f"{path}.expects", dep["expects"])
            for key in _keys("Dependency.expects", f"{path}.expects", expects, ctx):
                if key == "availability":
                    entry["expected_availability"] = _ratio(f"{path}.expects.availability", expects[key])
                else:  # latency
                    latency = _mapping(f"{path}.expects.latency", expects[key])
                    if "p99" in _keys("Dependency.expects.latency", f"{path}.expects.latency", latency, ctx):
                        entry["expected_latency_p99"] = _duration(f"{path}.expects.latency.p99", latency["p99"])
        deps.append(entry)
    if deps:
        out["dependencies"] = deps


def _convert_ownership(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
    own = _mapping("spec.ownership", value)
    for key in _keys("Ownership", "spec.ownership", own, ctx):
        path = f"spec.ownership.{key}"
        if key == "team":
            team = _entity_name(path, own[key])
            if f"group:default/{team}" != out["owner"]["group"]:
                raise ValueError(f"spec.ownership.team {team!r} disagrees with metadata.team ({out['owner']['group']})")
        elif key == "escalation":
            out["owner"]["escalation"] = f"group:default/{_entity_name(path, own[key])}"
        elif key == "pagerduty":
            pd = _mapping(path, own[key])
            if "service_id" in _keys("Ownership.pagerduty", path, pd, ctx):
                _annotate(ctx, "pagerduty.com/service-id", pd["service_id"], f"{path}.service_id")
        else:
            _annotate(ctx, _OWNERSHIP_ANNOTATIONS[key], own[key], path)


def _convert_observability(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
    _keys("Observability", "spec.observability", _mapping("spec.observability", value), ctx)


def _convert_instrumentation(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
    inst = _mapping("spec.instrumentation", value)
    if "events" in _keys("Instrumentation", "spec.instrumentation", inst, ctx):
        path = "spec.instrumentation.events"
        events = _mapping(path, inst["events"])
        required = []
        for key in _keys("Instrumentation.events", path, events, ctx):
            if not isinstance(events[key], str):
                raise ValueError(f"{path}.{key} must be an event-name string, got {events[key]!r}")
            required.append({"type": events[key]})
        if required:
            out["instrumentation"] = {"required_events": required}


_SPEC_HANDLERS.update(
    slos=_convert_slos, contract=_convert_contract, dependencies=_convert_dependencies, ownership=_convert_ownership,
    observability=_convert_observability, instrumentation=_convert_instrumentation,
)
