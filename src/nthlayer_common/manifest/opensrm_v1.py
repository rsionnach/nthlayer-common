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
from collections.abc import Callable, Iterator
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

# Reasons shared by several _SECTIONS entries.
# opensrm-47tt: NthLayer's v2 parser infers an SLO's type from its name and
# cannot be told these types.
_UNINFERABLE_SLO_WHY = "NthLayer's v2 parser cannot infer this SLO's type"
_PERCENTILE_WHY = ("kept only in the OpenSLO displayName: an objective without an "
                   "indicator carries no machine-readable percentile")
# opensrm-a742: no judgment-SLO shape passes both the v2 schema and the parser.
_JUDGMENT_SLO_WHY = (
    "judgment_slo is new capability, not migration (AUTHORING §7.7 step 8), and no "
    "v2 judgment-SLO shape passes both the schema and NthLayer's v2 parser"
)
_MAX_ONLY_WHY = "maximum-only SLO ({field}) with no target; an OpenSLO objective needs one"
_CONTRACT_WHY = "v2 contracts[].promise has {have} only (AUTHORING §7.4)"
_NO_V2_WHY = "no v2 equivalent (AUTHORING §7.6)"

# v1 classical SLO key → the slo_type NthLayer's OpenSLO parser must infer.
_CLASSICAL = {
    "availability": "availability", "connection_availability": "availability",
    "error_rate": "error_rate",
    "latency": "latency", "duration": "latency", "query_latency": "latency",
}
# v1 classical SLO key → its schema definition.
_DEFN_FOR_KEY = {
    "availability": "AvailabilitySLO", "connection_availability": "ConnectionAvailabilitySLO",
    "error_rate": "ErrorRateSLO",
    "latency": "LatencySLO", "duration": "DurationSLO", "query_latency": "QueryLatencySLO",
}
# Percentile properties per v1 definition (spec/v1/schema.json): DurationSLO has no p999.
_PERCENTILES_BY_DEFN = {
    "LatencySLO": ("p50", "p90", "p95", "p99", "p999"),
    "DurationSLO": ("p50", "p90", "p95", "p99"),
    "QueryLatencySLO": ("p50", "p90", "p95", "p99", "p999"),
}
# spec/v1/schema.json: the window default of every converted SLO definition.
_V1_SLO_WINDOW_DEFAULT = "30d"

# The one table of what the converter does with each v1 object's keys.
# Keys are v1 schema definition names (opensrm/spec/v1/schema.json), or
# "Definition.property" for an object declared inline in a definition.
# Values are (converted keys, {unmapped key: reason}).
# tests/test_opensrm_v1_schema_completeness.py checks it both ways.
_SECTIONS: dict[str, tuple[frozenset[str], dict[str, str]]] = {
    "Metadata": (
        frozenset({"name", "team", "tier", "description", "labels", "annotations"}),
        {"template": "v2 templates are a different mechanism (AUTHORING §4.9); re-declare by hand"},
    ),
    "Spec": (
        frozenset({"type", "slos", "contract", "dependencies", "ownership",
                   "observability", "instrumentation"}),
        {"notifications": _NO_V2_WHY, "deployment": _NO_V2_WHY},
    ),
    "SLOs": (
        frozenset({*_CLASSICAL, "openslo", "judgment"}),
        {
            "throughput": "v1 declares a rate (e.g. 1000 rps); an OpenSLO objective needs a ratio target (AUTHORING §7.3)",
            "success_rate": _UNINFERABLE_SLO_WHY,
            "processing_time": _UNINFERABLE_SLO_WHY,
            "lag": _UNINFERABLE_SLO_WHY,
            "schedule_adherence": _MAX_ONLY_WHY.format(field="max_delay"),
            "data_freshness": _MAX_ONLY_WHY.format(field="max_age"),
            "replication_lag": _MAX_ONLY_WHY.format(field="max_lag"),
            "custom": "custom SLOs have no v2 mapping",
        },
    ),
    **{defn: (frozenset({"target", "window"}), dict.fromkeys(pcts, _PERCENTILE_WHY))
       for defn, pcts in _PERCENTILES_BY_DEFN.items()},
    **{defn: (frozenset({"target", "window"}), {})
       for defn in ("AvailabilitySLO", "ErrorRateSLO", "ConnectionAvailabilitySLO")},
    "JudgmentSLOs": (
        frozenset({"calibration", "feedback_latency"}),
        {"reversal_rate": _JUDGMENT_SLO_WHY, "high_confidence_failure": _JUDGMENT_SLO_WHY},
    ),
    "SLOs.openslo[]": (frozenset({"path"}), {}),
    "FlatCalibrationSLO": (frozenset({"target", "window"}), {}),
    "FeedbackLatencySLO": (frozenset({"p50", "p90"}), {}),
    "Contract": (
        frozenset({"availability", "latency", "throughput"}),
        {"judgment": _CONTRACT_WHY.format(have="availability, latency_p99 and throughput")},
    ),
    "ContractLatency": (
        frozenset({"p99"}),
        dict.fromkeys(("p50", "p999"), _CONTRACT_WHY.format(have="latency_p99")),
    ),
    "ContractThroughput": (
        frozenset({"max_rps"}),
        {"burst": "v2 contracts[].promise.throughput is a single rate (AUTHORING §7.4)"},
    ),
    "Dependency": (
        frozenset({"service", "type", "expects"}),
        {
            "critical": "v2 expresses this as a fallback, which only the author can choose (AUTHORING §7.5)",
            "manifest": "v2 resolves dependencies through the catalogue (AUTHORING §7.5)",
        },
    ),
    "Dependency.expects": (frozenset({"availability", "latency"}), {}),
    "Dependency.expects.latency": (frozenset({"p99"}), {}),
    "Ownership": (
        frozenset({"team", "slack", "escalation", "pagerduty", "runbook", "documentation"}),
        {
            "email": "no v2 field or AUTHORING §7.2 annotation key",
            "oncall_required": "no v2 field",
        },
    ),
    "Ownership.pagerduty": (
        frozenset({"service_id"}),
        {"escalation_policy_id": "no AUTHORING §7.2 annotation key"},
    ),
    "Observability": (
        frozenset(),
        {
            "metrics": "v2 instrumentation needs a type per metric (AUTHORING §7.7 step 6)",
            "dashboards": _NO_V2_WHY,
            "alerts": _NO_V2_WHY,
            "tracing": _NO_V2_WHY,
        },
    ),
    "Instrumentation": (
        frozenset({"events"}),
        {"attributes": "v2 instrumentation has no attribute list"},
    ),
    "Instrumentation.events": (frozenset({"decision", "reversal", "outcome", "audit_result"}), {}),
}


@dataclass(frozen=True)
class UnmappedField:
    """A v1 field with no v2 home. *path* is dotted into the v1 document."""

    path: str
    reason: str


@dataclass(frozen=True)
class MigrationResult:
    """*unmapped* is in document order, depth-first: the order the fields appear in the v1 file."""

    manifest: dict[str, Any]
    unmapped: tuple[UnmappedField, ...]


@dataclass
class _Ctx:
    service: str
    unmapped: list[UnmappedField] = field(default_factory=list)
    annotations: dict[str, str] = field(default_factory=dict)
    open_walks: int = 0  # _keys walks started and not yet exhausted

    def report(self, path: str, reason: str) -> None:
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


def _keys(section: str, path: str, value: dict[str, Any], ctx: _Ctx) -> Iterator[str]:
    """Walk *value*'s keys in document order, yielding the converted ones.

    Unmapped and unknown keys are reported as the walk reaches them, so a caller
    that handles each yielded key before asking for the next gets
    ``MigrationResult.unmapped`` in document order, depth-first. A caller with
    nothing to convert must still exhaust it: ``list(_keys(...))``. A walk that
    is not exhausted would silently drop reports, so the conversion counts open
    walks and refuses to return while one is left (see convert_opensrm_v1_to_v2).
    """
    ctx.open_walks += 1  # here, not in the generator: a walk never iterated counts too
    return _walk(section, path, value, ctx)


def _walk(section: str, path: str, value: dict[str, Any], ctx: _Ctx) -> Iterator[str]:
    converted, unmapped = _SECTIONS[section]
    for key in value:
        if key in unmapped:
            ctx.report(f"{path}.{key}", unmapped[key])
        elif key not in converted:
            ctx.report(f"{path}.{key}", _NOT_IN_SCHEMA)
        else:
            yield key
    ctx.open_walks -= 1  # only on exhaustion: an abandoned walk stays counted


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

    list(_keys("Metadata", "metadata", metadata, ctx))

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
    if ctx.open_walks:
        raise RuntimeError("internal: a section walk was not exhausted; fields may have been dropped")
    return MigrationResult(manifest=manifest, unmapped=tuple(ctx.unmapped))


def _convert_spec(spec: dict[str, Any], out: dict[str, Any], ctx: _Ctx) -> None:
    for key in _keys("Spec", "spec", spec, ctx):
        if key != "type":
            _SPEC_HANDLERS[key](spec[key], out, ctx)


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
    defn = _DEFN_FOR_KEY[key]
    list(_keys(defn, path, cfg, ctx))  # reports percentiles and unknown keys
    target = _ratio(f"{path}.target", cfg.get("target"))
    window = _duration(f"{path}.window", cfg.get("window", _V1_SLO_WINDOW_DEFAULT))
    slo_type = _CLASSICAL[key]
    if slo_type == "error_rate":
        # v1 ErrorRateSLO.target is a MAXIMUM error ratio; OpenSLO targets are
        # good-ratios (the spec's reference output turns 0.001 into 0.999).
        good = round(1.0 - target, 12)
        display = f"Error rate under {_pct(target)} over {window}"
    elif slo_type == "latency":
        pcts = [(p, _duration(f"{path}.{p}", cfg[p])) for p in _PERCENTILES_BY_DEFN[defn] if p in cfg]
        good = target
        bits = ", ".join(f"{p} under {v}" for p, v in pcts)
        if bits:
            display = f"{_pct(target)} of requests {bits} over {window}"
        else:
            display = f"{_pct(target)} {key.replace('_', ' ')} over {window}"
    else:
        good = target
        display = f"{_pct(target)} {key.replace('_', ' ')} over {window}"

    slo_name = f"{ctx.service}-{key.replace('_', '-')}"
    objective = {"displayName": display, "target": good}
    # opensrm-47tt: the parser infers the type from the name, so check it.
    try:
        inferred: str | None = _infer_openslo_type(slo_name, good * 100, None, None, None, objective)
    except OpenSLOParseError:
        inferred = None
    if inferred != slo_type:
        raise ValueError(
            f"SLO '{slo_name}' would be read as {inferred!r}, not {slo_type!r}: "
            "NthLayer's v2 parser infers the type from the SLO name and the service "
            "name contains another type's word. Rename the service or migrate this SLO by hand."
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
    docs: list[dict[str, Any]] = []
    for key in _keys("SLOs", "spec.slos", slos, ctx):
        cfg = slos[key]
        if key == "openslo":
            if not isinstance(cfg, list):
                raise ValueError(f"spec.slos.openslo must be a list, got {type(cfg).__name__}")
            for i, item in enumerate(cfg):
                item = _mapping(f"spec.slos.openslo[{i}]", item)
                list(_keys("SLOs.openslo[]", f"spec.slos.openslo[{i}]", item, ctx))
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


def _convert_judgment(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
    judgment = _mapping("spec.slos.judgment", value)
    if out["service"]["type"] != "ai-gate":
        ctx.report("spec.slos.judgment",
                 "v2 allows judgment blocks only on spec.service.type ai-gate (AUTHORING §3.1)")
        return
    entries: list[dict[str, Any]] = []
    for key in _keys("JudgmentSLOs", "spec.slos.judgment", judgment, ctx):
        path = f"spec.slos.judgment.{key}"
        cfg = judgment[key]
        if key == "calibration":
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
            ctx.report(path, f"converted with measurement.bins: {_SRM_V1_CALIBRATION_BINS}, "
                       "a value opensrm/v1 does not declare; check it")
            list(_keys("FlatCalibrationSLO", path, cfg, ctx))
        else:  # feedback_latency
            cfg = _mapping(path, cfg)
            bounds = {f"maximum_{p}": _duration(f"{path}.{p}", cfg[p]) for p in ("p50", "p90") if p in cfg}
            if not bounds:
                ctx.report(path, "declares neither p50 nor p90, so there is no bound to convert")
            list(_keys("FeedbackLatencySLO", path, cfg, ctx))
            if not bounds:
                continue
            entries.append(threshold_to_v2_entry(JudgmentThreshold(type="feedback_latency", bounds=bounds)))
    if entries:
        out["judgment_threshold"] = entries


_RESOURCE_TYPES = ("database", "cache", "queue")
_OWNERSHIP_ANNOTATIONS = {   # AUTHORING §7.2
    "slack": "slack.com/channel",
    "runbook": "opensrm.nthlayer.io/runbook",
    "documentation": "opensrm.nthlayer.io/documentation",
}


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
            for _ in _keys("ContractLatency", "spec.contract.latency", latency, ctx):  # p99
                promise["latency_p99"] = _duration("spec.contract.latency.p99", latency["p99"])
        else:  # throughput
            throughput = _mapping("spec.contract.throughput", contract[key])
            for _ in _keys("ContractThroughput", "spec.contract.throughput", throughput, ctx):  # max_rps
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
        for dep_key in _keys("Dependency", path, dep, ctx):
            if dep_key != "expects":
                continue  # service and type are already in the ref
            expects = _mapping(f"{path}.expects", dep["expects"])
            for key in _keys("Dependency.expects", f"{path}.expects", expects, ctx):
                if key == "availability":
                    entry["expected_availability"] = _ratio(f"{path}.expects.availability", expects[key])
                else:  # latency
                    latency = _mapping(f"{path}.expects.latency", expects[key])
                    for _ in _keys("Dependency.expects.latency", f"{path}.expects.latency", latency, ctx):  # p99
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
            for _ in _keys("Ownership.pagerduty", path, pd, ctx):  # service_id
                _annotate(ctx, "pagerduty.com/service-id", pd["service_id"], f"{path}.service_id")
        else:
            _annotate(ctx, _OWNERSHIP_ANNOTATIONS[key], own[key], path)


def _convert_observability(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
    list(_keys("Observability", "spec.observability", _mapping("spec.observability", value), ctx))


def _convert_instrumentation(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
    inst = _mapping("spec.instrumentation", value)
    for _ in _keys("Instrumentation", "spec.instrumentation", inst, ctx):  # events
        path = "spec.instrumentation.events"
        events = _mapping(path, inst["events"])
        required = []
        for key in _keys("Instrumentation.events", path, events, ctx):
            if not isinstance(events[key], str):
                raise ValueError(f"{path}.{key} must be an event-name string, got {events[key]!r}")
            required.append({"type": events[key]})
        if required:
            out["instrumentation"] = {"required_events": required}


_SPEC_HANDLERS: dict[str, Callable[[object, dict[str, Any], _Ctx], None]] = {
    "slos": _convert_slos,
    "contract": _convert_contract,
    "dependencies": _convert_dependencies,
    "ownership": _convert_ownership,
    "observability": _convert_observability,
    "instrumentation": _convert_instrumentation,
}
