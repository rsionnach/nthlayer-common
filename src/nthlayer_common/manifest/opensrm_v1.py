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
    resolve_service_type,
    valid_service_types_phrase,
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


def _not_yet(key: str) -> Callable[[object, dict[str, Any], _Ctx], None]:
    def handler(value: object, out: dict[str, Any], ctx: _Ctx) -> None:
        ctx.skip(f"spec.{key}", "not yet converted")
    return handler


# Tasks 4-6 replace each stub with the real handler.
_SPEC_HANDLERS = {
    key: _not_yet(key)
    for key in ("slos", "contract", "dependencies", "ownership", "observability", "instrumentation")
}
