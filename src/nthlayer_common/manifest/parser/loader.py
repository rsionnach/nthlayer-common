"""
Unified manifest loader with format auto-detection.

Supports srm/v1, opensrm.nthlayer.io/v2, and legacy NthLayer formats.
Auto-detects based on apiVersion field.

Usage:
    from nthlayer_common.manifest import load_manifest
    manifest = load_manifest("service.yaml")
"""

from __future__ import annotations

import contextlib
import shlex
import warnings
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml

from nthlayer_common.manifest.models import (
    SRM_V1_JUDGMENT_THRESHOLD_TYPES,
    Dependency,
    DependencyCriticality,
    DependencySLO,
    JudgmentThreshold,
    Observability,
    Ownership,
    PagerDutyConfig,
    ReliabilityManifest,
    SLODefinition,
    SourceFormat,
)
from nthlayer_common.manifest.opensrm_v1 import OPENSRM_V1_REJECTION, is_opensrm_v1_format
from nthlayer_common.manifest.parser import legacy
from nthlayer_common.manifest.parser._shapes import shape as _shape
from nthlayer_common.manifest.parser._shared import judgment_slo_fields, parse_observability
from nthlayer_common.manifest.parser.v1 import (
    OpenSRMParseError,
    is_srm_v1_format,
    parse_srm_v1,
    resolve_template,
)
from nthlayer_common.manifest.parser.v2 import (
    OpenSRMV2ParseError,
    is_opensrm_v2_format,
    parse_opensrm_v2,
    resolve_v2_template,
)
from nthlayer_common.manifest.target_validation import (
    check_finite,
    require_number,
    warn_target_convention_mismatches,
)
from nthlayer_common.manifest.thresholds import srm_v1_threshold

logger = structlog.get_logger()


class ManifestLoadError(Exception):
    """Error loading manifest file.

    ``backstop`` is True only on the error ``load_manifest`` raises when an
    error of a type outside its contract reached the end of a format's path
    unhandled (see ``LEGACY_BACKSTOP_MARKER`` and ``SHAPE_BACKSTOP_MARKER``).
    """

    backstop: bool = False


# In the message of a backstop ManifestLoadError. The legacy loader checks
# the shape of each position it reads and names it in the error; the
# backstop is for the shape those checks miss. It keeps load_manifest's
# contract, and it marks a missing check: the cause is chained, and
# tests/test_legacy_loader_shape_contract.py fails on any cell that gets
# this far (opensrm-wo62.4.10).
LEGACY_BACKSTOP_MARKER = "[legacy-shape-backstop]"

# The same for the srm/v1 and v2 paths (opensrm-wo62.4.28;
# tests/test_load_manifest_shape_contract.py).
SHAPE_BACKSTOP_MARKER = "[shape-backstop]"

# What a parser reading a wrong shape lets out: ``.get`` on a list, a loop
# over a number, ``[0]`` on a mapping, ``float()`` of an integer too large
# for one. One tuple for all three formats: the legacy branch caught only
# the first two until opensrm-8his.
_SHAPE_ERRORS = (AttributeError, TypeError, KeyError, IndexError, OverflowError)


def _shape_backstop(
    file_path: str | Path, reader: str, error: Exception, marker: str = SHAPE_BACKSTOP_MARKER
) -> ManifestLoadError:
    """The ManifestLoadError for a shape error no check named.

    *reader* is who met the value ("the srm/v1 parser"); *marker* is the
    format's backstop marker.
    """
    backstop = ManifestLoadError(
        f"{file_path} cannot be loaded: a value in it is not the type "
        f"{reader} expects ({type(error).__name__}: {error}) {marker}"
    )
    backstop.backstop = True
    return backstop


def _recursion_error(file_path: str | Path) -> ManifestLoadError:
    """The ManifestLoadError for a RecursionError on the srm/v1 or v2 path.

    Two causes, which the error cannot tell apart: a YAML alias to its own
    anchor (a walk over the document never ends), or a template or ``$ref``
    file nested too deeply for the YAML reader.
    """
    return ManifestLoadError(
        f"{file_path} cannot be loaded: a YAML alias in it refers to its own "
        "anchor, or a file it refers to is nested too deeply"
    )


class LegacyFormatWarning(UserWarning):
    """Warning for legacy format usage."""


def load_manifest(
    file_path: str | Path,
    environment: str | None = None,
    format: Literal["auto", "srm_v1", "opensrm_v2", "legacy"] | None = "auto",
    suppress_deprecation_warning: bool = False,
) -> ReliabilityManifest:
    """Load a reliability manifest from a YAML file.

    Auto-detects the format based on apiVersion:
      - srm/v1 → v1 parser
      - opensrm.nthlayer.io/v2 → v2 parser
      - service: section → legacy parser

    For the legacy format only, ``environment`` selects an overlay
    (``environments/{stem}-{env}.yaml``, ``environments/{env}.yaml``,
    ``.nthlayer/environments/{stem}-{env}.yaml``) deep-merged before parsing;
    ``service.template`` is then expanded (built-in templates, overridden by
    custom ones from ``.nthlayer/templates`` found walking up from the
    manifest, else a ``templates/`` dir beside it) and ``${var}`` substituted
    (parser/legacy.py, opensrm-wo62.4.1). srm/v1 and v2 only record it.

    Raises only ``OSError`` (``FileNotFoundError`` for a missing file),
    ``ManifestLoadError`` and ``ValueError``. That holds for every format,
    whatever the shape of the file: a position the loader reads as a
    mapping or a list and finds to be something else is a
    ``ManifestLoadError`` naming it, never a bare ``AttributeError`` or
    ``TypeError``; so is a number too large to be one, a YAML alias to its
    own anchor, and a file nested too deeply for the YAML reader
    (opensrm-wo62.4.10 for the legacy format, opensrm-wo62.4.28 for srm/v1
    and v2; tests/test_legacy_loader_shape_contract.py and
    tests/test_load_manifest_shape_contract.py cross every such position
    with every wrong type).
    A wrong type that no check names still ends as a
    ``ManifestLoadError``, with ``backstop`` True and the original error as
    its cause.

    Legacy SLOs also carry a latency ``threshold_ms`` (from
    ``indicator.threshold_ms``, else the plural ``indicators[].threshold_ms``
    (opensrm-wo62.4.24), else ``threshold_ms``, else ``latency_threshold``), and legacy caches and queues carry their declared
    ``type`` in ``Dependency.database_type`` (opensrm-wo62.4.7).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {file_path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ManifestLoadError(f"Invalid YAML in {file_path}: {e}") from e
    except RecursionError as e:
        # PyYAML builds nested nodes recursively: a few hundred levels of
        # nesting run it out of stack before anything here sees the data.
        raise ManifestLoadError(f"{file_path} is nested too deeply to load") from e

    if not isinstance(data, dict):
        raise ManifestLoadError(f"Expected YAML object in {file_path}")

    # The published OpenSRM v1 schema is migrate-only: NthLayer's v1 dialect is
    # srm/v1. Rejected before format dispatch so no explicit format= can push it
    # into a parser that misreads it (opensrm-pnmv). See
    # docs/superpowers/decisions/opensrm-v1-support.md in the nthlayer front door.
    if is_opensrm_v1_format(data):
        raise ManifestLoadError(OPENSRM_V1_REJECTION.format(path=shlex.quote(str(file_path))))

    # Detect format
    detected_format = _detect_format(data)

    if format == "auto":
        use_format = detected_format
    elif format == "srm_v1":
        use_format = SourceFormat.SRM_V1
    elif format == "opensrm_v2":
        use_format = SourceFormat.OPENSRM_V2
    elif format == "legacy":
        use_format = SourceFormat.LEGACY
    else:
        use_format = detected_format

    if use_format == SourceFormat.SRM_V1:
        try:
            template_dir = _find_template_dir(path)
            data, template_warnings = resolve_template(data, template_dir)
            for tw in template_warnings:
                logger.warning(tw)
            manifest = parse_srm_v1(data, source_file=str(path))
        except (OpenSRMParseError, ValueError) as e:
            raise ManifestLoadError(str(e)) from e
        except RecursionError as e:
            raise _recursion_error(file_path) from e
        except _SHAPE_ERRORS as e:
            raise _shape_backstop(file_path, "the srm/v1 parser", e) from e

    elif use_format == SourceFormat.OPENSRM_V2:
        try:
            data = resolve_v2_template(data, path.parent)
            manifest = parse_opensrm_v2(data, source_file=str(path), base_dir=path.parent)
        except (OpenSRMV2ParseError, ValueError) as e:
            raise ManifestLoadError(str(e)) from e
        except RecursionError as e:
            raise _recursion_error(file_path) from e
        except _SHAPE_ERRORS as e:
            raise _shape_backstop(file_path, "the v2 parser", e) from e

    else:
        if not suppress_deprecation_warning:
            warnings.warn(
                f"Legacy NthLayer format detected in {file_path}. "
                f"Consider migrating to OpenSRM format.",
                LegacyFormatWarning,
                stacklevel=2,
            )
        # generate's legacy order (opensrm-wo62.4.1, parser/legacy.py):
        # overlay, environment pass, template, then the per-spec pass.
        try:
            try:
                data = legacy.apply_environment_overlay(data, path, environment)
                data = legacy.substitute_environment(data, environment)
                data = legacy.expand_template(data, path, _find_template_dir(path))
                data = legacy.substitute_resource_specs(data, environment)
            except ValueError as e:
                raise ManifestLoadError(str(e)) from e
            # Outside the ValueError wrap above, as it always was: a
            # ValueError from the parse or the model reaches the caller as one.
            manifest = _parse_legacy_to_manifest(data, str(path), environment)
        except RecursionError as e:
            # These passes walk the whole document, and a YAML alias to its
            # own anchor (``spec: &a {indicator: *a}``) never ends. An
            # overlay or template nested too deeply for PyYAML is read in
            # here too; the manifest itself was read above.
            raise ManifestLoadError(
                f"{file_path} cannot be loaded: a YAML alias in it refers to its own "
                "anchor, or its overlay or template is nested too deeply"
            ) from e
        except _SHAPE_ERRORS as e:
            raise _shape_backstop(
                file_path, "the legacy loader", e, LEGACY_BACKSTOP_MARKER
            ) from e

    # Cross-subsystem target-convention drift check (opensrm-pa2w). Emits
    # warnings via the warnings module; never rejects. See module docstring
    # in nthlayer_common.manifest.target_validation for the heuristic.
    warn_target_convention_mismatches(manifest)
    return manifest


def _detect_format(data: dict[str, Any]) -> SourceFormat:
    """Detect the format of manifest data."""
    if is_opensrm_v2_format(data):
        return SourceFormat.OPENSRM_V2
    if is_srm_v1_format(data):
        return SourceFormat.SRM_V1
    if "service" in data:
        return SourceFormat.LEGACY
    return SourceFormat.LEGACY


def _find_template_dir(manifest_path: Path) -> Path | None:
    """Find the template directory for a manifest file."""
    current = manifest_path.resolve().parent
    for _ in range(10):
        candidate = current / ".nthlayer" / "templates"
        if candidate.is_dir():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent

    templates_dir = manifest_path.resolve().parent / "templates"
    if templates_dir.is_dir():
        return templates_dir

    return None


def is_manifest_file(file_path: str | Path) -> bool:
    """Check if a file appears to be a reliability manifest."""
    path = Path(file_path)

    if path.suffix not in (".yaml", ".yml"):
        return False

    name = path.stem.lower()
    if name.endswith(".reliability"):
        return True
    if name == "service":
        return True

    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return False

        if is_opensrm_v2_format(data):
            return True
        if is_srm_v1_format(data):
            return True
        # Counts as a manifest so a scan reaches load_manifest and its
        # migrate-manifest rejection instead of skipping the file silently.
        if is_opensrm_v1_format(data):
            return True

        if "service" in data and isinstance(data["service"], dict):
            service = data["service"]
            return all(k in service for k in ("name", "team", "tier", "type"))

    except Exception:
        return False

    return False


# =============================================================================
# Legacy Format Parser
# =============================================================================


# What YAML can put where a list goes that iterates to nothing when empty.
_EMPTY_AS_NO_ENTRIES = (str, bytes, dict, set, frozenset)


def _legacy_mapping(value: Any, what: str) -> dict[str, Any]:
    """*value* if it is a mapping, else ManifestLoadError naming *what*.

    The legacy loader reads these positions with ``.get``; anything else
    used to end in a bare AttributeError (opensrm-wo62.4.10).
    """
    if not isinstance(value, dict):
        raise ManifestLoadError(f"{what} must be a mapping, not {_shape(value)}")
    return value


def _legacy_entries(value: Any, what: str) -> list[dict[str, Any]]:
    """*value* as a list of mappings, else ManifestLoadError naming *what*.

    An empty string, mapping, set or binary counts as no entries: the
    loader used to iterate these positions, so each loaded as nothing, and
    a file that loads must keep loading. Every other non-list, and any
    entry that is not a mapping, used to end in a bare AttributeError or
    TypeError.
    """
    if isinstance(value, _EMPTY_AS_NO_ENTRIES) and not value:
        return []
    if not isinstance(value, list):
        raise ManifestLoadError(f"{what} must be a list, not {_shape(value)}")
    for i, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ManifestLoadError(f"{what}[{i}] must be a mapping, not {_shape(entry)}")
    return value


def _legacy_resource_spec(resource: dict[str, Any]) -> dict[str, Any]:
    """The resource's ``spec`` (``{}`` when the key is absent), which the
    kinds that read into it need as a mapping."""
    return _legacy_mapping(
        resource.get("spec", {}),
        f"Legacy {resource.get('kind')} resource {resource.get('name')!r}: spec",
    )


def _legacy_success_ratio(spec: dict[str, Any]) -> tuple[str | None, str | None]:
    """``(good_query, total_query)`` from the first plural ``indicators[]``
    entry whose ``success_ratio`` has both as non-blank strings, else
    ``(None, None)`` (opensrm-zo0o).

    The SLO's own good/total, which the loader used to drop: only each
    entry's ``type`` was read. A partial or malformed ``success_ratio`` is
    skipped rather than half-kept: one query alone is no ratio. The
    singular ``indicator`` carries one ``query`` instead, kept as
    ``indicator_query``.
    """
    indicators = spec.get("indicators")
    if not isinstance(indicators, list):
        return None, None
    for entry in indicators:
        ratio = entry.get("success_ratio") if isinstance(entry, dict) else None
        if not isinstance(ratio, dict):
            continue
        good, total = ratio.get("good_query"), ratio.get("total_query")
        if _is_query(good) and _is_query(total):
            return good, total
    return None, None


def _is_query(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _legacy_threshold_ms(name: str, spec: dict[str, Any]) -> float | None:
    """The first declared latency threshold in *spec*, in milliseconds, or None.

    Most specific first: the singular ``indicator.threshold_ms``, the plural
    ``indicators[].threshold_ms`` (opensrm-wo62.4.24), then the SLO's own
    ``threshold_ms`` and ``latency_threshold``; the same order as the SLO's
    type, where the singular indicator outranks the plural list. Only the
    first one declared is validated; a lower-priority key beside it is never
    read. Raises ManifestLoadError, naming the SLO, if it is not a finite
    positive number.
    """
    singular, *own = legacy.LATENCY_THRESHOLD_KEYS
    node = legacy.lookup_path(spec, singular)
    if node is not None:
        return _threshold_number(name, node, ".".join(singular))
    listed = _legacy_indicators_threshold_ms(name, spec)
    if listed is not None:
        return listed
    for path in own:
        node = legacy.lookup_path(spec, path)
        if node is not None:
            return _threshold_number(name, node, ".".join(path))
    return None


def _threshold_number(name: str, node: Any, key_path: str) -> float:
    """*node* as a latency threshold in milliseconds, else ManifestLoadError."""
    try:
        threshold = require_number(name, node, what=f"Latency {key_path}")
        check_finite(name, threshold, what=f"Latency {key_path}")
    except ValueError as e:
        # Not a number, not finite, or an integer too large for a float
        # (YAML integers are unbounded).
        raise ManifestLoadError(str(e)) from e
    if threshold <= 0:
        raise ManifestLoadError(f"Latency {key_path} for '{name}' must be positive, got {node!r}")
    return threshold


def _legacy_indicators_threshold_ms(name: str, spec: dict[str, Any]) -> float | None:
    """The one latency threshold a plural ``indicators:`` list declares, or None.

    The rule is the list's rule for its type (``_legacy_indicators_type``):
    entries with no ``threshold_ms`` are ignored, every declared one is
    validated, and entries that declare different numbers leave the SLO
    with no single bound, so the load fails rather than pick one
    (opensrm-wo62.4.24). ``800`` and ``"800.0"`` are the same number. Real
    legacy manifests declare the bound only here, and reading only the
    singular and top-level keys loaded them with no threshold, so
    nthlayer-generate gave their latency SLOs no error budget.

    Unlike the type, the list's shape is not judged here: a list that is
    not a list, or an entry that is not a mapping, declares no threshold
    (as it declares no ``success_ratio``). ``_legacy_indicators_type``
    refuses that shape when the list is what types the SLO; beside a
    singular ``indicator.type`` such a file has always loaded, and keeps
    loading.
    """
    declared = legacy.indicators_thresholds(spec.get("indicators"))
    thresholds = [
        _threshold_number(name, node, "indicators[].threshold_ms") for node in declared
    ]
    if len(set(thresholds)) > 1:
        raise ManifestLoadError(
            f"SLO '{name}' declares indicators with different threshold_ms values "
            f"{sorted(set(thresholds))}; a legacy SLO's indicators must share one threshold"
        )
    return thresholds[0] if thresholds else None


def _legacy_indicators_type(name: str, spec: dict[str, Any]) -> str | None:
    """The one type a plural ``indicators:`` list declares, or None.

    Untyped or blank entries are ignored. Entries that declare different
    types leave the SLO with no single slo_type, so the load fails rather
    than guess (opensrm-e9ts, R1); so does a list that is not a list of
    mappings, which would otherwise fall back to availability unseen.
    Types compare as written, so ``Custom`` and ``custom`` disagree.
    """
    indicators = spec.get("indicators")
    if indicators is None:
        return None
    if not isinstance(indicators, list) or not all(isinstance(e, dict) for e in indicators):
        raise ManifestLoadError(
            f"SLO '{name}' has indicators: {indicators!r}; indicators is a list of mappings"
        )
    declared = [e["type"] for e in indicators if e.get("type")]
    for t in declared:
        if not isinstance(t, str):
            raise ManifestLoadError(
                f"SLO '{name}' has an indicator type {t!r}; an indicator type is a string"
            )
    types = {t for t in declared if t.strip()}
    if len(types) > 1:
        raise ManifestLoadError(
            f"SLO '{name}' declares indicators of different types {sorted(types)}; "
            "a legacy SLO's indicators must share one type"
        )
    return types.pop() if types else None


def _infer_legacy_slo_type(name: str, spec: dict[str, Any], indicator: dict[str, Any]) -> str:
    """Infer slo_type for a legacy SLO resource.

    In order: a singular ``indicator.type``; the shared type of a plural
    ``indicators:`` list; name keywords; a declared latency threshold
    (opensrm-e9ts, R2); and availability only when nothing above says
    otherwise. Real legacy manifests declare the plural list, and reading only
    the singular key typed lag and freshness SLOs as availability, so
    nthlayer-generate wrote them an error budget from request success.
    """
    explicit = indicator.get("type")
    if explicit:
        return explicit
    listed = _legacy_indicators_type(name, spec)
    if listed:
        return listed

    # Only here is the name read as text. An SLO whose type is declared
    # loads whatever its name is, as it always has.
    if not isinstance(name, str):
        raise ManifestLoadError(
            f"Legacy SLO resource {name!r} declares no indicator type, so its type is "
            f"inferred from its name, which must be a string, not {_shape(name)}"
        )
    name_lower = name.lower()
    if "avail" in name_lower:
        return "availability"
    if "latency" in name_lower or "duration" in name_lower:
        return "latency"
    if "error" in name_lower:
        return "error_rate"
    if "throughput" in name_lower:
        return "throughput"
    # After the name keywords, so `errors-p99` with a threshold stays
    # error_rate: an explicit word in the name outranks an inferred shape.
    if _legacy_threshold_ms(name, spec) is not None:
        return "latency"

    # Nothing declared a type. Availability is the oldest legacy default, and
    # an SLO like `success-rate` with no indicator at all is an availability SLO.
    return "availability"


def _parse_legacy_to_manifest(
    data: dict[str, Any],
    source_file: str,
    environment: str | None = None,
) -> ReliabilityManifest:
    """Parse legacy NthLayer format into a ReliabilityManifest."""
    service_data = _legacy_mapping(data.get("service", {}), "service")
    resources_data = _legacy_entries(data.get("resources", []), "resources")

    name = service_data.get("name")
    if not name:
        raise ManifestLoadError("service.name is required")
    team = service_data.get("team")
    if not team:
        raise ManifestLoadError("service.team is required")
    tier = service_data.get("tier")
    if not tier:
        raise ManifestLoadError("service.tier is required")
    if isinstance(tier, list | dict):
        # The model looks the tier up in a set: an unhashable one was a
        # bare TypeError there.
        raise ManifestLoadError(f"service.tier must be a string, not {_shape(tier)}")
    service_type = service_data.get("type")
    if not service_type:
        raise ManifestLoadError("service.type is required")
    metadata = _legacy_mapping(service_data.get("metadata", {}), "service.metadata")

    slos, judgment_thresholds = _extract_slos_from_resources(resources_data)
    dependencies = _extract_dependencies_from_resources(resources_data)
    ownership = _extract_ownership(service_data, resources_data, source_file)
    alerting = _extract_alerting_from_resources(resources_data)
    observability = _extract_observability(resources_data, name, source_file)

    return ReliabilityManifest(
        name=name,
        team=team,
        tier=tier,
        type=service_type,
        description=service_data.get("description"),
        labels=metadata.get("labels", {}),
        annotations=metadata.get("annotations", {}),
        slos=slos,
        judgment_thresholds=judgment_thresholds,
        dependencies=dependencies,
        ownership=ownership,
        alerting=alerting,
        observability=observability,
        language=service_data.get("language"),
        framework=service_data.get("framework"),
        template=service_data.get("template"),
        support_model=service_data.get("support_model", "self"),
        environment=environment,
        source_format=SourceFormat.LEGACY,
        source_file=source_file,
        raw_data=data,
    )


def _legacy_declared_target(spec: dict[str, Any]) -> Any:
    """``objective``, else ``target``; None when neither is declared.

    Null is undeclared, and an SLO that declares neither is dropped
    (opensrm-wo62.4.17). Any other value is the SLO's target, 0 included:
    this used to be ``objective or target``, so a falsy objective (0, an
    empty string, list or mapping, false) fell through to ``target`` and,
    with none there, the SLO vanished from the manifest unreported
    (opensrm-1lhl).
    """
    for key in ("objective", "target"):
        value = spec.get(key)
        if value is not None:
            return value
    return None


def _legacy_target_number(name: Any, target: Any) -> float:
    """*target* as a finite number, else ManifestLoadError naming the SLO.

    The fifth writer of ``SLODefinition.target``, guarded like the other
    four (opensrm-ocvu, opensrm-1lhl): a number or a string spelling one,
    never a boolean, never NaN or infinite. A NaN target makes every
    comparison in ``validate_contracts()`` false, so a breached contract
    reported clean. Range is not checked: a legacy target outside 0-100
    has always loaded.
    """
    try:
        number = require_number(name, target, what="Legacy SLO objective")
        check_finite(name, number, what="Legacy SLO objective")
    except ValueError as exc:
        # Not a number, not finite, or an integer too large for a float
        # (YAML integers are unbounded).
        raise ManifestLoadError(str(exc)) from exc
    return number


def _extract_slos_from_resources(
    resources: list[dict[str, Any]],
) -> tuple[list[SLODefinition], list[JudgmentThreshold]]:
    """Extract SLO definitions and judgment thresholds from legacy resources.

    ``kind: SLO`` resources named calibration or feedback_latency are judgment
    thresholds, lifted exactly as srm/v1 lifts its ``spec.slos`` entries of
    the same names (see _legacy_threshold).
    """
    slos: list[SLODefinition] = []
    thresholds: list[JudgmentThreshold] = []

    for resource in resources:
        if resource.get("kind") != "SLO":
            continue

        name = resource.get("name", "")
        spec = _legacy_resource_spec(resource)
        if isinstance(name, list | dict):
            # The name keys a set lookup just below.
            raise ManifestLoadError(
                f"Legacy SLO resource name must be a string, not {_shape(name)}"
            )

        # Only calibration and feedback_latency are lifted [opensrm-wo62.1].
        # srm/v1 rejects segments, stability and audit_backlog, but doing so
        # here would break legacy manifests that load today, so they stay
        # classical SLOs.
        if name in SRM_V1_JUDGMENT_THRESHOLD_TYPES:
            thresholds.append(_legacy_threshold(name, spec))
            continue

        target = _legacy_declared_target(spec)
        if target is None:
            continue

        indicator = _legacy_mapping(
            spec.get("indicator", {}), f"Legacy SLO resource {name!r}: spec.indicator"
        )
        slo_type = _infer_legacy_slo_type(name, spec, indicator)
        window = spec.get("window", "30d")
        # Same name-keyed detection as srm/v1 [opensrm-wo62.1]: without it a
        # legacy ai-gate manifest's judgment SLOs load as classical, and
        # nthlayer-generate silently drops their judgment recording rules.
        judgment_type, measurement, statistical_requirements = judgment_slo_fields(name, window)
        good_query, total_query = _legacy_success_ratio(spec)

        target_value = _legacy_target_number(name, target)

        slo = SLODefinition(
            name=name,
            target=target_value,
            slo_type=slo_type,
            window=window,
            unit=spec.get("unit"),
            percentile=spec.get("percentile"),
            threshold_ms=_legacy_threshold_ms(name, spec),
            indicator_query=indicator.get("query"),
            good_query=good_query,
            total_query=total_query,
            description=spec.get("description"),
            judgment_type=judgment_type,
            measurement=measurement,
            statistical_requirements=statistical_requirements,
        )
        slos.append(slo)

    return slos, thresholds


def _legacy_threshold(name: str, spec: dict[str, Any]) -> JudgmentThreshold:
    """Lift a legacy calibration/feedback_latency SLO resource [opensrm-wo62.1].

    Exactly one of ``objective`` and ``target`` must be set (``None`` counts
    as unset; 0 is a value, as srm/v1 lifts target 0). It becomes srm/v1's
    ``target``, and every other spec key goes to
    thresholds.srm_v1_threshold unchanged, so legacy accepts exactly what
    srm/v1 accepts (window, unit, description, labels; no unit on calibration).

    The one exception is ``indicator``. Legacy SLO resources normally carry
    one and judgment thresholds do not use it, so rejecting it would fail most
    legacy calibration/feedback_latency resources in a patch release. It is
    dropped with a LegacyFormatWarning naming the resource.

    Raises ManifestLoadError when the objective is missing or given twice,
    and for any other key the srm/v1 lift rejects.
    """
    objective, target = spec.get("objective"), spec.get("target")
    if objective is not None and target is not None:
        raise ManifestLoadError(
            f"Legacy SLO resource '{name}' sets both objective and target; "
            f"set exactly one."
        )
    if objective is None and target is None:
        raise ManifestLoadError(
            f"Legacy SLO resource '{name}' requires an objective (or target)."
        )
    if "indicator" in spec:
        warnings.warn(
            f"Legacy SLO resource '{name}' is a judgment threshold; its "
            f"spec.indicator is not used by judgment thresholds and is ignored.",
            LegacyFormatWarning,
            # _legacy_threshold -> _extract_slos_from_resources ->
            # _parse_legacy_to_manifest -> load_manifest -> caller.
            stacklevel=5,
        )
    config = {
        k: v for k, v in spec.items() if k not in ("objective", "target", "indicator")
    }
    config["target"] = objective if objective is not None else target
    try:
        return srm_v1_threshold(name, config)
    except ValueError as exc:
        raise ManifestLoadError(f"Legacy SLO resource '{name}': {exc}") from exc


def _legacy_dependency_slo(entry: dict[str, Any], what: str) -> DependencySLO | None:
    """The ``slo`` a database or upstream entry declares, or None."""
    if "slo" not in entry:
        return None
    slo = _legacy_mapping(entry["slo"], f"{what}.slo")
    return DependencySLO(
        availability=slo.get("availability"),
        latency_p99=slo.get("latency_p99"),
    )


def _extract_dependencies_from_resources(
    resources: list[dict[str, Any]],
) -> list[Dependency]:
    """Extract dependencies from legacy resource list."""
    dependencies = []

    for resource in resources:
        if resource.get("kind") != "Dependencies":
            continue

        spec = _legacy_resource_spec(resource)
        where = f"Legacy Dependencies resource {resource.get('name')!r}: spec"
        listed = {
            key: _legacy_entries(spec.get(key, []), f"{where}.{key}")
            for key in ("databases", "upstream", "downstream", "caches", "queues", "services")
        }

        for i, db in enumerate(listed["databases"]):
            dep_slo = _legacy_dependency_slo(db, f"{where}.databases[{i}]")

            criticality = None
            if "criticality" in db:
                with contextlib.suppress(ValueError):
                    criticality = DependencyCriticality(db["criticality"])

            dependencies.append(Dependency(
                name=db.get("name", ""),
                type="database",
                critical=db.get("criticality") in ("critical", "high"),
                criticality=criticality,
                database_type=db.get("type"),
                slo=dep_slo,
            ))

        for i, upstream in enumerate(listed["upstream"]):
            dep_slo = _legacy_dependency_slo(upstream, f"{where}.upstream[{i}]")

            criticality = None
            if "criticality" in upstream:
                with contextlib.suppress(ValueError):
                    criticality = DependencyCriticality(upstream["criticality"])

            dependencies.append(Dependency(
                name=upstream.get("name", ""),
                type="api",
                critical=upstream.get("criticality") in ("critical", "high"),
                criticality=criticality,
                slo=dep_slo,
            ))

        for downstream in listed["downstream"]:
            dependencies.append(Dependency(
                name=downstream.get("name", ""),
                type="api",
                critical=False,
            ))

        # The declared technology rides in database_type, as for databases
        # [opensrm-wo62.4.7, R2]; generate's legacy dashboard read it from here.
        for cache in listed["caches"]:
            dependencies.append(Dependency(
                name=cache.get("name", ""),
                type="cache",
                critical=cache.get("criticality") in ("critical", "high"),
                database_type=cache.get("type"),
            ))

        for queue in listed["queues"]:
            dependencies.append(Dependency(
                name=queue.get("name", ""),
                type="queue",
                critical=queue.get("criticality") in ("critical", "high"),
                database_type=queue.get("type"),
            ))

        # Also handle flat services list (newer legacy format)
        for svc in listed["services"]:
            dependencies.append(Dependency(
                name=svc.get("name", ""),
                type=svc.get("type", "api"),
                critical=svc.get("criticality") in ("critical", "high"),
            ))

    return dependencies


def _extract_alerting_from_resources(resources: list[dict[str, Any]]) -> Any:
    """Extract raw alerting config from legacy resources."""
    for resource in resources:
        if resource.get("kind") != "Alerts":
            continue
        return resource.get("spec")
    return None


def _extract_observability(
    resources: list[dict[str, Any]], service: Any, source_file: str
) -> Observability | None:
    """Map the first legacy ``kind: Observability`` resource.

    generate's legacy detector plans a dashboard for any Observability
    resource, whatever its spec, so an empty or missing spec still yields
    ``Observability()``: presence is the signal. The legacy spec keys
    (metrics, logs, health) have no Observability field and are not read.
    A non-mapping spec is likewise only presence: it is warned about and
    ignored rather than failing a load that succeeded before this resource
    was read. opensrm-wo62.4.10 kept that: it turned the shapes that
    crashed into ManifestLoadError and left the ones that loaded alone.
    """
    for resource in resources:
        if resource.get("kind") == "Observability":
            spec = resource.get("spec")
            if spec is not None and not isinstance(spec, dict):
                # Names the manifest: a scan loads many, and the warning
                # is no use without knowing which one (opensrm-wo62.4.14).
                logger.warning(
                    "legacy_observability_spec_not_mapping",
                    spec_type=type(spec).__name__,
                    service=service,
                    source_file=source_file,
                )
                spec = None
            return parse_observability(spec) or Observability()
    return None


def _legacy_service_pagerduty(service_data: dict[str, Any], source_file: str) -> dict[str, Any]:
    """Return ``service.pagerduty`` as a mapping (``{}`` when unset).

    It was never read before opensrm-wo62.4.13, so a non-mapping value is
    warned about and ignored rather than failing a load that used to succeed.
    """
    value = service_data.get("pagerduty")
    if value is None:
        return {}
    if not isinstance(value, dict):
        logger.warning(
            "legacy_service_pagerduty_not_mapping",
            value_type=type(value).__name__,
            service=service_data.get("name"),
            source_file=source_file,
        )
        return {}
    return value


def _extract_ownership(
    service_data: dict[str, Any],
    resources: list[dict[str, Any]],
    source_file: str,
) -> Ownership | None:
    """Extract ownership info from legacy service data and resources."""
    team = service_data.get("team")
    if not team:
        return None

    ownership = Ownership(
        team=team,
        slack=service_data.get("slack_channel"),
        email=service_data.get("email"),
        runbook=service_data.get("runbook_url"),
    )

    # The config is built only for a PagerDuty resource (generate plans
    # pagerduty only then); service.pagerduty keys alone add nothing.
    for resource in resources:
        if resource.get("kind") == "PagerDuty":
            spec = _legacy_resource_spec(resource)
            service_pagerduty = _legacy_service_pagerduty(service_data, source_file)
            ownership.pagerduty = PagerDutyConfig(
                service_id=spec.get("service_id"),
                escalation_policy_id=spec.get("escalation_policy"),
                urgency=spec.get("urgency"),
                auto_resolve_timeout=spec.get("auto_resolve_timeout"),
                create_escalation_policy=spec.get("create_escalation_policy"),
                timezone=service_pagerduty.get("timezone"),
                sre_escalation_policy_id=service_pagerduty.get("sre_escalation_policy_id"),
                integration_key=spec.get("integration_key"),
                sre_integration_key=spec.get("sre_integration_key"),
            )
            ownership.escalation = spec.get("escalation_policy")
            break

    return ownership
