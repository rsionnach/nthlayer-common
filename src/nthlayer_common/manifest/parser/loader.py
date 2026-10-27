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
    Ownership,
    PagerDutyConfig,
    ReliabilityManifest,
    SLODefinition,
    SourceFormat,
)
from nthlayer_common.manifest.opensrm_v1 import OPENSRM_V1_REJECTION, is_opensrm_v1_format
from nthlayer_common.manifest.parser import legacy
from nthlayer_common.manifest.parser._shared import judgment_slo_fields
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
    warn_target_convention_mismatches,
)
from nthlayer_common.manifest.thresholds import srm_v1_threshold

logger = structlog.get_logger()


class ManifestLoadError(Exception):
    """Error loading manifest file."""


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
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {file_path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ManifestLoadError(f"Invalid YAML in {file_path}: {e}") from e

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

    elif use_format == SourceFormat.OPENSRM_V2:
        try:
            data = resolve_v2_template(data, path.parent)
            manifest = parse_opensrm_v2(data, source_file=str(path), base_dir=path.parent)
        except (OpenSRMV2ParseError, ValueError) as e:
            raise ManifestLoadError(str(e)) from e

    else:
        if not suppress_deprecation_warning:
            warnings.warn(
                f"Legacy NthLayer format detected in {file_path}. "
                f"Consider migrating to OpenSRM format.",
                LegacyFormatWarning,
                stacklevel=2,
            )
        # Overlay, then template, then ${var} — generate's legacy order
        # (opensrm-wo62.4.1). See parser/legacy.py.
        try:
            data = legacy.apply_environment_overlay(data, path, environment)
            data = legacy.expand_template(data, path, _find_template_dir(path))
            data = legacy.substitute_variables(data, environment)
        except ValueError as e:
            raise ManifestLoadError(str(e)) from e
        manifest = _parse_legacy_to_manifest(data, str(path), environment)

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


def _infer_legacy_slo_type(name: str, spec: dict[str, Any]) -> str:
    """Infer slo_type for a legacy SLO resource."""
    indicator = spec.get("indicator", {})
    explicit = indicator.get("type")
    if explicit:
        return explicit

    name_lower = name.lower()
    if "avail" in name_lower:
        return "availability"
    if "latency" in name_lower or "duration" in name_lower:
        return "latency"
    if "error" in name_lower:
        return "error_rate"
    if "throughput" in name_lower:
        return "throughput"

    # Default to availability for legacy — better than failing
    return "availability"


def _parse_legacy_to_manifest(
    data: dict[str, Any],
    source_file: str,
    environment: str | None = None,
) -> ReliabilityManifest:
    """Parse legacy NthLayer format into a ReliabilityManifest."""
    service_data = data.get("service", {})
    resources_data = data.get("resources", [])

    name = service_data.get("name")
    if not name:
        raise ManifestLoadError("service.name is required")
    team = service_data.get("team")
    if not team:
        raise ManifestLoadError("service.team is required")
    tier = service_data.get("tier")
    if not tier:
        raise ManifestLoadError("service.tier is required")
    service_type = service_data.get("type")
    if not service_type:
        raise ManifestLoadError("service.type is required")

    slos, judgment_thresholds = _extract_slos_from_resources(resources_data)
    dependencies = _extract_dependencies_from_resources(resources_data)
    ownership = _extract_ownership(service_data, resources_data)
    alerting = _extract_alerting_from_resources(resources_data)

    return ReliabilityManifest(
        name=name,
        team=team,
        tier=tier,
        type=service_type,
        description=service_data.get("description"),
        labels=service_data.get("metadata", {}).get("labels", {}),
        annotations=service_data.get("metadata", {}).get("annotations", {}),
        slos=slos,
        judgment_thresholds=judgment_thresholds,
        dependencies=dependencies,
        ownership=ownership,
        alerting=alerting,
        language=service_data.get("language"),
        framework=service_data.get("framework"),
        template=service_data.get("template"),
        support_model=service_data.get("support_model", "self"),
        environment=environment,
        source_format=SourceFormat.LEGACY,
        source_file=source_file,
        raw_data=data,
    )


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
        spec = resource.get("spec", {})

        # Only calibration and feedback_latency are lifted [opensrm-wo62.1].
        # srm/v1 rejects segments, stability and audit_backlog, but doing so
        # here would break legacy manifests that load today, so they stay
        # classical SLOs.
        if name in SRM_V1_JUDGMENT_THRESHOLD_TYPES:
            thresholds.append(_legacy_threshold(name, spec))
            continue

        target = spec.get("objective") or spec.get("target")
        if target is None:
            continue

        slo_type = _infer_legacy_slo_type(name, spec)
        window = spec.get("window", "30d")
        # Same name-keyed detection as srm/v1 [opensrm-wo62.1]: without it a
        # legacy ai-gate manifest's judgment SLOs load as classical, and
        # nthlayer-generate silently drops their judgment recording rules.
        judgment_type, measurement, statistical_requirements = judgment_slo_fields(name, window)

        slo = SLODefinition(
            name=name,
            target=float(target),
            slo_type=slo_type,
            window=window,
            unit=spec.get("unit"),
            percentile=spec.get("percentile"),
            indicator_query=spec.get("indicator", {}).get("query"),
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


def _extract_dependencies_from_resources(
    resources: list[dict[str, Any]],
) -> list[Dependency]:
    """Extract dependencies from legacy resource list."""
    dependencies = []

    for resource in resources:
        if resource.get("kind") != "Dependencies":
            continue

        spec = resource.get("spec", {})

        for db in spec.get("databases", []):
            dep_slo = None
            if "slo" in db:
                dep_slo = DependencySLO(
                    availability=db["slo"].get("availability"),
                    latency_p99=db["slo"].get("latency_p99"),
                )

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

        for upstream in spec.get("upstream", []):
            dep_slo = None
            if "slo" in upstream:
                dep_slo = DependencySLO(
                    availability=upstream["slo"].get("availability"),
                    latency_p99=upstream["slo"].get("latency_p99"),
                )

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

        for downstream in spec.get("downstream", []):
            dependencies.append(Dependency(
                name=downstream.get("name", ""),
                type="api",
                critical=False,
            ))

        for cache in spec.get("caches", []):
            dependencies.append(Dependency(
                name=cache.get("name", ""),
                type="cache",
                critical=cache.get("criticality") in ("critical", "high"),
            ))

        for queue in spec.get("queues", []):
            dependencies.append(Dependency(
                name=queue.get("name", ""),
                type="queue",
                critical=queue.get("criticality") in ("critical", "high"),
            ))

        # Also handle flat services list (newer legacy format)
        for svc in spec.get("services", []):
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


def _extract_ownership(
    service_data: dict[str, Any],
    resources: list[dict[str, Any]],
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

    for resource in resources:
        if resource.get("kind") == "PagerDuty":
            spec = resource.get("spec", {})
            ownership.pagerduty = PagerDutyConfig(
                service_id=spec.get("service_id"),
                escalation_policy_id=spec.get("escalation_policy"),
                urgency=spec.get("urgency"),
                auto_resolve_timeout=spec.get("auto_resolve_timeout"),
                create_escalation_policy=spec.get("create_escalation_policy"),
            )
            ownership.escalation = spec.get("escalation_policy")
            break

    return ownership
