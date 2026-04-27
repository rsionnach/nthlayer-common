"""
OpenSLO v1 document parser.

Translates OpenSLO v1 `kind: SLO` documents (inline or $ref) into
SLODefinition instances. $ref documents are resolved at parse time;
the original ref path is stored in source_ref for provenance.

OpenSLO spec: https://github.com/OpenSLO/OpenSLO
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nthlayer_common.manifest.models import SLODefinition


class OpenSLOParseError(Exception):
    """Error parsing an OpenSLO document."""


def parse_openslo_slos(
    slo_list: list[Any],
    base_dir: Path | None = None,
) -> list[SLODefinition]:
    """Parse a list of OpenSLO v1 SLO entries (inline or $ref).

    Args:
        slo_list: list from v2 manifest's spec.slo block
        base_dir: directory for resolving $ref paths

    Returns:
        list of SLODefinition instances
    """
    results = []

    for entry in slo_list:
        if not isinstance(entry, dict):
            raise OpenSLOParseError(f"Expected dict in slo list, got {type(entry).__name__}")

        # $ref resolution
        if "$ref" in entry:
            ref_path = entry["$ref"]
            if base_dir is None:
                raise OpenSLOParseError(
                    f"Cannot resolve $ref '{ref_path}' — no base directory"
                )

            resolved_path = (base_dir / ref_path).resolve()
            if not resolved_path.is_relative_to(base_dir.resolve()):
                raise OpenSLOParseError(
                    f"$ref '{ref_path}' resolves outside manifest directory"
                )
            if not resolved_path.exists():
                raise OpenSLOParseError(
                    f"$ref '{ref_path}' not found at {resolved_path}"
                )

            try:
                with open(resolved_path) as f:
                    entry = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise OpenSLOParseError(
                    f"Invalid YAML in $ref '{ref_path}': {e}"
                ) from e

            if not isinstance(entry, dict):
                raise OpenSLOParseError(
                    f"$ref '{ref_path}' is not a YAML object"
                )

            slo_def = _parse_openslo_document(entry)
            slo_def.source_ref = ref_path
            results.append(slo_def)
        else:
            # Inline OpenSLO document
            results.append(_parse_openslo_document(entry))

    return results


def _parse_openslo_document(data: dict[str, Any]) -> SLODefinition:
    """Parse a single OpenSLO v1 kind: SLO document into SLODefinition.

    Validates apiVersion (openslo/v1) and kind (SLO), then extracts
    name, objectives, and indicator into the internal SLODefinition.
    """
    api_version = data.get("apiVersion", "")
    if api_version and api_version != "openslo/v1":
        raise OpenSLOParseError(
            f"Unsupported OpenSLO apiVersion: {api_version}. Expected openslo/v1"
        )

    kind = data.get("kind", "")
    if kind and kind != "SLO":
        raise OpenSLOParseError(
            f"Expected kind: SLO, got: {kind}"
        )

    metadata = data.get("metadata", {})
    spec = data.get("spec", {})

    name = metadata.get("name")
    if not name:
        raise OpenSLOParseError("OpenSLO document missing metadata.name")

    # Parse objectives — take the first one (OpenSLO supports multiple but
    # we normalise to one SLODefinition per objective)
    objectives = spec.get("objectives", [])
    if not objectives:
        raise OpenSLOParseError(f"OpenSLO '{name}' has no objectives")

    objective = objectives[0]
    target = objective.get("target")
    if target is None:
        raise OpenSLOParseError(f"OpenSLO '{name}' objective missing target")

    # Parse indicator
    indicator = spec.get("indicator", {})
    indicator_spec = indicator.get("spec", {})

    indicator_query = None
    total_query = None
    good_query = None

    # ratioMetric indicator (most common)
    ratio = indicator_spec.get("ratioMetric")
    if ratio:
        total_source = ratio.get("total", {}).get("metricSource", {})
        good_source = ratio.get("good", {}).get("metricSource", {})

        total_query = _extract_query(total_source)
        good_query = _extract_query(good_source)

        # Compose indicator_query from total and good
        if total_query and good_query:
            indicator_query = f"({good_query}) / ({total_query})"

    # thresholdMetric indicator
    threshold = indicator_spec.get("thresholdMetric")
    if threshold and not ratio:
        metric_source = threshold.get("metricSource", {})
        indicator_query = _extract_query(metric_source)

    # Direct query (non-standard but common extension)
    if not indicator_query:
        direct = indicator_spec.get("query") or indicator_spec.get("metricSource", {}).get("spec", {}).get("query")
        if direct:
            indicator_query = direct

    # Infer slo_type from the objective/indicator shape
    slo_type = _infer_openslo_type(name, target, indicator_query, total_query, good_query, objective)

    # Extract window from timeWindow if present
    window = "30d"
    time_windows = spec.get("timeWindow", [])
    if time_windows:
        tw = time_windows[0]
        duration = tw.get("duration")
        if duration:
            window = duration

    # Extract unit and percentile hints
    display_name = objective.get("displayName", "")
    unit = None
    percentile = None

    if "ms" in display_name.lower():
        unit = "ms"
    elif " s " in display_name.lower() or "seconds" in display_name.lower():
        unit = "s"

    indicator_name = indicator.get("metadata", {}).get("name", "")
    for p in ("p50", "p95", "p99", "p999"):
        if p in indicator_name or p in name:
            percentile = p
            break

    description = objective.get("displayName") or metadata.get("displayName")

    return SLODefinition(
        name=name,
        target=float(target),
        slo_type=slo_type,
        window=window,
        unit=unit,
        percentile=percentile,
        indicator_query=indicator_query,
        total_query=total_query,
        good_query=good_query,
        description=description,
    )


def _extract_query(metric_source: dict[str, Any]) -> str | None:
    """Extract PromQL query from an OpenSLO metricSource."""
    ms_type = metric_source.get("type", "")
    if ms_type.lower() == "prometheus":
        return metric_source.get("spec", {}).get("query")
    # Fallback: try spec.query regardless of type
    return metric_source.get("spec", {}).get("query")


def _infer_openslo_type(
    name: str,
    target: float,
    indicator_query: str | None,
    total_query: str | None,
    good_query: str | None,
    objective: dict[str, Any],
) -> str:
    """Infer slo_type from OpenSLO document content.

    Raises OpenSLOParseError if ambiguous.
    """
    name_lower = name.lower()

    # Target value hints
    if target <= 1.0:
        # Ratio metric — likely availability or error rate
        if "error" in name_lower:
            return "error_rate"
        if "avail" in name_lower or (total_query and good_query):
            return "availability"

    # Query hints
    query = indicator_query or ""
    if "histogram_quantile" in query or "duration" in query:
        return "latency"
    if "latency" in name_lower or "duration" in name_lower:
        return "latency"
    if "throughput" in name_lower or "rps" in name_lower:
        return "throughput"
    if "error" in name_lower:
        return "error_rate"
    if "avail" in name_lower:
        return "availability"

    # If we have a ratio metric, default to availability
    if total_query and good_query:
        return "availability"

    raise OpenSLOParseError(
        f"Cannot infer slo_type for OpenSLO '{name}'. The indicator shape "
        f"is ambiguous. Add an opensrm.nthlayer.io/slo-type annotation."
    )
