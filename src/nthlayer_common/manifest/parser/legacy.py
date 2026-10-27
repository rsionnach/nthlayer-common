"""Legacy-format preprocessing: environment overlay, template expansion and
``${var}`` substitution (opensrm-wo62.4.1, rulings D1-D3).

Applies to the legacy ``service:`` + ``resources:`` format only. The loader
runs the three steps in this order, before ``_parse_legacy_to_manifest``,
which is the order nthlayer-generate's ``specs/parser.py`` applied them:

1. :func:`apply_environment_overlay` — ports generate's
   ``specs/environments.py`` (lookup order, env-name validation) and
   ``specs/environment_merger.py`` (deep merge).
2. :func:`expand_template` — ports generate's ``specs/parser.py``
   ``_merge_resources`` and the template loading in ``template_loader.py``
   / ``custom_templates.py``. One deliberate change: custom templates are
   found by walking up from the manifest, not from the current directory.
3. :func:`substitute_variables` — ports generate's ``specs/template.py``
   over the context ``ServiceContext.to_dict`` built.

Errors are raised as ValueError; the loader wraps them in ManifestLoadError
(this module cannot import it without a cycle).
"""

from __future__ import annotations

import re
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import structlog
import yaml

from nthlayer_common.manifest.models import resolve_service_type
from nthlayer_common.tiers import TIER_NAMES

logger = structlog.get_logger()

# generate specs/environments.py VALID_ENVIRONMENT_PATTERN.
_ENVIRONMENT_NAME = re.compile(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$|^[a-z0-9]$")
# generate specs/template.py VARIABLE_PATTERN.
_VARIABLE = re.compile(r"\$\{(\w+)\}")

BUILTIN_TEMPLATES_PACKAGE = "nthlayer_common.manifest.legacy_templates"

_TEMPLATE_REQUIRED_FIELDS = ("name", "description", "tier", "type")

ResourceKey = tuple[Any, Any]


# =============================================================================
# 1. Environment overlay
# =============================================================================


def validate_environment_name(environment: str) -> str:
    """Return the lowercased, stripped environment name; ValueError if invalid.

    Rejects path traversal and anything outside ``[a-z0-9-]`` with no
    leading or trailing hyphen, as generate's validate_environment_name did.
    """
    env = environment.lower().strip()
    if ".." in env or "/" in env or "\\" in env:
        raise ValueError(f"Invalid environment name (path traversal attempt): {environment!r}")
    if not _ENVIRONMENT_NAME.match(env):
        raise ValueError(
            f"Invalid environment name format: {environment!r}. Must be alphanumeric "
            "with optional hyphens (e.g. 'dev', 'staging', 'prod-us')."
        )
    return env


def find_environment_file(manifest_path: Path, environment: str) -> Path | None:
    """Return the first overlay that exists, most specific first, or None.

    ``environment`` must already be validated. Lookup order:

    1. ``{dir}/environments/{stem}-{env}.yaml``
    2. ``{dir}/environments/{env}.yaml``
    3. ``{dir}/.nthlayer/environments/{stem}-{env}.yaml``
    """
    base_dir = manifest_path.parent
    stem = manifest_path.stem
    candidates = (
        base_dir / "environments" / f"{stem}-{environment}.yaml",
        base_dir / "environments" / f"{environment}.yaml",
        base_dir / ".nthlayer" / "environments" / f"{stem}-{environment}.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def apply_environment_overlay(
    data: dict[str, Any], manifest_path: Path, environment: str | None
) -> dict[str, Any]:
    """Deep-merge the environment overlay for ``environment`` into ``data``.

    No environment (None or "") or no overlay file is a no-op. A malformed
    overlay raises ValueError naming the overlay file.
    """
    if not environment:
        return data
    overlay_path = find_environment_file(manifest_path, validate_environment_name(environment))
    if overlay_path is None:
        return data

    overlay = _load_overlay(overlay_path)
    _require_mergeable_base(data, manifest_path)
    return _merge_service_config(data, overlay)


def _load_overlay(path: Path) -> dict[str, Any]:
    try:
        overlay = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in environment overlay {path}: {exc}") from exc

    def malformed(why: str) -> ValueError:
        return ValueError(f"Malformed environment overlay {path}: {why}")

    if not isinstance(overlay, dict):
        raise malformed("expected a YAML mapping at the top level")
    if "service" in overlay and not isinstance(overlay["service"], dict):
        raise malformed("'service' must be a mapping")
    if "resources" in overlay:
        # An overlay spec is deep-merged, so it must be a mapping.
        _require_resource_list(overlay["resources"], malformed, spec_mapping=True)
    return overlay


def _require_mergeable_base(data: dict[str, Any], manifest_path: Path) -> None:
    def malformed(why: str) -> ValueError:
        return ValueError(f"{why} in {manifest_path}")

    if "service" in data and not isinstance(data["service"], dict):
        raise malformed("'service' must be a mapping")
    if data.get("resources") is not None:
        _require_resource_list(data["resources"], malformed)


_SCALAR = (str, int, float, bool, type(None))


def _require_resource_list(resources: Any, error: Any, *, spec_mapping: bool = False) -> None:
    """Reject shapes the merges cannot key or merge, as a ValueError naming the file.

    ``kind`` and ``name`` key the (kind, name) merges, so they must be
    scalars. Legacy wrapped every such failure in a ServiceParseError
    naming the file.
    """
    if not isinstance(resources, list):
        raise error("'resources' must be a list")
    for i, resource in enumerate(resources):
        if not isinstance(resource, dict):
            raise error(f"resource {i} must be a mapping")
        for field in ("kind", "name"):
            if not isinstance(resource.get(field), _SCALAR):
                raise error(f"resource {i} '{field}' must be a scalar")
        if spec_mapping and "spec" in resource and not isinstance(resource["spec"], dict):
            raise error(f"resource {i} 'spec' must be a mapping")


def _merge_service_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """generate EnvironmentMerger.merge_service_config."""
    result = deepcopy(base)
    if "service" in overlay:
        result["service"] = _merge_dict(result.get("service") or {}, overlay["service"])
    if "resources" in overlay:
        result["resources"] = _merge_overlay_resources(
            result.get("resources") or [], overlay["resources"]
        )
    return result


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge; ``override`` wins, nested mappings merge recursively."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _key(resource: dict[str, Any]) -> ResourceKey:
    return (resource.get("kind"), resource.get("name"))


def _merge_overlay_resources(
    base: list[dict[str, Any]], overlay: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """generate EnvironmentMerger._merge_resources.

    Resources match on (kind, name). A matched resource has its ``spec``
    deep-merged and every other overlay key ignored; an unmatched overlay
    resource is appended. Base order is kept, new resources follow in
    overlay order.
    """
    indexed: dict[ResourceKey, dict[str, Any]] = {_key(r): deepcopy(r) for r in base}
    for resource in overlay:
        key = _key(resource)
        if key not in indexed:
            indexed[key] = deepcopy(resource)
        elif "spec" in resource:
            indexed[key]["spec"] = _merge_dict(indexed[key].get("spec") or {}, resource["spec"])

    base_keys = [_key(r) for r in base]
    seen = set(base_keys)
    new_keys = [_key(r) for r in overlay if _key(r) not in seen]
    return [indexed[k] for k in base_keys + new_keys]


# =============================================================================
# 2. Template expansion
# =============================================================================


def load_templates(template_dir: Path | None) -> dict[str, list[dict[str, Any]]]:
    """Return {template name: resources}: built-ins, then custom ones over them.

    A template is registered under its ``name`` field, not its file name. A
    template file that fails to load is skipped with a warning, as in
    generate, so one bad custom file cannot hide the others.
    """
    templates: dict[str, list[dict[str, Any]]] = {}
    builtin = files(BUILTIN_TEMPLATES_PACKAGE)
    for entry in sorted(builtin.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".yaml"):
            _register(templates, entry.read_text(), f"built-in {entry.name}")
    if template_dir is not None:
        for path in sorted(template_dir.glob("*.yaml")):
            _register(templates, path.read_text(), str(path))
    return templates


def _register(templates: dict[str, list[dict[str, Any]]], text: str, origin: str) -> None:
    try:
        name, resources = _parse_template(text, origin)
    except (ValueError, yaml.YAMLError) as exc:
        logger.warning("legacy_template_skipped", template=origin, error=str(exc))
        return
    templates[name] = resources


def _parse_template(text: str, origin: str) -> tuple[str, list[dict[str, Any]]]:
    """Validate one template file as generate's TemplateLoader/ServiceTemplate did."""
    data = yaml.safe_load(text)
    if not data:
        raise ValueError(f"Empty template file: {origin}")
    if not isinstance(data, dict):
        raise ValueError(f"Template must be a YAML mapping: {origin}")
    for field in _TEMPLATE_REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in template: {origin}")
    if data["tier"] not in TIER_NAMES:
        raise ValueError(f"Invalid tier {data['tier']!r} in template: {origin}")
    if resolve_service_type(data["type"]) is None:
        raise ValueError(f"Invalid type {data['type']!r} in template: {origin}")
    if not isinstance(data["name"], str):
        raise ValueError(f"Template name must be a string: {origin}")

    raw_resources = data.get("resources", [])
    if not isinstance(raw_resources, list):
        raise ValueError(f"'resources' must be a list in template: {origin}")
    resources = []
    for i, resource in enumerate(raw_resources):
        if not isinstance(resource, dict):
            raise ValueError(f"Resource {i} must be a mapping in template: {origin}")
        spec = resource.get("spec", {})
        if not resource.get("kind") or not resource.get("name") or spec is None:
            raise ValueError(f"Resource {i} needs kind, name and spec in template: {origin}")
        # Only kind, name and spec survive, as generate's Resource kept them.
        resources.append({"kind": resource["kind"], "name": resource["name"], "spec": spec})
    return data["name"], resources


def expand_template(
    data: dict[str, Any], manifest_path: Path, template_dir: Path | None
) -> dict[str, Any]:
    """Merge ``service.template``'s resources into ``data['resources']``.

    Template resources come first; a user resource with the same
    (kind, name) replaces the template one WHOLE (no spec merge); new user
    resources follow. No template set is a no-op. An unknown template
    raises ValueError listing the available ones.
    """
    service = data.get("service")
    template_name = service.get("template") if isinstance(service, dict) else None
    if not template_name:
        return data

    templates = load_templates(template_dir)
    if not isinstance(template_name, str) or template_name not in templates:
        raise ValueError(
            f"Unknown template {template_name!r} in {manifest_path}. "
            f"Available templates: {', '.join(sorted(templates))}"
        )

    user_resources = data.get("resources")
    if user_resources is None:
        user_resources = []
    _require_resource_list(user_resources, lambda why: ValueError(f"{why} in {manifest_path}"))

    result = dict(data)
    result["resources"] = _merge_template_resources(
        deepcopy(templates[template_name]), user_resources
    )
    return result


def _merge_template_resources(
    template: list[dict[str, Any]], user: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """generate specs/parser.py _merge_resources."""
    merged: dict[ResourceKey, dict[str, Any]] = {}
    for resource in template + user:
        merged[_key(resource)] = resource

    template_keys = [_key(r) for r in template]
    seen = set(template_keys)
    return [merged[k] for k in template_keys] + [r for r in user if _key(r) not in seen]


# =============================================================================
# 3. ${var} substitution
# =============================================================================


def substitute_variables(data: dict[str, Any], environment: str | None) -> dict[str, Any]:
    """Replace ``${var}`` throughout ``data`` (values only, not keys).

    Variables are generate's ServiceContext.to_dict: service, team, tier,
    type (as authored, not alias-resolved), support_model (default
    ``self``), language and framework (``""`` when unset), and env (only
    when an environment is given, exactly as passed). An unknown variable,
    or one whose value is None, is left as written.
    """
    service = data.get("service")
    if not isinstance(service, dict):
        service = {}
    context: dict[str, Any] = {
        "service": service.get("name"),
        "team": service.get("team"),
        "tier": service.get("tier"),
        "type": service.get("type"),
        "support_model": service.get("support_model", "self"),
        "language": service.get("language") or "",
        "framework": service.get("framework") or "",
    }
    if environment:
        context["env"] = environment
    return _substitute(data, context)


def _substitute(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _VARIABLE.sub(lambda m: _lookup(m, context), value)
    if isinstance(value, dict):
        return {k: _substitute(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(item, context) for item in value]
    return value


def _lookup(match: re.Match[str], context: dict[str, Any]) -> str:
    value = context.get(match.group(1))
    return match.group(0) if value is None else str(value)
