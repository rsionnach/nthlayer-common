"""Shared prompt loader, renderer, and response validator.

Loads YAML prompt definitions with explicit response schemas.
Schema defined once, used twice: appended to the prompt (so the model sees it)
and used in the parser (for validation).

Usage:
    from nthlayer_common.prompts import load_prompt, render_user_prompt, validate_response

    spec = load_prompt("prompts/triage.yaml")
    user = render_user_prompt(spec.user_template, context="...")
    # spec.system already includes the schema block
    response = llm_call(system=spec.system, user=user)
    validated = validate_response(json.loads(response.text), spec.response_schema)
    confidence = extract_confidence(validated)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PromptSpec:
    """Loaded prompt specification from YAML."""

    name: str
    version: str
    system: str  # System prompt with schema block already inserted
    user_template: str
    response_schema: dict[str, Any]


def load_prompt(path: str | Path) -> PromptSpec:
    """Load a YAML prompt file and return a PromptSpec.

    The {schema_block} placeholder in the system prompt (or user_template)
    is replaced with a rendered JSON example of the response schema.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    name = raw.get("name", path.stem)
    version = raw.get("version", "1.0")
    system = raw.get("system", "")
    user_template = raw.get("user_template", "")
    response_schema = raw.get("response_schema", {})

    # Render the schema as a JSON example block
    schema_block = _render_schema_block(response_schema)

    # Insert into system prompt or user template
    system = system.replace("{schema_block}", schema_block)
    user_template = user_template.replace("{schema_block}", schema_block)

    return PromptSpec(
        name=name,
        version=version,
        system=system.strip(),
        user_template=user_template.strip(),
        response_schema=response_schema,
    )


def render_user_prompt(template: str, **kwargs: Any) -> str:
    """Interpolate {{ variable }} placeholders in a user template.

    Simple replacement — not Jinja2. No new dependencies.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{ " + key + " }}", str(value))
        result = result.replace("{{" + key + "}}", str(value))
    return result


def validate_response(data: dict, schema: dict) -> dict:
    """Validate a model response against the prompt's response schema.

    Checks required fields exist and applies defaults for missing optional fields.
    Returns the validated dict with defaults filled in.
    Raises ValueError if a required field is missing.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    validated = {}
    for field_name, field_spec in properties.items():
        value = data.get(field_name)

        if value is None and field_name in required:
            # Check for common aliases before failing
            # (aliases are documented in YAML comments but not in schema)
            raise ValueError(f"Required field '{field_name}' missing from model response")

        if value is None:
            validated[field_name] = field_spec.get("default")
        else:
            # Type coercion for confidence/numeric fields
            if field_spec.get("type") == "number" and not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = field_spec.get("default", 0.0)

            # Clamp numeric fields with range
            if field_spec.get("range") and isinstance(value, (int, float)):
                low, high = field_spec["range"]
                value = max(low, min(high, float(value)))

            validated[field_name] = value

    # Pass through any extra fields the model returned (don't strip them)
    for key, value in data.items():
        if key not in validated:
            validated[key] = value

    return validated


def extract_confidence(data: dict) -> float:
    """Extract confidence from a response dict.

    Returns 0.0 if absent — meaning 'model didn't report confidence',
    not a hardcoded guess.
    """
    conf = data.get("confidence", 0.0)
    if not isinstance(conf, (int, float)):
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            return 0.0
    return max(0.0, min(1.0, float(conf)))


def _render_schema_block(schema: dict) -> str:
    """Render a response schema as a JSON example for inclusion in prompts."""
    example = _schema_to_example(schema)
    return (
        "Respond with ONLY valid JSON matching this schema:\n"
        + json.dumps(example, indent=2)
    )


def _schema_to_example(schema: dict) -> Any:
    """Convert a JSON Schema to an example JSON object."""
    schema_type = schema.get("type", "object")

    if schema_type == "object":
        props = schema.get("properties", {})
        result = {}
        for name, prop_schema in props.items():
            result[name] = _schema_to_example(prop_schema)
        return result

    if schema_type == "array":
        items = schema.get("items", {})
        return [_schema_to_example(items)]

    if schema_type == "string":
        desc = schema.get("description", "")
        enum = schema.get("enum")
        if enum:
            return enum[0]
        return f"<{desc}>" if desc else "<string>"

    if schema_type == "number":
        return schema.get("default", 0.0)

    if schema_type == "integer":
        return schema.get("default", 0)

    if schema_type == "boolean":
        return schema.get("default", False)

    return None
