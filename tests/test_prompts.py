# tests/test_prompts.py
"""Tests for the shared prompt loader, renderer, and validator."""
from __future__ import annotations

import pytest

from nthlayer_common.prompts import (
    PromptSpec,
    extract_confidence,
    load_prompt,
    render_user_prompt,
    validate_response,
)


SAMPLE_YAML = """\
name: test-prompt
version: "1.0"

system: |
  You are a test agent.
  {schema_block}

response_schema:
  type: object
  required: [result, confidence]
  properties:
    result:
      type: string
      description: "The test result"
    score:
      type: number
      description: "A score"
      range: [0.0, 1.0]
      default: 0.5
    confidence:
      type: number
      description: "Confidence in the result"
      range: [0.0, 1.0]
      default: 0.0

user_template: |
  Context: {{ context }}
  Agent: {{ agent_name }}
"""


@pytest.fixture
def yaml_file(tmp_path):
    p = tmp_path / "test-prompt.yaml"
    p.write_text(SAMPLE_YAML)
    return p


class TestLoadPrompt:
    def test_loads_yaml(self, yaml_file):
        spec = load_prompt(yaml_file)
        assert isinstance(spec, PromptSpec)
        assert spec.name == "test-prompt"
        assert spec.version == "1.0"

    def test_schema_block_inserted_in_system(self, yaml_file):
        spec = load_prompt(yaml_file)
        assert "Respond with ONLY valid JSON matching this schema:" in spec.system
        assert '"result"' in spec.system
        assert '"confidence"' in spec.system

    def test_schema_block_placeholder_removed(self, yaml_file):
        spec = load_prompt(yaml_file)
        assert "{schema_block}" not in spec.system

    def test_response_schema_preserved(self, yaml_file):
        spec = load_prompt(yaml_file)
        assert "result" in spec.response_schema["properties"]
        assert "confidence" in spec.response_schema["properties"]

    def test_user_template_preserved(self, yaml_file):
        spec = load_prompt(yaml_file)
        assert "{{ context }}" in spec.user_template


class TestRenderUserPrompt:
    def test_interpolates_variables(self):
        result = render_user_prompt(
            "Hello {{ name }}, you are {{ role }}.",
            name="Rob", role="SRE",
        )
        assert result == "Hello Rob, you are SRE."

    def test_no_variables_unchanged(self):
        result = render_user_prompt("No variables here.")
        assert result == "No variables here."

    def test_missing_variable_left_as_is(self):
        result = render_user_prompt("Hello {{ name }}.", role="SRE")
        assert "{{ name }}" in result

    def test_handles_no_space_braces(self):
        result = render_user_prompt("Hello {{name}}.", name="Rob")
        assert result == "Hello Rob."


class TestValidateResponse:
    def test_valid_response(self, yaml_file):
        spec = load_prompt(yaml_file)
        data = {"result": "passed", "score": 0.9, "confidence": 0.85}
        validated = validate_response(data, spec.response_schema)
        assert validated["result"] == "passed"
        assert validated["score"] == 0.9
        assert validated["confidence"] == 0.85

    def test_missing_required_field_raises(self, yaml_file):
        spec = load_prompt(yaml_file)
        data = {"score": 0.9, "confidence": 0.5}  # missing "result"
        with pytest.raises(ValueError, match="result"):
            validate_response(data, spec.response_schema)

    def test_optional_field_gets_default(self, yaml_file):
        spec = load_prompt(yaml_file)
        data = {"result": "passed", "confidence": 0.8}  # missing optional "score"
        validated = validate_response(data, spec.response_schema)
        assert validated["score"] == 0.5  # default from schema

    def test_confidence_clamped(self, yaml_file):
        spec = load_prompt(yaml_file)
        data = {"result": "ok", "confidence": 1.5}
        validated = validate_response(data, spec.response_schema)
        assert validated["confidence"] == 1.0

    def test_confidence_negative_clamped(self, yaml_file):
        spec = load_prompt(yaml_file)
        data = {"result": "ok", "confidence": -0.5}
        validated = validate_response(data, spec.response_schema)
        assert validated["confidence"] == 0.0

    def test_extra_fields_passed_through(self, yaml_file):
        spec = load_prompt(yaml_file)
        data = {"result": "ok", "confidence": 0.8, "extra_field": "bonus"}
        validated = validate_response(data, spec.response_schema)
        assert validated["extra_field"] == "bonus"

    def test_string_number_coerced(self, yaml_file):
        spec = load_prompt(yaml_file)
        data = {"result": "ok", "confidence": "0.75", "score": "0.9"}
        validated = validate_response(data, spec.response_schema)
        assert validated["confidence"] == 0.75
        assert validated["score"] == 0.9


class TestExtractConfidence:
    def test_present(self):
        assert extract_confidence({"confidence": 0.85}) == 0.85

    def test_absent_returns_zero(self):
        assert extract_confidence({}) == 0.0

    def test_clamped_high(self):
        assert extract_confidence({"confidence": 1.5}) == 1.0

    def test_clamped_low(self):
        assert extract_confidence({"confidence": -0.3}) == 0.0

    def test_string_coerced(self):
        assert extract_confidence({"confidence": "0.9"}) == 0.9

    def test_invalid_string_returns_zero(self):
        assert extract_confidence({"confidence": "high"}) == 0.0

    def test_none_returns_zero(self):
        assert extract_confidence({"confidence": None}) == 0.0
