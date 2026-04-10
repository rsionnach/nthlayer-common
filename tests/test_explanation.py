"""Tests for BudgetExplanation data model and formatter."""
from __future__ import annotations

import json

from nthlayer_common.explanation import BudgetExplanation, format_explanation


def _make_explanation(**overrides) -> BudgetExplanation:
    defaults = dict(
        service="payment-api",
        slo_name="availability",
        headline="availability: 73% consumed (WARNING)",
        body="Budget: 1440 min total, 1051 min consumed, 389 min remaining.",
        causes=["Error rate gradually increasing"],
        recommended_actions=["Investigate before next deployment"],
        severity="warning",
    )
    defaults.update(overrides)
    return BudgetExplanation(**defaults)


class TestBudgetExplanation:
    def test_create(self) -> None:
        exp = _make_explanation()
        assert exp.service == "payment-api"
        assert exp.severity == "warning"

    def test_defaults(self) -> None:
        exp = BudgetExplanation(
            service="svc", slo_name="avail", headline="h", body="b"
        )
        assert exp.causes == []
        assert exp.recommended_actions == []
        assert exp.severity == "info"

    def test_to_dict(self) -> None:
        exp = _make_explanation()
        d = exp.to_dict()
        assert isinstance(d, dict)
        assert d["service"] == "payment-api"
        assert d["causes"] == ["Error rate gradually increasing"]


class TestFormatTable:
    def test_contains_service_and_slo(self) -> None:
        output = format_explanation(_make_explanation())
        assert "payment-api" in output
        assert "availability" in output

    def test_severity_icon(self) -> None:
        assert "⚠" in format_explanation(_make_explanation(severity="warning"))
        assert "✗" in format_explanation(_make_explanation(severity="critical"))
        assert "ℹ" in format_explanation(_make_explanation(severity="info"))

    def test_includes_causes(self) -> None:
        output = format_explanation(_make_explanation())
        assert "Error rate" in output

    def test_includes_actions(self) -> None:
        output = format_explanation(_make_explanation())
        assert "Investigate" in output


class TestFormatJSON:
    def test_valid_json(self) -> None:
        output = format_explanation(_make_explanation(), fmt="json")
        parsed = json.loads(output)
        assert parsed["service"] == "payment-api"

    def test_roundtrip(self) -> None:
        exp = _make_explanation(severity="critical", causes=["c1", "c2"])
        parsed = json.loads(format_explanation(exp, fmt="json"))
        assert parsed["severity"] == "critical"
        assert parsed["causes"] == ["c1", "c2"]


class TestFormatMarkdown:
    def test_has_header(self) -> None:
        output = format_explanation(_make_explanation(), fmt="markdown")
        assert "##" in output

    def test_contains_service(self) -> None:
        output = format_explanation(_make_explanation(), fmt="markdown")
        assert "payment-api" in output

    def test_has_sections(self) -> None:
        output = format_explanation(_make_explanation(), fmt="markdown")
        assert "### Causes" in output
        assert "### Recommended Actions" in output

    def test_no_sections_when_empty(self) -> None:
        exp = _make_explanation(causes=[], recommended_actions=[])
        output = format_explanation(exp, fmt="markdown")
        assert "### Causes" not in output
