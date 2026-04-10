"""Budget explanation data model and formatter.

Shared across the ecosystem: nthlayer-observe produces explanations,
nthlayer-respond can format them for incident communications.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class BudgetExplanation:
    """Human-readable explanation of error budget status."""

    service: str
    slo_name: str
    headline: str
    body: str
    causes: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    severity: str = "info"  # "info" | "warning" | "critical"

    def to_dict(self) -> dict:
        return asdict(self)


def format_explanation(explanation: BudgetExplanation, fmt: str = "table") -> str:
    """Format a BudgetExplanation for output."""
    if fmt == "json":
        return json.dumps(explanation.to_dict(), indent=2)

    if fmt == "markdown":
        lines = [
            f"## {explanation.service} — {explanation.slo_name}",
            "",
            f"**{explanation.headline}**",
            "",
            explanation.body,
        ]
        if explanation.causes:
            lines.extend(["", "### Causes"])
            lines.extend(f"- {c}" for c in explanation.causes)
        if explanation.recommended_actions:
            lines.extend(["", "### Recommended Actions"])
            lines.extend(f"- {a}" for a in explanation.recommended_actions)
        return "\n".join(lines)

    # Default: table
    icon = {"info": "ℹ", "warning": "⚠", "critical": "✗"}.get(explanation.severity, "?")
    lines = [
        f"{icon} {explanation.service} / {explanation.slo_name}",
        f"  {explanation.headline}",
        f"  {explanation.body}",
    ]
    if explanation.causes:
        lines.append("  Causes:")
        lines.extend(f"    • {c}" for c in explanation.causes)
    if explanation.recommended_actions:
        lines.append("  Actions:")
        lines.extend(f"    → {a}" for a in explanation.recommended_actions)
    return "\n".join(lines)
