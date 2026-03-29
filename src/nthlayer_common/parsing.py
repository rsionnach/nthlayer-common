"""Shared parsing utilities for LLM responses.

Used by every component that calls llm_call() and parses the response.
"""
from __future__ import annotations


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from model response text.

    Handles ```json, ```, and bare ``` patterns.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value to [low, high]. Default: [0.0, 1.0]."""
    return max(low, min(high, value))
