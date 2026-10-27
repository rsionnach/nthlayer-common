"""Smoke test: non-Python package data is present in the installed wheel.

The legacy built-in templates are YAML loaded through importlib.resources
(opensrm-wo62.4.1). The dev tree always has them, so only a check against
the installed wheel catches a missing ``[tool.setuptools.package-data]``
entry, which would make every legacy ``service.template`` fail as unknown.
"""

from __future__ import annotations

from importlib.resources import files

LEGACY_TEMPLATES = {
    "background-job.yaml",
    "critical-api.yaml",
    "low-api.yaml",
    "pipeline.yaml",
    "standard-api.yaml",
}


def test_legacy_builtin_templates_are_in_the_wheel():
    package = files("nthlayer_common.manifest.legacy_templates")
    names = {entry.name for entry in package.iterdir() if entry.name.endswith(".yaml")}
    assert names == LEGACY_TEMPLATES
