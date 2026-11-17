"""load_manifest's contract on a malformed legacy manifest (opensrm-wo62.4.10).

Whatever the shape of the input, ``load_manifest`` returns a manifest or
raises ``OSError``, ``ManifestLoadError`` or ``ValueError``. It never lets an
``AttributeError`` or ``TypeError`` out: a legacy file with ``spec: 5`` used
to end in "'int' object has no attribute 'get'", which every caller had to
catch for itself.

The cases are generated, not picked. Two well-formed legacy documents hold
every position the legacy loader reads; each node of each document is
replaced, one at a time, by a value of each wrong type (None, a string, a
number, a list, a mapping, and the empty or zero value of each). A position the loader starts to read belongs in
the documents below, which adds its cells here.

The second half pins that nothing which loaded before this contract stopped
loading, and nothing that failed started to load unnoticed: the cells that
loaded on the commit before the fix are listed in
``tests/fixtures/legacy_shape_matrix/loaded_before.json`` (see the README
beside it).
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from nthlayer_common.manifest import ManifestLoadError, load_manifest

LOADED_BEFORE = Path(__file__).parent / "fixtures" / "legacy_shape_matrix" / "loaded_before.json"

# One value of each wrong type, and the empty or zero value of each: the
# loader iterates and truth-tests what it reads, so ``databases: {}`` and
# ``databases: {x: 1}`` take different paths. A node is also replaced by a
# value of its own type (a string by another string): that is a different
# value, not a wrong shape, and must not crash either.
WRONG_VALUES: dict[str, Any] = {
    "none": None,
    "string": "text",
    "empty-string": "",
    "number": 7,
    "zero": 0,
    "list": ["x"],
    "empty-list": [],
    "mapping": {"x": 1},
    "empty-mapping": {},
}

# Every key the legacy loader reads (parser/loader.py's legacy half and
# parser/legacy.py), in a document that loads.
FULL: dict[str, Any] = {
    "service": {
        "name": "checkout",
        "team": "payments",
        "tier": "critical",
        "type": "api",
        "description": "Checkout API",
        "language": "python",
        "framework": "fastapi",
        "support_model": "shared",
        "slack_channel": "#payments",
        "email": "payments@example.com",
        "runbook_url": "https://runbooks.example.com/checkout",
        "metadata": {"labels": {"domain": "commerce"}, "annotations": {"owner": "payments"}},
        "pagerduty": {"timezone": "Europe/Dublin", "sre_escalation_policy_id": "PSRE1"},
    },
    "resources": [
        {
            "kind": "SLO",
            "name": "availability",
            "spec": {
                "objective": 99.9,
                "window": "30d",
                "unit": "percent",
                "description": "Requests that succeed",
                "indicator": {"type": "availability", "query": "sum(rate(ok[5m]))"},
                "indicators": [
                    {
                        "type": "availability",
                        "success_ratio": {
                            "good_query": "sum(rate(ok[5m]))",
                            "total_query": "sum(rate(all[5m]))",
                        },
                    }
                ],
            },
        },
        {
            "kind": "SLO",
            "name": "latency-p99",
            "spec": {
                "target": 99.0,
                "window": "7d",
                "percentile": 99,
                "indicator": {"type": "latency", "threshold_ms": 250},
                "threshold_ms": 300,
                "latency_threshold": 400,
            },
        },
        {
            "kind": "SLO",
            "name": "calibration",
            "spec": {"target": 0.1, "window": "30d", "description": "Calibration error"},
        },
        {
            "kind": "SLO",
            "name": "feedback_latency",
            "spec": {"objective": 24, "unit": "h", "window": "30d"},
        },
        {
            "kind": "Dependencies",
            "name": "deps",
            "spec": {
                "databases": [
                    {
                        "name": "orders-db",
                        "type": "postgresql",
                        "criticality": "critical",
                        "slo": {"availability": 99.95, "latency_p99": "50ms"},
                    }
                ],
                "upstream": [
                    {
                        "name": "payment-api",
                        "criticality": "high",
                        "slo": {"availability": 99.9, "latency_p99": "200ms"},
                    }
                ],
                "downstream": [{"name": "notifications"}],
                "caches": [{"name": "sessions", "type": "redis", "criticality": "medium"}],
                "queues": [{"name": "orders", "type": "kafka", "criticality": "high"}],
                "services": [{"name": "fraud", "type": "api", "criticality": "low"}],
            },
        },
        {
            "kind": "Alerts",
            "name": "alerts",
            "spec": {"for": "5m", "channels": ["#payments-alerts"]},
        },
        {
            "kind": "Observability",
            "name": "observability",
            "spec": {"metrics": {"enabled": True}},
        },
        {
            "kind": "PagerDuty",
            "name": "pagerduty",
            "spec": {
                "service_id": "PSVC1",
                "escalation_policy": "PESC1",
                "urgency": "high",
                "auto_resolve_timeout": 14400,
                "create_escalation_policy": False,
                "integration_key": "integration",
                "sre_integration_key": "sre-integration",
            },
        },
    ],
}

# The template path: service.template is expanded into resources before
# anything else reads them (parser/legacy.py, expand_template).
TEMPLATED: dict[str, Any] = {
    "service": {
        "name": "orders",
        "team": "commerce",
        "tier": "standard",
        "type": "api",
        "template": "standard-api",
    },
    "resources": [
        {"kind": "SLO", "name": "availability", "spec": {"objective": 99.5}},
    ],
}

DOCUMENTS = {"full": FULL, "templated": TEMPLATED}


def _paths(node: Any, prefix: tuple = ()) -> list[tuple]:
    """Every position in *node* below the root, parents before children."""
    children = node.items() if isinstance(node, dict) else enumerate(node)
    found: list[tuple] = []
    for key, child in children:
        path = (*prefix, key)
        found.append(path)
        if isinstance(child, dict | list):
            found.extend(_paths(child, path))
    return found


def _replaced(document: dict[str, Any], path: tuple, value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(document)
    parent = mutated
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = copy.deepcopy(value)
    return mutated


def _cells() -> dict[str, dict[str, Any]]:
    """Cell id -> the mutated document: every position crossed with every
    wrong value."""
    cells = {}
    for doc_name, document in DOCUMENTS.items():
        for path in _paths(document):
            where = ".".join(str(key) for key in path)
            for kind, value in WRONG_VALUES.items():
                cells[f"{doc_name}:{where}={kind}"] = _replaced(document, path, value)
    return cells


CELLS = _cells()
CELL_IDS = sorted(CELLS)

# The matrix is generated at import; an empty one would collect no tests and
# pass. FULL alone has well over a hundred positions.
assert len(CELL_IDS) >= 1200


def load_cell(cell_id: str, tmp_path: Path) -> Any:
    """Write the cell's document beside nothing else and load it."""
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(CELLS[cell_id], sort_keys=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_manifest(path, suppress_deprecation_warning=True)


def loads(cell_id: str, tmp_path: Path) -> bool:
    """Whether the cell loads. Only the three contract errors count as "does
    not load"; anything else propagates and fails the test."""
    try:
        load_cell(cell_id, tmp_path)
    except (OSError, ManifestLoadError, ValueError):
        return False
    return True


@pytest.mark.parametrize("document", ["full", "templated"])
def test_the_unmutated_documents_load(document, tmp_path):
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(DOCUMENTS[document], sort_keys=False))

    manifest = load_manifest(path, suppress_deprecation_warning=True)

    assert manifest.name == DOCUMENTS[document]["service"]["name"]


def test_the_full_document_exercises_every_resource_kind(tmp_path):
    # If a mutation target is not read, its cells prove nothing.
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(FULL, sort_keys=False))

    manifest = load_manifest(path, suppress_deprecation_warning=True)

    assert [slo.name for slo in manifest.slos] == ["availability", "latency-p99"]
    assert manifest.slos[1].threshold_ms == 250
    assert manifest.slos[0].good_query == "sum(rate(ok[5m]))"
    assert [t.type for t in manifest.judgment_thresholds] == ["calibration", "feedback_latency"]
    assert [d.type for d in manifest.dependencies] == [
        "database", "api", "api", "cache", "queue", "api",
    ]
    assert manifest.dependencies[0].slo.availability == 99.95
    assert manifest.alerting == {"for": "5m", "channels": ["#payments-alerts"]}
    assert manifest.observability is not None
    assert manifest.ownership.pagerduty.service_id == "PSVC1"
    assert manifest.ownership.pagerduty.timezone == "Europe/Dublin"
    assert manifest.labels == {"domain": "commerce"}


@pytest.mark.parametrize("cell_id", CELL_IDS)
def test_a_malformed_legacy_manifest_loads_or_raises_a_contract_error(cell_id, tmp_path):
    try:
        load_cell(cell_id, tmp_path)
    except (OSError, ManifestLoadError, ValueError):
        pass


def test_the_baseline_names_only_cells_of_this_matrix():
    loaded_before = set(json.loads(LOADED_BEFORE.read_text())["loaded"])

    assert loaded_before
    assert loaded_before <= set(CELL_IDS)


@pytest.mark.parametrize("cell_id", CELL_IDS)
def test_a_cell_loads_exactly_when_it_loaded_before(cell_id, tmp_path):
    loaded_before = set(json.loads(LOADED_BEFORE.read_text())["loaded"])

    assert loads(cell_id, tmp_path) == (cell_id in loaded_before)
