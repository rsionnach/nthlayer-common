"""load_manifest's contract on a malformed legacy manifest (opensrm-wo62.4.10).

Whatever the shape of the input, ``load_manifest`` returns a manifest or
raises ``OSError``, ``ManifestLoadError`` or ``ValueError``. It never lets an
``AttributeError`` or ``TypeError`` out: a legacy file with ``spec: 5`` used
to end in "'int' object has no attribute 'get'", which every caller had to
catch for itself.

The cases are generated, not picked. Two well-formed legacy documents hold
every position the legacy loader reads; each node of each document is
replaced, one at a time, by a value of each wrong type (None, a string, a
number, a list, a mapping, the empty or zero value of each, and the other
types YAML can build: a boolean, a float, a set, binary data, a date). The
same is done to an environment overlay merged into the first document. A position the loader starts to read belongs in
the documents below, which adds its cells here.

The second half pins that nothing which loaded before this contract stopped
loading, and nothing that failed started to load unnoticed: the cells that
loaded on the commit before the fix are listed in
``tests/fixtures/legacy_shape_matrix/loaded_before.json`` (see the README
beside it).
"""

from __future__ import annotations

import copy
import datetime
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from nthlayer_common.manifest import ManifestLoadError, load_manifest

LOADED_BEFORE = Path(__file__).parent / "fixtures" / "legacy_shape_matrix" / "loaded_before.json"

_RECURSIVE_LIST: list[Any] = []
_RECURSIVE_LIST.append(_RECURSIVE_LIST)
_RECURSIVE_MAPPING: dict[str, Any] = {}
_RECURSIVE_MAPPING["again"] = _RECURSIVE_MAPPING

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
    # The rest of what yaml.safe_load can build (opensrm-wo62.4.10's
    # correctness review found an empty !!set where a list goes, which
    # used to load as nothing).
    "true": True,
    "false": False,
    "float": 1.5,
    "nan": float("nan"),
    "set": {"x"},
    "empty-set": set(),
    "binary": b"x",
    "empty-binary": b"",
    "date": datetime.date(2024, 1, 1),
    "list-of-none": [None],
    "int-keyed-mapping": {1: "x"},
    # A number float() cannot hold, and values that contain themselves (a
    # YAML alias to its own anchor): the edge-cases review found
    # OverflowError and RecursionError getting out.
    "huge-int": 10**400,
    "recursive-list": _RECURSIVE_LIST,
    "recursive-mapping": _RECURSIVE_MAPPING,
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

# The overlay path: load_manifest(path, environment="prod") deep-merges
# environments/prod.yaml into FULL before anything else reads it
# (parser/legacy.py, apply_environment_overlay). Its cells mutate the
# overlay and leave FULL as it is.
OVERLAY: dict[str, Any] = {
    "service": {
        "tier": "high",
        "metadata": {"labels": {"env": "prod"}},
        "pagerduty": {"timezone": "UTC"},
    },
    "resources": [
        {
            "kind": "SLO",
            "name": "availability",
            "spec": {
                "objective": 99.0,
                "indicator": {"type": "availability", "threshold_ms": 10},
            },
        },
        {"kind": "SLO", "name": "prod-only", "spec": {"objective": 95, "threshold_ms": 5}},
        {
            "kind": "Dependencies",
            "name": "deps",
            "spec": {"databases": [{"name": "replica", "slo": {"availability": 99}}]},
        },
        {"kind": "PagerDuty", "name": "pagerduty", "spec": {"urgency": "low"}},
    ],
}
ENVIRONMENT = "prod"


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


def _cells() -> dict[str, tuple[dict[str, Any], dict[str, Any] | None]]:
    """Cell id -> (document, overlay or None): every position of every
    document, and of the overlay, crossed with every wrong value."""
    cells: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    for doc_name, document in DOCUMENTS.items():
        for path in _paths(document):
            where = ".".join(str(key) for key in path)
            for kind, value in WRONG_VALUES.items():
                cells[f"{doc_name}:{where}={kind}"] = (_replaced(document, path, value), None)
    for path in _paths(OVERLAY):
        where = ".".join(str(key) for key in path)
        for kind, value in WRONG_VALUES.items():
            cells[f"overlay:{where}={kind}"] = (FULL, _replaced(OVERLAY, path, value))
    return cells


CELLS = _cells()
CELL_IDS = sorted(CELLS)
# One test per position (its cells differ only in the wrong value), which
# keeps a failure readable and the run short.
POSITIONS = sorted({cell_id.rsplit("=", 1)[0] for cell_id in CELL_IDS})

# The matrix is generated at import; an empty one would collect no tests and
# pass. FULL alone has well over a hundred positions.
assert len(CELL_IDS) >= 3900
assert len(POSITIONS) >= 170


def _write(tmp_path: Path, document: dict[str, Any], overlay: dict[str, Any] | None) -> Path:
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    if overlay is not None:
        (tmp_path / "environments").mkdir(exist_ok=True)
        (tmp_path / "environments" / f"{ENVIRONMENT}.yaml").write_text(
            yaml.safe_dump(overlay, sort_keys=False)
        )
    return path


def load_cell(cell_id: str, tmp_path: Path) -> Any:
    """Write the cell's document (and its overlay, if it has one) and load it."""
    document, overlay = CELLS[cell_id]
    path = _write(tmp_path, document, overlay)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_manifest(
            path,
            environment=ENVIRONMENT if overlay is not None else None,
            suppress_deprecation_warning=True,
        )


def outcome(cell_id: str, tmp_path: Path) -> str:
    """"loads" or "refused" (one of the three contract errors). Anything
    else that gets out fails the test, naming the cell."""
    try:
        load_cell(cell_id, tmp_path)
    except (OSError, ManifestLoadError, ValueError):
        return "refused"
    except Exception as exc:
        pytest.fail(f"{cell_id} let a {type(exc).__name__} out of load_manifest: {exc}")
    return "loads"


@pytest.mark.parametrize("document", ["full", "templated"])
def test_the_unmutated_documents_load(document, tmp_path):
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(DOCUMENTS[document], sort_keys=False))

    manifest = load_manifest(path, suppress_deprecation_warning=True)

    assert manifest.name == DOCUMENTS[document]["service"]["name"]


def test_the_unmutated_overlay_loads_and_is_applied(tmp_path):
    path = _write(tmp_path, FULL, OVERLAY)

    manifest = load_manifest(path, environment=ENVIRONMENT, suppress_deprecation_warning=True)

    assert manifest.tier == "high"
    assert [slo.name for slo in manifest.slos] == ["availability", "latency-p99", "prod-only"]
    assert manifest.slos[0].target == 99.0
    assert [d.name for d in manifest.dependencies if d.type == "database"] == ["replica"]
    assert manifest.ownership.pagerduty.urgency == "low"
    assert manifest.ownership.pagerduty.timezone == "UTC"


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


def test_the_baseline_names_only_cells_of_this_matrix():
    loaded_before = set(json.loads(LOADED_BEFORE.read_text())["loaded"])

    assert loaded_before
    assert loaded_before <= set(CELL_IDS)


@pytest.mark.parametrize("position", POSITIONS)
def test_every_wrong_value_is_refused_or_loads_as_it_did_before(position, tmp_path):
    # The contract and the baseline in one load per cell: a cell either
    # loads or is refused with OSError, ManifestLoadError or ValueError
    # (never anything else), and it loads exactly when it loaded before
    # (nothing that loaded stops loading, nothing that failed starts to).
    loaded_before = set(json.loads(LOADED_BEFORE.read_text())["loaded"])
    expected = {
        f"{position}={kind}": "loads" if f"{position}={kind}" in loaded_before else "refused"
        for kind in WRONG_VALUES
    }

    assert {cell_id: outcome(cell_id, tmp_path) for cell_id in expected} == expected


# A few cells that used to crash, and what the error must let the author
# find: the resource by its name, or the top-level key.
NAMED = [
    ("full:resources.0.spec=number", "availability"),
    ("full:resources.1.spec.indicator=string", "latency-p99"),
    ("full:resources.1.spec.target=list", "latency-p99"),
    ("full:resources.4.spec=none", "deps"),
    ("full:resources.4.spec.databases.0.slo=string", "deps"),
    ("full:resources.4.spec.queues=number", "deps"),
    ("full:resources.7.spec=list", "pagerduty"),
    ("full:resources=number", "resources"),
    ("full:resources.5=string", "resources[5]"),
    ("full:service=list", "service"),
    ("full:service.metadata=number", "service.metadata"),
    ("full:service.tier=mapping", "service.tier"),
    ("full:resources.0.spec.objective=huge-int", "availability"),
    ("full:resources.1.spec.indicator.threshold_ms=huge-int", "latency-p99"),
    ("full:resources.4.spec.databases=recursive-list", "service.yaml"),
]


@pytest.mark.parametrize(("cell_id", "named"), NAMED)
def test_the_error_names_the_resource_or_the_position(cell_id, named, tmp_path):
    with pytest.raises(ManifestLoadError) as raised:
        load_cell(cell_id, tmp_path)

    assert named in str(raised.value)
