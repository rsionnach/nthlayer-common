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
same is done to an environment overlay merged into the first document, and
to each document with the unmutated overlay merged into it. Last, the merge
of one resource's spec is tabled in full: every state of the manifest's spec
against every state of the overlay's spec for the same resource. A position
the loader starts to read belongs in the documents below, which adds its
cells here.

No cell may be refused by the backstop (``ManifestLoadError.backstop``):
that error keeps the contract for a shape nothing here reaches, and a cell
that ends in it is a position whose check is missing.

The second half pins that nothing which loaded before this contract stopped
loading, and nothing that failed started to load unnoticed: the cells that
loaded on the commit before the fix are listed in
``tests/fixtures/legacy_shape_matrix/loaded_before.json`` (see the README
beside it).
"""

from __future__ import annotations

import collections
import copy
import datetime
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from nthlayer_common.manifest import ManifestLoadError, load_manifest
from nthlayer_common.manifest.parser import loader

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
        # The loader takes the first latency threshold key it finds
        # (indicator.threshold_ms, threshold_ms, latency_threshold), so in
        # latency-p99 above only the first is ever read. These two declare
        # each of the others alone. Appended, so the positions above keep
        # their numbers.
        {
            "kind": "SLO",
            "name": "latency-p95",
            "spec": {"target": 95.0, "threshold_ms": 300},
        },
        {
            "kind": "SLO",
            "name": "duration-p50",
            "spec": {"target": 50.0, "latency_threshold": 400},
        },
        # Likewise the SLO's type: indicator.type is read first, then the
        # plural indicators[] list, then keywords in the name. Only this
        # resource gets as far as the list: it has no indicator.type, and
        # "success" holds no keyword, so its type can only come from there.
        {
            "kind": "SLO",
            "name": "success",
            "spec": {
                "objective": 99.5,
                "indicators": [
                    # The list's threshold_ms is read too (opensrm-wo62.4.24).
                    {"type": "throughput", "threshold_ms": 350},
                    {
                        "type": "throughput",
                        "success_ratio": {
                            "good_query": "sum(rate(done[5m]))",
                            "total_query": "sum(rate(started[5m]))",
                        },
                    },
                ],
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
# (parser/legacy.py, apply_environment_overlay). The "overlay:" cells
# mutate the overlay and leave FULL as it is; the "<document>+overlay:"
# cells mutate the document and leave the overlay as it is.
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

# The merge of one resource (parser/legacy.py, _merge_overlay_resources)
# reads two specs at once: the manifest's and the overlay's for the same
# (kind, name). The cells above change one side and hold the other still,
# so they never pair, say, a manifest spec that is not a mapping with an
# overlay spec that is empty. The "pair:" cells are the whole table, for
# each resource of FULL: each state a spec can be in, on both sides.
ABSENT = object()  # the resource has no "spec" key
SPEC_STATES: dict[str, Any] = {
    "absent": ABSENT,
    "null": None,
    "empty-mapping": {},
    "mapping": "the resource's own spec",  # filled in per resource
    "string": "text",
    "empty-string": "",
    "number": 7,
    "zero": 0,
    "list": ["x"],
    "empty-list": [],
}


def _with_spec(resource: dict[str, Any], state: str, own_spec: Any) -> dict[str, Any]:
    """*resource*'s kind and name, with its spec in *state*."""
    result = {"kind": resource["kind"], "name": resource["name"]}
    if state == "mapping":
        result["spec"] = copy.deepcopy(own_spec)
    elif SPEC_STATES[state] is not ABSENT:
        result["spec"] = copy.deepcopy(SPEC_STATES[state])
    return result


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
    # The other side of the merge: each document mutated as above, with the
    # unmutated overlay merged into it. The merge reads the manifest's own
    # resources (their kind, name and spec) before the loader does.
    for doc_name, document in DOCUMENTS.items():
        for path in _paths(document):
            where = ".".join(str(key) for key in path)
            for kind, value in WRONG_VALUES.items():
                cells[f"{doc_name}+overlay:{where}={kind}"] = (
                    _replaced(document, path, value),
                    OVERLAY,
                )
    for index, resource in enumerate(FULL["resources"]):
        for manifest_state in SPEC_STATES:
            document = copy.deepcopy(FULL)
            document["resources"][index] = _with_spec(resource, manifest_state, resource["spec"])
            for overlay_state in SPEC_STATES:
                overlay = {"resources": [_with_spec(resource, overlay_state, resource["spec"])]}
                cells[
                    f"pair:resources.{index}:spec.{manifest_state}:overlay-spec={overlay_state}"
                ] = (document, overlay)
    return cells


CELLS = _cells()
CELL_IDS = sorted(CELLS)
# One test per position (its cells differ only in the wrong value), which
# keeps a failure readable and the run short.
CELLS_AT: dict[str, list[str]] = {}
for _cell_id in CELL_IDS:
    CELLS_AT.setdefault(_cell_id.rsplit("=", 1)[0], []).append(_cell_id)
POSITIONS = sorted(CELLS_AT)

# The matrix is generated at import; an empty one would collect no tests and
# pass. FULL alone has well over a hundred positions.
assert len(CELL_IDS) >= 9300
assert len(POSITIONS) >= 465
assert len([cell_id for cell_id in CELL_IDS if cell_id.startswith("pair:")]) == 11 * 10 * 10


# One deliberate departure from "loads exactly when it loaded before". The
# loader now reads the plural list's threshold_ms (opensrm-wo62.4.24), and
# a declared threshold must be a finite positive number wherever it is
# declared (opensrm-wo62.4.7). On ec44215 the key was not read, so any
# value there loaded; a value that is no threshold is now refused. The
# cells are derived from the wrong values, not from what the loader does.
LIST_THRESHOLD_POSITIONS = (
    "full:resources.10.spec.indicators.0.threshold_ms",
    "full+overlay:resources.10.spec.indicators.0.threshold_ms",
)


def _is_a_threshold(value: Any) -> bool:
    """None (undeclared) or a finite positive number. No string among the
    wrong values spells a number, so strings need no case here."""
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return 0 < float(value) < float("inf")
    except OverflowError:
        return False


REFUSED_SINCE_WO62_4_24 = {
    f"{position}={kind}"
    for position in LIST_THRESHOLD_POSITIONS
    for kind, value in WRONG_VALUES.items()
    if not _is_a_threshold(value)
}
assert set(CELL_IDS) >= REFUSED_SINCE_WO62_4_24

# A second one, of the same kind (opensrm-1lhl): an SLO's target is a
# finite number. On ec44215 a boolean loaded as 1.0 or 0.0, NaN loaded as
# NaN, and a falsy ``objective`` (an empty string, list, mapping, set or
# binary, or false) dropped the SLO and loaded the rest. Those are refused
# now; 0 is a number and still loads. The positions are every ``objective``
# or ``target`` of a classical SLO that reaches the loader: not the two
# judgment thresholds (resources 2 and 3 of FULL, lifted by srm/v1's
# rules), and not an objective the unmutated overlay replaces
# (``availability``, in the "+overlay" cells of both documents).
TARGET_POSITIONS = (
    "full:resources.0.spec.objective",
    "full:resources.1.spec.target",
    "full:resources.8.spec.target",
    "full:resources.9.spec.target",
    "full:resources.10.spec.objective",
    "full+overlay:resources.1.spec.target",
    "full+overlay:resources.8.spec.target",
    "full+overlay:resources.9.spec.target",
    "full+overlay:resources.10.spec.objective",
    "templated:resources.0.spec.objective",
    "overlay:resources.0.spec.objective",
    "overlay:resources.1.spec.objective",
)


def _is_a_target(value: Any) -> bool:
    """None (undeclared) or a finite number, of any sign and 0 included."""
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return float("-inf") < float(value) < float("inf")
    except OverflowError:
        return False


REFUSED_SINCE_1LHL = {
    f"{position}={kind}"
    for position in TARGET_POSITIONS
    for kind, value in WRONG_VALUES.items()
    if not _is_a_target(value)
}
assert set(CELL_IDS) >= REFUSED_SINCE_1LHL


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
    """"loads" or "refused" (one of the three contract errors, raised by a
    check that names the position). Anything else that gets out fails the
    test, naming the cell; so does a refusal by the backstop."""
    try:
        load_cell(cell_id, tmp_path)
    except (OSError, ManifestLoadError, ValueError) as exc:
        if getattr(exc, "backstop", False):
            pytest.fail(f"{cell_id} was refused by the backstop, not by a check: {exc}")
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
    assert [slo.name for slo in manifest.slos] == [
        "availability", "latency-p99", "latency-p95", "duration-p50", "success", "prod-only",
    ]
    assert manifest.slos[0].target == 99.0
    assert [d.name for d in manifest.dependencies if d.type == "database"] == ["replica"]
    assert manifest.ownership.pagerduty.urgency == "low"
    assert manifest.ownership.pagerduty.timezone == "UTC"


def test_the_overlay_also_merges_into_the_templated_document(tmp_path):
    path = _write(tmp_path, TEMPLATED, OVERLAY)

    manifest = load_manifest(path, environment=ENVIRONMENT, suppress_deprecation_warning=True)

    assert manifest.tier == "high"
    assert "prod-only" in [slo.name for slo in manifest.slos]
    assert {slo.name: slo.target for slo in manifest.slos}["availability"] == 99.0


def test_the_full_document_exercises_every_resource_kind(tmp_path):
    # If a mutation target is not read, its cells prove nothing.
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(FULL, sort_keys=False))

    manifest = load_manifest(path, suppress_deprecation_warning=True)

    assert [(slo.name, slo.threshold_ms) for slo in manifest.slos] == [
        ("availability", None),
        ("latency-p99", 250),  # indicator.threshold_ms, over the two below it
        ("latency-p95", 300),  # threshold_ms alone
        ("duration-p50", 400),  # latency_threshold alone
        ("success", 350),  # indicators[].threshold_ms (opensrm-wo62.4.24)
    ]
    assert [slo.slo_type for slo in manifest.slos] == [
        "availability",  # indicator.type
        "latency",  # indicator.type
        "latency",  # the name
        "latency",  # the name
        "throughput",  # indicators[].type: nothing else says so
    ]
    assert manifest.slos[4].good_query == "sum(rate(done[5m]))"
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


def test_the_departure_is_the_unusable_list_thresholds_that_used_to_load():
    """Not vacuous, and no wider than its rule: 18 wrong values that loaded
    when the key was unread, at each of the two positions; None and the
    two real numbers (7 and 1.5) still load."""
    loaded_before = set(json.loads(LOADED_BEFORE.read_text())["loaded"])
    still = {
        f"{position}={kind}"
        for position in LIST_THRESHOLD_POSITIONS
        for kind in ("none", "number", "float")
    }

    assert len(REFUSED_SINCE_WO62_4_24 & loaded_before) == 36
    assert still <= loaded_before - REFUSED_SINCE_WO62_4_24


def test_the_target_departure_is_the_non_numbers_that_used_to_load():
    """Not vacuous, and no wider than its rule. At an ``objective``: true,
    false, NaN and the five falsy non-numbers that dropped the SLO (8). At
    a ``target``, which nothing fell back from: true, false and NaN (3).
    None, 0 and the two real numbers still load everywhere."""
    loaded_before = set(json.loads(LOADED_BEFORE.read_text())["loaded"])
    departed = collections.Counter(
        cell_id.rsplit("=", 1)[0] for cell_id in REFUSED_SINCE_1LHL & loaded_before
    )
    still = {
        f"{position}={kind}"
        for position in TARGET_POSITIONS
        for kind in ("none", "zero", "number", "float")
    }

    assert departed == {
        position: 8 if position.endswith(".objective") else 3 for position in TARGET_POSITIONS
    }
    assert still <= loaded_before - REFUSED_SINCE_1LHL


def test_the_baseline_names_only_cells_of_this_matrix():
    loaded_before = set(json.loads(LOADED_BEFORE.read_text())["loaded"])

    assert loaded_before
    assert loaded_before <= set(CELL_IDS)


@pytest.mark.parametrize("position", POSITIONS)
def test_every_wrong_value_is_refused_or_loads_as_it_did_before(position, tmp_path):
    # The contract and the baseline in one load per cell: a cell either
    # loads or is refused with OSError, ManifestLoadError or ValueError
    # (never anything else), and it loads exactly when it loaded before
    # (nothing that loaded stops loading, nothing that failed starts to),
    # but for the cells of REFUSED_SINCE_WO62_4_24 and REFUSED_SINCE_1LHL.
    loaded_before = set(json.loads(LOADED_BEFORE.read_text())["loaded"])
    loads = loaded_before - REFUSED_SINCE_WO62_4_24 - REFUSED_SINCE_1LHL
    expected = {
        cell_id: "loads" if cell_id in loads else "refused" for cell_id in CELLS_AT[position]
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
    ("full:resources.8.spec.threshold_ms=huge-int", "latency-p95"),
    ("full:resources.9.spec.latency_threshold=list", "duration-p50"),
    ("full:resources.10.spec.indicators.0.type=number", "success"),
    ("full:resources.10.spec.indicators=string", "success"),
    ("full+overlay:resources.0.spec=number", "availability"),
    ("full+overlay:resources.7.spec=list", "pagerduty"),
]


@pytest.mark.parametrize(("cell_id", "named"), NAMED)
def test_the_error_names_the_resource_or_the_position(cell_id, named, tmp_path):
    with pytest.raises(ManifestLoadError) as raised:
        load_cell(cell_id, tmp_path)

    assert named in str(raised.value)


# The backstop itself. No shape in the matrix reaches it, so the error is
# planted where a missed check would raise it: inside the legacy parse.
@pytest.mark.parametrize("leaked", [AttributeError, TypeError])
def test_an_unchecked_wrong_type_ends_in_the_backstop(leaked, tmp_path, monkeypatch):
    def parse(*args, **kwargs):
        raise leaked("'int' object has no attribute 'get'")

    monkeypatch.setattr(loader, "_parse_legacy_to_manifest", parse)
    path = _write(tmp_path, FULL, None)

    with pytest.raises(ManifestLoadError) as raised:
        load_manifest(path, suppress_deprecation_warning=True)

    assert raised.value.backstop is True
    assert isinstance(raised.value.__cause__, leaked)
    assert loader.LEGACY_BACKSTOP_MARKER in str(raised.value)


def test_a_refusal_by_a_check_is_not_a_backstop_error(tmp_path):
    with pytest.raises(ManifestLoadError) as raised:
        load_cell("full:resources.0.spec=number", tmp_path)

    assert raised.value.backstop is False
    assert loader.LEGACY_BACKSTOP_MARKER not in str(raised.value)


def test_the_matrix_fails_a_cell_the_backstop_refuses(tmp_path, monkeypatch):
    def parse(*args, **kwargs):
        raise TypeError("unhashable type: 'list'")

    monkeypatch.setattr(loader, "_parse_legacy_to_manifest", parse)

    with pytest.raises(pytest.fail.Exception):
        outcome("full:resources.0.spec=number", tmp_path)


# What only YAML text can say: the matrix replaces nodes with Python values,
# so it cannot nest a file past the YAML reader's stack, and it writes a
# self-reference only as the alias yaml.safe_dump chooses.
SERVICE_TEXT = "service: {name: deep, team: t, tier: standard, type: api}\n"
DEPTH = 2000  # PyYAML gives out a few hundred levels down


def _nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def test_a_manifest_nested_too_deeply_for_the_yaml_reader_is_refused(tmp_path):
    path = tmp_path / "service.yaml"
    path.write_text(
        SERVICE_TEXT
        + "resources:\n"
        + "  - kind: SLO\n    name: availability\n"
        + f"    spec: {{objective: 99, indicator: {_nested(DEPTH)}}}\n"
    )

    with pytest.raises(ManifestLoadError):
        load_manifest(path, suppress_deprecation_warning=True)


def test_an_overlay_nested_too_deeply_for_the_yaml_reader_is_refused(tmp_path):
    path = tmp_path / "service.yaml"
    path.write_text(SERVICE_TEXT)
    (tmp_path / "environments").mkdir()
    (tmp_path / "environments" / "prod.yaml").write_text(
        f"service: {{description: {_nested(DEPTH)}}}\n"
    )

    with pytest.raises(ManifestLoadError):
        load_manifest(path, environment="prod", suppress_deprecation_warning=True)


@pytest.mark.parametrize(
    "resource",
    [
        "  - kind: SLO\n    name: availability\n    spec: &a {objective: 99, indicator: *a}\n",
        "  - kind: Dependencies\n    name: deps\n    spec: {databases: &a [*a]}\n",
        "  - &a {kind: Alerts, name: alerts, spec: {again: *a}}\n",
    ],
)
def test_an_alias_to_its_own_anchor_is_refused(resource, tmp_path):
    path = tmp_path / "service.yaml"
    path.write_text(SERVICE_TEXT + "resources:\n" + resource)

    with pytest.raises(ManifestLoadError):
        load_manifest(path, suppress_deprecation_warning=True)


def test_nesting_the_yaml_reader_can_hold_still_loads(tmp_path):
    # The refusals above are about running out of stack, not about depth as
    # such: a description nested 20 deep loaded before and still does.
    path = tmp_path / "service.yaml"
    path.write_text(
        "service: {name: deep, team: t, tier: standard, type: api, "
        f"description: {_nested(20)}}}\n"
    )

    assert load_manifest(path, suppress_deprecation_warning=True).name == "deep"
