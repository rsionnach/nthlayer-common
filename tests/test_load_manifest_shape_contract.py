"""load_manifest's contract on a malformed srm/v1 or v2 manifest (opensrm-wo62.4.28).

Whatever the shape of the input, ``load_manifest`` returns a manifest or
raises ``OSError``, ``ManifestLoadError`` or ``ValueError``.
opensrm-wo62.4.10 made that hold for the legacy format
(tests/test_legacy_loader_shape_contract.py); the srm/v1 and v2 branches
still let ``AttributeError``, ``TypeError``, ``KeyError`` and
``OverflowError`` out: ``metadata: []`` ended in "'list' object has no
attribute 'get'".

The cases are generated, not picked, and the positions come from documents
this parser did not write:

- v2: every ``kind: ServiceManifest`` example the spec ships, the template
  one of them extends, and the five JudgmentSLO examples, copied unchanged
  (fixtures/load_shape_matrix/README.md). Most of them do not load (open
  beads on the schema-versus-parser seam), so three documents below add a
  loading path to every block; ``test_every_schema_position_is_in_a_document``
  holds the set to the schema: each property spec/v2/schema.json declares
  for a ServiceManifest is in at least one document.
- srm/v1 has no schema. Its positions are the six example manifests
  nthlayer-generate ships, the four demo manifests
  (fixtures/demo-specs), and one document holding the keys the format's
  parser reads that no example uses (on-call, deployment gates and policy,
  audit, the scalar SLO form).

Each node of each document is replaced, one at a time, by each value of
``WRONG_VALUES`` (the legacy matrix's table: one of every type YAML can
build, and the empty or zero value of each). The files a manifest refers
to (its template, an OpenSLO ``$ref`` target) are mutated the same way
under an unmutated manifest.

No cell may be refused by the backstop (``ManifestLoadError.backstop``):
that error keeps the contract for a shape nothing here reaches, and a cell
that ends in it is a position whose check is missing.

The second half pins that nothing which loaded before this contract stopped
loading, and nothing that failed started to load:
``fixtures/load_shape_matrix/loaded_before.json`` lists the cells that
loaded on the commit before the fix. One later departure is listed beside
that test: v2 availability ratios (opensrm-f0co).

Cells are fed to the loader in place of what ``yaml.safe_load`` returns for
the file (writing 40,000 YAML files takes minutes).
``test_a_fed_cell_is_what_the_file_would_load`` checks, on real files, that
this is the same input.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_legacy_loader_shape_contract import WRONG_VALUES, _paths, _replaced

from nthlayer_common.manifest import ManifestLoadError, load_manifest
from nthlayer_common.manifest.parser import loader

FIXTURES = Path(__file__).parent / "fixtures" / "load_shape_matrix"
V2_EXAMPLES = FIXTURES / "v2-spec-examples"
V1_EXAMPLES = FIXTURES / "srm-v1-examples"
DEMO_SPECS = Path(__file__).parent / "fixtures" / "demo-specs"
LOADED_BEFORE = FIXTURES / "loaded_before.json"


def _read(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


# --------------------------------------------------------------------------
# v2 documents
# --------------------------------------------------------------------------

_QUERY = "sum(rate(http_requests_total[5m]))"


def _metric(query: str) -> dict[str, Any]:
    return {"metricSource": {"type": "prometheus", "spec": {"query": query}}}


def _openslo(name: str, indicator_spec: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": "openslo/v1",
        "kind": "SLO",
        "metadata": {"name": name, "displayName": name},
        "spec": {
            "service": "checkout",
            "indicator": {"metadata": {"name": f"{name}-p99"}, "spec": indicator_spec},
            "timeWindow": [{"duration": "28d", "isRolling": True}],
            "objectives": [objective],
        },
    }


# An api service with every block the schema gives a ServiceManifest, in a
# form that loads. The OpenSLO documents carry each indicator OpenSLO v1
# defines: ratioMetric as good/total, bad/total and raw, as the vendored
# excerpt of its spec has them (fixtures/openslo/ratio-metric-v1.md), and
# thresholdMetric, with ``timeWindow`` and the objective's ``op``, ``value``
# and ``targetPercent``, as the second excerpt has them
# (fixtures/openslo/slo-objectives-threshold-v1.md).
# ``outcomes`` and ``observability`` are not in the schema; the parser
# reads them.
V2_FULL: dict[str, Any] = {
    "apiVersion": "opensrm.nthlayer.io/v2",
    "kind": "ServiceManifest",
    "metadata": {
        "name": "checkout",
        "namespace": "shop",
        "annotations": {"backstage.io/techdocs-ref": "dir:."},
        "labels": {"tier": "critical", "domain": "commerce"},
    },
    "spec": {
        "owner": {
            "group": "group:default/payments",
            "escalation": "group:default/payments-oncall",
            "technical_contact": "user:default/payments-lead",
        },
        "service": {"name": "checkout", "type": "api", "description": "Checkout API"},
        "slo": [
            _openslo(
                "checkout-availability",
                {"ratioMetric": {"good": _metric(_QUERY), "total": _metric(_QUERY)}},
                {"displayName": "99.9% of requests succeed", "target": 0.999},
            ),
            _openslo(
                "checkout-latency",
                {"thresholdMetric": _metric(f"histogram_quantile(0.99, {_QUERY})")},
                {"displayName": "under 300 ms", "op": "lte", "value": 0.3, "targetPercent": 99},
            ),
            _openslo(
                "checkout-error-rate",
                {"ratioMetric": {"bad": _metric(_QUERY), "total": _metric(_QUERY)}},
                {"displayName": "errors under 0.1%", "target": 0.999},
            ),
            _openslo(
                "checkout-availability-raw",
                {"ratioMetric": {"raw": _metric(_QUERY), "rawType": "success"}},
                {"displayName": "precomputed ratio", "target": 0.99},
            ),
        ],
        "contracts": [
            {
                "name": "checkout-api",
                "api_ref": {
                    "openapi": "./api/checkout.yaml",
                    "asyncapi": "./api/checkout-events.yaml",
                    "operation_id": "createOrder",
                },
                "promise": {"availability": 0.999, "latency_p99": "300ms", "throughput": "500rps"},
                "conditions": ["authenticated requests"],
                "breach_semantics": {
                    "consumer_notification": "status page",
                    "compensation": {"description": "service credits"},
                    "sla_credits": {"enabled": True, "calculation_ref": "./sla/credits.md"},
                },
                "contract_owner": "group:default/payments",
            }
        ],
        "dependencies": [
            {
                "service": "resource:default/orders-db",
                "contract_ref": "./contracts/orders-db.yaml",
                "expected_availability": 0.9995,
                "expected_latency_p99": "50ms",
                "fallback": {"type": "fail_closed", "description": "no store, no order"},
                "type": "database",
                "criticality": "critical",
            }
        ],
        "instrumentation": {
            "required_metrics": [
                {"name": "http_requests_total", "type": "counter", "labels": ["method"]}
            ],
            "required_traces": [
                {
                    "span_name": "checkout",
                    "required_attributes": ["order.id"],
                    "emitted_by": "application",
                }
            ],
            "required_logs": [{"format": "structured", "required_fields": ["trace_id"]}],
            "required_events": [{"type": "order.created", "schema_ref": "./events/order.json"}],
        },
        "outcomes": {
            "decision_value": {
                "correct": 1.5,
                "currency": "EUR",
                "false_positive": {"cost": 20, "category": "financial_loss"},
                "false_negative": {"cost": 5, "category": "friction"},
            },
            "revenue": {"attribution": "direct", "signal": "orders_total"},
            "volume": {"estimated_daily_decisions": 1000, "peak_multiplier": 2.0},
        },
        "observability": {
            "metrics_prefix": "checkout",
            "logs_label": "app=checkout",
            "traces_service": "checkout",
            "prometheus_job": "checkout",
            "grafana_url": "https://grafana.example.com",
            "labels": {"env": "prod"},
        },
    },
}


def _judgment_slos() -> list[dict[str, Any]]:
    """The spec's five JudgmentSLO example documents, each with one breach
    action added (the examples declare none; one of each kind the schema's
    BreachAction defines)."""
    actions = [
        {"notify": "slack:#trust-and-safety"},
        {"create_case": {"priority": "P1", "template": "judgment-breach"}},
        {"reduce_autonomy": {"agent": "refund-approver", "new_autonomy_level": "supervised"}},
        {"action_request": {"action_id": "pause-agent", "parameters": {"reason": "breach"}}},
        {"notify": "pagerduty:trust-and-safety"},
    ]
    documents = []
    for path, action in zip(sorted((V2_EXAMPLES / "judgment-slos").glob("*.yaml")), actions, strict=True):
        document = _read(path)
        document["spec"]["breach_actions"] = [action]
        documents.append(document)
    return documents


# An ai-gate: the judgment SLOs in the wrapped ``kind: JudgmentSLO`` form
# this parser reads (the schema's flat inline item is services/ai-gate.yaml,
# among the spec examples; opensrm-a742), a judgment promise on a contract,
# and thresholds.
V2_AI_GATE: dict[str, Any] = {
    "apiVersion": "opensrm.nthlayer.io/v2",
    "kind": "ServiceManifest",
    "metadata": {"name": "refund-approver", "labels": {"tier": "critical"}},
    "spec": {
        "owner": {"group": "group:default/trust-and-safety"},
        "service": {"name": "refund-approver", "type": "ai-gate"},
        "judgment_slo": _judgment_slos(),
        "judgment_threshold": _read(V2_EXAMPLES / "judgment-thresholds" / "all-five-types.yaml")[
            "spec"
        ]["judgment_threshold"],
        "contracts": [
            {
                "name": "refund-decisions",
                "promise": {"availability": 0.999, "judgment": {"reversal_rate": 0.05}},
            }
        ],
    },
}



def _schema_form_judgment_slos() -> list[dict[str, Any]]:
    """The same five, as the schema's JudgmentSLOItem has them: the flat
    JudgmentSLOSpec, without the document wrapper."""
    items = [document["spec"] for document in _judgment_slos()]
    # Decision 3c moved this bound to judgment_threshold; the schema still
    # declares the position, as one it refuses.
    items[2]["target"]["audit_backlog_maximum_age"] = "24h"
    return items


# The positions the schema declares and no loading document can hold: the
# flat judgment SLO item (this parser reads the wrapped form), and the
# keys the schema declares on a threshold in order to refuse them
# (opensrm-l33e). The parser refuses this document; its cells must still
# be refusals, not crashes.
V2_SCHEMA_FORM: dict[str, Any] = {
    "apiVersion": "opensrm.nthlayer.io/v2",
    "kind": "ServiceManifest",
    "metadata": {"name": "refund-approver", "labels": {"tier": "critical"}},
    "spec": {
        "owner": {"group": "group:default/trust-and-safety"},
        "service": {"name": "refund-approver", "type": "ai-gate"},
        "judgment_slo": _schema_form_judgment_slos(),
        "judgment_threshold": [
            {
                "type": "calibration",
                "measurement": {"bins": 10, "window": "30d", "method": "ece"},
                "bounds": {"maximum_brier_score": 0.05},
                "breach_actions": [{"notify": "slack:#trust-and-safety"}],
                "target": {"maximum_brier_score": 0.05},
                "judgment_type": "calibration",
                "service": "refund-approver",
            }
        ],
    },
}

# A manifest that extends a template, with each override directive, and
# the template it extends. Both sides are mutated.
V2_TEMPLATE: dict[str, Any] = {
    "apiVersion": "opensrm.nthlayer.io/v2",
    "kind": "ServiceManifestTemplate",
    "metadata": {"name": "ai-gate-base", "namespace": "default"},
    "spec": {
        "owner": {"group": "group:default/platform"},
        "service": {"description": "from the template"},
        "contracts": [{"name": "base", "promise": {"availability": 0.99}}],
        "dependencies": [{"service": "component:default/identity"}],
        "instrumentation": {
            "required_metrics": [{"name": "decisions_total", "type": "counter"}]
        },
        "judgment_threshold": [
            {
                "type": "calibration",
                "measurement": {"bins": 10, "window": "30d"},
                "bounds": {"maximum_brier_score": 0.05},
            }
        ],
    },
}
V2_TEMPLATED: dict[str, Any] = {
    "apiVersion": "opensrm.nthlayer.io/v2",
    "kind": "ServiceManifest",
    "metadata": {"name": "fraud-gate", "labels": {"tier": "high"}},
    "spec": {
        "owner": {"group": "group:default/fraud"},
        "service": {"name": "fraud-gate", "type": "ai-gate"},
        "judgment_threshold": [
            {"type": "feedback_latency", "bounds": {"maximum_p50": "1h"}}
        ],
        "template": {
            "extends": "template:default/ai-gate-base",
            "overrides": {
                "dependencies": {"append": [{"service": "component:default/ledger"}]},
                "contracts": {
                    "replace": [{"name": "fraud", "promise": {"availability": 0.999}}]
                },
                "judgment_threshold": {
                    "append": [
                        {"type": "audit_backlog", "bounds": {"maximum_age": "24h"}}
                    ]
                },
            },
        },
    },
}

# A manifest whose one SLO is an OpenSLO document in a file of its own
# (``$ref``), and that file. The spec's examples refer to ``./slos/*.yaml``
# files it does not ship, so none of them reaches a referenced document.
V2_REF_TARGET: dict[str, Any] = _openslo(
    "checkout-availability",
    {"ratioMetric": {"good": _metric(_QUERY), "total": _metric(_QUERY)}},
    {"displayName": "99.9% of requests succeed", "target": 0.999},
)
V2_REF: dict[str, Any] = {
    "apiVersion": "opensrm.nthlayer.io/v2",
    "kind": "ServiceManifest",
    "metadata": {"name": "checkout", "labels": {"tier": "critical"}},
    "spec": {
        "owner": {"group": "group:default/payments"},
        "service": {"name": "checkout", "type": "api"},
        "slo": [{"$ref": "./slos/checkout-availability.yaml"}],
    },
}

# --------------------------------------------------------------------------
# srm/v1 documents
# --------------------------------------------------------------------------

# The keys parser/v1.py reads that none of the ten example manifests uses.
# There is no srm/v1 schema to take them from; this is the one document
# here written from the parser.
V1_FULL: dict[str, Any] = {
    "apiVersion": "srm/v1",
    "kind": "ServiceReliabilityManifest",
    "metadata": {
        "name": "checkout",
        "team": "payments",
        "tier": "critical",
        "description": "Checkout API",
        "labels": {"domain": "commerce"},
        "annotations": {"owner": "payments"},
    },
    "spec": {
        "type": "api",
        "slos": {
            "availability": {
                "target": 99.9,
                "window": "30d",
                "unit": "percent",
                "description": "Requests that succeed",
                "labels": {"class": "gold"},
                "indicator": {"type": "availability", "query": _QUERY},
            },
            "latency_p99": 250,
            "saturation": {"minimum": 80, "type": "throughput", "percentile": 99},
        },
        "dependencies": [
            {
                "name": "orders-db",
                "type": "database",
                "critical": True,
                "criticality": "critical",
                "database_type": "postgresql",
                "manifest": "./orders-db.reliability.yaml",
                "slo": {"availability": 99.95, "latency_p99": "50ms", "latency": {"p99": "50ms"}},
            },
            {
                "name": "payment-api",
                "expected_availability": 99.9,
                "expected_latency_p99": "200ms",
            },
        ],
        "ownership": {
            "team": "payments",
            "slack": "#payments",
            "email": "payments@example.com",
            "escalation": "payments-oncall",
            "runbook": "https://runbooks.example.com/checkout",
            "documentation": "https://docs.example.com/checkout",
            "pagerduty": {"service_id": "PSVC1", "escalation_policy_id": "PESC1"},
            "oncall": {
                "timezone": "Europe/Dublin",
                "rotation": {
                    "type": "weekly",
                    "handoff": "monday 09:00",
                    "roster": [
                        {
                            "name": "Aoife",
                            "slack_id": "U1",
                            "ntfy_topic": "aoife",
                            "phone": "+353000000",
                        }
                    ],
                },
                "overrides": [
                    {
                        "start": "2026-12-24T00:00:00Z",
                        "end": "2026-12-27T00:00:00Z",
                        "user": "Aoife",
                        "reason": "holiday swap",
                    }
                ],
                "escalation": [
                    {"after": "5m", "notify": "slack_dm", "target": "U1", "phone": "+353000000"}
                ],
            },
        },
        "observability": {
            "metrics_prefix": "checkout",
            "logs_label": "app=checkout",
            "traces_service": "checkout",
            "prometheus_job": "checkout",
            "grafana_url": "https://grafana.example.com",
            "labels": {"env": "prod"},
        },
        "deployment": {
            "environments": ["staging", "production"],
            "gates": {
                "error_budget": {
                    "enabled": True,
                    "threshold": 0.1,
                    "policy": {
                        "window": "30d",
                        "thresholds": {"warning": 0.2, "critical": 0.1},
                        "on_exhausted": ["freeze_deploys", "notify"],
                    },
                },
                "slo_compliance": {"threshold": 0.99},
                "recent_incidents": {"p1_max": 0, "p2_max": 2, "lookback": "7d"},
            },
            "rollback": {
                "automatic": True,
                "criteria": {"error_rate_increase": "5%", "latency_increase": "50%"},
            },
            "audit": {"enabled": True, "retention_days": 90},
        },
        "instrumentation": {
            "telemetry_events": [{"name": "order.created", "fields": ["order_id"]}],
            "feedback_loop": "weekly review",
            "ground_truth_source": "ledger",
        },
        "contract": {"availability": 99.5, "latency": {"p99": "300ms"}},
        "alerting": {"for": "5m", "channels": ["#payments-alerts"]},
    },
}

# srm/v1 template inheritance (parser/v1.py, resolve_template): the
# template's spec is merged under the manifest's.
V1_TEMPLATE: dict[str, Any] = {
    "apiVersion": "srm/v1",
    "kind": "ServiceReliabilityManifest",
    "metadata": {"name": "standard-api"},
    "spec": {
        "type": "api",
        "slos": {"availability": {"target": 99.5}, "latency_p99": {"target": 300, "unit": "ms"}},
        "ownership": {"team": "platform", "slack": "#platform"},
        "deployment": {"environments": ["production"]},
    },
}
V1_TEMPLATED: dict[str, Any] = {
    "apiVersion": "srm/v1",
    "kind": "ServiceReliabilityManifest",
    "metadata": {"name": "orders", "team": "commerce", "tier": "standard", "template": "standard-api"},
    "spec": {
        "type": "api",
        "slos": {"availability": {"target": 99.9}},
        "ownership": {"team": "commerce"},
    },
}

# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------

# Document id -> (path under the work directory, document, and the path
# and content of the file it refers to (a template, a $ref target) or None).
INLINE: dict[str, tuple[str, dict[str, Any], tuple[str, dict[str, Any]] | None]] = {
    "v2-full": ("inline/v2-full/service.yaml", V2_FULL, None),
    "v2-ai-gate": ("inline/v2-ai-gate/service.yaml", V2_AI_GATE, None),
    "v2-templated": (
        "inline/v2-templated/service.yaml",
        V2_TEMPLATED,
        ("inline/v2-templated/templates/ai-gate-base.yaml", V2_TEMPLATE),
    ),
    "v2-schema-form": ("inline/v2-schema-form/service.yaml", V2_SCHEMA_FORM, None),
    "v2-ref": (
        "inline/v2-ref/service.yaml",
        V2_REF,
        ("inline/v2-ref/slos/checkout-availability.yaml", V2_REF_TARGET),
    ),
    "v1-full": ("inline/v1-full/service.yaml", V1_FULL, None),
    "v1-templated": (
        "inline/v1-templated/service.yaml",
        V1_TEMPLATED,
        ("inline/v1-templated/templates/standard-api.yaml", V1_TEMPLATE),
    ),
}

V2_SPEC_MANIFESTS = sorted(
    path
    for path in V2_EXAMPLES.rglob("*.yaml")
    if path.parent.name not in ("judgment-slos", "templates")
)
V1_EXAMPLE_MANIFESTS = sorted(V1_EXAMPLES.glob("*.yaml")) + sorted(DEMO_SPECS.glob("*.yaml"))
# A missed glob would leave a matrix of the inline documents alone.
assert len(V2_SPEC_MANIFESTS) == 17
assert len(V1_EXAMPLE_MANIFESTS) == 10


def _documents() -> dict[str, tuple[str, dict[str, Any]]]:
    """Document id -> (path under the work directory, document)."""
    documents = {doc_id: (rel, document) for doc_id, (rel, document, _) in INLINE.items()}
    for path in V2_SPEC_MANIFESTS:
        rel = path.relative_to(V2_EXAMPLES)
        documents[f"v2-spec:{rel.with_suffix('')}"] = (f"v2-spec-examples/{rel}", _read(path))
    for path in sorted(V1_EXAMPLES.glob("*.yaml")):
        documents[f"v1-example:{path.name.split('.')[0]}"] = (
            f"srm-v1-examples/{path.name}",
            _read(path),
        )
    for path in sorted(DEMO_SPECS.glob("*.yaml")):
        documents[f"v1-demo:{path.stem}"] = (f"demo-specs/{path.name}", _read(path))
    return documents


DOCUMENTS = _documents()
DOC_IDS = sorted(DOCUMENTS)
# Seven inline documents, the spec's 17 and the ten srm/v1 manifests.
assert len(DOC_IDS) == 34

# Id -> (the manifest that refers to the file, the file's path under the
# work directory, its content): the templates, and the $ref target.
REFERRED_FILES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "v2-ref/slo": ("v2-ref", *INLINE["v2-ref"][2]),
    "v2-templated/template": ("v2-templated", *INLINE["v2-templated"][2]),
    "v1-templated/template": ("v1-templated", *INLINE["v1-templated"][2]),
    "v2-spec:template-extension/template": (
        "v2-spec:template-extension",
        "v2-spec-examples/templates/payment-services-base.yaml",
        _read(V2_EXAMPLES / "templates" / "payment-services-base.yaml"),
    ),
}


def _cells() -> dict[str, tuple[str, str, dict[str, Any]]]:
    """Cell id -> (the manifest to load, the file whose content is
    replaced, what replaces it): every position of every document, and of
    every file one refers to, crossed with every wrong value."""
    cells: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for doc_id, (rel, document) in DOCUMENTS.items():
        for path in _paths(document):
            where = ".".join(str(key) for key in path)
            for kind, value in WRONG_VALUES.items():
                cells[f"{doc_id}:{where}={kind}"] = (rel, rel, _replaced(document, path, value))
    for template_id, (doc_id, template_rel, template) in REFERRED_FILES.items():
        manifest_rel = DOCUMENTS[doc_id][0]
        for path in _paths(template):
            where = ".".join(str(key) for key in path)
            for kind, value in WRONG_VALUES.items():
                cells[f"{template_id}:{where}={kind}"] = (
                    manifest_rel,
                    template_rel,
                    _replaced(template, path, value),
                )
    return cells


CELLS = _cells()
CELL_IDS = sorted(CELLS)
# One test per position (its cells differ only in the wrong value).
CELLS_AT: dict[str, list[str]] = {}
for _cell_id in CELL_IDS:
    CELLS_AT.setdefault(_cell_id.rsplit("=", 1)[0], []).append(_cell_id)
POSITIONS = sorted(CELLS_AT)

# The matrix is generated at import; an empty one would collect no tests
# and pass.
WRONG_KINDS = sorted(WRONG_VALUES)
assert len(WRONG_KINDS) == 23
assert len(POSITIONS) >= 1500
assert len(CELL_IDS) == len(POSITIONS) * len(WRONG_KINDS)


def build_work(root: Path) -> Path:
    """The fixture trees and the inline documents, as files under *root*:
    load_manifest needs a path that exists, and looks for templates beside
    it. Also what the baseline was recorded from (see the fixtures' README)."""
    root = root.resolve()
    shutil.copytree(V2_EXAMPLES, root / "v2-spec-examples")
    shutil.copytree(V1_EXAMPLES, root / "srm-v1-examples")
    shutil.copytree(DEMO_SPECS, root / "demo-specs")
    for rel, document, template in INLINE.values():
        files = [(rel, document)] + ([template] if template else [])
        for file_rel, content in files:
            path = root / file_rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(content, sort_keys=False))
    return root


@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return build_work(tmp_path_factory.mktemp("load-shape-matrix"))


def _load(manifest: Path, fed: dict[Path, Any] | None = None) -> Any:
    """load_manifest(manifest), with each file in *fed* read as the value
    given instead of its content."""
    fed = fed or {}
    real_safe_load = yaml.safe_load

    def safe_load(stream: Any) -> Any:
        name = getattr(stream, "name", None)
        if name is not None and Path(name) in fed:
            return copy.deepcopy(fed[Path(name)])
        return real_safe_load(stream)

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch.setattr(yaml, "safe_load", safe_load)
        return load_manifest(manifest)


def load_cell(cell_id: str, work: Path) -> Any:
    manifest_rel, fed_rel, content = CELLS[cell_id]
    return _load(work / manifest_rel, {work / fed_rel: content})


def classify(load: Any) -> tuple[str, BaseException | None]:
    """("loads", None) or ("refused", the contract error). Anything else
    *load* raises is outside the contract and is not caught here."""
    try:
        load()
    except (OSError, ManifestLoadError, ValueError) as exc:
        return "refused", exc
    return "loads", None


def outcome(cell_id: str, work: Path) -> str:
    """"loads" or "refused" by a check that names the position. Anything
    else that gets out fails the test, naming the cell; so does a refusal
    by the backstop."""
    try:
        result, exc = classify(lambda: load_cell(cell_id, work))
    except Exception as crash:
        pytest.fail(f"{cell_id} let a {type(crash).__name__} out of load_manifest: {crash}")
    if getattr(exc, "backstop", False):
        pytest.fail(f"{cell_id} was refused by the backstop, not by a check: {exc}")
    return result


# --------------------------------------------------------------------------
# The documents are what they claim to be
# --------------------------------------------------------------------------

# What each unmutated document does. Twelve of the spec's examples do not
# load; the reason is the parser's, and each is an open bead or a file the
# example refers to and the spec does not ship.
LOADS_UNMUTATED = {
    **{doc_id: True for doc_id in INLINE},
    **{doc_id: True for doc_id in DOCUMENTS if doc_id.startswith("v1-")},
    **{doc_id: False for doc_id in DOCUMENTS if doc_id.startswith("v2-spec:")},
    "v2-schema-form": False,
    "v2-spec:judgment-thresholds/all-five-types": True,
    "v2-spec:judgment-thresholds/calibration-both-bounds": True,
    "v2-spec:judgment-thresholds/feedback-latency-p50-only": True,
    "v2-spec:judgment-thresholds/segments-two-dimensions": True,
    "v2-spec:judgment-thresholds/single-bounds": True,
    "v2-spec:migration/api-full-v2": True,
}
assert set(LOADS_UNMUTATED) == set(DOCUMENTS)


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_the_unmutated_documents_load_or_are_refused(doc_id, work):
    result, exc = classify(lambda: _load(work / DOCUMENTS[doc_id][0]))

    assert result == ("loads" if LOADS_UNMUTATED[doc_id] else "refused"), exc


def test_the_inline_documents_reach_every_block(work):
    # If a mutation target is not read, its cells prove nothing.
    full = _load(work / INLINE["v2-full"][0])
    assert [(slo.name, slo.slo_type, slo.target) for slo in full.slos] == [
        ("checkout-availability", "availability", pytest.approx(99.9)),
        ("checkout-latency", "latency", 99.0),
        ("checkout-error-rate", "error_rate", pytest.approx(99.9)),
        ("checkout-availability-raw", "availability", 99.0),
    ]
    assert [slo.window for slo in full.slos] == ["28d"] * 4
    assert full.slos[2].bad_query == _QUERY
    assert full.slos[1].indicator_query.startswith("histogram_quantile")
    assert full.contracts[0].breach_semantics.sla_credits_enabled is True
    assert full.contracts[0].api_ref.operation_id == "createOrder"
    assert full.dependencies[0].fallback.kind == "fail_closed"
    assert full.dependencies[0].slo.latency_p99 == "50ms"
    assert len(full.instrumentation.required_metrics) == 1
    assert len(full.instrumentation.required_traces) == 1
    assert len(full.instrumentation.required_logs) == 1
    assert len(full.instrumentation.required_events) == 1
    assert full.outcomes.volume.peak_multiplier == 2.0
    assert full.observability.prometheus_job == "checkout"
    assert full.ownership.escalation == "payments-oncall"

    gate = _load(work / INLINE["v2-ai-gate"][0])
    assert sorted(slo.judgment_type for slo in gate.slos) == [
        "audit_sampling", "escalation", "high_confidence_failure", "outcomes", "reversal_rate",
    ]
    (audit,) = [slo for slo in gate.slos if slo.judgment_type == "audit_sampling"]
    # The spec's 03-audit-sampling.yaml: overall 0.01, strata 0.1 and 0.25.
    assert audit.measurement.sampling.overall_rate == 0.01
    assert [(s.segment, s.rate) for s in audit.measurement.sampling.stratified] == [
        ("high_stakes", 0.1),
        ("low_confidence", 0.25),
    ]
    assert sorted(a.action_type for slo in gate.slos for a in slo.breach_actions) == [
        "action_request", "create_case", "notify", "notify", "reduce_autonomy",
    ]
    assert len(gate.judgment_thresholds) == 5
    assert [p.judgment_type for p in gate.contracts[0].promise.judgment] == ["reversal_rate"]

    templated = _load(work / INLINE["v2-templated"][0])
    assert templated.description == "from the template"
    assert [c.name for c in templated.contracts] == ["fraud"]
    assert [d.name for d in templated.dependencies] == ["identity", "ledger"]
    assert [t.type for t in templated.judgment_thresholds] == [
        "calibration", "feedback_latency", "audit_backlog",
    ]

    ref = _load(work / INLINE["v2-ref"][0])
    assert [(slo.name, slo.source_ref) for slo in ref.slos] == [
        ("checkout-availability", "./slos/checkout-availability.yaml")
    ]

    v1 = _load(work / INLINE["v1-full"][0])
    assert [(slo.name, slo.slo_type, slo.target) for slo in v1.slos] == [
        ("availability", "availability", 99.9),
        ("latency_p99", "latency", 250.0),
        ("saturation", "throughput", 80.0),
    ]
    assert [d.slo.latency_p99 for d in v1.dependencies] == ["50ms", "200ms"]
    assert v1.ownership.pagerduty.service_id == "PSVC1"
    assert v1.ownership.oncall.rotation.roster[0].name == "Aoife"
    assert v1.ownership.oncall.overrides[0].user == "Aoife"
    assert v1.ownership.oncall.escalation[0].notify == "slack_dm"
    assert v1.deployment.gates.error_budget.policy.on_exhausted == ["freeze_deploys", "notify"]
    # The numbers the documents declare, as read: a parser that returned
    # some other number for them would load every cell just the same.
    budget_thresholds = v1.deployment.gates.error_budget.policy.thresholds
    assert (budget_thresholds.warning, budget_thresholds.critical) == (0.2, 0.1)
    assert v1.deployment.gates.slo_compliance.threshold == 0.99
    assert v1.deployment.gates.recent_incidents.lookback == "7d"
    assert v1.deployment.rollback.latency_increase == "50%"
    assert v1.deployment.audit.retention_days == 90
    assert v1.instrumentation.telemetry_events[0].fields == ["order_id"]
    assert v1.contracts[0].promise.latency_p99 == "300ms"
    assert v1.observability.grafana_url == "https://grafana.example.com"

    v1_templated = _load(work / INLINE["v1-templated"][0])
    assert {slo.name: slo.target for slo in v1_templated.slos} == {
        "availability": 99.9,
        "latency_p99": 300.0,
    }
    assert v1_templated.ownership.slack == "#platform"
    assert v1_templated.deployment.environments == ["production"]


def _schema_positions(schema: dict[str, Any]) -> set[tuple[str, ...]]:
    """Every property position spec/v2/schema.json declares for a
    ServiceManifest, with ``[]`` for a list's items: through ``$ref``,
    ``items`` and each branch of ``allOf``/``oneOf``/``anyOf``/``then``."""
    definitions = schema["definitions"]
    found: set[tuple[str, ...]] = set()

    def walk(node: Any, prefix: tuple[str, ...], seen: tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            return
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name in seen:
                return
            walk(definitions[name], prefix, (*seen, name))
            return
        for key, child in node.get("properties", {}).items():
            found.add((*prefix, key))
            walk(child, (*prefix, key), seen)
        if "items" in node:
            walk(node["items"], (*prefix, "[]"), seen)
        for branches in ("allOf", "oneOf", "anyOf"):
            for branch in node.get(branches, []):
                walk(branch, prefix, seen)
        for branch in ("then", "else"):
            walk(node.get(branch), prefix, seen)

    walk({"$ref": "#/definitions/ServiceManifest"}, (), ())
    return found


def _document_positions(node: Any, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()
    if isinstance(node, dict):
        for key, child in node.items():
            found.add((*prefix, key))
            found |= _document_positions(child, (*prefix, key))
    elif isinstance(node, list):
        for child in node:
            found |= _document_positions(child, (*prefix, "[]"))
    return found


def test_every_schema_position_is_in_a_document():
    schema = json.loads((FIXTURES / "v2-schema.json").read_text())
    declared = _schema_positions(schema)
    present: set[tuple[str, ...]] = set()
    for doc_id, (_, document) in DOCUMENTS.items():
        if doc_id.startswith("v2-"):
            present |= _document_positions(document)

    # The walk found the schema: a ServiceManifest has well over a hundred
    # positions, down to a threshold's bounds.
    assert len(declared) >= 100
    assert ("spec", "contracts", "[]", "breach_semantics", "sla_credits", "enabled") in declared
    assert sorted(declared - present) == []


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left):
        return math.isnan(right)
    return type(left) is type(right) and left == right


@pytest.mark.parametrize("kind", WRONG_KINDS)
def test_a_fed_cell_is_what_the_file_would_load(kind, tmp_path, work):
    """A cell is fed to the loader, not written. For each wrong value: YAML
    can spell it, and a manifest written with it at one position ends as
    the fed cell does."""
    value = WRONG_VALUES[kind]
    text = yaml.safe_dump({"position": value})
    read_back = yaml.safe_load(text)["position"]
    if kind == "recursive-list":
        assert read_back[0] is read_back
    elif kind == "recursive-mapping":
        assert read_back["again"] is read_back
    else:
        assert _same(read_back, value)

    for doc_id, where in (("v2-full", "spec.contracts"), ("v1-full", "spec.dependencies")):
        cell_id = f"{doc_id}:{where}={kind}"
        _, _, content = CELLS[cell_id]
        written = tmp_path / doc_id / "service.yaml"
        written.parent.mkdir()
        written.write_text(yaml.safe_dump(content, sort_keys=False))

        from_file, _ = classify(lambda written=written: _load(written))
        fed, _ = classify(lambda cell_id=cell_id: load_cell(cell_id, work))

        assert from_file == fed, cell_id


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("position", POSITIONS)
def test_a_wrong_shape_is_a_named_error_never_a_crash(position, work):
    for cell_id in CELLS_AT[position]:
        assert outcome(cell_id, work) in ("loads", "refused")


# One deliberate departure from "loads exactly when it loaded before"
# (opensrm-f0co). A contract's ``promise.availability`` and a dependency's
# ``expected_availability`` are schema.json's Ratio: a number from 0 to 1.
# On adaa35b the v2 parser stored whatever was written there, so every
# wrong value loaded; a value that is no Ratio is now refused. The cells
# are derived from the wrong values and the schema's rule, not from what
# the parser does.
#
# The positions are every one of those two fields that a v2 document which
# loads unmutated declares, in the manifest or in what it merges in. Not
# srm/v1's field of the same name (v1-full: its convention is undecided,
# opensrm-ev8r), and not the one in v2-templated's template, whose
# contracts the manifest's ``replace`` override discards unread.
_RATIO_FIELDS = (".promise.availability", ".expected_availability")
_RATIO_NEVER_PARSED = {"v2-templated/template:spec.contracts.0.promise.availability"}


def _ratio_positions() -> list[str]:
    found = []
    for position in POSITIONS:
        owner = position.rsplit(":", 1)[0]
        doc_id = REFERRED_FILES[owner][0] if owner in REFERRED_FILES else owner
        if (
            position.endswith(_RATIO_FIELDS)
            and doc_id.startswith("v2-")
            and LOADS_UNMUTATED[doc_id]
            and position not in _RATIO_NEVER_PARSED
        ):
            found.append(position)
    return found


def _is_a_ratio(value: Any) -> bool:
    """None (undeclared) or a number from 0 to 1 that is not a boolean."""
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return 0 <= value <= 1


def test_what_loaded_before_still_loads_and_nothing_else_does(work):
    loaded_before = {
        f"{position}={kind}"
        for position, kinds in json.loads(LOADED_BEFORE.read_text()).items()
        for kind in kinds.split()
    }
    assert loaded_before <= set(CELL_IDS)
    # The baseline is not empty: thousands of cells loaded before the fix.
    assert len(loaded_before) >= 5000

    loads_now = {
        cell_id
        for cell_id in CELL_IDS
        if classify(lambda cell_id=cell_id: load_cell(cell_id, work))[0] == "loads"
    }

    ratio_positions = _ratio_positions()
    refused_since_f0co = {
        f"{position}={kind}"
        for position in ratio_positions
        for kind, value in WRONG_VALUES.items()
        if not _is_a_ratio(value)
    }
    # Eight positions; of the 23 wrong values only null and 0 are Ratios.
    assert len(ratio_positions) == 8
    assert len(refused_since_f0co) == 8 * 21
    # Each of them did load before: the departure is real, not vacuous.
    assert refused_since_f0co <= loaded_before

    assert sorted(loaded_before - loads_now) == sorted(refused_since_f0co)
    assert sorted(loads_now - loaded_before) == []


# --------------------------------------------------------------------------
# The backstop
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error", [AttributeError, TypeError, KeyError, IndexError, OverflowError]
)
@pytest.mark.parametrize(
    ("doc_id", "parser"), [("v1-full", "parse_srm_v1"), ("v2-full", "parse_opensrm_v2")]
)
def test_a_shape_no_check_names_is_still_a_load_error(doc_id, parser, error, work, monkeypatch):
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise error("a position nothing checks")

    monkeypatch.setattr(loader, parser, broken)

    with pytest.raises(ManifestLoadError) as raised:
        load_manifest(work / DOCUMENTS[doc_id][0])

    assert raised.value.backstop is True
    assert isinstance(raised.value.__cause__, error)


# What only YAML text can say: the matrix replaces nodes with Python values,
# so it cannot nest a file past the YAML reader's stack, and a value that
# contains itself never reaches a walk that follows it. Both are a
# RecursionError inside the parsers, on files they read themselves.
_DEEP = "[" * 3000 + "]" * 3000


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root / "service.yaml"


def test_a_v2_template_nested_too_deeply_is_a_load_error(tmp_path):
    manifest = _write_tree(
        tmp_path,
        {
            "service.yaml": yaml.safe_dump(V2_TEMPLATED),
            "templates/ai-gate-base.yaml": f"kind: ServiceManifestTemplate\nspec: {_DEEP}\n",
        },
    )

    with pytest.raises(ManifestLoadError) as raised:
        _load(manifest)

    assert raised.value.backstop is False
    assert isinstance(raised.value.__cause__, RecursionError)


def test_a_v2_slo_ref_nested_too_deeply_is_a_load_error(tmp_path):
    manifest = _write_tree(
        tmp_path,
        {
            "service.yaml": yaml.safe_dump(V2_REF),
            "slos/checkout-availability.yaml": f"spec: {_DEEP}\n",
        },
    )

    with pytest.raises(ManifestLoadError) as raised:
        _load(manifest)

    assert raised.value.backstop is False
    assert isinstance(raised.value.__cause__, RecursionError)


def test_an_srm_v1_alias_to_its_own_anchor_is_a_load_error(tmp_path):
    # The template merge descends wherever both sides hold a mapping, and
    # here both hold one that contains itself.
    looping = "  deployment: &loop\n    again: *loop\n"
    manifest = _write_tree(
        tmp_path,
        {
            "service.yaml": (
                "apiVersion: srm/v1\n"
                "kind: ServiceReliabilityManifest\n"
                "metadata: {name: orders, team: commerce, tier: standard, template: standard-api}\n"
                "spec:\n"
                "  type: api\n" + looping
            ),
            "templates/standard-api.yaml": (
                "apiVersion: srm/v1\nmetadata: {name: standard-api}\nspec:\n" + looping
            ),
        },
    )

    with pytest.raises(ManifestLoadError) as raised:
        _load(manifest)

    assert raised.value.backstop is False
    assert isinstance(raised.value.__cause__, RecursionError)


@pytest.mark.parametrize(
    "error", [AttributeError, TypeError, KeyError, IndexError, OverflowError]
)
def test_the_legacy_path_has_the_same_backstop(error, tmp_path, monkeypatch):
    # One tuple of shape errors for all three formats (opensrm-8his): the
    # legacy branch used to convert only AttributeError and TypeError.
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise error("a position nothing checks")

    monkeypatch.setattr(loader, "_parse_legacy_to_manifest", broken)
    manifest = _write_tree(
        tmp_path,
        {"service.yaml": "service: {name: orders, team: commerce, tier: standard, type: api}\n"},
    )

    with pytest.raises(ManifestLoadError) as raised:
        _load(manifest)

    assert raised.value.backstop is True
    assert isinstance(raised.value.__cause__, error)


@pytest.mark.parametrize("doc_id", ["v1-full", "v2-full"])
def test_a_check_that_names_the_position_is_not_the_backstop(doc_id, work):
    cell_id = f"{doc_id}:metadata=list"

    with pytest.raises(ManifestLoadError) as raised:
        load_cell(cell_id, work)

    assert raised.value.backstop is False


@pytest.mark.parametrize(
    ("doc_id", "parser"), [("v1-full", "parse_srm_v1"), ("v2-full", "parse_opensrm_v2")]
)
def test_the_matrix_fails_a_cell_the_backstop_refuses(doc_id, parser, work, monkeypatch):
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("unhashable type: 'list'")

    monkeypatch.setattr(loader, parser, broken)

    # A cell the template pass before the parser lets through.
    with pytest.raises(pytest.fail.Exception):
        outcome(f"{doc_id}:metadata.annotations=string", work)
