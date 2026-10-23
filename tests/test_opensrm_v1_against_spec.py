"""The spec's own opensrm/v1 examples convert, validate and parse (opensrm-pnmv).

The EXPECTED unmapped lists are derived from each example file's keys and the
§4 mapping table of the pnmv design, not from the converter's output.
"""
import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from nthlayer_common.manifest import ManifestLoadError, is_manifest_file, load_manifest
from nthlayer_common.manifest.opensrm_v1 import convert_opensrm_v1_to_v2
from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2

# parents: [0]=tests/ [1]=nthlayer-common/ [2]=<ecosystem>/
OPENSRM = Path(__file__).resolve().parents[2] / "opensrm"
EXAMPLES = OPENSRM / "examples"
V2_SCHEMA = OPENSRM / "spec" / "v2" / "schema.json"
REFERENCE = OPENSRM / "spec" / "v2" / "examples" / "migration" / "api-full-v2.yaml"

_OBS = ["spec.observability.metrics", "spec.observability.dashboards", "spec.observability.alerts"]
EXPECTED: dict[str, tuple[str | None, list[str]]] = {
    # file: (service_type to supply, unmapped paths in document order)
    "ai-gate-full.yaml": (None, [
        "spec.contract.judgment", "spec.slos.latency.p99",
        "spec.slos.judgment.reversal_rate", "spec.slos.judgment.high_confidence_failure",
        "spec.slos.judgment.calibration", "spec.instrumentation.attributes",
        "spec.dependencies[0].critical", "spec.dependencies[1].critical", "spec.dependencies[2].critical",
        *_OBS]),
    "ai-gate-minimal.yaml": (None, ["spec.observability.metrics", "spec.slos.judgment.reversal_rate"]),
    "ai-gate-security-scanner.yaml": (None, [
        "spec.observability.metrics", "spec.slos.judgment.reversal_rate",
        "spec.slos.judgment.high_confidence_failure", "spec.instrumentation.attributes"]),
    "api-full.yaml": ("api", [
        "spec.slos.latency.p50", "spec.slos.latency.p99", "spec.slos.throughput",
        "spec.dependencies[0].critical", "spec.dependencies[1].critical",
        "spec.dependencies[2].critical", "spec.dependencies[2].manifest", "spec.dependencies[3].critical",
        "spec.ownership.email", "spec.ownership.pagerduty.escalation_policy_id",
        "spec.notifications", *_OBS, "spec.observability.tracing", "spec.deployment"]),
    "api-internal.yaml": ("api", [
        "spec.slos.latency.p95", "spec.dependencies[0].critical", "spec.dependencies[1].critical",
        "spec.ownership.email", *_OBS, "spec.observability.tracing", "spec.deployment"]),
    "api-minimal.yaml": (None, ["spec.observability.metrics"]),
    "batch-etl.yaml": (None, [
        "spec.slos.success_rate", "spec.slos.duration.p95", "spec.slos.schedule_adherence",
        "spec.slos.data_freshness", "spec.slos.throughput",
        "spec.dependencies[0].critical", "spec.dependencies[1].critical", "spec.dependencies[2].critical",
        "spec.ownership.email", *_OBS, "spec.deployment"]),
    "database-mysql.yaml": (None, [
        "spec.slos.query_latency.p50", "spec.slos.query_latency.p99", "spec.slos.replication_lag",
        "spec.slos.throughput", "spec.dependencies[0].critical", "spec.dependencies[1].critical",
        "spec.ownership.email", "spec.ownership.pagerduty.escalation_policy_id", *_OBS, "spec.deployment"]),
}


# The reason and kind each unmapped path above must carry, as literals
# cross-checked against the pnmv design §4 table (not read from the converter's
# table). Only calibration is "assumed" (converted, with a bins value v1 does
# not declare); every other entry is "dropped" — percentiles included, since
# they survive only as displayName text. Dependency indices are normalised to [*].
_PCT = ("kept only in the OpenSLO displayName: an objective without an "
        "indicator carries no machine-readable percentile")
_JUDGMENT_SLO = ("judgment_slo is new capability, not migration (AUTHORING §7.7 step 8), and no "
                 "v2 judgment-SLO shape passes both the schema and NthLayer's v2 parser")
_NO_V2 = "no v2 equivalent (AUTHORING §7.6)"
REASONS: dict[str, tuple[str, str]] = {
    "spec.contract.judgment":
        ("v2 contracts[].promise has availability, latency_p99 and throughput only (AUTHORING §7.4)",
         "dropped"),
    "spec.slos.latency.p50": (_PCT, "dropped"),
    "spec.slos.latency.p95": (_PCT, "dropped"),
    "spec.slos.latency.p99": (_PCT, "dropped"),
    "spec.slos.duration.p95": (_PCT, "dropped"),
    "spec.slos.query_latency.p50": (_PCT, "dropped"),
    "spec.slos.query_latency.p99": (_PCT, "dropped"),
    "spec.slos.throughput":
        ("v1 declares a rate (e.g. 1000 rps); an OpenSLO objective needs a ratio target (AUTHORING §7.3)",
         "dropped"),
    "spec.slos.success_rate": ("NthLayer's v2 parser cannot infer this SLO's type", "dropped"),
    "spec.slos.schedule_adherence":
        ("maximum-only SLO (max_delay) with no target; an OpenSLO objective needs one", "dropped"),
    "spec.slos.data_freshness":
        ("maximum-only SLO (max_age) with no target; an OpenSLO objective needs one", "dropped"),
    "spec.slos.replication_lag":
        ("maximum-only SLO (max_lag) with no target; an OpenSLO objective needs one", "dropped"),
    "spec.slos.judgment.reversal_rate": (_JUDGMENT_SLO, "dropped"),
    "spec.slos.judgment.high_confidence_failure": (_JUDGMENT_SLO, "dropped"),
    "spec.slos.judgment.calibration":
        ("converted with measurement.bins: 10, a value opensrm/v1 does not declare; check it",
         "assumed"),
    "spec.instrumentation.attributes": ("v2 instrumentation has no attribute list", "dropped"),
    "spec.dependencies[*].critical":
        ("v2 expresses this as a fallback, which only the author can choose (AUTHORING §7.5)",
         "dropped"),
    "spec.dependencies[*].manifest":
        ("v2 resolves dependencies through the catalogue (AUTHORING §7.5)", "dropped"),
    "spec.ownership.email": ("no v2 field or AUTHORING §7.2 annotation key", "dropped"),
    "spec.ownership.pagerduty.escalation_policy_id": ("no AUTHORING §7.2 annotation key", "dropped"),
    "spec.notifications": (_NO_V2, "dropped"),
    "spec.observability.metrics":
        ("v2 instrumentation needs a type per metric (AUTHORING §7.7 step 6)", "dropped"),
    "spec.observability.dashboards": (_NO_V2, "dropped"),
    "spec.observability.alerts": (_NO_V2, "dropped"),
    "spec.observability.tracing": (_NO_V2, "dropped"),
    "spec.deployment": (_NO_V2, "dropped"),
}


def _generic(path: str) -> str:
    """spec.dependencies[2].critical -> spec.dependencies[*].critical (one index level)."""
    head, bracket, rest = path.partition("[")
    return f"{head}[*]{rest.partition(']')[2]}" if bracket else path


def test_every_expected_path_has_a_reason_and_every_reason_is_used():
    used = {_generic(p) for _, paths in EXPECTED.values() for p in paths}
    assert used == set(REASONS)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_unmapped_reasons_and_kinds_are_the_specific_ones(name):
    service_type, _ = EXPECTED[name]
    doc = yaml.safe_load((EXAMPLES / name).read_text())
    r = convert_opensrm_v1_to_v2(doc, service_type=service_type)
    got = [(f.path, (f.reason, f.kind)) for f in r.unmapped]
    assert got == [(path, REASONS[_generic(path)]) for path, _ in got]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_is_lossy_exactly_when_an_expected_path_is_dropped(name):
    service_type, unmapped = EXPECTED[name]
    doc = yaml.safe_load((EXAMPLES / name).read_text())
    r = convert_opensrm_v1_to_v2(doc, service_type=service_type)
    assert r.lossy is any(REASONS[_generic(p)][1] == "dropped" for p in unmapped)
    assert r.lossy is True  # every spec example drops something


def _ai_gate_calibration_only() -> dict:
    """ai-gate-full.yaml cut down to its envelope and spec.slos.judgment.calibration:
    the shape is the spec example's, not one the converter was written against."""
    full = yaml.safe_load((EXAMPLES / "ai-gate-full.yaml").read_text())
    return {
        "apiVersion": full["apiVersion"], "kind": full["kind"], "metadata": full["metadata"],
        "spec": {"type": full["spec"]["type"],
                 "slos": {"judgment": {"calibration": full["spec"]["slos"]["judgment"]["calibration"]}}},
    }


def test_calibration_only_document_is_not_lossy():
    r = convert_opensrm_v1_to_v2(_ai_gate_calibration_only())
    assert [(f.path, f.kind) for f in r.unmapped] == [("spec.slos.judgment.calibration", "assumed")]
    assert r.lossy is False


def test_any_dropped_field_makes_the_result_lossy():
    # The calibration-only document plus one field ai-gate-full.yaml also
    # declares and v2 has no home for.
    full = yaml.safe_load((EXAMPLES / "ai-gate-full.yaml").read_text())
    doc = _ai_gate_calibration_only()
    doc["spec"]["slos"]["judgment"]["reversal_rate"] = full["spec"]["slos"]["judgment"]["reversal_rate"]
    r = convert_opensrm_v1_to_v2(doc)
    assert sorted((f.path, f.kind) for f in r.unmapped) == [
        ("spec.slos.judgment.calibration", "assumed"),
        ("spec.slos.judgment.reversal_rate", "dropped"),
    ]
    assert r.lossy is True


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_is_manifest_file_counts_every_example(name):
    # The example names are neither *.reliability.yaml nor service.yaml, so
    # only the opensrm/v1 content check can count them.
    assert is_manifest_file(EXAMPLES / name)

def test_the_spec_examples_are_present():
    found = sorted(p.name for p in EXAMPLES.glob("*.yaml")
                   if yaml.safe_load(p.read_text()).get("apiVersion") == "opensrm/v1")
    assert found == sorted(EXPECTED), "the opensrm sibling is missing or its v1 examples changed"
    assert V2_SCHEMA.is_file() and REFERENCE.is_file()


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_converts_validates_and_parses(name):
    service_type, unmapped = EXPECTED[name]
    doc = yaml.safe_load((EXAMPLES / name).read_text())
    r = convert_opensrm_v1_to_v2(doc, service_type=service_type)
    assert sorted(f.path for f in r.unmapped) == sorted(unmapped)
    jsonschema.validate(r.manifest, json.loads(V2_SCHEMA.read_text()))
    parse_opensrm_v2(r.manifest)


def _document_paths(node, prefix=""):
    """Every dotted path in *node*, depth-first in the file's key order, parents first."""
    if isinstance(node, dict):
        items = [(f"{prefix}.{k}" if prefix else k, v) for k, v in node.items()]
    elif isinstance(node, list):
        items = [(f"{prefix}[{i}]", v) for i, v in enumerate(node)]
    else:
        return
    for path, child in items:
        yield path
        yield from _document_paths(child, path)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_unmapped_is_in_document_order_depth_first(name):
    """Users read the list against their file top to bottom. The expected order
    is the example's own key order, filtered to the paths EXPECTED names."""
    service_type, unmapped = EXPECTED[name]
    doc = yaml.safe_load((EXAMPLES / name).read_text())
    expected = [p for p in _document_paths(doc) if p in set(unmapped)]
    assert sorted(expected) == sorted(unmapped), "an expected path is not in the file"
    r = convert_opensrm_v1_to_v2(doc, service_type=service_type)
    assert [f.path for f in r.unmapped] == expected


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_load_manifest_rejects_every_example(name):
    with pytest.raises(ManifestLoadError, match="nthlayer-generate migrate-manifest"):
        load_manifest(EXAMPLES / name)


def test_api_full_matches_the_spec_reference_on_mapped_fields():
    """Excluded (the reference author's decisions or inventions, or unmapped by
    §4): fallback, api_ref, technical_contact, the backstage.io/component-ref
    annotation, metadata.namespace, instrumentation, and the throughput SLO.
    Dependencies are compared without ``type`` as well as ``fallback``: the
    converter writes the v1 dependency type (so a database does not parse as
    'api') and the spec's reference file does not carry it.

    SLO documents are compared whole except for two keys:
    ``objectives[].displayName`` (wording) and ``spec.timeWindow``. The
    reference omits timeWindow, which is a defect in the spec's reference file:
    its 7d error-rate SLO would parse as the parser's 30d default. timeWindow is
    pinned separately in test_opensrm_v1_slos.py."""
    got = convert_opensrm_v1_to_v2(yaml.safe_load((EXAMPLES / "api-full.yaml").read_text()),
                                   service_type="api").manifest
    ref = yaml.safe_load(REFERENCE.read_text())
    assert (got["apiVersion"], got["kind"]) == (ref["apiVersion"], ref["kind"])
    assert got["metadata"]["name"] == ref["metadata"]["name"]
    assert got["metadata"]["labels"] == ref["metadata"]["labels"]
    ref_ann = {k: v for k, v in ref["metadata"]["annotations"].items() if k != "backstage.io/component-ref"}
    assert got["metadata"]["annotations"] == ref_ann
    assert {k: got["spec"]["owner"][k] for k in ("group", "escalation")} == \
           {k: ref["spec"]["owner"][k] for k in ("group", "escalation")}
    assert got["spec"]["service"] == ref["spec"]["service"]

    def slo_docs(slos):
        def comparable(d):
            spec = {k: v for k, v in d["spec"].items() if k != "timeWindow"}
            spec["objectives"] = [{k: v for k, v in o.items() if k != "displayName"}
                                  for o in spec["objectives"]]
            return {**d, "spec": spec}
        return {d["metadata"]["name"]: comparable(d)
                for d in slos if not d["metadata"]["name"].endswith("throughput")}
    assert slo_docs(got["spec"]["slo"]) == slo_docs(ref["spec"]["slo"])
    (c,) = got["spec"]["contracts"]
    (rc,) = ref["spec"]["contracts"]
    assert (c["name"], c["promise"], c["contract_owner"]) == (rc["name"], rc["promise"], rc["contract_owner"])

    def strip(deps):
        return [{k: v for k, v in d.items() if k not in ("fallback", "type")} for d in deps]
    assert strip(got["spec"]["dependencies"]) == strip(ref["spec"]["dependencies"])
