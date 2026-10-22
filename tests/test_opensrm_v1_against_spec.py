"""The spec's own opensrm/v1 examples convert, validate and parse (opensrm-pnmv).

The EXPECTED unmapped lists are derived from each example file's keys and the
§4 mapping table of the pnmv design, not from the converter's output.
"""
import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from nthlayer_common.manifest import ManifestLoadError, load_manifest
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


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_load_manifest_rejects_every_example(name):
    with pytest.raises(ManifestLoadError, match="nthlayer migrate-manifest"):
        load_manifest(EXAMPLES / name)


def test_api_full_matches_the_spec_reference_on_mapped_fields():
    """Excluded (the reference author's decisions or inventions, or unmapped by
    §4): fallback, api_ref, technical_contact, the backstage.io/component-ref
    annotation, metadata.namespace, instrumentation, and the throughput SLO.

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
        return [{k: v for k, v in d.items() if k != "fallback"} for d in deps]
    assert got["spec"]["dependencies"] == strip(ref["spec"]["dependencies"])
