"""opensrm/v1 contract / dependencies / ownership / observability /
instrumentation (opensrm-pnmv). Shapes: spec/v1/schema.json; mapping:
AUTHORING.md §7.2 (ownership → owner + annotations), §7.4 (contract →
contracts), §7.5 (dependencies), §7.6 (homeless blocks).
"""
import json
from pathlib import Path

import jsonschema
import pytest

from nthlayer_common.manifest.opensrm_v1 import (
    _SPEC_HANDLERS,
    OpenSRMV1MigrationError,
    _keys,
    convert_opensrm_v1_to_v2,
)
from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2

V2_SCHEMA = Path(__file__).resolve().parents[2] / "opensrm" / "spec" / "v2" / "schema.json"


def _v1(**spec) -> dict:
    return {"apiVersion": "opensrm/v1", "kind": "ServiceReliabilityManifest",
            "metadata": {"name": "payment-api", "team": "payments", "tier": "critical"},
            "spec": {"type": "api", **spec}}


def _paths(r):
    return sorted(f.path for f in r.unmapped)


def test_contract_becomes_contracts():
    r = convert_opensrm_v1_to_v2(_v1(contract={
        "availability": 0.9995, "latency": {"p99": "300ms"}, "throughput": {"max_rps": 5000}}))
    assert r.manifest["spec"]["contracts"] == [{
        "name": "payment-api",
        "promise": {"availability": 0.9995, "latency_p99": "300ms", "throughput": "5000rps"},
        "contract_owner": "group:default/payments",
    }]
    assert r.unmapped == ()
    parse_opensrm_v2(r.manifest)


def test_contract_fields_with_no_promise_slot_are_reported():
    r = convert_opensrm_v1_to_v2(_v1(contract={
        "latency": {"p50": "50ms", "p999": "1s"}, "throughput": {"burst": 100},
        "judgment": {"max_reversal_rate": 0.05}}))
    assert "contracts" not in r.manifest["spec"]
    assert [(f.path, f.reason) for f in r.unmapped] == [
        ("spec.contract.latency.p50", "v2 contracts[].promise has latency_p99 only (AUTHORING §7.4)"),
        ("spec.contract.latency.p999", "v2 contracts[].promise has latency_p99 only (AUTHORING §7.4)"),
        ("spec.contract.throughput.burst", "v2 contracts[].promise.throughput is a single rate (AUTHORING §7.4)"),
        ("spec.contract.judgment",
         "v2 contracts[].promise has availability, latency_p99 and throughput only (AUTHORING §7.4)"),
    ]


@pytest.mark.parametrize("dtype, ref", [
    ("database", "resource:default/postgresql"), ("cache", "resource:default/postgresql"),
    ("queue", "resource:default/postgresql"), ("external", "component:external/postgresql"),
    ("service", "component:default/postgresql"), (None, "component:default/postgresql"),
])
def test_dependency_ref_kind_follows_the_v1_type(dtype, ref):
    dep = {"service": "postgresql", "expects": {"availability": 0.9995, "latency": {"p99": "20ms"}}}
    if dtype:
        dep["type"] = dtype
    r = convert_opensrm_v1_to_v2(_v1(dependencies=[dep]))
    expected = {"service": ref, "expected_availability": 0.9995, "expected_latency_p99": "20ms"}
    if dtype not in (None, "service"):
        expected["type"] = dtype  # see test_dependency_type_survives_into_the_parsed_dependency
    assert r.manifest["spec"]["dependencies"] == [expected]
    parse_opensrm_v2(r.manifest)


def test_dependency_critical_and_manifest_are_reported():
    r = convert_opensrm_v1_to_v2(_v1(dependencies=[
        {"service": "user-service", "type": "service", "critical": True,
         "manifest": "https://example.com/user-service.yaml"}]))
    assert _paths(r) == sorted(["spec.dependencies[0].critical", "spec.dependencies[0].manifest"])


def test_ownership_becomes_owner_and_annotations():
    r = convert_opensrm_v1_to_v2(_v1(ownership={
        "team": "payments", "slack": "#payments-team", "email": "p@example.com",
        "escalation": "payments-oncall", "runbook": "https://wiki/rb", "documentation": "https://docs",
        "pagerduty": {"service_id": "PXXXXXX", "escalation_policy_id": "PYYY"},
        "oncall_required": True}))
    m = r.manifest
    assert m["spec"]["owner"] == {"group": "group:default/payments",
                                  "escalation": "group:default/payments-oncall"}
    assert m["metadata"]["annotations"] == {
        "slack.com/channel": "#payments-team",
        "opensrm.nthlayer.io/runbook": "https://wiki/rb",
        "opensrm.nthlayer.io/documentation": "https://docs",
        "pagerduty.com/service-id": "PXXXXXX",
    }
    assert _paths(r) == sorted(["spec.ownership.email", "spec.ownership.pagerduty.escalation_policy_id", "spec.ownership.oncall_required"])
    parse_opensrm_v2(m)


def test_ownership_team_must_agree_with_metadata_team():
    with pytest.raises(OpenSRMV1MigrationError, match="ownership.team"):
        convert_opensrm_v1_to_v2(_v1(ownership={"team": "billing"}))


def test_an_annotation_collision_raises():
    d = _v1(ownership={"team": "payments", "slack": "#new"})
    d["metadata"]["annotations"] = {"slack.com/channel": "#old"}
    with pytest.raises(OpenSRMV1MigrationError, match="slack.com/channel"):
        convert_opensrm_v1_to_v2(d)


def test_observability_notifications_and_deployment_are_reported():
    r = convert_opensrm_v1_to_v2(_v1(
        observability={"metrics": {"required": ["x"]}, "dashboards": {"required": True},
                       "alerts": {"required": True}, "tracing": {"required": True}},
        notifications={"slack": {"channel": "#x"}}, deployment={"gates": {}}))
    assert _paths(r) == sorted(["spec.observability.metrics", "spec.observability.dashboards", "spec.observability.alerts", "spec.observability.tracing", "spec.notifications", "spec.deployment"])


def test_instrumentation_events_become_required_events():
    r = convert_opensrm_v1_to_v2(_v1(instrumentation={
        "events": {"decision": "pr_review.decision", "reversal": "pr_review.reversal"}, "attributes": {"decision_id": "string"}}))
    assert r.manifest["spec"]["instrumentation"] == {
        "required_events": [{"type": "pr_review.decision"}, {"type": "pr_review.reversal"}]}
    assert _paths(r) == sorted(["spec.instrumentation.attributes"])


def test_instrumentation_event_value_must_be_a_string():
    with pytest.raises(OpenSRMV1MigrationError, match=r"spec\.instrumentation\.events\.decision"):
        convert_opensrm_v1_to_v2(_v1(instrumentation={"events": {"decision": {}}}))


def test_instrumentation_unknown_event_key_is_reported():
    r = convert_opensrm_v1_to_v2(_v1(instrumentation={
        "events": {"decision": "d", "bogus": "b"}}))
    assert r.manifest["spec"]["instrumentation"] == {"required_events": [{"type": "d"}]}
    assert _paths(r) == ["spec.instrumentation.events.bogus"]


@pytest.mark.parametrize("rps", [0, 1.5, -3, True, "5"])
def test_max_rps_below_the_schema_minimum_or_non_integer_raises(rps):
    with pytest.raises(OpenSRMV1MigrationError, match=r"spec\.contract\.throughput\.max_rps"):
        convert_opensrm_v1_to_v2(_v1(contract={"throughput": {"max_rps": rps}}))


def test_max_rps_of_one_converts():
    r = convert_opensrm_v1_to_v2(_v1(contract={"throughput": {"max_rps": 1}}))
    assert r.manifest["spec"]["contracts"][0]["promise"] == {"throughput": "1rps"}


def test_dependency_type_outside_the_enum_raises():
    with pytest.raises(OpenSRMV1MigrationError, match=r"spec\.dependencies\[0\]\.type"):
        convert_opensrm_v1_to_v2(_v1(dependencies=[{"service": "db", "type": "mainframe"}]))


@pytest.mark.parametrize("name", ["", "-lead", "has space", 7, None])
def test_dependency_service_name_must_be_usable_in_a_ref(name):
    with pytest.raises(OpenSRMV1MigrationError, match=r"spec\.dependencies\[0\]\.service"):
        convert_opensrm_v1_to_v2(_v1(dependencies=[{"service": name}]))


def test_unknown_keys_in_pagerduty_and_expects_are_reported_as_outside_the_schema():
    r = convert_opensrm_v1_to_v2(_v1(
        ownership={"team": "payments", "pagerduty": {"service_id": "PX", "routing_key": "k"}},
        dependencies=[{"service": "db", "expects": {"availability": 0.999, "error_rate": 0.01,
                                                    "latency": {"p99": "20ms", "p50": "5ms"}}}]))
    assert {(f.path, f.reason) for f in r.unmapped} == {
        ("spec.ownership.pagerduty.routing_key", "not part of the opensrm/v1 schema"),
        ("spec.dependencies[0].expects.error_rate", "not part of the opensrm/v1 schema"),
        ("spec.dependencies[0].expects.latency.p50", "not part of the opensrm/v1 schema"),
    }
    assert r.manifest["spec"]["dependencies"] == [
        {"service": "component:default/db", "expected_availability": 0.999, "expected_latency_p99": "20ms"}]


@pytest.mark.parametrize(("section", "path", "dep", "own"), [
    ("Ownership.pagerduty", "spec.ownership.pagerduty.service_id", None, {"pagerduty": {"service_id": "PX"}}),
    ("Dependency.expects", "spec.dependencies[0].expects.availability",
     {"expects": {"availability": 0.999}}, None),
    ("Dependency.expects.latency", "spec.dependencies[0].expects.latency.p99",
     {"expects": {"latency": {"p99": "20ms"}}}, None),
])
def test_pagerduty_and_expects_take_their_reasons_from_sections(monkeypatch, section, path, dep, own):
    """The inline handlers read _SECTIONS, so the schema-completeness test governs them too."""
    from nthlayer_common.manifest import opensrm_v1

    key = path.rsplit(".", 1)[1]
    converted, unmapped = opensrm_v1._SECTIONS[section]
    monkeypatch.setitem(opensrm_v1._SECTIONS, section, (converted - {key}, {**unmapped, key: "probe"}))
    spec = {}
    if dep is not None:
        spec["dependencies"] = [{"service": "db", **dep}]
    if own is not None:
        spec["ownership"] = {"team": "payments", **own}
    r = convert_opensrm_v1_to_v2(_v1(**spec))
    assert (path, "probe") in {(f.path, f.reason) for f in r.unmapped}


# --- the open-walk guard: an abandoned _keys walk would drop reports silently ---

def _abandon_with_next(value, out, ctx):
    next(_keys("Instrumentation", "spec.instrumentation", value, ctx), None)


def _never_iterate(value, out, ctx):
    _keys("Instrumentation", "spec.instrumentation", value, ctx)


@pytest.mark.parametrize("handler", [_abandon_with_next, _never_iterate])
def test_an_abandoned_section_walk_fails_loudly(monkeypatch, handler):
    monkeypatch.setitem(_SPEC_HANDLERS, "instrumentation", handler)
    with pytest.raises(RuntimeError, match="section walk was not exhausted"):
        convert_opensrm_v1_to_v2(_v1(instrumentation={"events": {"decision": "d", "reversal": "r"}, "attributes": {}}))


@pytest.mark.parametrize("dtype, parsed", [
    ("database", "database"), ("cache", "cache"), ("queue", "queue"),
    ("external", "external"), ("service", "api"), (None, "api"),
])
def test_dependency_type_survives_into_the_parsed_dependency(dtype, parsed):
    """common's v2 parser reads a missing dependency type as 'api', which
    nthlayer-generate would turn into API alerts for a database."""
    dep = {"service": "postgresql"}
    if dtype:
        dep["type"] = dtype
    r = convert_opensrm_v1_to_v2(_v1(dependencies=[dep]))
    (entry,) = r.manifest["spec"]["dependencies"]
    assert ("type" in entry) == (dtype not in (None, "service"))
    (parsed_dep,) = parse_opensrm_v2(r.manifest).dependencies
    assert parsed_dep.type == parsed
    jsonschema.validate(r.manifest, json.loads(V2_SCHEMA.read_text()))
