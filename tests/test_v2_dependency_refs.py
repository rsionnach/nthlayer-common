"""spec.dependencies[].service accepts resource: refs (opensrm-pnmv).

The v2 spec's own migration reference (opensrm/spec/v2/examples/migration/
api-full-v2.yaml) declares databases and caches as resource: entities, and
spec/v2/schema.json's EntityRef allows any kind.
"""
import pytest

from nthlayer_common.manifest.parser.v2 import OpenSRMV2ParseError, parse_opensrm_v2


def _manifest(service_ref: str) -> dict:
    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {"name": "payment-api", "labels": {"tier": "critical"}},
        "spec": {
            "owner": {"group": "group:default/payments"},
            "service": {"name": "payment-api", "type": "api"},
            "dependencies": [{"service": service_ref, "expected_availability": 0.9995}],
        },
    }


@pytest.mark.parametrize("ref, name", [
    ("resource:default/postgresql", "postgresql"),
    ("component:default/user-service", "user-service"),
    ("component:external/stripe-api", "stripe-api"),
])
def test_dependency_accepts_component_and_resource_refs(ref, name):
    (dep,) = parse_opensrm_v2(_manifest(ref)).dependencies
    assert dep.name == name


@pytest.mark.parametrize("ref", ["group:default/payments", "user:default/alice", "system:default/x"])
def test_dependency_rejects_other_kinds(ref):
    with pytest.raises(OpenSRMV2ParseError, match="reference"):
        parse_opensrm_v2(_manifest(ref))


def test_owner_escalation_still_requires_a_group_ref():
    m = _manifest("component:default/user-service")
    m["spec"]["owner"]["escalation"] = "resource:default/pager"
    with pytest.raises(OpenSRMV2ParseError):
        parse_opensrm_v2(m)
