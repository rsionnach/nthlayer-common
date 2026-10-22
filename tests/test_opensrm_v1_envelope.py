"""opensrm/v1 → v2: envelope, metadata, owner, service type (opensrm-pnmv).

Shapes follow opensrm/spec/v1/schema.json (Metadata: name/team/tier required;
name ^[a-z0-9-]+$; tier critical|high|standard|low) and AUTHORING.md §7.1-7.2.
"""
import pytest

from nthlayer_common.manifest.opensrm_v1 import (
    MigrationResult,
    UnmappedField,
    convert_opensrm_v1_to_v2,
    is_opensrm_v1_format,
)


def _v1(**spec) -> dict:
    return {
        "apiVersion": "opensrm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "payment-api", "team": "payments", "tier": "critical",
                     "description": "Handles payments"},
        "spec": {"type": "api", **spec},
    }


def test_detects_only_opensrm_v1():
    assert is_opensrm_v1_format(_v1())
    assert not is_opensrm_v1_format({"apiVersion": "srm/v1"})
    assert not is_opensrm_v1_format([])


def test_envelope_metadata_owner_and_service():
    r = convert_opensrm_v1_to_v2(_v1())
    assert isinstance(r, MigrationResult) and r.unmapped == ()
    m = r.manifest
    assert (m["apiVersion"], m["kind"]) == ("opensrm.nthlayer.io/v2", "ServiceManifest")
    assert m["metadata"] == {"name": "payment-api", "labels": {"tier": "critical"}}
    assert m["spec"]["owner"] == {"group": "group:default/payments"}
    assert m["spec"]["service"] == {"name": "payment-api", "type": "api",
                                    "description": "Handles payments"}


def test_labels_and_annotations_are_copied_and_tier_joins_labels():
    d = _v1()
    d["metadata"]["labels"] = {"domain": "payments"}
    d["metadata"]["annotations"] = {"backstage.io/techdocs-ref": "dir:."}
    m = convert_opensrm_v1_to_v2(d).manifest
    assert m["metadata"]["labels"] == {"domain": "payments", "tier": "critical"}
    assert m["metadata"]["annotations"] == {"backstage.io/techdocs-ref": "dir:."}


def test_template_is_reported_unmapped():
    d = _v1()
    d["metadata"]["template"] = "standard-api"
    assert [f.path for f in convert_opensrm_v1_to_v2(d).unmapped] == ["metadata.template"]


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d.update(apiVersion="srm/v1"), "opensrm/v1"),
    (lambda d: d.update(kind="Template"), "ServiceReliabilityManifest"),
    (lambda d: d["metadata"].pop("name"), "metadata.name"),
    (lambda d: d["metadata"].update(name="-payment-api"), "metadata.name"),
    (lambda d: d["metadata"].pop("team"), "team"),
    (lambda d: d["metadata"].update(team="pay ments"), "team"),
    (lambda d: d["spec"].update(type="mainframe"), "spec.type"),
    (lambda d: d["metadata"].update(tier="gold"), "tier"),
    (lambda d: d["metadata"].update(labels={"tier": "low"}), "tier"),
])
def test_unconvertible_documents_raise(mutate, match):
    d = _v1()
    mutate(d)
    with pytest.raises(ValueError, match=match):
        convert_opensrm_v1_to_v2(d)


def test_missing_type_needs_service_type():
    d = _v1()
    del d["spec"]["type"]
    with pytest.raises(ValueError, match="service-type"):
        convert_opensrm_v1_to_v2(d)
    assert convert_opensrm_v1_to_v2(d, service_type="api").manifest["spec"]["service"]["type"] == "api"


def test_supplied_type_must_agree_with_a_declared_one():
    assert convert_opensrm_v1_to_v2(_v1(), service_type="api").manifest["spec"]["service"]["type"] == "api"
    with pytest.raises(ValueError, match="spec.type"):
        convert_opensrm_v1_to_v2(_v1(), service_type="worker")


def test_unknown_spec_keys_are_reported_not_dropped():
    r = convert_opensrm_v1_to_v2(_v1(sla={"x": 1}))
    assert r.unmapped == (UnmappedField("spec.sla", "not part of the opensrm/v1 schema"),)
