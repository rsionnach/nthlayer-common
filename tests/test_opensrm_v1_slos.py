"""opensrm/v1 classical SLOs → OpenSLO (opensrm-pnmv).

Expected values follow opensrm/spec/v1/schema.json (targets are 0-1 ratios;
ErrorRateSLO.target is a MAXIMUM error ratio; window defaults are 30d for the
converted types) and AUTHORING.md §7.3, whose reference output
(spec/v2/examples/migration/api-full-v2.yaml) names SLOs <service>-<key>.
"""
import pytest

from nthlayer_common.manifest.opensrm_v1 import convert_opensrm_v1_to_v2
from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2


def _v1(slos: dict, name: str = "payment-api", stype: str = "api") -> dict:
    return {"apiVersion": "opensrm/v1", "kind": "ServiceReliabilityManifest",
            "metadata": {"name": name, "team": "payments", "tier": "critical"},
            "spec": {"type": stype, "slos": slos}}


def _slo(result, key):
    (doc,) = [d for d in result.manifest["spec"]["slo"] if d["metadata"]["name"].endswith(key)]
    return doc


def test_availability_keeps_target_and_window():
    r = convert_opensrm_v1_to_v2(_v1({"availability": {"target": 0.9995, "window": "30d"}}))
    doc = _slo(r, "availability")
    assert doc["metadata"]["name"] == "payment-api-availability"
    assert doc["spec"]["service"] == "payment-api"
    assert doc["spec"]["timeWindow"] == [{"duration": "30d", "isRolling": True}]
    assert doc["spec"]["objectives"][0]["target"] == 0.9995
    assert "indicator" not in doc["spec"]
    assert r.unmapped == ()


def test_error_rate_is_complemented_and_keeps_a_7d_window():
    r = convert_opensrm_v1_to_v2(_v1({"error_rate": {"target": 0.001, "window": "7d"}}))
    doc = _slo(r, "error-rate")
    assert doc["spec"]["objectives"][0]["target"] == 0.999
    assert doc["spec"]["timeWindow"] == [{"duration": "7d", "isRolling": True}]
    (slo,) = parse_opensrm_v2(r.manifest).slos
    assert (slo.slo_type, slo.window) == ("error_rate", "7d")


def test_absent_window_uses_the_v1_schema_default():
    r = convert_opensrm_v1_to_v2(_v1({"availability": {"target": 0.999}}))
    assert _slo(r, "availability")["spec"]["timeWindow"] == [{"duration": "30d", "isRolling": True}]


def test_latency_percentiles_go_to_display_name_and_are_reported():
    r = convert_opensrm_v1_to_v2(_v1({"latency": {"p50": "50ms", "p99": "200ms", "target": 0.99}}))
    obj = _slo(r, "latency")["spec"]["objectives"][0]
    assert obj["target"] == 0.99 and "p50 under 50ms" in obj["displayName"] and "p99 under 200ms" in obj["displayName"]
    assert [f.path for f in r.unmapped] == ["spec.slos.latency.p50", "spec.slos.latency.p99"]


@pytest.mark.parametrize("key, expected_type", [
    ("duration", "latency"), ("query_latency", "latency"),
    ("connection_availability", "availability"),
])
def test_other_converted_types_parse_as_intended(key, expected_type):
    r = convert_opensrm_v1_to_v2(_v1({key: {"target": 0.99}}, stype="database"))
    (slo,) = parse_opensrm_v2(r.manifest).slos
    assert slo.slo_type == expected_type


@pytest.mark.parametrize("key, cfg", [
    ("throughput", {"target": 1000, "unit": "rps"}),
    ("success_rate", {"target": 0.995}),
    ("processing_time", {"target": 0.95, "p95": "5m"}),
    ("lag", {"target": 0.99, "max": "1m"}),
    ("schedule_adherence", {"max_delay": "10m"}),
    ("data_freshness", {"max_age": "2h"}),
    ("replication_lag", {"max_lag": "1s"}),
])
def test_unconvertible_slo_types_are_reported_not_dropped(key, cfg):
    r = convert_opensrm_v1_to_v2(_v1({key: cfg}, stype="batch"))
    assert "slo" not in r.manifest["spec"]
    assert [f.path for f in r.unmapped] == [f"spec.slos.{key}"]


def test_custom_is_reported_and_openslo_paths_become_refs():
    r = convert_opensrm_v1_to_v2(_v1({"custom": [{"name": "x"}], "openslo": [{"path": "slos/a.yaml"}]}))
    assert r.manifest["spec"]["slo"] == [{"$ref": "slos/a.yaml"}]
    assert [f.path for f in r.unmapped] == ["spec.slos.custom"]


@pytest.mark.parametrize("cfg, match", [
    ({"target": 1.5}, "target"), ({"target": "0.99"}, "target"), ({"target": True}, "target"),
    ({"target": 0.99, "window": "30 days"}, "window"), ({}, "target"),
])
def test_malformed_converted_slos_raise(cfg, match):
    with pytest.raises(ValueError, match=match):
        convert_opensrm_v1_to_v2(_v1({"availability": cfg}))


def test_a_type_hint_word_in_the_service_name_raises_instead_of_mistyping():
    # "error-tracker-availability" would be typed error_rate by the parser.
    with pytest.raises(ValueError, match="error-tracker-availability"):
        convert_opensrm_v1_to_v2(_v1({"availability": {"target": 0.999}}, name="error-tracker"))


@pytest.mark.parametrize("value", [{"path": "a.yaml"}, None, "a.yaml"])
def test_openslo_must_be_a_list(value):
    with pytest.raises(ValueError, match=r"spec\.slos\.openslo"):
        convert_opensrm_v1_to_v2(_v1({"openslo": value}))


def test_openslo_item_extra_keys_are_reported():
    from nthlayer_common.manifest.opensrm_v1 import _NOT_IN_SCHEMA
    r = convert_opensrm_v1_to_v2(_v1({"openslo": [{"path": "a.yaml", "inline": 1}]}))
    assert r.manifest["spec"]["slo"] == [{"$ref": "a.yaml"}]
    assert [(f.path, f.reason) for f in r.unmapped] == [("spec.slos.openslo[0].inline", _NOT_IN_SCHEMA)]


@pytest.mark.parametrize("name", ["rps-svc", "latency-svc"])
def test_other_type_hint_words_in_service_name_raise(name):
    with pytest.raises(ValueError, match=f"{name}-availability"):
        convert_opensrm_v1_to_v2(_v1({"availability": {"target": 0.999}}, name=name))


def test_unconvertible_reasons_are_the_specific_ones():
    from nthlayer_common.manifest.opensrm_v1 import _NOT_IN_SCHEMA, _SECTIONS
    for key in _SECTIONS["SLOs"][1]:
        if key == "custom":
            continue
        r = convert_opensrm_v1_to_v2(_v1({key: {"target": 0.9}}, stype="batch"))
        (f,) = r.unmapped
        assert f.reason == _SECTIONS["SLOs"][1][key] != _NOT_IN_SCHEMA
