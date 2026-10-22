"""opensrm/v1 classical SLOs → OpenSLO (opensrm-pnmv).

Expected values follow opensrm/spec/v1/schema.json (targets are 0-1 ratios;
ErrorRateSLO.target is a MAXIMUM error ratio; window defaults are 30d for the
converted types) and AUTHORING.md §7.3, whose reference output
(spec/v2/examples/migration/api-full-v2.yaml) names SLOs <service>-<key>.
"""
import pytest

from nthlayer_common.manifest.opensrm_v1 import OpenSRMV1MigrationError, convert_opensrm_v1_to_v2
from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2

_PERCENTILE_WHY = ("kept only in the OpenSLO displayName: an objective without an "
                   "indicator carries no machine-readable percentile")


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
    ("lag", {"max_seconds": 60, "target": 0.99}),
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
    with pytest.raises(OpenSRMV1MigrationError, match=match):
        convert_opensrm_v1_to_v2(_v1({"availability": cfg}))


def test_a_type_hint_word_in_the_service_name_raises_instead_of_mistyping():
    # "error-tracker-availability" would be typed error_rate by the parser.
    with pytest.raises(OpenSRMV1MigrationError, match="error-tracker-availability"):
        convert_opensrm_v1_to_v2(_v1({"availability": {"target": 0.999}}, name="error-tracker"))


@pytest.mark.parametrize("value", [{"path": "a.yaml"}, None, "a.yaml"])
def test_openslo_must_be_a_list(value):
    with pytest.raises(OpenSRMV1MigrationError, match=r"spec\.slos\.openslo"):
        convert_opensrm_v1_to_v2(_v1({"openslo": value}))


def test_openslo_item_extra_keys_are_reported():
    r = convert_opensrm_v1_to_v2(_v1({"openslo": [{"path": "a.yaml", "inline": 1}]}))
    assert r.manifest["spec"]["slo"] == [{"$ref": "a.yaml"}]
    assert [(f.path, f.reason) for f in r.unmapped] == [
        ("spec.slos.openslo[0].inline", "not part of the opensrm/v1 schema")]


@pytest.mark.parametrize("name", ["rps-svc", "latency-svc"])
def test_other_type_hint_words_in_service_name_raise(name):
    with pytest.raises(OpenSRMV1MigrationError, match=f"{name}-availability"):
        convert_opensrm_v1_to_v2(_v1({"availability": {"target": 0.999}}, name=name))


# Literal reasons, checked against the pnmv design §4 table. Each fixture is
# shaped by its v1 schema definition: ThroughputSLO {target, unit},
# SuccessRateSLO {target}, ProcessingTimeSLO {target, p50|p90|p95|p99},
# LagSLO {max_seconds, target}, ScheduleAdherenceSLO {max_delay},
# DataFreshnessSLO {max_age}, ReplicationLagSLO {max_lag}, custom [{name, target}].
@pytest.mark.parametrize("key, cfg, reason", [
    ("throughput", {"target": 1000, "unit": "rps"},
     "v1 declares a rate (e.g. 1000 rps); an OpenSLO objective needs a ratio target (AUTHORING §7.3)"),
    ("success_rate", {"target": 0.995}, "NthLayer's v2 parser cannot infer this SLO's type"),
    ("processing_time", {"target": 0.95, "p95": "5m"}, "NthLayer's v2 parser cannot infer this SLO's type"),
    ("lag", {"max_seconds": 60, "target": 0.99}, "NthLayer's v2 parser cannot infer this SLO's type"),
    ("schedule_adherence", {"max_delay": "10m"},
     "maximum-only SLO (max_delay) with no target; an OpenSLO objective needs one"),
    ("data_freshness", {"max_age": "2h"},
     "maximum-only SLO (max_age) with no target; an OpenSLO objective needs one"),
    ("replication_lag", {"max_lag": "1s"},
     "maximum-only SLO (max_lag) with no target; an OpenSLO objective needs one"),
    ("custom", [{"name": "queue-depth", "target": 100}], "custom SLOs have no v2 mapping"),
])
def test_unconvertible_reasons_are_the_specific_ones(key, cfg, reason):
    r = convert_opensrm_v1_to_v2(_v1({key: cfg}, stype="batch"))
    assert [(f.path, f.reason) for f in r.unmapped] == [(f"spec.slos.{key}", reason)]


def test_keys_after_target_in_a_classical_slo_are_all_reported_in_order():
    r = convert_opensrm_v1_to_v2(_v1({"latency": {"target": 0.99, "p99": "200ms", "bogus": 1}}))
    assert [(f.path, f.reason) for f in r.unmapped] == [
        ("spec.slos.latency.p99", _PERCENTILE_WHY),
        ("spec.slos.latency.bogus", "not part of the opensrm/v1 schema"),
    ]


def test_display_name_does_not_round_a_near_one_target_up_to_100_percent():
    r = convert_opensrm_v1_to_v2(_v1({"availability": {"target": 0.9999999}}))
    assert _slo(r, "availability")["spec"]["objectives"][0]["displayName"].startswith("99.99999% ")


def test_error_rate_complement_is_exact_for_an_inexact_float():
    # 1 - 0.07 is 0.9299999999999999 in binary floating point.
    r = convert_opensrm_v1_to_v2(_v1({"error_rate": {"target": 0.07}}))
    assert _slo(r, "error-rate")["spec"]["objectives"][0]["target"] == 0.93


@pytest.mark.parametrize("key, target, objective", [
    ("error_rate", 1.0, 0.0), ("error_rate", 0, 1.0),
    ("availability", 0, 0.0), ("availability", 1.0, 1.0),
])
def test_ratio_edges_convert_and_parse(key, target, objective):
    r = convert_opensrm_v1_to_v2(_v1({key: {"target": target}}))
    assert _slo(r, key.replace("_", "-"))["spec"]["objectives"][0]["target"] == objective
    parse_opensrm_v2(r.manifest)
